#!/usr/bin/env python3
"""BindingDB diagnostic retrieve-then-rerank. Not a DTI SOTA run.

Stage-1 vehicles (all weak on purpose):
  - popularity: train ligand frequency
  - knn3mer: protein 3-mer kNN (k=10), transfer neighbour ligands (same rule as Table 4)
  - cosine: protein 3-mer (PCA) linearly mapped to ligand fingerprint space, cosine

Ligand fingerprint: RDKit Morgan r=2, 2048 bits if rdkit is present; otherwise a
stable hashed SMILES n-gram (labelled as such, not claimed as Morgan).

Stage-2: ligand popularity on the T=20 shortlist, fused with Stage-1 at α=0.7.
Optional sklearn GBM if installed; default is the non-learning popularity channel.

Protocols: pair-subset vs fixed-gallery. Split is by target, never by pair.
T=20, K=10, α=0.7 match the main text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
T_DEFAULT = 20
K_DEFAULT = 10
ALPHA = 0.7
K_NEIGH = 10
PCA_DIM = 128


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def kmer_vec(seq: str, k: int = 3) -> np.ndarray:
    seq = "".join(c for c in (seq or "").upper() if c in AA)[:1024]
    dim = len(AA) ** k
    v = np.zeros(dim, dtype=np.float64)
    if len(seq) < k:
        return v
    index = {a: i for i, a in enumerate(AA)}
    for i in range(len(seq) - k + 1):
        trip = seq[i : i + k]
        if any(c not in index for c in trip):
            continue
        idx = 0
        for c in trip:
            idx = idx * len(AA) + index[c]
        v[idx] += 1.0
    return l2norm(v, axis=0)


def hashed_smiles_fp(smi: str, n_bits: int = 2048, ngram: int = 4) -> np.ndarray:
    v = np.zeros(n_bits, dtype=np.float64)
    s = smi or ""
    if len(s) < ngram:
        key = hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()
        v[int(key, 16) % n_bits] = 1.0
        return l2norm(v, axis=0)
    for i in range(len(s) - ngram + 1):
        h = hashlib.md5(s[i : i + ngram].encode("utf-8", errors="ignore")).hexdigest()
        v[int(h, 16) % n_bits] += 1.0
    return l2norm(v, axis=0)


def morgan_or_hash(ligands: list[str]) -> tuple[np.ndarray, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        fps = np.stack([hashed_smiles_fp(s) for s in ligands], axis=0)
        return fps, "hashed_smiles_4gram_2048"
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    n_fail = 0
    for smi in ligands:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            rows.append(hashed_smiles_fp(smi))
            continue
        arr = np.zeros(2048, dtype=np.float64)
        fp = gen.GetFingerprint(mol)
        arr[list(fp.GetOnBits())] = 1.0
        rows.append(l2norm(arr, axis=0))
    tag = "morgan_r2_2048"
    if n_fail:
        tag += f"_hashfallback_{n_fail}"
    return np.stack(rows, axis=0), tag


def pca_fit_transform(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    xc = x - mu
    # economy SVD; x is n_targets x d with n << d
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    dim = min(dim, vt.shape[0])
    w = vt[:dim].T
    return (xc @ w), w, mu


def pca_apply(x: np.ndarray, w: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (x - mu) @ w


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    d = x.shape[1]
    xtx = x.T @ x + lam * np.eye(d)
    return np.linalg.solve(xtx, x.T @ y)


def ranks_from_scores(scores: np.ndarray, true_js: np.ndarray) -> np.ndarray:
    """scores: [n_gallery], true_js: ligand indices into this gallery vector."""
    order = np.argsort(-scores, kind="mergesort")
    pos = np.empty(len(scores), dtype=np.int32)
    pos[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return pos[true_js]


def fusion_rerank_rank(
    shortlist: np.ndarray,
    stage1: np.ndarray,
    rerank: np.ndarray,
    true_j: int,
    alpha: float,
) -> int:
    if true_j not in shortlist:
        return len(shortlist) + 1
    s1 = minmax(stage1[shortlist].astype(np.float64))
    s2 = minmax(rerank[shortlist].astype(np.float64))
    fused = alpha * s1 + (1.0 - alpha) * s2
    order = shortlist[np.argsort(-fused, kind="mergesort")]
    return int(np.where(order == true_j)[0][0]) + 1


def metrics_block(retr_rank: np.ndarray, rerank_rank: np.ndarray, T: int, K: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partner_rank_eval" / "src"))
    from partner_rank_eval.metrics import QueryRecord, metrics_from_ranks

    recs = []
    for i, (rr, rk) in enumerate(zip(retr_rank, rerank_rank)):
        recs.append(
            QueryRecord(
                query_id=str(i),
                true_id=str(i),
                retrieval_rank=int(rr),
                rerank_rank=int(rk) if rr <= T else None,
                in_gallery=True,
            )
        )
    m = metrics_from_ranks(recs, T=T, K=K)
    m["recall@100"] = float(np.mean(retr_rank <= 100))
    m["recall@20"] = float(np.mean(retr_rank <= T))
    return m


def evaluate_arm(
    name: str,
    stage1: np.ndarray,
    pop: np.ndarray,
    queries: list[tuple[int, int]],
    partners_idx: list[list[int]],
    n_ligands: int,
    T: int,
    K: int,
    alpha: float,
) -> dict:
    """stage1, pop: [n_targets, n_ligands]. queries: (target_row, ligand_j)."""
    by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for qi, (ti, lj) in enumerate(queries):
        by_t[ti].append((qi, lj))
    out = {}
    for proto in ("fixed_gallery", "pair_subset"):
        retr = np.empty(len(queries), dtype=np.int32)
        rer = np.empty(len(queries), dtype=np.int32)
        gal_sizes = np.empty(len(queries), dtype=np.int32)
        for ti, items in by_t.items():
            if proto == "fixed_gallery":
                sc = stage1[ti]
                rp = pop[ti]
                order = np.argsort(-sc, kind="mergesort")
                pos = np.empty(n_ligands, dtype=np.int32)
                pos[order] = np.arange(1, n_ligands + 1, dtype=np.int32)
                sl = order[: min(T, n_ligands)]
                gal_n = n_ligands
                for qi, lj in items:
                    rr = int(pos[lj])
                    retr[qi] = rr
                    rer[qi] = fusion_rerank_rank(sl, sc, rp, lj, alpha)
                    gal_sizes[qi] = gal_n
            else:
                g = np.asarray(partners_idx[ti], dtype=np.int32)
                sc = stage1[ti, g]
                rp = pop[ti, g]
                order = np.argsort(-sc, kind="mergesort")
                pos = np.empty(len(g), dtype=np.int32)
                pos[order] = np.arange(1, len(g) + 1, dtype=np.int32)
                sl = order[: min(T, len(g))]
                loc = {int(j): k for k, j in enumerate(g.tolist())}
                for qi, lj in items:
                    local = loc[lj]
                    retr[qi] = int(pos[local])
                    rer[qi] = fusion_rerank_rank(sl, sc, rp, local, alpha)
                    gal_sizes[qi] = int(len(g))
        block = metrics_block(retr, rer, T, K)
        block["median_gallery_size"] = float(np.median(gal_sizes))
        block["mean_gallery_size"] = float(np.mean(gal_sizes))
        block["arm"] = name
        block["protocol"] = proto
        out[proto] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.K > args.T:
        raise SystemExit("identity requires K <= T")

    data = json.loads(Path(args.evalset).read_text())
    ligands: list[str] = data["ligands"]
    target_ids = sorted(data["targets"])
    n_t, n_l = len(target_ids), len(ligands)
    split = np.array([data["targets"][u]["split"] for u in target_ids])
    seqs = [data["targets"][u]["seq"] for u in target_ids]
    partners_idx = [list(data["targets"][u]["ligand_idx"]) for u in target_ids]
    train_mask = split == "train"
    test_mask = split == "test"
    n_missing_seq = sum(1 for s in seqs if not s)
    print(
        f"targets={n_t} ligands={n_l} train={int(train_mask.sum())} "
        f"test={int(test_mask.sum())} missing_seq={n_missing_seq}",
        flush=True,
    )

    print("fingerprints...", flush=True)
    lig_fp, fp_tag = morgan_or_hash(ligands)
    lig_fp = np.nan_to_num(lig_fp, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  ligand_fp={fp_tag} shape={lig_fp.shape}", flush=True)

    print("protein 3-mer...", flush=True)
    prot = np.stack([kmer_vec(s) for s in seqs], axis=0)
    prot = np.nan_to_num(prot, nan=0.0, posinf=0.0, neginf=0.0)

    pop_vec = np.zeros(n_l, dtype=np.float64)
    for i, train in enumerate(train_mask):
        if not train:
            continue
        for j in partners_idx[i]:
            pop_vec[j] += 1.0
    pop = np.broadcast_to(pop_vec, (n_t, n_l)).copy()

    print("knn3mer scores...", flush=True)
    train_rows = np.where(train_mask)[0]
    H = prot[train_rows]
    knn = np.zeros((n_t, n_l), dtype=np.float64)
    for i in range(n_t):
        sims = np.nan_to_num(H @ prot[i], nan=-1e9, posinf=-1e9, neginf=-1e9)
        self = np.where(train_rows == i)[0]
        if len(self):
            sims[self[0]] = -1e9
        nn = np.argsort(-sims)[:K_NEIGH]
        acc = np.zeros(n_l, dtype=np.float64)
        for r in nn:
            w = float(sims[r])
            if w <= 0:
                continue
            src = int(train_rows[r])
            acc[partners_idx[src]] += w
        knn[i] = acc + 1e-9 * pop_vec

    print("cosine projector...", flush=True)
    Xtr = prot[train_rows]
    xp, w_pca, mu = pca_fit_transform(Xtr, PCA_DIM)
    ytr = np.stack(
        [
            l2norm(lig_fp[partners_idx[i]].mean(axis=0), axis=0)
            if partners_idx[i]
            else np.zeros(lig_fp.shape[1])
            for i in train_rows
        ],
        axis=0,
    )
    B = ridge_fit(xp, ytr, lam=1.0)
    mapped = np.nan_to_num(pca_apply(prot, w_pca, mu) @ B, nan=0.0)
    prot_map = l2norm(mapped, axis=1)
    cosine = np.nan_to_num(prot_map @ lig_fp.T, nan=0.0)

    test_rows = np.where(test_mask)[0]
    pair_queries: list[tuple[int, int]] = []
    ge2_queries: list[tuple[int, int]] = []
    rng = np.random.default_rng(args.seed)
    target_queries: list[tuple[int, int]] = []
    for i in test_rows:
        js = partners_idx[i]
        for j in js:
            pair_queries.append((i, j))
        if len(js) >= 2:
            for j in js:
                ge2_queries.append((i, j))
        # one designated true partner per target (not all actives as queries)
        target_queries.append((i, int(js[int(rng.integers(0, len(js)))])))
    print(
        f"pair queries={len(pair_queries)} ge2={len(ge2_queries)} "
        f"target queries={len(target_queries)}",
        flush=True,
    )

    arms = {
        "popularity": pop,
        "knn3mer_k10": knn,
        "cosine_3mer_to_fp": cosine,
    }
    report = {
        "T": args.T,
        "K": args.K,
        "alpha": args.alpha,
        "k_neighbors": K_NEIGH,
        "ligand_fingerprint": fp_tag,
        "protein_encoder": "aa_3mer_l2 (Table 4 knn rule; ESM is a drop-in later)",
        "reranker": "train-ligand-popularity on T-shortlist, fused at alpha",
        "split": "by target, never random pair",
        "n_train_targets": int(train_mask.sum()),
        "n_test_targets": int(test_mask.sum()),
        "n_gallery_fixed": n_l,
        "n_targets_missing_seq": n_missing_seq,
        "model_role": "diagnostic vehicle; not a DTI SOTA claim",
        "protocol_note": (
            "pair-subset gallery = known ligands of the query target. "
            "If every gallery member is also used as a labelled query, "
            "Recall@T = mean min(T, n_partners)/n_partners and is model-independent. "
            "Target-level queries (one designated true ligand per target) are the "
            "valid pair-subset vs fixed-gallery contrast. Pair-level queries are "
            "reported on fixed-gallery only."
        ),
        "fixed_gallery_pair_queries": {},
        "target_queries": {},
        "fixed_gallery_pair_queries_ge2": {},
    }
    for name, sc in arms.items():
        print(f"eval {name}...", flush=True)
        pair_blk = evaluate_arm(
            name, sc, pop, pair_queries, partners_idx, n_l, args.T, args.K, args.alpha
        )
        tgt_blk = evaluate_arm(
            name, sc, pop, target_queries, partners_idx, n_l, args.T, args.K, args.alpha
        )
        ge2_blk = evaluate_arm(
            name, sc, pop, ge2_queries, partners_idx, n_l, args.T, args.K, args.alpha
        )
        # pair-level pair-subset is a gallery-size identity; do not headline it
        report["fixed_gallery_pair_queries"][name] = pair_blk["fixed_gallery"]
        report["fixed_gallery_pair_queries_ge2"][name] = ge2_blk["fixed_gallery"]
        report["target_queries"][name] = {
            "fixed_gallery": tgt_blk["fixed_gallery"],
            "pair_subset": tgt_blk["pair_subset"],
        }
        fg = tgt_blk["fixed_gallery"]
        ps = tgt_blk["pair_subset"]
        print(
            f"  {name:22s} target/pair-subset  R@20={ps['recall@20']:.4f} "
            f"Hit@10={ps['end_to_end_hit@K']:.4f} oracle={ps['oracle@K']:.4f} "
            f"n={ps['n_oracle_subset']} err={ps['identity_abs_error']:.2e} "
            f"gal~{ps['median_gallery_size']:.0f}",
            flush=True,
        )
        print(
            f"  {name:22s} target/fixed       R@20={fg['recall@20']:.4f} "
            f"R@100={fg['recall@100']:.4f} Hit@10={fg['end_to_end_hit@K']:.4f} "
            f"oracle={fg['oracle@K']:.4f} n={fg['n_oracle_subset']} "
            f"err={fg['identity_abs_error']:.2e} gal~{fg['median_gallery_size']:.0f}",
            flush=True,
        )
        print(
            f"  {name:22s} pair/fixed         R@20={pair_blk['fixed_gallery']['recall@20']:.4f} "
            f"R@100={pair_blk['fixed_gallery']['recall@100']:.4f} "
            f"Hit@10={pair_blk['fixed_gallery']['end_to_end_hit@K']:.4f} "
            f"oracle={pair_blk['fixed_gallery']['oracle@K']:.4f} "
            f"n={pair_blk['fixed_gallery']['n_oracle_subset']} "
            f"err={pair_blk['fixed_gallery']['identity_abs_error']:.2e}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute pair-subset gallery saturation + Hit@1 / MRR / mean percentile rank.

HVIDB/IntAct: label-informed galleries = known positive partners of the query human
in the evaluation pair file. Ranking within G_q uses train-set virus popularity
(non-learning diagnostic; comparable in spirit to Table 3/4 vehicles). Main-text
fusion Hit@10 under the same galleries remains the S5 headline.

BindingDB: reuses the diagnostic Stage-1 arms from bindingdb_weak_retrieve_rerank.py
and reports fusion end-to-end ranks within pair-subset galleries.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BIB = Path(__file__).resolve().parents[2]
REPO = BIB / "_git_push_work" / "repo"
SECOND = BIB / "_second_source"
OUT = BIB / "transfer_bioinformatics" / "analysis" / "pair_subset_saturation.json"
SCRIPTS = BIB / "scripts"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(BIB / "partner_rank_eval" / "src"))


def load_pos(path: Path) -> list[dict]:
    pairs = pickle.load(path.open("rb"))
    return [p for p in pairs if int(p.get("interaction", 0)) == 1]


def summarize_ranks(ranks: np.ndarray, gal_sizes: np.ndarray) -> dict:
    ranks = ranks.astype(float)
    g = gal_sizes.astype(float)
    return {
        "n_queries": int(len(ranks)),
        "gallery_median": float(np.median(g)),
        "gallery_iqr": [float(x) for x in np.percentile(g, [25, 75])],
        "gallery_min": int(g.min()),
        "gallery_max": int(g.max()),
        "p_gallery_leq_10": float(np.mean(g <= 10)),
        "hit@1": float(np.mean(ranks <= 1)),
        "hit@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        # Top-oriented percentile: 1 = best, (G-rank+1)/G
        "mean_percentile_rank": float(np.mean((g - ranks + 1.0) / g)),
    }


def eval_hvidb_style(
    pairs_pkl: Path,
    train_freq: Counter,
    label: str,
) -> dict:
    pos = load_pos(pairs_pkl)
    by_h: dict[str, set[str]] = defaultdict(set)
    for p in pos:
        by_h[p["human_id"]].add(p["virus_id"])
    ranks = []
    sizes = []
    for p in pos:
        h, v = p["human_id"], p["virus_id"]
        gallery = list(by_h[h])
        order = sorted(gallery, key=lambda x: (-train_freq.get(x, 0), x))
        ranks.append(order.index(v) + 1)
        sizes.append(len(gallery))
    m = summarize_ranks(np.asarray(ranks), np.asarray(sizes))
    m["dataset"] = label
    m["ranking"] = "train_virus_popularity_within_label_informed_gallery"
    return m


def eval_bindingdb() -> dict:
    import bindingdb_weak_retrieve_rerank as bdb

    data = json.loads((SECOND / "evalset.json").read_text())
    ligands: list[str] = data["ligands"]
    target_ids = sorted(data["targets"])
    n_t, n_l = len(target_ids), len(ligands)
    split = np.array([data["targets"][u]["split"] for u in target_ids])
    seqs = [data["targets"][u]["seq"] for u in target_ids]
    partners_idx = [list(data["targets"][u]["ligand_idx"]) for u in target_ids]
    train_mask = split == "train"
    test_mask = split == "test"

    lig_fp, fp_tag = bdb.morgan_or_hash(ligands)
    lig_fp = np.nan_to_num(lig_fp, nan=0.0, posinf=0.0, neginf=0.0)
    prot = np.stack([bdb.kmer_vec(s) for s in seqs], axis=0)
    prot = np.nan_to_num(prot, nan=0.0, posinf=0.0, neginf=0.0)

    pop_vec = np.zeros(n_l, dtype=np.float64)
    for i, train in enumerate(train_mask):
        if not train:
            continue
        for j in partners_idx[i]:
            pop_vec[j] += 1.0
    pop = np.broadcast_to(pop_vec, (n_t, n_l)).copy()

    train_rows = np.where(train_mask)[0]
    H = prot[train_rows]
    knn = np.zeros((n_t, n_l), dtype=np.float64)
    for i in range(n_t):
        sims = np.nan_to_num(H @ prot[i], nan=-1e9, posinf=-1e9, neginf=-1e9)
        self = np.where(train_rows == i)[0]
        if len(self):
            sims[self[0]] = -1e9
        nn = np.argsort(-sims)[: bdb.K_NEIGH]
        acc = np.zeros(n_l, dtype=np.float64)
        for r in nn:
            w = float(sims[r])
            if w <= 0:
                continue
            src = int(train_rows[r])
            acc[partners_idx[src]] += w
        knn[i] = acc + 1e-9 * pop_vec

    Xtr = prot[train_rows]
    xp, w_pca, mu = bdb.pca_fit_transform(Xtr, bdb.PCA_DIM)
    ytr = np.stack(
        [
            bdb.l2norm(lig_fp[partners_idx[i]].mean(axis=0), axis=0)
            if partners_idx[i]
            else np.zeros(lig_fp.shape[1])
            for i in train_rows
        ],
        axis=0,
    )
    B = bdb.ridge_fit(xp, ytr, lam=1.0)
    mapped = np.nan_to_num(bdb.pca_apply(prot, w_pca, mu) @ B, nan=0.0)
    prot_map = bdb.l2norm(mapped, axis=1)
    cosine = np.nan_to_num(prot_map @ lig_fp.T, nan=0.0)

    rng = np.random.default_rng(0)
    target_queries: list[tuple[int, int]] = []
    for i in np.where(test_mask)[0]:
        js = partners_idx[i]
        target_queries.append((i, int(js[int(rng.integers(0, len(js)))])))

    arms = {
        "popularity": pop,
        "knn3mer_k10": knn,
        "cosine_3mer_to_fp": cosine,
    }
    out_arms = {}
    for name, sc in arms.items():
        # replicate evaluate_arm pair_subset branch and collect fusion ranks
        by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for qi, (ti, lj) in enumerate(target_queries):
            by_t[ti].append((qi, lj))
        retr = np.empty(len(target_queries), dtype=np.int32)
        rer = np.empty(len(target_queries), dtype=np.int32)
        gal = np.empty(len(target_queries), dtype=np.int32)
        for ti, items in by_t.items():
            g = np.asarray(partners_idx[ti], dtype=np.int32)
            sc_g = sc[ti, g]
            rp = pop[ti, g]
            order = np.argsort(-sc_g, kind="mergesort")
            pos = np.empty(len(g), dtype=np.int32)
            pos[order] = np.arange(1, len(g) + 1, dtype=np.int32)
            sl = order[: min(20, len(g))]
            loc = {int(j): k for k, j in enumerate(g.tolist())}
            for qi, lj in items:
                local = loc[lj]
                retr[qi] = int(pos[local])
                rer[qi] = bdb.fusion_rerank_rank(sl, sc_g, rp, local, 0.7)
                gal[qi] = int(len(g))
        # Within pair-subset, T=20 covers almost all galleries; use fusion rank when retrieved
        use_rank = np.where(retr <= 20, rer, retr)
        m = summarize_ranks(use_rank, gal)
        m["arm"] = name
        m["ligand_fingerprint"] = fp_tag
        out_arms[name] = m
    return {
        "dataset": "BindingDB target-level (n=156)",
        "ranking": "diagnostic Stage-1 + popularity fusion (α=0.7) within pair-subset",
        "arms": out_arms,
        "headline_arm": "knn3mer_k10",
    }


def main() -> None:
    train_pkl = REPO / "data" / "processed" / "train_protein_pairs.pkl"
    if not train_pkl.exists() or train_pkl.stat().st_size < 300_000_000:
        raise SystemExit(f"need train pairs at {train_pkl}")
    train_pos = load_pos(train_pkl)
    train_freq = Counter(p["virus_id"] for p in train_pos)

    report = {
        "hvidb_test": eval_hvidb_style(
            REPO / "data" / "processed" / "test_protein_pairs.pkl",
            train_freq,
            "HVIDB held-out test",
        ),
        "intact": eval_hvidb_style(
            REPO / "data" / "cross_intact" / "processed" / "val_protein_pairs.pkl",
            train_freq,
            "IntAct cross-test",
        ),
        "bindingdb": eval_bindingdb(),
        "notes": {
            "label_informed_gallery": (
                "Query-specific candidate set constructed from the query's known "
                "positive interaction partners in the evaluation file."
            ),
            "hvidb_fusion_hit10_main_text": {
                "HVIDB": 0.913,
                "IntAct": 0.857,
                "source": "Supplementary Table S5 / protocol_comparison.json",
            },
            "hvidb_fusion_mrr_protocol_comparison": {
                "HVIDB_val_pair_subset": 0.6139,
                "IntAct_pair_subset": 0.5217,
                "note": "Existing fusion MRR under pair-subset; Hit@1/percentile newly computed via train-popularity diagnostic within the same galleries.",
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

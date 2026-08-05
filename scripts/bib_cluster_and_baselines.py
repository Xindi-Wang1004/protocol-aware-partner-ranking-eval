#!/usr/bin/env python3
"""Dump per-query hits, cluster bootstrap CIs, and extended non-learning baselines.

Outputs under analysis_results/:
  - constrained_2104_per_query_{dataset}.npz   (per-query vectors + ids)
  - constrained_2104_bootstrap_cluster_{dataset}.json
  - constrained_2104_sensitivity_baselines_extended_{dataset}.json
  - constrained_2104_recall_curves_baselines_hvidb_test.json  (for Figure 3)

Datasets: hvidb_test (default), intact, hvidb_val
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import types
from collections import Counter, defaultdict
from datetime import datetime
from importlib.machinery import SourcelessFileLoader
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/wangxindi/protein_interaction_model")
SEED = 42
N_BOOT = 2000
ALPHA = 0.7
TOP_K = 20
K_NEIGH = 10
DEVICE = "cuda:0"
AA = "ACDEFGHIKLMNPQRSTVWY"
RECALL_KS = [1, 5, 10, 20, 50, 100, 200, 500]

DATASETS = {
    "hvidb_test": {
        "pairs_pkl": "data/processed/test_protein_pairs.pkl",
        "train_pkl": "data/processed/train_protein_pairs.pkl",
        "retr_cache": "analysis_results/embed_cache/2104_retrieval_virus_1115_512.pt",
        "clf_cache": "analysis_results/embed_cache/2104_classifier_virus_1115_512.pt",
        "do_baselines": True,
    },
    "hvidb_val": {
        "pairs_pkl": "data/processed/val_protein_pairs.pkl",
        "train_pkl": "data/processed/train_protein_pairs.pkl",
        "retr_cache": "analysis_results/embed_cache/2104_retrieval_virus_1115_512.pt",
        "clf_cache": "analysis_results/embed_cache/2104_classifier_virus_1115_512.pt",
        "do_baselines": False,
    },
    "intact": {
        "pairs_pkl": "data/cross_intact/processed/val_protein_pairs.pkl",
        "train_pkl": "data/processed/train_protein_pairs.pkl",
        "retr_cache": "analysis_results/embed_cache/2104_retrieval_virus_1229_512.pt",
        "clf_cache": "analysis_results/embed_cache/2104_classifier_virus_1229_512.pt",
        "do_baselines": True,
    },
}


def load_pyc(name: str):
    pyc = ROOT / "scripts" / "__pycache__" / f"{name}.cpython-310.pyc"
    loader = SourcelessFileLoader(f"scripts.{name}", str(pyc))
    spec = importlib.util.spec_from_loader(f"scripts.{name}", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"scripts.{name}"] = mod
    loader.exec_module(mod)
    return mod


def kmer_vec(seq: str, k: int = 3) -> np.ndarray:
    seq = "".join(c for c in (seq or "").upper() if c in AA)[:512]
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
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def query_bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "unit": "query",
    }


def cluster_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: list[str],
    n_boot: int,
    seed: int,
    unit: str,
) -> dict:
    """Resample unique clusters with replacement; keep all queries in selected clusters."""
    rng = np.random.default_rng(seed)
    clusters = sorted(set(cluster_ids))
    c2idx = {c: [] for c in clusters}
    for i, c in enumerate(cluster_ids):
        c2idx[c].append(i)
    n_c = len(clusters)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, n_c, size=n_c)
        idxs = []
        for ci in chosen:
            idxs.extend(c2idx[clusters[int(ci)]])
        means[b] = values[np.asarray(idxs, dtype=np.int64)].mean() if idxs else np.nan
    lo, hi = np.percentile(means[~np.isnan(means)], [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_queries": int(len(values)),
        "n_clusters": int(n_c),
        "unit": unit,
    }


def fmt_cell(v: dict) -> str:
    return f"{100*v['mean']:.1f}% [{100*v['ci_low']:.1f}–{100*v['ci_high']:.1f}]"


def shortlist_fusion_hit(
    shortlist_idx: np.ndarray,
    base_scores: np.ndarray,
    h_clf_vec: np.ndarray,
    v_clf: np.ndarray,
    clf_model,
    device: str,
    v_true_j: int,
    alpha: float,
    minmax,
) -> tuple[float, int]:
    """Return (hit@10, rank) after α-fusion of baseline score + classifier on shortlist."""
    k = len(shortlist_idx)
    if k == 0:
        return 0.0, 10**9
    base_cand = base_scores[shortlist_idx]
    h_vec = torch.as_tensor(h_clf_vec, device=device).float()
    v_chunk = torch.as_tensor(v_clf[shortlist_idx], device=device).float()
    combined = torch.cat([h_vec.unsqueeze(0).expand(k, -1), v_chunk], dim=1)
    clf_cand = clf_model.classifier(combined).squeeze(-1).detach().cpu().numpy()
    fused = alpha * minmax(base_cand) + (1.0 - alpha) * minmax(clf_cand)
    order = np.argsort(-fused)
    reranked = shortlist_idx[order]
    try:
        rank = int(np.where(reranked == v_true_j)[0][0]) + 1
    except IndexError:
        rank = k + 1
    return (1.0 if rank <= 10 else 0.0), rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="hvidb_test")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--skip-direct", action="store_true")
    ap.add_argument("--with-replace", action="store_true", help="Also compute full-sequence replace-rerank (slow)")
    args = ap.parse_args()
    args.skip_replace = not args.with_replace
    cfg = DATASETS[args.dataset]

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    pkg = types.ModuleType("scripts")
    pkg.__path__ = [str(ROOT / "scripts")]
    sys.modules["scripts"] = pkg
    for dep in ("retrieval_eval_utils", "necessity_eval", "constrained_2104_eval"):
        m = load_pyc(dep)
        setattr(pkg, dep, m)
        if hasattr(m, "PROJECT_ROOT"):
            m.PROJECT_ROOT = ROOT

    ce = sys.modules["scripts.constrained_2104_eval"]
    ne = sys.modules["scripts.necessity_eval"]
    reu = sys.modules["scripts.retrieval_eval_utils"]
    minmax = ne._minmax_norm

    pairs = ne.load_pairs(cfg["pairs_pkl"])
    positive_pairs, query_human_seq, query_virus_seq = ne.build_queries_from_pairs(pairs)
    hvidb_ids = ce.load_hvidb_2104_ids(ce.HVIDB_TSV)
    seq_map = ce.build_sequence_map("data/SV.fasta", ce.DEFAULT_PAIR_PKLS)
    gallery, gstats = ce.build_hvidb_2104_gallery(
        hvidb_ids, seq_map, 512, positive_pairs, query_virus_seq
    )
    virus_ids = sorted(gallery.keys())
    human_ids = sorted({h for h, _ in positive_pairs})
    pool_human = {h: query_human_seq[h][:512] for h in human_ids}
    pool_virus = {v: gallery[v][:512] for v in virus_ids}
    queries = sorted(positive_pairs)
    n_q = len(queries)
    print(
        f"[{args.dataset}] q={n_q} humans={len(human_ids)} gallery={len(virus_ids)} "
        f"gstats={gstats}",
        flush=True,
    )

    retr_cache = torch.load(ROOT / cfg["retr_cache"], map_location="cpu", weights_only=False)
    clf_cache = torch.load(ROOT / cfg["clf_cache"], map_location="cpu", weights_only=False)
    assert retr_cache["ids"] == virus_ids
    assert clf_cache["ids"] == virus_ids
    v_emb = retr_cache["emb"]
    v_clf = clf_cache["emb"]
    if isinstance(v_emb, torch.Tensor):
        v_emb = v_emb.numpy()
    if isinstance(v_clf, torch.Tensor):
        v_clf = v_clf.numpy()

    device = args.device if torch.cuda.is_available() else "cpu"
    retr_model, _ = reu.load_retrieval_model(
        "retrieval_task/gnn_tau007_longrun/best_model.pt",
        device=device,
        model_type="esm3_lora_gnn",
    )
    h_ids, h_emb = ne.encode_retrieval_pool(
        retr_model, human_ids, pool_human, device, 8, "human"
    )
    assert list(h_ids) == human_ids
    h_mat = h_emb.detach().cpu().numpy() if isinstance(h_emb, torch.Tensor) else np.asarray(h_emb)
    sim = h_mat @ v_emb.T

    clf_model, _ = reu.load_classification_model(
        "classification_task/esm3_frozen/best_model.pt", device=device
    )
    h_clf_ids, h_clf_emb = ne.precompute_classifier_embeddings(
        clf_model, human_ids, pool_human, 8
    )
    assert list(h_clf_ids) == human_ids
    h_clf = (
        h_clf_emb.detach().cpu().numpy()
        if isinstance(h_clf_emb, torch.Tensor)
        else np.asarray(h_clf_emb)
    )

    h2i = {h: i for i, h in enumerate(human_ids)}
    v2j = {v: j for j, v in enumerate(virus_ids)}
    human_ids_q = [h for h, _ in queries]
    virus_ids_q = [v for _, v in queries]

    recall20 = np.zeros(n_q, dtype=np.float64)
    recall100 = np.zeros(n_q, dtype=np.float64)
    retr_hit10 = np.zeros(n_q, dtype=np.float64)
    fusion_hit10 = np.zeros(n_q, dtype=np.float64)
    direct_hit10 = np.zeros(n_q, dtype=np.float64)
    replace_hit10 = np.zeros(n_q, dtype=np.float64)
    oracle_mask = np.zeros(n_q, dtype=bool)
    rank_retr_arr = np.zeros(n_q, dtype=np.int32)

    clf_model.eval()
    virus_chunk = 256
    with torch.no_grad():
        for qi, (h, v_true) in enumerate(queries):
            i = h2i[h]
            j = v2j[v_true]
            retr_scores = sim[i]
            order = np.argsort(-retr_scores)
            rank_retr = int(np.where(order == j)[0][0]) + 1
            rank_retr_arr[qi] = rank_retr
            recall20[qi] = 1.0 if rank_retr <= 20 else 0.0
            recall100[qi] = 1.0 if rank_retr <= 100 else 0.0
            retr_hit10[qi] = 1.0 if rank_retr <= 10 else 0.0
            oracle_mask[qi] = rank_retr <= TOP_K

            k = min(TOP_K, len(virus_ids))
            top_idx = order[:k]
            hit, _ = shortlist_fusion_hit(
                top_idx, retr_scores, h_clf[i], v_clf, clf_model, device, j, ALPHA, minmax
            )
            fusion_hit10[qi] = hit

            if not args.skip_direct:
                h_vec = torch.as_tensor(h_clf[i], device=device).float()
                all_scores = []
                for start in range(0, len(virus_ids), virus_chunk):
                    end = min(start + virus_chunk, len(virus_ids))
                    v_chunk = torch.as_tensor(v_clf[start:end], device=device).float()
                    n = v_chunk.shape[0]
                    combined = torch.cat(
                        [h_vec.unsqueeze(0).expand(n, -1), v_chunk], dim=1
                    )
                    logits = clf_model.classifier(combined).squeeze(-1)
                    all_scores.append(logits.detach().cpu())
                scores = torch.cat(all_scores).numpy()
                d_order = np.argsort(-scores)
                rank_direct = int(np.where(d_order == j)[0][0]) + 1
                direct_hit10[qi] = 1.0 if rank_direct <= 10 else 0.0

            if (qi + 1) % 200 == 0:
                print(
                    f"  main {qi+1}/{n_q} R@100={recall100[:qi+1].mean():.4f} "
                    f"fus@10={fusion_hit10[:qi+1].mean():.4f}",
                    flush=True,
                )

    if not args.skip_replace:
        print(f"[{args.dataset}] replace-rerank...", flush=True)
        clf_batch_size = 8
        with torch.no_grad():
            for qi, (h, v_true) in enumerate(queries):
                i = h2i[h]
                scores = sim[i]
                k = min(TOP_K, len(virus_ids))
                top_idx = np.argsort(-scores)[:k]
                candidates = [virus_ids[int(j)] for j in top_idx]
                h_seq = pool_human[h]
                pairs_h = [h_seq] * len(candidates)
                pairs_v = [pool_virus[vc] for vc in candidates]
                clf_scores = []
                for start in range(0, len(pairs_h), clf_batch_size):
                    bh = pairs_h[start : start + clf_batch_size]
                    bv = pairs_v[start : start + clf_batch_size]
                    logits = clf_model(bh, bv)
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    clf_scores.extend(logits.detach().cpu().numpy().tolist())
                rerank_order = np.argsort(-np.asarray(clf_scores))
                reranked = [candidates[idx] for idx in rerank_order]
                try:
                    rank = reranked.index(v_true) + 1
                except ValueError:
                    rank = k + 1
                replace_hit10[qi] = 1.0 if rank <= 10 else 0.0

    oracle_fusion = fusion_hit10[oracle_mask]
    oracle_humans = [human_ids_q[i] for i, m in enumerate(oracle_mask) if m]
    oracle_viruses = [virus_ids_q[i] for i, m in enumerate(oracle_mask) if m]

    # ---------- baselines (non-learning retrievers + Stage-2 fusion) ----------
    baseline_payload = None
    recall_curve_payload = None
    if cfg.get("do_baselines"):
        print(f"[{args.dataset}] computing popularity/kNN baselines...", flush=True)
        train = pickle.load(open(ROOT / cfg["train_pkl"], "rb"))
        tr_pos = [p for p in train if int(p["interaction"]) == 1]
        pop = Counter()
        train_partners: dict[str, set[str]] = defaultdict(set)
        train_human_seq: dict[str, str] = {}
        for p in tr_pos:
            hh, vv = p["human_id"], p["virus_id"]
            train_human_seq.setdefault(hh, p["human_seq"])
            if vv in v2j:
                pop[vv] += 1
                train_partners[hh].add(vv)
        pop_scores = np.array([float(pop[v]) for v in virus_ids], dtype=np.float64)

        train_h_ids = sorted(train_human_seq)
        H = np.stack([kmer_vec(train_human_seq[hh]) for hh in train_h_ids], axis=0)
        h2row = {hh: i for i, hh in enumerate(train_h_ids)}

        def empty_metric():
            return {
                "recall@20": np.zeros(n_q),
                "recall@100": np.zeros(n_q),
                "hit@10_retrieval": np.zeros(n_q),
                "fusion_hit@10": np.zeros(n_q),
                "oracle_mask": np.zeros(n_q, dtype=bool),
                "ranks": np.zeros(n_q, dtype=np.int32),
            }

        pop_m = empty_metric()
        knn_m = empty_metric()
        curve = {
            "main_stage1": {k: 0.0 for k in RECALL_KS},
            "popularity": {k: 0.0 for k in RECALL_KS},
            "knn": {k: 0.0 for k in RECALL_KS},
            "random_expectation": {k: float(min(k, len(virus_ids)) / len(virus_ids)) for k in RECALL_KS},
        }
        # accumulate main recall curve from rank_retr_arr
        for k in RECALL_KS:
            curve["main_stage1"][k] = float(np.mean(rank_retr_arr <= k))

        with torch.no_grad():
            for qi, (h, v_true) in enumerate(queries):
                j = v2j[v_true]
                hseq = pool_human[h]

                # popularity
                pop_order = np.argsort(-pop_scores)
                pop_rank = int(np.where(pop_order == j)[0][0]) + 1
                pop_m["ranks"][qi] = pop_rank
                pop_m["recall@20"][qi] = 1.0 if pop_rank <= 20 else 0.0
                pop_m["recall@100"][qi] = 1.0 if pop_rank <= 100 else 0.0
                pop_m["hit@10_retrieval"][qi] = 1.0 if pop_rank <= 10 else 0.0
                pop_m["oracle_mask"][qi] = pop_rank <= TOP_K
                top_pop = pop_order[: min(TOP_K, len(virus_ids))]
                hit, _ = shortlist_fusion_hit(
                    top_pop, pop_scores, h_clf[h2i[h]], v_clf, clf_model, device, j, ALPHA, minmax
                )
                pop_m["fusion_hit@10"][qi] = hit

                # knn
                qv = kmer_vec(hseq)
                sims = H @ qv
                if h in h2row:
                    sims[h2row[h]] = -1e9
                nn = np.argsort(-sims)[:K_NEIGH]
                knn_scores = np.zeros(len(virus_ids), dtype=np.float64)
                for ridx in nn:
                    nh = train_h_ids[int(ridx)]
                    w = float(sims[ridx])
                    if w <= 0:
                        continue
                    for vv in train_partners.get(nh, ()):
                        knn_scores[v2j[vv]] += w
                knn_scores = knn_scores + 1e-9 * pop_scores
                knn_order = np.argsort(-knn_scores)
                knn_rank = int(np.where(knn_order == j)[0][0]) + 1
                knn_m["ranks"][qi] = knn_rank
                knn_m["recall@20"][qi] = 1.0 if knn_rank <= 20 else 0.0
                knn_m["recall@100"][qi] = 1.0 if knn_rank <= 100 else 0.0
                knn_m["hit@10_retrieval"][qi] = 1.0 if knn_rank <= 10 else 0.0
                knn_m["oracle_mask"][qi] = knn_rank <= TOP_K
                top_knn = knn_order[: min(TOP_K, len(virus_ids))]
                hit2, _ = shortlist_fusion_hit(
                    top_knn, knn_scores, h_clf[h2i[h]], v_clf, clf_model, device, j, ALPHA, minmax
                )
                knn_m["fusion_hit@10"][qi] = hit2

                if (qi + 1) % 200 == 0:
                    print(f"  baseline {qi+1}/{n_q}", flush=True)

        for k in RECALL_KS:
            curve["popularity"][k] = float(np.mean(pop_m["ranks"] <= k))
            curve["knn"][k] = float(np.mean(knn_m["ranks"] <= k))

        def pack_baseline(name, m):
            om = m["oracle_mask"]
            oracle_vals = m["fusion_hit@10"][om]
            o_h = [human_ids_q[i] for i, x in enumerate(om) if x]
            return {
                "name": name,
                "recall@20": float(m["recall@20"].mean()),
                "recall@100": float(m["recall@100"].mean()),
                "retrieval_hit@10": float(m["hit@10_retrieval"].mean()),
                "end_to_end_fusion_hit@10_T20": float(m["fusion_hit@10"].mean()),
                "oracle@20_n": int(om.sum()),
                "oracle@20_unique_humans": int(len(set(o_h))),
                "oracle@20_fusion_hit@10": float(oracle_vals.mean()) if om.any() else None,
                "cluster_human_ci": {
                    "recall@100": cluster_bootstrap_ci(m["recall@100"], human_ids_q, N_BOOT, SEED, "human"),
                    "fusion_hit@10": cluster_bootstrap_ci(m["fusion_hit@10"], human_ids_q, N_BOOT, SEED, "human"),
                },
            }

        baseline_payload = {
            "evaluated_at": datetime.now().isoformat(),
            "dataset": args.dataset,
            "protocol": "fixed_gallery_same_as_table3",
            "gallery_n": len(virus_ids),
            "n_queries": n_q,
            "fusion_alpha": ALPHA,
            "rerank_T": TOP_K,
            "main_stage1_reference": {
                "recall@20": float(recall20.mean()),
                "recall@100": float(recall100.mean()),
                "retrieval_hit@10": float(retr_hit10.mean()),
                "end_to_end_fusion_hit@10_T20": float(fusion_hit10.mean()),
                "oracle@20_n": int(oracle_mask.sum()),
                "oracle@20_unique_humans": int(len(set(oracle_humans))),
                "oracle@20_unique_viruses": int(len(set(oracle_viruses))),
                "oracle@20_fusion_hit@10": float(oracle_fusion.mean()) if len(oracle_fusion) else None,
            },
            "baselines": {
                "virus_popularity_train": pack_baseline("virus_popularity_train", pop_m),
                "knn_partner_transfer_3mer_k10": pack_baseline("knn_partner_transfer_3mer_k10", knn_m)
                | {"k_neighbors": K_NEIGH, "exclude_same_human_id": True},
            },
            "notes": [
                "Non-learning retrievers scored over the identical fixed retrievable gallery.",
                "End-to-end fusion Hit@10 uses each baseline's top-20 shortlist then Stage-2 α=0.7 fusion of baseline score + classifier head.",
                "Diagnostic comparators, not optimised SOTA systems.",
            ],
        }
        recall_curve_payload = {
            "evaluated_at": datetime.now().isoformat(),
            "dataset": args.dataset,
            "gallery_n": len(virus_ids),
            "k_list": RECALL_KS,
            "recall_at_k": curve,
        }

    # ---------- bootstrap packages ----------
    metric_vecs = {
        "recall@20": recall20,
        "recall@100": recall100,
        "retrieval_hit@10": retr_hit10,
        "fusion_hit@10": fusion_hit10,
        "oracle@20_fusion_hit@10": oracle_fusion,
    }
    if not args.skip_direct:
        metric_vecs["direct_full_gallery_hit@10"] = direct_hit10
    if not args.skip_replace:
        metric_vecs["replace_rerank_hit@10"] = replace_hit10

    def package_metric(name, values, humans=None, viruses=None):
        humans = humans if humans is not None else human_ids_q
        viruses = viruses if viruses is not None else virus_ids_q
        # oracle metric uses subset ids
        if name.startswith("oracle@"):
            humans = oracle_humans
            viruses = oracle_viruses
        return {
            "query_bootstrap": query_bootstrap_ci(values, N_BOOT, SEED),
            "human_cluster_bootstrap": cluster_bootstrap_ci(values, humans, N_BOOT, SEED, "human"),
            "virus_cluster_bootstrap": cluster_bootstrap_ci(values, viruses, N_BOOT, SEED, "virus"),
            "table_string_human_cluster": fmt_cell(
                cluster_bootstrap_ci(values, humans, N_BOOT, SEED, "human")
            ),
        }

    metrics_out = {k: package_metric(k, v) for k, v in metric_vecs.items()}

    cluster_payload = {
        "evaluated_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "pairs_pkl": cfg["pairs_pkl"],
        "n_boot": N_BOOT,
        "seed": SEED,
        "n_queries": n_q,
        "n_unique_humans": len(set(human_ids_q)),
        "n_unique_viruses": len(set(virus_ids_q)),
        "oracle@20_n": int(oracle_mask.sum()),
        "oracle@20_unique_humans": int(len(set(oracle_humans))),
        "oracle@20_unique_viruses": int(len(set(oracle_viruses))),
        "repeated_human_fraction": float(
            1.0 - (len(set(human_ids_q)) / n_q)
        ),
        "metrics": metrics_out,
        "primary_reporting": "human_cluster_bootstrap",
        "notes": [
            "Primary CIs: resample unique human proteins with replacement; include all queries of selected humans.",
            "Virus-cluster CIs provided as sensitivity.",
            "Oracle@20 CIs are conditional on the Stage-1 top-20 subset.",
        ],
    }

    out_dir = ROOT / "analysis_results"
    pq_path = out_dir / f"constrained_2104_per_query_{args.dataset}.npz"
    np.savez_compressed(
        pq_path,
        human_ids=np.asarray(human_ids_q),
        virus_ids=np.asarray(virus_ids_q),
        recall20=recall20,
        recall100=recall100,
        retr_hit10=retr_hit10,
        fusion_hit10=fusion_hit10,
        direct_hit10=direct_hit10,
        replace_hit10=replace_hit10,
        oracle_mask=oracle_mask.astype(np.uint8),
        rank_retr=rank_retr_arr,
    )
    print("Wrote", pq_path, flush=True)

    cluster_path = out_dir / f"constrained_2104_bootstrap_cluster_{args.dataset}.json"
    cluster_path.write_text(json.dumps(cluster_payload, indent=2))
    print("Wrote", cluster_path, flush=True)
    print(json.dumps({k: v["table_string_human_cluster"] for k, v in metrics_out.items()}, indent=2))

    if baseline_payload is not None:
        bp = out_dir / f"constrained_2104_sensitivity_baselines_extended_{args.dataset}.json"
        bp.write_text(json.dumps(baseline_payload, indent=2))
        print("Wrote", bp, flush=True)
        print(json.dumps(baseline_payload["baselines"], indent=2), flush=True)
    if recall_curve_payload is not None:
        rc = out_dir / f"constrained_2104_recall_curves_baselines_{args.dataset}.json"
        rc.write_text(json.dumps(recall_curve_payload, indent=2))
        print("Wrote", rc, flush=True)

    # pair-subset candidate-set sizes (HVIDB test / intact only)
    if args.dataset in ("hvidb_test", "intact"):
        all_pairs = pickle.load(open(ROOT / cfg["pairs_pkl"], "rb"))
        viruses_in_file = sorted({p["virus_id"] for p in all_pairs})
        # legacy pair-subset: each query ranked against all viral IDs in the evaluation pair file
        sizes = [len(viruses_in_file)] * n_q
        ps = {
            "definition": "Each query ranked against all unique viral IDs present in the evaluation pair file (not only partners of that human).",
            "unique_viruses_in_pair_file": len(viruses_in_file),
            "per_query_gallery_size_constant": True,
            "median": int(np.median(sizes)),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
            "iqr": [float(np.percentile(sizes, 25)), float(np.percentile(sizes, 75))],
        }
        # also report partners-of-same-human alternative size for transparency
        partners = defaultdict(set)
        for p in all_pairs:
            if int(p.get("interaction", 1)) == 1:
                partners[p["human_id"]].add(p["virus_id"])
        alt = [len(partners.get(h, ())) for h, _ in queries]
        ps["alt_partners_of_same_human_only"] = {
            "median": float(np.median(alt)) if alt else None,
            "min": int(np.min(alt)) if alt else None,
            "max": int(np.max(alt)) if alt else None,
            "iqr": [float(np.percentile(alt, 25)), float(np.percentile(alt, 75))] if alt else None,
        }
        psp = out_dir / f"constrained_2104_pair_subset_sizes_{args.dataset}.json"
        psp.write_text(json.dumps(ps, indent=2))
        print("Wrote", psp, flush=True)


if __name__ == "__main__":
    main()

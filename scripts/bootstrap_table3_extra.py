#!/usr/bin/env python3
"""Bootstrap 95% CIs for Table-3 cells missing from the primary bootstrap JSON.

Matches manuscript protocol: seed=42, n_boot=2000 query resamples.
Computes (per dataset):
  - retrieval_hit@10          (needed for IntAct; sanity for HVIDB test)
  - direct_full_gallery_hit@10
  - replace_rerank_hit@10     (full-sequence pair forward; slow; --skip-replace to omit)
  - best_oracle@20_hit@10     (HVIDB test α=0.0; IntAct α=1.0)

Usage:
  python scripts/bootstrap_table3_extra.py --dataset hvidb_test
  python scripts/bootstrap_table3_extra.py --dataset intact --skip-replace
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from datetime import datetime
from importlib.machinery import SourcelessFileLoader
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/wangxindi/protein_interaction_model")
SEED = 42
N_BOOT = 2000
TOP_K = 20
DEVICE = "cuda:0"

DATASETS = {
    "hvidb_test": {
        "pairs_pkl": "data/processed/test_protein_pairs.pkl",
        "retr_cache": "analysis_results/embed_cache/2104_retrieval_virus_1115_512.pt",
        "clf_cache": "analysis_results/embed_cache/2104_classifier_virus_1115_512.pt",
        "best_oracle_alpha": 0.0,
        "expected": {
            "retrieval_hit@10": 0.019666666666666666,
            "direct_full_gallery_hit@10": 0.073,
            "replace_rerank_hit@10": 0.035333333333333335,
            "best_oracle@20_hit@10": 0.8688524590163934,
            "oracle_n": 122,
        },
        "out": "analysis_results/constrained_2104_bootstrap_hvidb_test_extra.json",
    },
    "intact": {
        "pairs_pkl": "data/cross_intact/processed/val_protein_pairs.pkl",
        "retr_cache": "analysis_results/embed_cache/2104_retrieval_virus_1229_512.pt",
        "clf_cache": "analysis_results/embed_cache/2104_classifier_virus_1229_512.pt",
        "best_oracle_alpha": 1.0,
        "expected": {
            "retrieval_hit@10": 0.027044293015332198,
            "direct_full_gallery_hit@10": 0.002342419080068143,
            "replace_rerank_hit@10": 0.019804088586030666,
            "best_oracle@20_hit@10": 0.5407725321888412,
            "oracle_n": 233,
        },
        "out": "analysis_results/constrained_2104_bootstrap_intact_extra.json",
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


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> dict:
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
    }


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--skip-replace", action="store_true")
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()
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
    print(
        f"[{args.dataset}] q={len(positive_pairs)} humans={len(human_ids)} "
        f"gallery={len(virus_ids)} gstats={gstats}",
        flush=True,
    )

    retr_cache = torch.load(ROOT / cfg["retr_cache"], map_location="cpu", weights_only=False)
    clf_cache = torch.load(ROOT / cfg["clf_cache"], map_location="cpu", weights_only=False)
    assert retr_cache["ids"] == virus_ids, "retrieval virus id order mismatch"
    assert clf_cache["ids"] == virus_ids, "classifier virus id order mismatch"
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
    n_q = len(positive_pairs)
    queries = sorted(positive_pairs)

    retr_hit10 = np.zeros(n_q, dtype=np.float64)
    direct_hit10 = np.zeros(n_q, dtype=np.float64)
    replace_hit10 = np.zeros(n_q, dtype=np.float64)
    best_oracle_vals = []
    best_alpha = float(cfg["best_oracle_alpha"])

    clf_model.eval()
    virus_chunk = 256
    with torch.no_grad():
        for qi, (h, v_true) in enumerate(queries):
            i = h2i[h]
            j = v2j[v_true]
            retr_scores = sim[i]
            order = np.argsort(-retr_scores)
            rank_retr = int(np.where(order == j)[0][0]) + 1
            retr_hit10[qi] = 1.0 if rank_retr <= 10 else 0.0

            # Direct full-gallery classifier-head ranking
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

            # Best-oracle@20 at dataset-specific α (within top-20 shortlist)
            k = min(TOP_K, len(virus_ids))
            top_idx = order[:k]
            if rank_retr <= TOP_K:
                if best_alpha >= 1.0 - 1e-12:
                    # pure retrieval order on the shortlist
                    try:
                        rank_best = list(top_idx).index(j) + 1
                    except ValueError:
                        rank_best = k + 1
                elif best_alpha <= 1e-12:
                    h_vec2 = torch.as_tensor(h_clf[i], device=device).float()
                    v_chunk = torch.as_tensor(v_clf[top_idx], device=device).float()
                    combined = torch.cat(
                        [h_vec2.unsqueeze(0).expand(k, -1), v_chunk], dim=1
                    )
                    clf_cand = (
                        clf_model.classifier(combined).squeeze(-1).detach().cpu().numpy()
                    )
                    rerank_order = np.argsort(-clf_cand)
                    reranked_ids = [virus_ids[idx] for idx in top_idx[rerank_order]]
                    try:
                        rank_best = reranked_ids.index(v_true) + 1
                    except ValueError:
                        rank_best = k + 1
                else:
                    retr_cand = retr_scores[top_idx]
                    h_vec2 = torch.as_tensor(h_clf[i], device=device).float()
                    v_chunk = torch.as_tensor(v_clf[top_idx], device=device).float()
                    combined = torch.cat(
                        [h_vec2.unsqueeze(0).expand(k, -1), v_chunk], dim=1
                    )
                    clf_cand = (
                        clf_model.classifier(combined).squeeze(-1).detach().cpu().numpy()
                    )
                    fused = best_alpha * minmax(retr_cand) + (1.0 - best_alpha) * minmax(
                        clf_cand
                    )
                    rerank_order = np.argsort(-fused)
                    reranked_ids = [virus_ids[idx] for idx in top_idx[rerank_order]]
                    try:
                        rank_best = reranked_ids.index(v_true) + 1
                    except ValueError:
                        rank_best = k + 1
                best_oracle_vals.append(1.0 if rank_best <= 10 else 0.0)

            if (qi + 1) % 200 == 0:
                print(
                    f"  progress {qi+1}/{n_q} "
                    f"retr@10={retr_hit10[:qi+1].mean():.4f} "
                    f"direct@10={direct_hit10[:qi+1].mean():.4f}",
                    flush=True,
                )

    best_oracle_vals = np.asarray(best_oracle_vals, dtype=np.float64)

    # Optional expensive full-sequence replace-rerank
    if not args.skip_replace:
        print(f"[{args.dataset}] running full-sequence replace-rerank...", flush=True)
        # Reuse evaluate_rerank but capture per-query hits via a local loop
        clf_batch_size = 8
        with torch.no_grad():
            for qi, (h, v_true) in enumerate(queries):
                i = h2i[h]
                scores = sim[i]
                k = min(TOP_K, len(virus_ids))
                top_idx = np.argsort(-scores)[:k]
                candidates = [virus_ids[j] for j in top_idx]
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
                if (qi + 1) % 100 == 0:
                    print(
                        f"  replace {qi+1}/{n_q} hit@10={replace_hit10[:qi+1].mean():.4f}",
                        flush=True,
                    )

    metrics = {
        "retrieval_hit@10": bootstrap_ci(retr_hit10, N_BOOT, SEED),
        "direct_full_gallery_hit@10": bootstrap_ci(direct_hit10, N_BOOT, SEED),
        "best_oracle@20_hit@10": bootstrap_ci(best_oracle_vals, N_BOOT, SEED),
    }
    if not args.skip_replace:
        metrics["replace_rerank_hit@10"] = bootstrap_ci(replace_hit10, N_BOOT, SEED)

    exp = cfg["expected"]
    sanity = {
        "retrieval_hit@10": float(retr_hit10.mean()),
        "direct_full_gallery_hit@10": float(direct_hit10.mean()),
        "best_oracle@20_hit@10": float(best_oracle_vals.mean()) if len(best_oracle_vals) else None,
        "oracle_n": int(len(best_oracle_vals)),
        "best_oracle_alpha": best_alpha,
        "expected": exp,
    }
    if not args.skip_replace:
        sanity["replace_rerank_hit@10"] = float(replace_hit10.mean())

    payload = {
        "evaluated_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "pairs_pkl": cfg["pairs_pkl"],
        "n_boot": N_BOOT,
        "seed": SEED,
        "skip_replace": bool(args.skip_replace),
        "metrics": metrics,
        "sanity": sanity,
        "table3_strings": {
            k: f"{fmt_pct(v['mean'])} [{fmt_pct(v['ci_low'])}–{fmt_pct(v['ci_high'])}]".replace(
                "% [", "% ["
            )
            for k, v in metrics.items()
        },
    }
    # nicer percent formatting without trailing .0 issues already handled
    def cell(v):
        return (
            f"{100*v['mean']:.1f}% [{100*v['ci_low']:.1f}–{100*v['ci_high']:.1f}]"
        )

    payload["table3_strings"] = {k: cell(v) for k, v in metrics.items()}

    out = ROOT / cfg["out"]
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()

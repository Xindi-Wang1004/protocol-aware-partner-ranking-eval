#!/usr/bin/env python3
"""Neural Stage-2 fusion on MMseqs2 Stage-1 shortlists (HVIDB test).

Requires the classification checkpoint + embed cache on a machine with torch/ESM3
(typically server 14). Reuses MMseqs bitscore ranks from mmseqs_partner_ranks.csv
and/or mmseqs_hits.tsv; scores only the top-T shortlist with the Stage-2 head
(α=0.7 fusion), matching Table 3 / S16 row protocol for other external arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def load_hits(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open() as f:
        for line in f:
            q, t, bits, *_ = line.rstrip("\n").split("\t")
            b = float(bits)
            if t not in scores[q] or b > scores[q][t]:
                scores[q][t] = b
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="protocol-aware-partner-ranking-eval root")
    ap.add_argument("--mmseqs-dir", type=Path, required=True, help="dir with mmseqs_hits.tsv / ranks csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()

    repo = args.repo
    sys.path.insert(0, str(repo))
    scripts = repo / "scripts"
    sys.path.insert(0, str(scripts))

    def _load_mod(name: str):
        import importlib
        import importlib.util
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            pass
        # Fall back to __pycache__/*.pyc (source .py may be withheld from release)
        cache = scripts / "__pycache__"
        cands = sorted(cache.glob(f"{name}.cpython-*.pyc")) if cache.exists() else []
        if not cands:
            raise ModuleNotFoundError(name)
        # Prefer matching current interpreter tag
        tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        pick = next((c for c in cands if tag in c.name), cands[-1])
        spec = importlib.util.spec_from_file_location(name, pick)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    reu = _load_mod("retrieval_eval_utils")
    ne = _load_mod("necessity_eval")

    work = args.mmseqs_dir
    hits_path = work / "mmseqs_hits.tsv"
    if not hits_path.exists():
        raise SystemExit(f"missing {hits_path}")

    evalj = json.loads(
        (repo / "analysis_results" / "constrained_2104_hvidb_test.json").read_text()
    )
    virus_ids: list[str] = evalj["candidate_pool"]["virus_ids"]
    v2j = {v: i for i, v in enumerate(virus_ids)}
    n_g = len(virus_ids)

    npz = np.load(
        repo / "analysis_results" / "constrained_2104_per_query_hvidb_test.npz"
        if (repo / "analysis_results" / "constrained_2104_per_query_hvidb_test.npz").exists()
        else Path.home() / "protein_interaction_model" / "analysis_results" / "constrained_2104_per_query_hvidb_test.npz",
        allow_pickle=True,
    )
    # Prefer local cis-aligned order from ranks csv if present
    rank_csv = work / "mmseqs_partner_ranks.csv"
    if rank_csv.exists():
        rows = list(csv.DictReader(rank_csv.open()))
        h_ids = [r["human_id"] for r in rows]
        v_true = [r["true_id"] for r in rows]
    else:
        h_ids = [str(x) for x in npz["human_ids"]]
        v_true = [str(x) for x in npz["virus_ids"]]

    test = pickle.load(open(repo / "data/processed/test_protein_pairs.pkl", "rb"))
    pool_human = {}
    for p in test:
        if p.get("human_seq"):
            pool_human.setdefault(p["human_id"], p["human_seq"])

    human_uniq = sorted(set(h_ids))
    missing = [h for h in human_uniq if h not in pool_human]
    if missing:
        raise SystemExit(f"missing human seqs: {missing[:5]} ({len(missing)})")

    clf_cache = torch.load(
        repo / "analysis_results/embed_cache/2104_classifier_virus_1115_512.pt"
        if (repo / "analysis_results/embed_cache/2104_classifier_virus_1115_512.pt").exists()
        else repo / "embed_cache/2104_classifier_virus_1115_512.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert clf_cache["ids"] == virus_ids
    v_clf = clf_cache["emb"]
    if isinstance(v_clf, torch.Tensor):
        v_clf = v_clf.numpy()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = repo / "classification_task/esm3_frozen/best_model.pt"
    if not ckpt.exists():
        ckpt = repo / "checkpoints/classification_esm3_frozen_best_model.pt"
    clf_model, _ = reu.load_classification_model(str(ckpt), device=device)
    clf_model.eval()

    h_clf_ids, h_clf_emb = ne.precompute_classifier_embeddings(
        clf_model, human_uniq, {h: pool_human[h] for h in human_uniq}, 8
    )
    h2i = {h: i for i, h in enumerate(h_clf_ids)}
    h_clf = (
        h_clf_emb.detach().cpu().numpy()
        if isinstance(h_clf_emb, torch.Tensor)
        else np.asarray(h_clf_emb)
    )

    hitmap = load_hits(hits_path)
    # query ids in ranks csv are q0..; hits use same
    n_q = len(h_ids)
    retr_ranks = np.zeros(n_q, dtype=np.int32)
    fusion_hit = np.zeros(n_q, dtype=np.float64)
    replace_hit = np.zeros(n_q, dtype=np.float64)
    oracle_mask = np.zeros(n_q, dtype=bool)
    recall20 = np.zeros(n_q)
    recall100 = np.zeros(n_q)
    out_rows = []

    with torch.no_grad():
        for i, (h, vt) in enumerate(zip(h_ids, v_true)):
            qid = f"q{i}"
            sc = np.zeros(n_g, dtype=np.float64)
            for t, b in hitmap.get(qid, {}).items():
                if t in v2j:
                    sc[v2j[t]] = b
            order = np.lexsort((np.arange(n_g), -sc))
            j = v2j[vt]
            rr = int(np.where(order == j)[0][0]) + 1
            retr_ranks[i] = rr
            recall20[i] = 1.0 if rr <= args.T else 0.0
            recall100[i] = 1.0 if rr <= 100 else 0.0
            oracle_mask[i] = rr <= args.T

            top = order[: args.T]
            base = sc[top]
            hi = h2i[h]
            h_vec = torch.as_tensor(h_clf[hi], device=device).float()
            v_chunk = torch.as_tensor(v_clf[top], device=device).float()
            combined = torch.cat([h_vec.unsqueeze(0).expand(len(top), -1), v_chunk], dim=1)
            clf_cand = clf_model.classifier(combined).squeeze(-1).detach().cpu().numpy()
            fused = args.alpha * minmax(base) + (1.0 - args.alpha) * minmax(clf_cand)
            top_order = top[np.argsort(-fused, kind="mergesort")]
            # pure classifier replace-rerank on shortlist
            repl_order = top[np.argsort(-clf_cand, kind="mergesort")]

            if j in top:
                fr = int(np.where(top_order == j)[0][0]) + 1
                pr = int(np.where(repl_order == j)[0][0]) + 1
                fusion_hit[i] = 1.0 if fr <= args.K else 0.0
                replace_hit[i] = 1.0 if pr <= args.K else 0.0
                rerank_rank = fr
            else:
                fusion_hit[i] = 0.0
                replace_hit[i] = 0.0
                rerank_rank = ""

            out_rows.append(
                {
                    "query_id": qid,
                    "true_id": vt,
                    "retrieval_rank": rr,
                    "rerank_rank": rerank_rank,
                    "gallery_size": n_g,
                    "in_gallery": True,
                    "human_id": h,
                    "stage2": "neural_classifier_fusion",
                }
            )
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_q} fus@10={fusion_hit[:i+1].mean():.4f}", flush=True)

    om = oracle_mask
    summary = {
        "method": "MMseqs2 Stage-1 + neural Stage-2 fusion (α=0.7, T=20)",
        "n_queries": n_q,
        "gallery_n": n_g,
        "fusion": {
            "alpha": args.alpha,
            "T": args.T,
            "K": args.K,
            "stage2_channel": "classification_esm3_frozen_classifier_head",
            "checkpoint": str(ckpt),
        },
        "recall@20": float(recall20.mean()),
        "recall@100": float(recall100.mean()),
        "retrieval_hit@10": float((retr_ranks <= 10).mean()),
        "end_to_end_fusion_hit@10_T20": float(fusion_hit.mean()),
        "end_to_end_replace_hit@10_T20": float(replace_hit.mean()),
        "oracle@20_n": int(om.sum()),
        "oracle@20_fusion_hit@10": float(fusion_hit[om].mean()) if om.any() else None,
        "oracle@20_replace_hit@10": float(replace_hit[om].mean()) if om.any() else None,
        "identity_abs_error": abs(
            float(fusion_hit.mean())
            - float(recall20.mean()) * (float(fusion_hit[om].mean()) if om.any() else 0.0)
        ),
    }
    out_csv = work / "mmseqs_partner_ranks_neural_stage2.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    out_json = work / "mmseqs_hvidb_test_neural_stage2_summary.json"
    summary["rank_csv"] = str(out_csv)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out_json)


if __name__ == "__main__":
    main()

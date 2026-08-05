#!/usr/bin/env python3
"""Bootstrap 95% CIs for held-out HVIDB test fixed-gallery metrics (seed=42, n_boot=2000)."""
from __future__ import annotations

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
OUT = ROOT / "analysis_results/constrained_2104_bootstrap_hvidb_test.json"
SEED = 42
N_BOOT = 2000
ALPHA = 0.7
TOP_K = 20
DEVICE = "cuda:0"


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


def main() -> None:
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

    pairs = ne.load_pairs("data/processed/test_protein_pairs.pkl")
    positive_pairs, query_human_seq, query_virus_seq = ne.build_queries_from_pairs(pairs)
    hvidb_ids = ce.load_hvidb_2104_ids(ce.HVIDB_TSV)
    seq_map = ce.build_sequence_map("data/SV.fasta", ce.DEFAULT_PAIR_PKLS)
    gallery, gstats = ce.build_hvidb_2104_gallery(
        hvidb_ids, seq_map, 512, positive_pairs, query_virus_seq
    )
    virus_ids = sorted(gallery.keys())
    human_ids = sorted({h for h, _ in positive_pairs})
    pool_human = {h: query_human_seq[h][:512] for h in human_ids}
    print(
        f"q={len(positive_pairs)} humans={len(human_ids)} gallery={len(virus_ids)} "
        f"gstats={gstats}",
        flush=True,
    )

    # Virus caches from main eval
    retr_cache = torch.load(
        ROOT / "analysis_results/embed_cache/2104_retrieval_virus_1115_512.pt",
        map_location="cpu",
        weights_only=False,
    )
    clf_cache = torch.load(
        ROOT / "analysis_results/embed_cache/2104_classifier_virus_1115_512.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert retr_cache["ids"] == virus_ids, "retrieval virus id order mismatch"
    assert clf_cache["ids"] == virus_ids, "classifier virus id order mismatch"
    v_emb = retr_cache["emb"]
    v_clf = clf_cache["emb"]
    if isinstance(v_emb, torch.Tensor):
        v_emb = v_emb.numpy()
    if isinstance(v_clf, torch.Tensor):
        v_clf = v_clf.numpy()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    retr_model, _ = reu.load_retrieval_model(
        "retrieval_task/gnn_tau007_longrun/best_model.pt",
        device=device,
        model_type="esm3_lora_gnn",
    )
    h_ids, h_emb = ne.encode_retrieval_pool(
        retr_model, human_ids, pool_human, device, 8, "human"
    )
    assert list(h_ids) == human_ids
    if isinstance(h_emb, torch.Tensor):
        h_mat = h_emb.detach().cpu().numpy()
    else:
        h_mat = np.asarray(h_emb)
    sim = h_mat @ v_emb.T

    clf_model, _ = reu.load_classification_model(
        "classification_task/esm3_frozen/best_model.pt", device=device
    )
    h_clf_ids, h_clf_emb = ne.precompute_classifier_embeddings(
        clf_model, human_ids, pool_human, 8
    )
    assert list(h_clf_ids) == human_ids
    if isinstance(h_clf_emb, torch.Tensor):
        h_clf = h_clf_emb.detach().cpu().numpy()
    else:
        h_clf = np.asarray(h_clf_emb)

    h2i = {h: i for i, h in enumerate(human_ids)}
    v2j = {v: j for j, v in enumerate(virus_ids)}

    n_q = len(positive_pairs)
    recall100 = np.zeros(n_q, dtype=np.float64)
    retr_hit10 = np.zeros(n_q, dtype=np.float64)
    fusion_hit10 = np.zeros(n_q, dtype=np.float64)
    oracle_vals = []

    clf_model.eval()
    with torch.no_grad():
        for qi, (h, v_true) in enumerate(sorted(positive_pairs)):
            i = h2i[h]
            j = v2j[v_true]
            retr_scores = sim[i]
            order = np.argsort(-retr_scores)
            rank_retr = int(np.where(order == j)[0][0]) + 1
            recall100[qi] = 1.0 if rank_retr <= 100 else 0.0
            retr_hit10[qi] = 1.0 if rank_retr <= 10 else 0.0

            k = min(TOP_K, len(virus_ids))
            top_idx = order[:k]
            retr_cand = retr_scores[top_idx]
            h_vec = torch.as_tensor(h_clf[i], device=device).float()
            v_chunk = torch.as_tensor(v_clf[top_idx], device=device).float()
            combined = torch.cat(
                [h_vec.unsqueeze(0).expand(k, -1), v_chunk], dim=1
            )
            clf_cand = clf_model.classifier(combined).squeeze(-1).detach().cpu().numpy()
            fused = ALPHA * minmax(retr_cand) + (1.0 - ALPHA) * minmax(clf_cand)
            rerank_order = np.argsort(-fused)
            reranked_ids = [virus_ids[idx] for idx in top_idx[rerank_order]]
            try:
                rank_fus = reranked_ids.index(v_true) + 1
            except ValueError:
                rank_fus = k + 1
            fusion_hit10[qi] = 1.0 if rank_fus <= 10 else 0.0
            if rank_retr <= TOP_K:
                oracle_vals.append(fusion_hit10[qi])

    oracle_vals = np.asarray(oracle_vals, dtype=np.float64)
    metrics = {
        "recall@100": bootstrap_ci(recall100, N_BOOT, SEED),
        "retrieval_hit@10": bootstrap_ci(retr_hit10, N_BOOT, SEED),
        "end_to_end_fusion_hit@10_top20": bootstrap_ci(fusion_hit10, N_BOOT, SEED),
        "oracle@20_fusion_hit@10": bootstrap_ci(oracle_vals, N_BOOT, SEED),
    }
    payload = {
        "evaluated_at": datetime.now().isoformat(),
        "pairs_pkl": "data/processed/test_protein_pairs.pkl",
        "n_boot": N_BOOT,
        "seed": SEED,
        "metrics": metrics,
        "sanity": {
            "recall@100": float(recall100.mean()),
            "retrieval_hit@10": float(retr_hit10.mean()),
            "fusion_hit@10": float(fusion_hit10.mean()),
            "oracle@20_n": int(len(oracle_vals)),
            "oracle@20_fusion_hit@10": float(oracle_vals.mean()),
            "expected_recall@100": 0.156,
            "expected_retrieval_hit@10": 0.019666666666666666,
            "expected_fusion_hit@10": 0.025666666666666667,
            "expected_oracle@20": 0.6311475409836066,
            "expected_oracle_n": 122,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"metrics": metrics, "sanity": payload["sanity"]}, indent=2), flush=True)
    print("Wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

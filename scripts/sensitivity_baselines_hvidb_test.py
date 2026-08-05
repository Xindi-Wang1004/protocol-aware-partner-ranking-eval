#!/usr/bin/env python3
"""Supplementary sensitivity baselines under the same fixed HVIDB-2104 gallery.

Non-learning controls (CPU):
  1) Virus popularity: rank gallery viruses by #train positive partners
  2) kNN partner transfer: 3-mer TF cosine neighbors among train humans
     (exclude identical human_id), transfer their known viral partners

Reports Hit@10 and Recall@100 on held-out HVIDB test positives.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangxindi/protein_interaction_model")
OUT = ROOT / "analysis_results/constrained_2104_sensitivity_baselines_hvidb_test.json"
MAIN = ROOT / "analysis_results/constrained_2104_hvidb_test.json"
K_NEIGH = 10
AA = "ACDEFGHIKLMNPQRSTVWY"


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


def ranks_from_scores(scores: np.ndarray, true_j: int, ks=(10, 100)) -> dict:
    # higher is better
    order = np.argsort(-scores)
    rank = int(np.where(order == true_j)[0][0]) + 1
    return {f"hit@{k}": float(rank <= k) for k in ks} | {"rank": rank}


def main() -> None:
    main_eval = json.loads(MAIN.read_text())
    virus_ids = main_eval["candidate_pool"]["virus_ids"]
    v2j = {v: i for i, v in enumerate(virus_ids)}
    n_g = len(virus_ids)

    train = pickle.load(open(ROOT / "data/processed/train_protein_pairs.pkl", "rb"))
    test = pickle.load(open(ROOT / "data/processed/test_protein_pairs.pkl", "rb"))
    tr_pos = [p for p in train if int(p["interaction"]) == 1]
    te_pos = [p for p in test if int(p["interaction"]) == 1]
    # stable query order matching main eval habit
    queries = sorted((p["human_id"], p["virus_id"], p["human_seq"]) for p in te_pos)

    # Popularity from train positives (gallery-restricted)
    pop = Counter()
    train_partners: dict[str, set[str]] = defaultdict(set)
    train_human_seq: dict[str, str] = {}
    for p in tr_pos:
        h, v = p["human_id"], p["virus_id"]
        train_human_seq.setdefault(h, p["human_seq"])
        if v in v2j:
            pop[v] += 1
            train_partners[h].add(v)

    pop_scores = np.array([float(pop[v]) for v in virus_ids], dtype=np.float64)

    # 3-mer features for train humans
    train_h_ids = sorted(train_human_seq)
    print(f"Encoding {len(train_h_ids)} train humans (3-mer)...", flush=True)
    H = np.stack([kmer_vec(train_human_seq[h]) for h in train_h_ids], axis=0)
    h2row = {h: i for i, h in enumerate(train_h_ids)}

    pop_hit10 = []
    pop_r100 = []
    knn_hit10 = []
    knn_r100 = []

    print(f"Scoring {len(queries)} test queries...", flush=True)
    for qi, (h, v_true, hseq) in enumerate(queries):
        j = v2j[v_true]
        # popularity
        r = ranks_from_scores(pop_scores, j)
        pop_hit10.append(r["hit@10"])
        pop_r100.append(r["hit@100"])

        # kNN transfer
        qv = kmer_vec(hseq)
        sims = H @ qv
        # exclude identical human id if present in train
        if h in h2row:
            sims[h2row[h]] = -1e9
        nn = np.argsort(-sims)[:K_NEIGH]
        scores = np.zeros(n_g, dtype=np.float64)
        for ridx in nn:
            nh = train_h_ids[int(ridx)]
            w = float(sims[ridx])
            if w <= 0:
                continue
            for v in train_partners.get(nh, ()):
                scores[v2j[v]] += w
        # tiny popularity tie-break so zero-score viruses are ordered
        scores = scores + 1e-9 * pop_scores
        r2 = ranks_from_scores(scores, j)
        knn_hit10.append(r2["hit@10"])
        knn_r100.append(r2["hit@100"])

        if (qi + 1) % 500 == 0:
            print(f"  {qi+1}/{len(queries)}", flush=True)

    def summarize(hit10, r100):
        return {
            "hit@10": float(np.mean(hit10)),
            "recall@100": float(np.mean(r100)),
            "n": len(hit10),
        }

    payload = {
        "evaluated_at": datetime.now().isoformat(),
        "protocol": "fixed_gallery_hvidb_2104_same_as_main",
        "gallery_n": n_g,
        "n_queries": len(queries),
        "baselines": {
            "virus_popularity_train": summarize(pop_hit10, pop_r100),
            "knn_partner_transfer_3mer_k10": summarize(knn_hit10, knn_r100)
            | {"k_neighbors": K_NEIGH, "exclude_same_human_id": True},
        },
        "reference_main_pipeline": {
            "retrieval_hit@10": main_eval["retrieval_only"]["hit@10"],
            "retrieval_recall@100": main_eval["recall_at_k"]["recall@100"],
            "direct_classifier_hit@10": main_eval["direct_classifier"]["hit@10"],
            "fusion_hit@10": main_eval["score_fusion"]["hit@10"],
        },
        "notes": [
            "Sensitivity checks under the identical fixed retrievable gallery as Table 2.",
            "Not claimed as SOTA predictors; intended to show protocol-aware decomposition remains informative beyond the main frozen-encoder case study.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["baselines"], indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()

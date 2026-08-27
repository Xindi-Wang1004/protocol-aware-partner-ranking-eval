#!/usr/bin/env python3
"""Multilabel sensitivity: designated-positive vs any-known-positive Hit@K.

Stage-1 ranks depend only on the human query, so ranks of all evaluation-file
positive partners of the same human can be pooled across that human's queries.
Fusion/direct Hit@10 indicators similarly transfer across partners of the same
human (shortlist/full-gallery scores do not depend on which partner is designated).

Primary identity Hit@K = Recall@T × oracle@K remains designated-positive only.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

BIB = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "analysis"
REPO = BIB / "_git_push_work" / "repo"


def load_pos_partners(pkl: Path) -> dict[str, set[str]]:
    pairs = pickle.load(open(pkl, "rb"))
    h2v: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        if int(p.get("interaction", 0)) == 1:
            h2v[str(p["human_id"])].add(str(p["virus_id"]))
    return dict(h2v)


def metrics_from_npz(npz_path: Path, h2v: dict[str, set[str]], label: str) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    humans = [str(x) for x in z["human_ids"]]
    viruses = [str(x) for x in z["virus_ids"]]
    rank_retr = np.asarray(z["rank_retr"], dtype=np.int32)
    fusion_hit10 = np.asarray(z["fusion_hit10"], dtype=np.float64)
    direct_hit10 = np.asarray(z["direct_hit10"], dtype=np.float64)
    n = len(humans)

    # Per-human Stage-1 ranks for every designated partner observed as a query
    h_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    h_fus10: dict[str, dict[str, float]] = defaultdict(dict)
    h_dir10: dict[str, dict[str, float]] = defaultdict(dict)
    for i in range(n):
        h, v = humans[i], viruses[i]
        h_ranks[h][v] = int(rank_retr[i])
        h_fus10[h][v] = float(fusion_hit10[i])
        h_dir10[h][v] = float(direct_hit10[i])

    des_hit = {k: np.zeros(n) for k in (1, 5, 10)}
    any_hit = {k: np.zeros(n) for k in (1, 5, 10)}
    fus_des = np.zeros(n)
    fus_any = np.zeros(n)
    dir_des = np.zeros(n)
    dir_any = np.zeros(n)
    n_known = np.zeros(n, dtype=np.int32)
    multi = np.zeros(n, dtype=bool)

    missing_partner_rank = 0
    for i in range(n):
        h, v_des = humans[i], viruses[i]
        known = h2v.get(h, {v_des})
        # restrict to gallery-scored partners we have ranks for (all test positives)
        ranked = h_ranks[h]
        known_ranked = [v for v in known if v in ranked]
        if v_des not in ranked:
            missing_partner_rank += 1
        n_known[i] = len(known_ranked)
        multi[i] = len(known_ranked) >= 2
        r_des = ranked[v_des]
        for k in (1, 5, 10):
            des_hit[k][i] = 1.0 if r_des <= k else 0.0
            any_hit[k][i] = 1.0 if min(ranked[v] for v in known_ranked) <= k else 0.0
        fus_des[i] = h_fus10[h][v_des]
        fus_any[i] = 1.0 if any(h_fus10[h].get(v, 0.0) >= 0.5 for v in known_ranked) else 0.0
        dir_des[i] = h_dir10[h][v_des]
        dir_any[i] = 1.0 if any(h_dir10[h].get(v, 0.0) >= 0.5 for v in known_ranked) else 0.0

    uniq_h = sorted(set(humans))
    return {
        "label": label,
        "n_queries": n,
        "n_unique_humans": len(uniq_h),
        "frac_queries_multi_positive": float(multi.mean()),
        "frac_humans_multi_positive": float(
            np.mean([1.0 if len(h2v.get(h, ())) >= 2 else 0.0 for h in uniq_h])
        ),
        "mean_known_positives_per_query": float(n_known.mean()),
        "median_known_positives_per_query": float(np.median(n_known)),
        "stage1_retrieval": {
            "designated_hit@1": float(des_hit[1].mean()),
            "designated_hit@5": float(des_hit[5].mean()),
            "designated_hit@10": float(des_hit[10].mean()),
            "any_known_positive_hit@1": float(any_hit[1].mean()),
            "any_known_positive_hit@5": float(any_hit[5].mean()),
            "any_known_positive_hit@10": float(any_hit[10].mean()),
        },
        "fusion_T20": {
            "designated_hit@10": float(fus_des.mean()),
            "any_known_positive_hit@10": float(fus_any.mean()),
            "note": (
                "Fusion Hit@10 for a partner equals the stored fusion_hit10 on the query "
                "where that partner is designated; shortlist fusion order is human-specific "
                "and independent of designation."
            ),
        },
        "direct_full_gallery": {
            "designated_hit@10": float(dir_des.mean()),
            "any_known_positive_hit@10": float(dir_any.mean()),
        },
        "definition": {
            "known_positives": (
                "Evaluation-file positive viral partners of the query human "
                "(HVIDB test / IntAct cross-test positives), not training-only partners."
            ),
            "designated_positive": (
                "The single labelled true partner of the pair-level query; primary "
                "protocol for Table 2 and the Hit@K = Recall@T × oracle@K identity."
            ),
            "any_known_positive_hit@K": (
                "1 if at least one evaluation-file positive partner of the query human "
                "has Stage-1 rank ≤ K (or fusion/direct Hit@10 membership for those arms)."
            ),
        },
        "missing_partner_rank_rows": missing_partner_rank,
    }


def partners_from_npz(npz_path: Path) -> dict[str, set[str]]:
    z = np.load(npz_path, allow_pickle=True)
    h2v: dict[str, set[str]] = defaultdict(set)
    for h, v in zip(z["human_ids"], z["virus_ids"]):
        h2v[str(h)].add(str(v))
    return dict(h2v)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hvidb_npz = BIB / "_server14_cis/constrained_2104_per_query_hvidb_test.npz"
    intact_npz = BIB / "_server14_cis/constrained_2104_per_query_intact.npz"
    hvidb = load_pos_partners(REPO / "data/processed/test_protein_pairs.pkl")
    # IntAct pair pickle may be absent locally; NPZ query pairs define the evaluation positives.
    intact = partners_from_npz(intact_npz)
    summary = {
        "hvidb_test": metrics_from_npz(
            hvidb_npz, hvidb, "HVIDB test (pair-held-out)"
        ),
        "intact_cross_test": metrics_from_npz(
            intact_npz, intact, "IntAct cross-test"
        ),
    }
    out = OUT / "multilabel_any_known_positive.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

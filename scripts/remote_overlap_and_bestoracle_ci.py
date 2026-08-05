#!/usr/bin/env python3
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangxindi/protein_interaction_model")
OUT = ROOT / "analysis_results"


def load_pos(path):
    obj = pickle.load(open(path, "rb"))
    hs, vs = set(), set()
    pairs = []
    for it in obj:
        if int(it.get("interaction", 1)) != 1:
            continue
        h, v = it["human_id"], it["virus_id"]
        hs.add(h)
        vs.add(v)
        pairs.append((h, v))
    return hs, vs, pairs


def main():
    tr_h, tr_v, tr_pairs = load_pos(ROOT / "data/processed/train_protein_pairs.pkl")
    te_h, te_v, te_pairs = load_pos(ROOT / "data/processed/test_protein_pairs.pkl")
    va_h, va_v, va_pairs = load_pos(ROOT / "data/processed/val_protein_pairs.pkl")

    # gallery viruses from evaluation (sequence-resolved)
    # approximate: viruses appearing in test positives that are also in train
    overlap = {
        "n_train_pos_pairs": len(tr_pairs),
        "n_test_pos_pairs": len(te_pairs),
        "n_train_viruses": len(tr_v),
        "n_test_viruses": len(te_v),
        "n_test_viruses_in_train": len(te_v & tr_v),
        "frac_test_viruses_in_train": len(te_v & tr_v) / len(te_v),
        "n_test_humans_in_train": len(te_h & tr_h),
        "frac_test_humans_in_train": len(te_h & tr_h) / len(te_h),
        "n_val_viruses_in_train": len(va_v & tr_v),
        "frac_val_viruses_in_train": len(va_v & tr_v) / len(va_v),
    }
    (OUT / "train_test_virus_overlap.json").write_text(json.dumps(overlap, indent=2))
    print(json.dumps(overlap, indent=2))

    human_to_viruses = defaultdict(set)
    for h, v in te_pairs:
        human_to_viruses[h].add(v)
    sizes = [len(human_to_viruses[h]) for h, _ in te_pairs]
    audit = {
        "unique_viruses_in_pair_file": len({v for _, v in te_pairs}),
        "n_pos_pairs": len(te_pairs),
        "per_query_known_partners_median": float(np.median(sizes)),
        "per_query_known_partners_min": int(np.min(sizes)),
        "per_query_known_partners_max": int(np.max(sizes)),
        "per_query_known_partners_iqr": [
            float(np.percentile(sizes, 25)),
            float(np.percentile(sizes, 75)),
        ],
        "note": "Legacy pair-subset Hit@K uses per-query known-partner subsets.",
    }
    (OUT / "pair_subset_true_definition_hvidb_test.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

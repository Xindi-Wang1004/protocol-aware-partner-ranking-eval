#!/usr/bin/env python3
"""BindingDB first-pass: activity filter + partners-per-human-target distribution.

Does not train a model. Reads a complete official BindingDB TSV (not a truncated download).

Positive pair rule (v0):
  - target organism contains Homo sapiens / human
  - UniProt primary ID present
  - ligand SMILES present
  - min(Ki, Kd, IC50, EC50) <= 10000 nM (10 µM), ignoring missing values
  - multiple measurements of the same (target, ligand) take the median affinity
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

NA = {"", "n/a", "na", "none", "-", "nan"}
HUMAN_TOKENS = ("homo sapiens", "human")
AFF_COLS_CAND = (
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "ki (nM)",
    "ic50 (nM)",
    "kd (nM)",
    "ec50 (nM)",
)


def parse_nM(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s.lower() in NA:
        return None
    # BindingDB sometimes prefixes inequalities; keep numeric core.
    for ch in "<>~=":
        s = s.replace(ch, "")
    s = s.strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def is_human(org: str) -> bool:
    o = (org or "").strip().lower()
    return any(tok in o for tok in HUMAN_TOKENS)


def open_tsv(path: Path):
    if path.suffix.lower() == ".zip":
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.lower().endswith(".tsv") or n.lower().endswith(".txt")]
        if not names:
            raise SystemExit(f"no tsv in {path}: {z.namelist()[:10]}")
        f = z.open(names[0])
        return z, (line.decode("utf-8", errors="replace") for line in f)
    return None, path.open(encoding="utf-8", errors="replace")


def partner_bin(n: int) -> str:
    if n <= 1:
        return "eq1"
    if n == 2:
        return "eq2"
    if n <= 5:
        return "3to5"
    if n <= 10:
        return "6to10"
    if n <= 50:
        return "11to50"
    return "gt50"


def stratified_target_split(
    partners: dict[str, set[str]], test_frac: float, seed: int
) -> dict:
    """Split by target (never by pair). Stratify on partners-per-target bins."""
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for uid, ligs in partners.items():
        buckets[partner_bin(len(ligs))].append(uid)
    train, test = [], []
    for uids in buckets.values():
        rng.shuffle(uids)
        n_test = int(round(len(uids) * test_frac))
        n_test = min(max(n_test, 0), len(uids))
        if n_test == len(uids) and len(uids) > 1:
            n_test = len(uids) - 1
        test.extend(uids[:n_test])
        train.extend(uids[n_test:])
    train_set, test_set = set(train), set(test)
    assert not (train_set & test_set)
    assert train_set | test_set == set(partners)
    return {
        "seed": seed,
        "test_frac": test_frac,
        "unit": "target UniProt (never pair)",
        "n_train_targets": len(train),
        "n_test_targets": len(test),
        "train_targets": sorted(train),
        "test_targets": sorted(test),
    }


def percentile(xs: list[int], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = (len(ys) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    if lo == hi:
        return float(ys[lo])
    return ys[lo] * (hi - i) + ys[hi] * (i - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold-nM", type=float, default=10000.0)
    ap.add_argument("--split-out", default=None)
    ap.add_argument("--evalset-out", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()
    path = Path(args.tsv)
    zhandle, lines = open_tsv(path)
    reader = csv.DictReader(lines, delimiter="\t")
    fields = reader.fieldnames or []
    # column aliases
    def col(*names):
        lower = {c.lower(): c for c in fields}
        for n in names:
            if n in fields:
                return n
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    c_smi = col("Ligand SMILES")
    c_org = col("Target Source Organism According to Curator or DataSource")
    # BindingDB TSV uses "... Target Chain 1/2/..." suffixes; chain 1 is the primary target.
    c_up = col(
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (SwissProt) Primary ID of Target Chain",
    )
    c_up2 = col(
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain",
    )
    c_name = col("Target Name")
    c_seq = col("BindingDB Target Chain Sequence 1", "BindingDB Target Chain Sequence")
    aff_cols = [c for c in fields if c in AFF_COLS_CAND or c.lower() in {x.lower() for x in AFF_COLS_CAND}]
    if not aff_cols:
        aff_cols = [c for c in fields if any(k in c.lower() for k in ("ki (nm)", "ic50 (nm)", "kd (nm)", "ec50 (nm)"))]

    n_rows = 0
    n_human = 0
    pair_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    target_seq: dict[str, str] = {}
    target_name: dict[str, str] = {}
    from collections import Counter as C
    skip = C()

    for row in reader:
        n_rows += 1
        org = row.get(c_org, "") if c_org else ""
        if not is_human(org):
            skip["not_human"] += 1
            continue
        n_human += 1
        uid = (row.get(c_up) or "").strip() if c_up else ""
        if not uid and c_up2:
            uid = (row.get(c_up2) or "").strip()
        smi = (row.get(c_smi) or "").strip() if c_smi else ""
        if not uid:
            skip["no_uniprot"] += 1
            continue
        if not smi:
            skip["no_smiles"] += 1
            continue
        vals = [parse_nM(row.get(c)) for c in aff_cols]
        vals = [v for v in vals if v is not None]
        if not vals:
            skip["no_affinity"] += 1
            continue
        best = min(vals)
        if best > args.threshold_nM:
            skip["above_threshold"] += 1
            continue
        pair_vals[(uid, smi)].append(best)
        skip["kept_measurement"] += 1
        if c_name:
            nm = (row.get(c_name) or "").strip()
            if nm and uid not in target_name:
                target_name[uid] = nm
        if c_seq:
            seq = "".join((row.get(c_seq) or "").split())
            if seq and (uid not in target_seq or len(seq) > len(target_seq[uid])):
                target_seq[uid] = seq.upper()

    # median over replicate measurements
    positives = {}
    for (uid, smi), vs in pair_vals.items():
        positives[(uid, smi)] = statistics.median(vs)

    partners: dict[str, set[str]] = defaultdict(set)
    for uid, smi in positives:
        partners[uid].add(smi)
    counts = [len(s) for s in partners.values()]
    n_ge2 = sum(1 for c in counts if c >= 2)
    hist = {
        "eq1": sum(1 for c in counts if c == 1),
        "eq2": sum(1 for c in counts if c == 2),
        "3to5": sum(1 for c in counts if 3 <= c <= 5),
        "6to10": sum(1 for c in counts if 6 <= c <= 10),
        "11to50": sum(1 for c in counts if 11 <= c <= 50),
        "gt50": sum(1 for c in counts if c > 50),
    }
    payload = {
        "source_file": str(path),
        "source_bytes": path.stat().st_size,
        "threshold_nM": args.threshold_nM,
        "n_tsv_rows": n_rows,
        "n_human_rows": n_human,
        "skip": dict(skip),
        "n_positive_pairs": len(positives),
        "n_human_targets": len(partners),
        "n_unique_ligands": len({smi for _, smi in positives}),
        "n_targets_with_seq": sum(1 for u in partners if u in target_seq),
        "frac_targets_with_seq": (
            sum(1 for u in partners if u in target_seq) / len(partners) if partners else None
        ),
        "partners_per_target": {
            "min": min(counts) if counts else None,
            "p10": percentile(counts, 0.10) if counts else None,
            "median": percentile(counts, 0.50) if counts else None,
            "p90": percentile(counts, 0.90) if counts else None,
            "max": max(counts) if counts else None,
            "mean": (sum(counts) / len(counts)) if counts else None,
            "n_targets_ge2": n_ge2,
            "frac_targets_ge2": (n_ge2 / len(counts)) if counts else None,
            "histogram": hist,
        },
        "columns_used": {
            "smiles": c_smi,
            "organism": c_org,
            "uniprot": c_up,
            "uniprot_trembl": c_up2,
            "target_name": c_name,
            "sequence": c_seq,
            "affinity": aff_cols,
        },
        "protocol_mapping": {
            "query": "human target protein (UniProt)",
            "gallery": "scoreable ligands with SMILES",
            "true_partner": "known active ligand of that target",
            "pair_subset": "known ligands of the query target in this file",
            "fixed_gallery": "all scoreable ligands",
            "split": "by target, never random pair split",
        },
        "license_note": (
            "BindingDB-curated articles TSV (BindingDB_BindingDB_Articles_202608): CC-BY. "
            "This file is BindingDB's own curation, not the All dump with ChEMBL imports. "
            "Release scripts + summary stats; do not dump the merged TSV without license labels."
        ),
    }
    if args.split_out or args.evalset_out:
        split = stratified_target_split(partners, args.test_frac, args.split_seed)
        train_t, test_t = set(split["train_targets"]), set(split["test_targets"])
        n_train_pairs = sum(1 for uid, _ in positives if uid in train_t)
        n_test_pairs = sum(1 for uid, _ in positives if uid in test_t)
        test_counts = [len(partners[u]) for u in test_t]
        train_counts = [len(partners[u]) for u in train_t]
        split_summary = {
            "seed": split["seed"],
            "test_frac": split["test_frac"],
            "unit": split["unit"],
            "n_train_targets": split["n_train_targets"],
            "n_test_targets": split["n_test_targets"],
            "n_train_pairs": n_train_pairs,
            "n_test_pairs": n_test_pairs,
            "train_partners_median": percentile(train_counts, 0.50),
            "test_partners_median": percentile(test_counts, 0.50),
            "n_test_targets_ge2": sum(1 for c in test_counts if c >= 2),
            "fixed_gallery_size": len({smi for _, smi in positives}),
            "pair_subset_gallery": "known ligands of the query target in this file",
        }
        payload["split"] = split_summary
        if args.split_out:
            split_path = Path(args.split_out)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(
                json.dumps(
                    {
                        **split_summary,
                        "train_targets": split["train_targets"],
                        "test_targets": split["test_targets"],
                        "protocol_mapping": payload["protocol_mapping"],
                        "source_file": str(path),
                        "threshold_nM": args.threshold_nM,
                    },
                    indent=2,
                )
            )
        if args.evalset_out:
            ligands = sorted({smi for _, smi in positives})
            lig_i = {s: i for i, s in enumerate(ligands)}
            targets_out = {}
            for uid, smis in partners.items():
                split_name = "train" if uid in train_t else "test"
                targets_out[uid] = {
                    "split": split_name,
                    "seq": target_seq.get(uid, ""),
                    "name": target_name.get(uid, ""),
                    "ligand_idx": sorted(lig_i[s] for s in smis),
                }
            eval_path = Path(args.evalset_out)
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            eval_path.write_text(
                json.dumps(
                    {
                        "license_note": payload["license_note"],
                        "source_file": str(path),
                        "threshold_nM": args.threshold_nM,
                        "protocol_mapping": payload["protocol_mapping"],
                        "split": split_summary,
                        "ligands": ligands,
                        "targets": targets_out,
                    }
                )
            )
            payload["evalset"] = {
                "path": str(eval_path),
                "n_ligands": len(ligands),
                "n_targets": len(targets_out),
                "n_targets_missing_seq": sum(1 for t in targets_out.values() if not t["seq"]),
            }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if zhandle:
        zhandle.close()


if __name__ == "__main__":
    main()

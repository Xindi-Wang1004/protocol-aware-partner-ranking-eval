#!/usr/bin/env python3
"""External MMseqs2 sequence-similarity Stage-1 baseline on HVIDB fixed gallery.

Produces partner-rank-eval rank CSV + summary metrics (Recall@20/100, Hit@10,
oracle@20 with popularity shortlist fusion α=0.7), for manuscript Table 3 / S16.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BIB = Path(__file__).resolve().parents[2]
REPO = BIB / "_git_push_work" / "repo"
DEFAULT_MMSEQS = BIB / "transfer_bioinformatics" / "tools" / "mmseqs"
EVAL_JSON = (
    BIB
    / "_zenodo_v011"
    / "processed_evaluation_files"
    / "analysis_results"
    / "constrained_2104_hvidb_test.json"
)


def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    sid = None
    chunks: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    seqs[sid] = "".join(chunks)
                # UniProt-style: >sp|ID|... or >ID ...
                header = line[1:].split()[0]
                parts = header.split("|")
                sid = parts[1] if len(parts) >= 2 and parts[0] in {"sp", "tr"} else header
                chunks = []
            else:
                chunks.append(line)
        if sid is not None:
            seqs[sid] = "".join(chunks)
    return seqs


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def write_fasta(path: Path, items: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for sid, seq in items:
            f.write(f">{sid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--K", type=int, default=10)
    args = ap.parse_args()
    mmseqs = Path(args.mmseqs)
    if not mmseqs.exists():
        raise SystemExit(f"mmseqs binary not found: {mmseqs}")

    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)

    evalj = json.loads(EVAL_JSON.read_text())
    virus_ids: list[str] = evalj["candidate_pool"]["virus_ids"]
    v2j = {v: i for i, v in enumerate(virus_ids)}
    n_g = len(virus_ids)

    sv = load_fasta(REPO / "data" / "SV.fasta")
    # also pull sequences from pair pickles as fallback
    train = pickle.load(open(REPO / "data/processed/train_protein_pairs.pkl", "rb"))
    test = pickle.load(open(REPO / "data/processed/test_protein_pairs.pkl", "rb"))
    for p in train + test:
        vid = p["virus_id"]
        if vid not in sv and p.get("virus_seq"):
            sv[vid] = p["virus_seq"]
        hid = p["human_id"]
        if p.get("human_seq"):
            # keep last
            pass

    gallery_items = []
    missing = []
    for v in virus_ids:
        if v in sv and sv[v]:
            gallery_items.append((v, sv[v][:512]))
        else:
            missing.append(v)
    if missing:
        raise SystemExit(f"missing {len(missing)} gallery sequences e.g. {missing[:5]}")

    te_pos = [p for p in test if int(p.get("interaction", 0)) == 1]
    # stable order matching main eval / npz: sorted by (human, virus) was NOT used;
    # npz order equals set of pairs — use human_id order from constrained json if needed.
    # Prefer order from per-query npz for alignment with other analyses.
    npz = np.load(
        BIB / "_server14_cis" / "constrained_2104_per_query_hvidb_test.npz",
        allow_pickle=True,
    )
    h_ids = [str(x) for x in npz["human_ids"]]
    v_true = [str(x) for x in npz["virus_ids"]]
    # human sequences from test pickle
    hseq = {}
    for p in te_pos:
        hseq.setdefault(p["human_id"], p["human_seq"])

    queries = []
    for i, (h, v) in enumerate(zip(h_ids, v_true)):
        qid = f"q{i}"
        queries.append((qid, h, v, hseq[h][:512]))

    q_fasta = work / "queries.fasta"
    g_fasta = work / "gallery.fasta"
    write_fasta(q_fasta, [(qid, seq) for qid, _, _, seq in queries])
    write_fasta(g_fasta, gallery_items)

    # MMseqs2 easy-search
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    hits = work / "mmseqs_hits.tsv"
    run(
        [
            str(mmseqs),
            "easy-search",
            str(q_fasta),
            str(g_fasta),
            str(hits),
            str(tmp),
            "--threads",
            str(args.threads),
            "--max-seqs",
            str(n_g),
            "-s",
            "7.5",
            "-e",
            "10000",  # permissive: report weak hits for gallery ranking
            "--format-output",
            "query,target,bits,evalue,pident",
        ]
    )

    # Parse hits: for each query, score[target] = bits (higher better)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with hits.open() as f:
        for line in f:
            q, t, bits, evalue, pident = line.rstrip("\n").split("\t")
            # keep best bits if duplicates
            b = float(bits)
            if t not in scores[q] or b > scores[q][t]:
                scores[q][t] = b

    # Train popularity for Stage-2-like fusion channel (diagnostic; neural Stage-2 not re-run)
    pop = Counter()
    for p in train:
        if int(p.get("interaction", 0)) == 1 and p["virus_id"] in v2j:
            pop[p["virus_id"]] += 1
    pop_vec = np.array([float(pop[v]) for v in virus_ids], dtype=np.float64)

    ranks_out = []
    retr_ranks = np.zeros(len(queries), dtype=np.int32)
    fusion_hit = np.zeros(len(queries), dtype=np.float64)
    oracle_mask = np.zeros(len(queries), dtype=bool)
    recall20 = np.zeros(len(queries))
    recall100 = np.zeros(len(queries))

    for i, (qid, h, vtrue, _) in enumerate(queries):
        sc = np.zeros(n_g, dtype=np.float64)
        hitmap = scores.get(qid, {})
        for t, b in hitmap.items():
            if t in v2j:
                sc[v2j[t]] = b
        # Stage-1: bitscore only (non-hits stay 0; stable id among ties — no popularity leak)
        order = np.lexsort((np.arange(n_g), -sc))
        pos = np.empty(n_g, dtype=np.int32)
        pos[order] = np.arange(1, n_g + 1, dtype=np.int32)
        j = v2j[vtrue]
        rr = int(pos[j])
        retr_ranks[i] = rr
        recall20[i] = 1.0 if rr <= 20 else 0.0
        recall100[i] = 1.0 if rr <= 100 else 0.0
        oracle_mask[i] = rr <= args.T

        # fusion on top-T: α·norm(mmseqs) + (1-α)·norm(pop)
        top = order[: args.T]
        s1 = minmax(sc[top])
        s2 = minmax(pop_vec[top])
        fused = args.alpha * s1 + (1.0 - args.alpha) * s2
        top_order = top[np.argsort(-fused, kind="mergesort")]
        if j in top:
            fr = int(np.where(top_order == j)[0][0]) + 1
            fusion_hit[i] = 1.0 if fr <= args.K else 0.0
            rerank_rank = fr
        else:
            fusion_hit[i] = 0.0
            rerank_rank = ""

        ranks_out.append(
            {
                "query_id": qid,
                "true_id": vtrue,
                "retrieval_rank": rr,
                "rerank_rank": rerank_rank,
                "gallery_size": n_g,
                "in_gallery": True,
                "human_id": h,
            }
        )

    rank_csv = work / "mmseqs_partner_ranks.csv"
    with rank_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "true_id",
                "retrieval_rank",
                "rerank_rank",
                "gallery_size",
                "in_gallery",
                "human_id",
            ],
        )
        w.writeheader()
        for row in ranks_out:
            w.writerow(row)

    om = oracle_mask
    summary = {
        "method": "MMseqs2 easy-search (bitscore) over fixed HVIDB retrievable gallery",
        "mmseqs_binary_sha": subprocess.check_output([str(mmseqs), "version"], text=True).strip(),
        "n_queries": len(queries),
        "gallery_n": n_g,
        "fusion": {
            "alpha": args.alpha,
            "T": args.T,
            "K": args.K,
            "stage2_channel": "train_virus_popularity_on_shortlist",
            "note": (
                "Neural Stage-2 checkpoint not re-run locally; shortlist fusion uses the "
                "same α/T/K recipe as BindingDB diagnostics (popularity channel) so the "
                "framework can audit an external rank file end-to-end."
            ),
        },
        "recall@20": float(recall20.mean()),
        "recall@100": float(recall100.mean()),
        "retrieval_hit@10": float((retr_ranks <= 10).mean()),
        "end_to_end_fusion_hit@10_T20": float(fusion_hit.mean()),
        "oracle@20_n": int(om.sum()),
        "oracle@20_fusion_hit@10": float(fusion_hit[om].mean()) if om.any() else None,
        "identity_abs_error": abs(
            float(fusion_hit.mean())
            - float(recall20.mean()) * (float(fusion_hit[om].mean()) if om.any() else 0.0)
        ),
        "n_queries_with_any_mmseqs_hit": int(sum(1 for q, *_ in queries if scores.get(q))),
        "rank_csv": str(rank_csv),
    }
    # partner-rank-eval if available
    sys.path.insert(0, str(BIB / "partner_rank_eval" / "src"))
    try:
        from partner_rank_eval.io import load_records
        from partner_rank_eval.metrics import metrics_from_ranks

        recs = load_records(rank_csv)
        m = metrics_from_ranks(recs, T=args.T, K=args.K)
        summary["partner_rank_eval_metrics"] = m
    except Exception as e:
        summary["partner_rank_eval_error"] = str(e)

    out_json = work / "mmseqs_hvidb_test_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out_json)


if __name__ == "__main__":
    main()

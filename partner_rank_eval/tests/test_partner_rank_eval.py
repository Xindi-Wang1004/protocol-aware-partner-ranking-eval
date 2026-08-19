from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from partner_rank_eval import __version__
from partner_rank_eval.metrics import QueryRecord, identity_residual, metrics_from_ranks
from partner_rank_eval.simulate import (
    simulate_gallery_nesting,
    simulate_grid,
    simulate_retrieve_rerank,
)
from partner_rank_eval.cli import main
from partner_rank_eval.report import CHECKLIST_ITEMS, empty_checklist, fill_checklist
from partner_rank_eval.io import load_records

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_identity_on_hand_ranks():
    recs = [
        QueryRecord("a", "t", retrieval_rank=2, rerank_rank=1, gallery_size=100),
        QueryRecord("b", "t", retrieval_rank=50, rerank_rank=None, gallery_size=100),
        QueryRecord("c", "t", retrieval_rank=3, rerank_rank=8, gallery_size=100),
        QueryRecord("d", "t", retrieval_rank=1, rerank_rank=12, gallery_size=100),
    ]
    m = metrics_from_ranks(recs, T=20, K=10)
    assert m["n_oracle_subset"] == 3
    assert m["recall@T"] == 0.75
    assert m["end_to_end_hit@K"] == 0.5
    assert identity_residual(m)
    assert m["identity_abs_error"] == 0.0


def test_identity_when_recall_is_zero():
    recs = [QueryRecord("a", "t", retrieval_rank=80, rerank_rank=None, gallery_size=100)]
    m = metrics_from_ranks(recs, T=20, K=10)
    assert m["recall@T"] == 0.0
    assert m["end_to_end_hit@K"] == 0.0
    assert m["recall_x_oracle"] == 0.0
    assert m["identity_abs_error"] == 0.0
    assert math.isnan(m["oracle@K"])


def test_unreachable_partner_is_a_miss():
    recs = [
        QueryRecord("a", "t", retrieval_rank=2, rerank_rank=1, in_gallery=True),
        QueryRecord("b", "t", retrieval_rank=10**9, rerank_rank=None, in_gallery=False),
    ]
    m = metrics_from_ranks(recs, T=20, K=10)
    assert m["reachability"] == 0.5
    assert m["recall@T"] == 0.5
    assert m["end_to_end_hit@K"] == 0.5
    assert identity_residual(m)


def test_k_greater_than_t_is_error():
    recs = [QueryRecord("a", "t", retrieval_rank=1, rerank_rank=1)]
    with pytest.raises(ValueError, match="K <= T"):
        metrics_from_ranks(recs, T=10, K=20)


def test_simulate_identity_zero_residual():
    for r in (0.05, 0.2, 0.5, 0.8):
        for q in (0.5, 0.9):
            m = simulate_retrieve_rerank(
                n_queries=1500, recall_at_T=r, oracle_hit=q, seed=1
            )
            assert identity_residual(m)
            assert abs(m["end_to_end_hit@K"] - r * q) < 0.04


def test_grid_and_nesting_monotonic():
    grid = simulate_grid(n_queries=800, seed=2)
    assert len(grid) == 8
    nest = simulate_gallery_nesting(n_queries=800, seed=2)
    hits = [row["end_to_end_hit@K"] for row in nest]
    assert all(hits[i] <= hits[i + 1] + 1e-12 for i in range(len(hits) - 1))
    assert nest[-1]["monotonic_nondecreasing_hit"] is True
    assert nest[-1]["inflation_vs_full_gallery"] > 1.0
    sizes = [row["gallery_size"] for row in nest]
    assert sizes == [1000, 200, 20, 10]


def test_cli_simulate_and_checklist(tmp_path: Path):
    out = tmp_path / "sim.json"
    main(["simulate", "--n", "400", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert "identity_grid" in payload
    assert "gallery_nesting" in payload
    assert len(payload["identity_grid"]) == 8
    chk = tmp_path / "checklist.json"
    main(["checklist", "--out", str(chk)])
    doc = json.loads(chk.read_text())
    assert doc["n_items"] == 10
    assert doc["n_reported"] == 0


def test_cli_metrics_from_csv(tmp_path: Path):
    csv_path = tmp_path / "ranks.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["query_id", "true_id", "retrieval_rank", "rerank_rank", "gallery_size"],
        )
        w.writeheader()
        w.writerow(
            dict(query_id="q1", true_id="t", retrieval_rank=2, rerank_rank=1, gallery_size=100)
        )
        w.writerow(
            dict(query_id="q2", true_id="t", retrieval_rank=80, rerank_rank="", gallery_size=100)
        )
    out = tmp_path / "m.json"
    main(
        [
            "metrics",
            "--rankings",
            str(csv_path),
            "--protocol",
            "fixed-gallery",
            "--gallery",
            "100 ligands",
            "--checklist",
            "--out",
            str(out),
        ]
    )
    payload = json.loads(out.read_text())
    assert payload["metrics"]["n_queries"] == 2
    assert payload["metrics"]["identity_abs_error"] == 0.0
    assert payload["checklist"]["n_reported"] >= 6
    proto = next(it for it in payload["checklist"]["items"] if it["id"] == "protocol")
    assert proto["value"] == "fixed-gallery"
    assert proto["reported"] is True
    gal = next(it for it in payload["checklist"]["items"] if it["id"] == "evaluation_gallery")
    assert gal["value"] == "100 ligands"


def test_cli_metrics_from_jsonl_matches_csv():
    csv_recs = load_records(EXAMPLES / "toy_ranks.csv")
    jsonl_recs = load_records(EXAMPLES / "toy_ranks.jsonl")
    a = metrics_from_ranks(csv_recs, T=20, K=10)
    b = metrics_from_ranks(jsonl_recs, T=20, K=10)
    assert a["n_queries"] == b["n_queries"] == 8
    assert a["end_to_end_hit@K"] == b["end_to_end_hit@K"]
    assert identity_residual(a) and identity_residual(b)


def test_cli_metrics_jsonl(tmp_path: Path):
    out = tmp_path / "m.json"
    main(["metrics", "--rankings", str(EXAMPLES / "toy_ranks.jsonl"), "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["metrics"]["n_queries"] == 8
    assert "NaN" not in out.read_text()


def test_cli_k_gt_t_exits(tmp_path: Path):
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("query_id,true_id,retrieval_rank\nq1,t,1\n")
    with pytest.raises(SystemExit) as ei:
        main(["metrics", "--rankings", str(csv_path), "--T", "10", "--K", "20"])
    assert ei.value.code == 2


def test_cli_nan_oracle_is_json_null(tmp_path: Path):
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("query_id,true_id,retrieval_rank,rerank_rank\nq1,t,99,\n")
    out = tmp_path / "m.json"
    main(["metrics", "--rankings", str(csv_path), "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["metrics"]["oracle@K"] is None
    assert payload["metrics"]["identity_abs_error"] == 0.0


def test_cli_checklist_from_metrics_json(tmp_path: Path):
    ranks = tmp_path / "ranks.csv"
    ranks.write_text("query_id,true_id,retrieval_rank,rerank_rank\nq1,t,2,1\n")
    metrics_json = tmp_path / "m.json"
    main(["metrics", "--rankings", str(ranks), "--out", str(metrics_json)])
    chk = tmp_path / "c.json"
    main(["checklist", "--metrics", str(metrics_json), "--out", str(chk)])
    doc = json.loads(chk.read_text())
    assert doc["n_reported"] >= 5


def test_cli_requires_command():
    with pytest.raises(SystemExit) as ei:
        main([])
    assert ei.value.code == 2


def test_empty_checklist_schema():
    doc = empty_checklist()
    filled = fill_checklist(
        {
            "T": 20,
            "K": 10,
            "recall@T": 0.1,
            "oracle@K": 0.5,
            "end_to_end_hit@K": 0.05,
            "recall_x_oracle": 0.05,
            "identity_abs_error": 0.0,
            "n_oracle_subset": 10,
            "reachability": 1.0,
        }
    )
    assert filled["n_reported"] >= 5
    assert doc["n_items"] == 10
    assert len(doc["items"]) == 10
    assert len(CHECKLIST_ITEMS) == 10


def test_frozen_table4_identity():
    payload = json.loads((EXAMPLES / "table4_hvidb_identity.json").read_text())
    assert len(payload["arms"]) == 4
    for row in payload["arms"]:
        prod = row["recall@20"] * row["oracle@K"]
        assert abs(row["end_to_end_hit@K"] - prod) <= 1e-12


def test_version_string():
    assert __version__ == "0.1.0"

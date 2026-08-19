"""Reporting-checklist JSON as a fillable artifact (manuscript Table 6).

The Python/CLI name is checklist, not a table number, so renumbering the
manuscript does not require renaming commands or JSON keys.
"""

from __future__ import annotations

from typing import Any, Optional

CHECKLIST_ITEMS: tuple[dict[str, str], ...] = (
    {
        "id": "nominal_universe",
        "item": "Nominal biological universe stated (what the claim is about)",
    },
    {
        "id": "evaluation_gallery",
        "item": "Stated evaluation gallery and its size (what was actually ranked against)",
    },
    {
        "id": "retrievable_gallery",
        "item": "Retrievable gallery (scoreable subset) distinguished from the nominal universe",
    },
    {
        "id": "reachability",
        "item": "Reachability: fraction of labelled true partners present in the retrievable gallery",
    },
    {
        "id": "protocol",
        "item": "Protocol named: pair-subset vs fixed-gallery vs other",
    },
    {
        "id": "retrieval_recall",
        "item": "Stage-1 retrieval Recall@T (and T) if a shortlist is used",
    },
    {
        "id": "end_to_end_hit",
        "item": "End-to-end Hit@K on the stated gallery (and K)",
    },
    {
        "id": "oracle",
        "item": "Oracle@T: conditional Hit@K on queries whose true partner entered the top-T shortlist",
    },
    {
        "id": "identity_check",
        "item": "Report Recall@T × oracle@K next to end-to-end Hit@K (should match when K ≤ T)",
    },
    {
        "id": "model_role",
        "item": "Model role stated: diagnostic vehicle vs proposed predictor; no SOTA claim unless a comparison protocol is fully specified",
    },
)


def empty_checklist() -> dict[str, Any]:
    items = [{**item, "value": None, "reported": False} for item in CHECKLIST_ITEMS]
    return {
        "standard": "protocol-aware minimum reporting for constrained biological partner ranking",
        "version": "0.1.0",
        "n_items": len(items),
        "n_reported": 0,
        "items": items,
    }


def fill_checklist(
    metrics: Optional[dict] = None,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    doc = empty_checklist()
    extra = extra or {}
    auto = {}
    if metrics:
        auto.update(
            {
                "reachability": metrics.get("reachability"),
                "retrieval_recall": {
                    "T": metrics.get("T"),
                    "recall@T": metrics.get("recall@T"),
                },
                "end_to_end_hit": {
                    "K": metrics.get("K"),
                    "hit@K": metrics.get("end_to_end_hit@K"),
                },
                "oracle": {
                    "K": metrics.get("K"),
                    "oracle@K": metrics.get("oracle@K"),
                    "n_subset": metrics.get("n_oracle_subset"),
                },
                "identity_check": {
                    "recall_x_oracle": metrics.get("recall_x_oracle"),
                    "end_to_end_hit@K": metrics.get("end_to_end_hit@K"),
                    "abs_error": metrics.get("identity_abs_error"),
                },
            }
        )
    auto.update(extra)
    for item in doc["items"]:
        if item["id"] in auto and auto[item["id"]] is not None:
            item["value"] = auto[item["id"]]
            item["reported"] = True
    doc["n_reported"] = sum(1 for it in doc["items"] if it["reported"])
    return doc

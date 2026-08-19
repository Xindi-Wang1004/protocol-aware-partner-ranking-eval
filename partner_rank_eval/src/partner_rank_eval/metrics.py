"""Rank metrics and the retrieve-then-rerank decomposition.

Identity (same query set, K <= T)
---------------------------------
Let I_i be 1 iff the true partner of query i has retrieval rank <= T.
Let J_i be 1 iff it has rerank rank <= K among the top-T shortlist
(undefined / 0 when I_i = 0). Then

    Hit@K_end-to-end = mean(I_i and J_i)
    Recall@T         = mean(I_i)
    oracle@K         = mean(J_i | I_i = 1)

hence Hit@K_end-to-end = Recall@T * oracle@K whenever Recall@T > 0.
This is a definitional identity, not an empirical finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class QueryRecord:
    """One labelled query against a stated evaluation gallery."""

    query_id: str
    true_id: str
    retrieval_rank: int  # 1-based; inf-like large int if missing from gallery
    rerank_rank: Optional[int] = None  # 1-based among the shortlist; None if not retrieved
    gallery_size: Optional[int] = None
    in_gallery: bool = True


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def metrics_from_ranks(
    records: Iterable[QueryRecord],
    *,
    T: int = 20,
    K: int = 10,
) -> dict:
    if K > T:
        raise ValueError(f"identity requires K <= T (got K={K}, T={T})")
    recs = list(records)
    if not recs:
        raise ValueError("no queries")

    n = len(recs)
    n_reachable = sum(1 for r in recs if r.in_gallery)
    retrieved = [r for r in recs if r.in_gallery and r.retrieval_rank <= T]
    hits = [
        r
        for r in retrieved
        if r.rerank_rank is not None and r.rerank_rank <= K
    ]
    # retrieval-only Hit@K: true partner already in top-K before rerank
    ret_hits_k = sum(1 for r in recs if r.in_gallery and r.retrieval_rank <= K)

    recall_T = len(retrieved) / n
    oracle_K = (len(hits) / len(retrieved)) if retrieved else float("nan")
    hit_e2e = len(hits) / n
    product = recall_T * oracle_K if retrieved else 0.0

    return {
        "n_queries": n,
        "n_reachable": n_reachable,
        "reachability": n_reachable / n,
        "T": T,
        "K": K,
        "recall@T": recall_T,
        "retrieval_hit@K": ret_hits_k / n,
        "oracle@K": oracle_K,
        "end_to_end_hit@K": hit_e2e,
        "recall_x_oracle": product,
        "identity_abs_error": abs(hit_e2e - product),
        "n_oracle_subset": len(retrieved),
        "gallery_factor": _gallery_factor(recs),
        "gate_factor": 1.0 - recall_T,
    }


def _gallery_factor(recs: Sequence[QueryRecord]) -> float:
    """Mean 1/gallery_size when sizes are reported; else NaN."""
    sizes = [r.gallery_size for r in recs if r.gallery_size]
    if not sizes:
        return float("nan")
    return _mean([1.0 / s for s in sizes])


def decompose(metrics: dict) -> dict:
    """Name the two factors that turn a strong oracle into a weak headline Hit@K."""
    recall = metrics["recall@T"]
    oracle = metrics["oracle@K"]
    return {
        "headline_hit@K": metrics["end_to_end_hit@K"],
        "gate_factor_missed_by_retrieval": 1.0 - recall,
        "retrieval_coverage_recall@T": recall,
        "conditional_rerank_oracle@K": oracle,
        "candidate_set_note": (
            "headline Hit@K is also not comparable across studies unless the "
            "evaluation gallery (and its size) is stated"
        ),
    }


def identity_residual(metrics: dict, tol: float = 1e-12) -> bool:
    return metrics["identity_abs_error"] <= tol

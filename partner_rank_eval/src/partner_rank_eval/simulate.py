"""Controlled simulations for the retrieve-then-rerank identity and gallery nesting."""

from __future__ import annotations

import numpy as np

from .metrics import QueryRecord, metrics_from_ranks


def _retrieval_rank(
    rng: np.random.Generator,
    gallery_size: int,
    T: int,
    in_shortlist: bool,
) -> int:
    if in_shortlist:
        return int(rng.integers(1, T + 1))
    if gallery_size <= T:
        return T
    return int(rng.integers(T + 1, gallery_size + 1))


def _rerank_rank(
    rng: np.random.Generator,
    T: int,
    K: int,
    success: bool,
) -> int:
    if success:
        return int(rng.integers(1, K + 1))
    if T <= K:
        return K
    return int(rng.integers(K + 1, T + 1))


def simulate_retrieve_rerank(
    *,
    n_queries: int = 2000,
    gallery_size: int = 1000,
    T: int = 20,
    K: int = 10,
    recall_at_T: float = 0.2,
    oracle_hit: float = 0.9,
    seed: int = 0,
) -> dict:
    """Draw labelled queries with known retrieval coverage r and rerank quality q.

    End-to-end Hit@K equals r * q in expectation (and in the finite sample,
    up to the identity residual which is numerically zero by construction).
    """
    if not (0.0 <= recall_at_T <= 1.0 and 0.0 <= oracle_hit <= 1.0):
        raise ValueError("recall_at_T and oracle_hit must be in [0, 1]")
    if K > T:
        raise ValueError("K must be <= T")
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_queries):
        retrieved = bool(rng.random() < recall_at_T)
        ret_rank = _retrieval_rank(rng, gallery_size, T, retrieved)
        rerank = None
        if retrieved:
            success = bool(rng.random() < oracle_hit)
            rerank = _rerank_rank(rng, T, K, success)
        records.append(
            QueryRecord(
                query_id=f"q{i}",
                true_id="t",
                retrieval_rank=ret_rank,
                rerank_rank=rerank,
                gallery_size=gallery_size,
                in_gallery=True,
            )
        )
    metrics = metrics_from_ranks(records, T=T, K=K)
    metrics.update(
        {
            "design_recall@T": recall_at_T,
            "design_oracle@K": oracle_hit,
            "design_hit@K": recall_at_T * oracle_hit,
            "gallery_size": gallery_size,
            "seed": seed,
        }
    )
    return metrics


def simulate_grid(
    recalls=(0.05, 0.2, 0.5, 0.8),
    oracles=(0.5, 0.9),
    **kwargs,
) -> list[dict]:
    rows = []
    for r in recalls:
        for q in oracles:
            rows.append(simulate_retrieve_rerank(recall_at_T=r, oracle_hit=q, **kwargs))
    return rows


# Default sizes span fixed large galleries through Hit@K-saturating pair-subset scales.
DEFAULT_NESTED_GALLERY_SIZES: tuple[int, ...] = (1000, 500, 100, 50, 20, 10, 5, 2)


def simulate_gallery_nesting(
    *,
    n_queries: int = 2000,
    full_gallery: int = 1000,
    nested_sizes: tuple[int, ...] = DEFAULT_NESTED_GALLERY_SIZES,
    T: int = 20,
    K: int = 10,
    score_noise: float = 1.0,
    seed: int = 0,
) -> list[dict]:
    """Same underlying scores; evaluation gallery shrinks but always contains the true partner.

    Pair-subset protocols correspond to the smallest nested galleries (|G| ≤ K),
    where Hit@K is structurally 1 whenever the true partner is in the gallery.
    Headline Hit@K is monotonically non-decreasing as the gallery shrinks
    (equivalently, monotonically non-increasing as the gallery grows).
    This matches the empirical pattern that fixed large-gallery Hit@K is lower.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Draw one true-item score and G-1 decoy scores per query; ranks recomputed on prefixes.
    true_scores = rng.normal(0.0, score_noise, size=n_queries)
    decoy_scores = rng.normal(0.0, score_noise, size=(n_queries, full_gallery - 1))

    for G in nested_sizes:
        if G < 1 or G > full_gallery:
            raise ValueError(f"nested size {G} not in 1..{full_gallery}")
        records = []
        n_decoy = G - 1
        for i in range(n_queries):
            scores = np.empty(G, dtype=np.float64)
            scores[0] = true_scores[i]
            if n_decoy:
                scores[1:] = decoy_scores[i, :n_decoy]
            # rank of true item (index 0); higher score is better
            rank = int(np.sum(scores >= scores[0]))  # 1-based with ties-go-to-true
            retrieved = rank <= min(T, G)
            rerank = rank if retrieved else None
            # When G <= K, Hit@K is 1 by construction if true is in gallery.
            records.append(
                QueryRecord(
                    query_id=f"q{i}",
                    true_id="t",
                    retrieval_rank=rank,
                    rerank_rank=rerank,
                    gallery_size=G,
                    in_gallery=True,
                )
            )
        m = metrics_from_ranks(records, T=min(T, G), K=min(K, G))
        m["gallery_size"] = G
        m["protocol"] = "pair-subset" if G <= K else "fixed-gallery"
        rows.append(m)
    hits = [r["end_to_end_hit@K"] for r in rows]
    rows[-1]["monotonic_nondecreasing_hit"] = all(
        hits[i] <= hits[i + 1] + 1e-12 for i in range(len(hits) - 1)
    )
    if len(hits) >= 2 and hits[0] > 0:
        rows[-1]["inflation_vs_full_gallery"] = hits[-1] / hits[0]
    return rows

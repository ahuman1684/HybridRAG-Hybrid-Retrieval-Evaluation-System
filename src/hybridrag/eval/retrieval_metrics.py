"""Standard IR metrics: Precision@k, Recall@k, and Mean Reciprocal Rank.

These are computed per query, then averaged across the eval set by the
harness. Kept dependency-free (stdlib only) since they're pure set/list
arithmetic - no reason to pull in a metrics library for this.
"""

from __future__ import annotations


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are relevant.

    Divides by k (not by however many were actually retrieved), the standard
    IR convention - retrieving fewer than k chunks is itself a precision
    failure the metric should reflect, not paper over.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top-k retrieved."""
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (rank of the first relevant chunk), or 0.0 if none were retrieved.

    Only the *first* relevant hit counts, by definition of MRR - this metric
    answers "how far down the list did I have to look to find something
    useful," not "how many relevant things did I find."
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0

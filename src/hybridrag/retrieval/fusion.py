"""Reciprocal Rank Fusion (RRF), implemented from scratch.

Why RRF instead of combining raw scores:
  Dense cosine similarity lives in [-1, 1] (in practice usually [0, 1] for
  this embedding model) while BM25 scores are unbounded and depend on corpus
  statistics (term rarity, average chunk length). There is no principled way
  to add or weight-average those two distributions without calibration -
  e.g. min-max normalizing each list per query, which is itself an arbitrary
  choice that can flip rankings for edge cases (a single very high BM25
  outlier compresses every other BM25 score toward zero after normalization).

  RRF sidesteps the calibration problem entirely by throwing away the raw
  scores and using only *rank position* within each list:

      RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

  A chunk gets a high fused score by being ranked near the top of one or
  more retrievers, regardless of what that retriever's raw score happened to
  be. This is also why RRF composes cleanly with a third signal later
  (Phase 3's reranker) - it's just another ranked list to fold in, with no
  new normalization problem to solve.

  `k` (default 60, from the original RRF paper, Cormack et al. 2009) damps
  the contribution of low ranks and is deliberately not made per-query -
  it's a property of how much you trust "rank 50" to mean something, not of
  any individual query.
"""

from __future__ import annotations

from hybridrag.types import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    k: int = 60,
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists of the same chunk universe into one.

    A chunk absent from a given ranking contributes 0 to that ranking's term
    of the sum (equivalent to treating its rank as infinite) - it is not
    penalized beyond simply not benefiting from that retriever's vote.
    """
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            chunk_id = scored.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[chunk_id] = scored.chunk

    fused = [ScoredChunk(chunk=chunk_by_id[cid], score=score) for cid, score in rrf_scores.items()]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused

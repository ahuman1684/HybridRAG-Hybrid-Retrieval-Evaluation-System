"""Phase 3 demo: retrieve top-20 via hybrid search, rerank to top-5 with the
cross-encoder, and measure the real latency reranking adds - not an
estimate, the actual wall-clock cost on this machine for this corpus.

Run with:
    python scripts/demo_phase3.py
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybridrag.config import PipelineConfig  # noqa: E402
from hybridrag.pipeline import RagPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_corpus"

QUERIES = [
    "What is the time complexity of BFS?",
    "How does a hash table handle collisions?",
    "When should I use memoization instead of tabulation?",
    "What is the BST property?",
]


def main() -> None:
    print(f"Corpus: {CORPUS_DIR}")
    config = PipelineConfig()
    config.chunking.strategy = "fixed"
    config.reranker.enabled = True

    pipeline = RagPipeline(config)
    t0 = time.perf_counter()
    chunks = pipeline.ingest(str(CORPUS_DIR))
    print(f"Ingested {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s\n")

    retrieval_only_ms = []
    rerank_only_ms = []
    end_to_end_ms = []

    for query in QUERIES:
        print("=" * 90)
        print(f"QUERY: {query}")
        print("-" * 90)

        # Isolate retrieval-only latency: top-20 hybrid candidates, no rerank.
        t0 = time.perf_counter()
        candidates = pipeline.hybrid_retriever.retrieve(
            query,
            top_k=config.retrieval.candidate_pool_size,
            dense_top_k=config.retrieval.dense_top_k,
            sparse_top_k=config.retrieval.sparse_top_k,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000
        retrieval_only_ms.append(retrieval_ms)

        print(f"  Retrieved {len(candidates)} candidates (hybrid, pre-rerank) in {retrieval_ms:.1f}ms")

        # Now rerank those exact candidates and measure just the reranker's cost.
        reranker = pipeline._get_reranker()
        t0 = time.perf_counter()
        reranked = reranker.rerank(query, candidates, top_k=config.retrieval.final_top_k)
        rerank_ms = (time.perf_counter() - t0) * 1000
        rerank_only_ms.append(rerank_ms)
        end_to_end_ms.append(retrieval_ms + rerank_ms)

        print(f"  Reranked to top {len(reranked)} in {rerank_ms:.1f}ms "
              f"({rerank_ms / len(candidates):.2f}ms/candidate)")
        for sc in reranked:
            preview = sc.chunk.text[:80].replace("\n", " ")
            print(f"    [{sc.score:.3f}] {sc.chunk.chunk_id}: {preview}...")
        print()

    print("=" * 90)
    print("LATENCY SUMMARY (mean over %d queries)" % len(QUERIES))
    print("-" * 90)
    print(f"  Hybrid retrieval (top-20):  {statistics.mean(retrieval_only_ms):.1f}ms")
    print(f"  Reranking (20 -> 5):        {statistics.mean(rerank_only_ms):.1f}ms  <- added by this stage")
    print(f"  End-to-end retrieval path:  {statistics.mean(end_to_end_ms):.1f}ms")
    overhead_pct = statistics.mean(rerank_only_ms) / statistics.mean(end_to_end_ms) * 100
    print(f"  Reranking is {overhead_pct:.0f}% of total retrieval-path latency")


if __name__ == "__main__":
    main()

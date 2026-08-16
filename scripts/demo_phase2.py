"""Phase 2 demo: dense-only vs BM25-only vs hybrid (RRF-fused) retrieval,
side by side, on the same queries used in the Phase 1 demo - including the
BFS query that exposed dense retrieval's exact-term blind spot.

Run with:
    python scripts/demo_phase2.py
"""

from __future__ import annotations

import logging
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


def _print_results(label: str, results, latency_ms: float) -> None:
    print(f"  {label} ({latency_ms:.1f}ms):")
    for sc in results:
        preview = sc.chunk.text[:90].replace("\n", " ")
        print(f"    [{sc.score:.4f}] {sc.chunk.chunk_id}: {preview}...")


def main() -> None:
    print(f"Corpus: {CORPUS_DIR}")
    config = PipelineConfig()
    config.chunking.strategy = "fixed"

    pipeline = RagPipeline(config)
    t0 = time.perf_counter()
    chunks = pipeline.ingest(str(CORPUS_DIR))
    print(f"Ingested {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s\n")

    for query in QUERIES:
        print("=" * 90)
        print(f"QUERY: {query}")
        print("-" * 90)

        t0 = time.perf_counter()
        dense_results = pipeline.dense_index.search(query, top_k=3)
        dense_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse_results = pipeline.sparse_index.search(query, top_k=3)
        sparse_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hybrid_results = pipeline.hybrid_retriever.retrieve(
            query, top_k=3, dense_top_k=20, sparse_top_k=20
        )
        hybrid_ms = (time.perf_counter() - t0) * 1000

        _print_results("DENSE ONLY", dense_results, dense_ms)
        _print_results("BM25 ONLY ", sparse_results, sparse_ms)
        _print_results("HYBRID(RRF)", hybrid_results, hybrid_ms)

        dense_top1 = dense_results[0].chunk.chunk_id if dense_results else None
        hybrid_top1 = hybrid_results[0].chunk.chunk_id if hybrid_results else None
        if dense_top1 != hybrid_top1:
            print(f"  >>> Fusion changed the top result: {dense_top1} -> {hybrid_top1}")
        print()


if __name__ == "__main__":
    main()

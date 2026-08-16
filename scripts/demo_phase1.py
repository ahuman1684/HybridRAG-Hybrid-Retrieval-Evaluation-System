"""Phase 1 demo: ingest the sample corpus, build a dense index, run a few
queries, and print retrieval (and generation, if ANTHROPIC_API_KEY is set)
results end to end.

Run with:
    python scripts/demo_phase1.py
"""

from __future__ import annotations

import logging
import os
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

    pipeline = RagPipeline(config)

    t0 = time.perf_counter()
    chunks = pipeline.ingest(str(CORPUS_DIR))
    print(f"Ingested {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s\n")

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_api_key:
        print(
            "ANTHROPIC_API_KEY not set - showing retrieval only. "
            "Export it to also see generated, cited answers.\n"
        )

    for query in QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("-" * 80)

        t0 = time.perf_counter()
        retrieved = pipeline.retrieve(query, top_k=3)
        retrieval_latency = time.perf_counter() - t0
        print(f"Top {len(retrieved)} chunks ({retrieval_latency * 1000:.1f}ms):")
        for sc in retrieved:
            preview = sc.chunk.text[:120].replace("\n", " ")
            print(f"  [{sc.score:.3f}] {sc.chunk.chunk_id}: {preview}...")

        if has_api_key:
            result = pipeline.answer(query, top_k=3)
            print(f"\nANSWER ({result.latency_seconds:.2f}s, "
                  f"{result.input_tokens} in / {result.output_tokens} out tokens):")
            print(result.answer)
            print(f"\nCitations found: {[c.chunk_id for c in result.citations]}")
        print()


if __name__ == "__main__":
    main()

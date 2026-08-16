"""End-to-end orchestration: ingest -> chunk -> embed -> index -> retrieve -> generate.

Phase 1 only wires up dense retrieval. Hybrid fusion (Phase 2) and reranking
(Phase 3) will plug into `retrieve()` without changing `ingest()` or
`answer()`, which is the point of keeping those methods separate rather than
one monolithic `run()`.
"""

from __future__ import annotations

import logging

from hybridrag.config import PipelineConfig
from hybridrag.embeddings import SentenceEmbedder
from hybridrag.generation.generator import Generator
from hybridrag.ingestion.chunking import chunk_fixed_size, chunk_semantic
from hybridrag.ingestion.loaders import load_corpus
from hybridrag.retrieval.dense import DenseIndex
from hybridrag.types import Chunk, GenerationResult, ScoredChunk

logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.embedder = SentenceEmbedder(
            model_name=self.config.embedding.model_name,
        )
        self.dense_index = DenseIndex(self.embedder)
        self.chunks: list[Chunk] = []
        self._generator: Generator | None = None

    def ingest(self, corpus_dir: str) -> list[Chunk]:
        documents = load_corpus(corpus_dir)
        logger.info("Loaded %d documents from %s", len(documents), corpus_dir)

        cfg = self.config.chunking
        if cfg.strategy == "fixed":
            chunks = chunk_fixed_size(
                documents,
                chunk_size=cfg.fixed_chunk_size,
                overlap=cfg.fixed_chunk_overlap,
            )
        elif cfg.strategy == "semantic":
            chunks = chunk_semantic(
                documents,
                embed_fn=lambda texts: self.embedder.embed(
                    texts, batch_size=self.config.embedding.batch_size
                ),
                breakpoint_percentile=cfg.semantic_breakpoint_percentile,
                min_chunk_chars=cfg.semantic_min_chunk_chars,
                max_chunk_chars=cfg.semantic_max_chunk_chars,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {cfg.strategy}")

        logger.info("Produced %d chunks using '%s' strategy", len(chunks), cfg.strategy)
        self.chunks = chunks
        self.dense_index.build(chunks)
        return chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self.config.retrieval.final_top_k
        return self.dense_index.search(query, top_k=k)

    def answer(self, query: str, top_k: int | None = None) -> GenerationResult:
        if self._generator is None:
            gen_cfg = self.config.generation
            self._generator = Generator(
                model=gen_cfg.model,
                max_tokens=gen_cfg.max_tokens,
                temperature=gen_cfg.temperature,
            )
        retrieved = self.retrieve(query, top_k=top_k)
        return self._generator.generate(query, retrieved)

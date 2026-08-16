"""End-to-end orchestration: ingest -> chunk -> embed -> index -> retrieve -> generate.

Phase 1 wired up dense retrieval. Phase 2 added BM25 + RRF fusion behind the
same `retrieve()` method, selected by `config.retrieval.mode`. Phase 3 adds
an optional reranking stage after retrieval, selected by
`config.reranker.enabled` - the combination of these two switches is what
lets Phase 5's ablation study reproduce configs (a) dense-only, (b)
dense+BM25, (c) dense+BM25+reranker by changing two config values, not three
separate code paths.
"""

from __future__ import annotations

import logging

from hybridrag.config import PipelineConfig
from hybridrag.embeddings import SentenceEmbedder
from hybridrag.generation.generator import Generator
from hybridrag.ingestion.chunking import chunk_fixed_size, chunk_semantic
from hybridrag.ingestion.loaders import load_corpus
from hybridrag.retrieval.dense import DenseIndex
from hybridrag.retrieval.hybrid import HybridRetriever
from hybridrag.retrieval.reranker import CrossEncoderReranker
from hybridrag.retrieval.sparse import BM25Index
from hybridrag.types import Chunk, GenerationResult, ScoredChunk

logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.embedder = SentenceEmbedder(
            model_name=self.config.embedding.model_name,
        )
        self.dense_index = DenseIndex(self.embedder)
        self.sparse_index = BM25Index()
        self.hybrid_retriever: HybridRetriever | None = None
        self.chunks: list[Chunk] = []
        self._generator: Generator | None = None
        self._reranker: CrossEncoderReranker | None = None

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
        self.sparse_index.build(chunks)
        self.hybrid_retriever = HybridRetriever(
            self.dense_index, self.sparse_index, rrf_k=self.config.retrieval.rrf_k
        )
        return chunks

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=self.config.reranker.model_name)
        return self._reranker

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        cfg = self.config.retrieval
        rerank_cfg = self.config.reranker
        final_k = top_k or cfg.final_top_k

        # When reranking, pull a wider candidate pool (default 20) so the
        # cross-encoder has real alternatives to compare beyond just the
        # final_k retrieval already would have returned - otherwise
        # reranking could only ever reorder the same 5 candidates.
        pool_k = cfg.candidate_pool_size if rerank_cfg.enabled else final_k

        if cfg.mode == "dense":
            candidates = self.dense_index.search(query, top_k=pool_k)
        elif cfg.mode == "hybrid":
            assert self.hybrid_retriever is not None, "call ingest() before retrieve()"
            candidates = self.hybrid_retriever.retrieve(
                query, top_k=pool_k, dense_top_k=cfg.dense_top_k, sparse_top_k=cfg.sparse_top_k
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {cfg.mode}")

        if rerank_cfg.enabled:
            candidates = self._get_reranker().rerank(query, candidates, top_k=final_k)

        return candidates

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

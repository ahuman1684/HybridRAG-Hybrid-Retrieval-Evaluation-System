"""Central configuration for the pipeline.

A single dataclass rather than scattered magic numbers/strings so every
pipeline stage (and every ablation config in Phase 5) can be described by
one object that's easy to log alongside eval results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"  # "fixed" | "semantic"

    # Fixed-size chunking, measured in whitespace-split words (a cheap proxy
    # for tokens that avoids pulling in a tokenizer dependency just to chunk).
    fixed_chunk_size: int = 220
    fixed_chunk_overlap: int = 40

    # Semantic chunking: sentences are grouped until the embedding-similarity
    # between consecutive sentences drops below a breakpoint, found as a
    # percentile of the distance distribution within each document (see
    # ingestion/chunking.py for why a percentile beats a fixed threshold).
    semantic_breakpoint_percentile: float = 90.0
    semantic_min_chunk_chars: int = 200
    semantic_max_chunk_chars: int = 2000


@dataclass
class EmbeddingConfig:
    # all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU. Good enough for a
    # low-hundreds-of-pages corpus; swappable for a larger model later to
    # measure the retrieval-quality/latency tradeoff in the ablation study.
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    # "dense" | "hybrid". Kept as one switch (rather than separate booleans
    # for "use BM25" / "use fusion") so Phase 5's ablation configs are each
    # exactly one config value away from each other: (a) dense-only is
    # mode="dense", (b) dense+BM25 is mode="hybrid" with reranker disabled,
    # (c) adds reranker.enabled=True on top of (b).
    mode: str = "hybrid"

    # Candidates pulled from each individual retriever before fusion. Larger
    # than final_top_k so RRF has enough of each ranking to actually combine -
    # see hybrid.py for why.
    dense_top_k: int = 20
    sparse_top_k: int = 20

    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF
    # paper (Cormack et al., 2009) and is not sensitive to small corpora, so
    # it's left fixed rather than exposed as a tuning knob.
    rrf_k: int = 60

    final_top_k: int = 5


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enabled: bool = False


@dataclass
class GenerationConfig:
    model: str = "claude-sonnet-5"
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

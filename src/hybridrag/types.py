"""Core data model shared across ingestion, retrieval, and generation.

Kept dependency-free (stdlib only) so every other module can import these
without pulling in heavy libraries just to type-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    """File format a Document was loaded from, kept for eval breakdowns by type."""

    MARKDOWN = "markdown"
    PDF = "pdf"
    TEXT = "text"


@dataclass(frozen=True)
class Document:
    """A single ingested file, before chunking."""

    doc_id: str
    source_path: str
    source_type: SourceType
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text produced by a chunking strategy.

    `chunk_id` is the identifier the LLM is asked to cite and the identifier
    the eval harness's labeled (query, chunk_id) pairs point at, so it must be
    stable across re-runs of ingestion for the same corpus + chunking config.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk plus a retrieval score, returned by any retriever."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """One claim-to-evidence link extracted from a generated answer."""

    chunk_id: str
    claim: str


@dataclass
class GenerationResult:
    """Output of the generation stage: the answer plus its claimed citations."""

    answer: str
    citations: list[Citation]
    raw_response: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float

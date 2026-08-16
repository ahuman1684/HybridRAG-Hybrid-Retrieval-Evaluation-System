from __future__ import annotations

import numpy as np
import pytest

from hybridrag.types import Document, SourceType


@pytest.fixture
def sample_document() -> Document:
    text = (
        "The quick brown fox jumps over the lazy dog. " * 2
        + "A second topic begins here about something entirely different. " * 2
        + "This third topic is again unrelated to the first two topics discussed above. " * 2
    )
    return Document(
        doc_id="sample.txt",
        source_path="/tmp/sample.txt",
        source_type=SourceType.TEXT,
        text=text.strip(),
    )


@pytest.fixture
def fake_embed_fn():
    """Deterministic fake embedder: returns a distinct-but-clustered vector per
    sentence based on which "topic" keyword it contains, so semantic chunking
    tests don't need a real model download."""

    def _embed(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "fox" in lowered:
                base = np.array([1.0, 0.0, 0.0])
            elif "second topic" in lowered:
                base = np.array([0.0, 1.0, 0.0])
            else:
                base = np.array([0.0, 0.0, 1.0])
            vectors.append(base)
        return np.array(vectors, dtype=np.float32)

    return _embed

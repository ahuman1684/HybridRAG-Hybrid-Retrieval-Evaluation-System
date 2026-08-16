from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridrag.retrieval.dense import DenseIndex
from hybridrag.types import Chunk


class FakeEmbedder:
    """Maps specific known strings to orthogonal vectors, so search results
    are deterministic without loading a real sentence-transformers model."""

    _VOCAB = {
        "cat": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "dog": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "car": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.array([self._VOCAB[t] for t in texts], dtype=np.float32)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc.txt",
        text=text,
        start_char=0,
        end_char=len(text),
        chunk_index=0,
    )


class TestDenseIndex:
    def test_search_returns_closest_chunk_first(self):
        index = DenseIndex(FakeEmbedder())
        chunks = [_chunk("c_cat", "cat"), _chunk("c_dog", "dog"), _chunk("c_car", "car")]
        index.build(chunks)

        results = index.search("cat", top_k=3)
        assert results[0].chunk.chunk_id == "c_cat"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_search_respects_top_k(self):
        index = DenseIndex(FakeEmbedder())
        chunks = [_chunk("c_cat", "cat"), _chunk("c_dog", "dog"), _chunk("c_car", "car")]
        index.build(chunks)

        results = index.search("cat", top_k=1)
        assert len(results) == 1

    def test_build_rejects_empty_chunk_list(self):
        index = DenseIndex(FakeEmbedder())
        with pytest.raises(ValueError):
            index.build([])

    def test_search_before_build_raises(self):
        index = DenseIndex(FakeEmbedder())
        with pytest.raises(RuntimeError):
            index.search("cat")

    def test_save_and_load_round_trip(self, tmp_path: Path):
        index = DenseIndex(FakeEmbedder())
        chunks = [_chunk("c_cat", "cat"), _chunk("c_dog", "dog")]
        index.build(chunks)
        index.save(tmp_path / "index")

        loaded = DenseIndex.load(tmp_path / "index", FakeEmbedder())
        results = loaded.search("dog", top_k=1)
        assert results[0].chunk.chunk_id == "c_dog"
        assert results[0].chunk.text == "dog"

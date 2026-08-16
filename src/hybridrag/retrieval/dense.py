"""Dense (embedding) retrieval backed by a FAISS flat index.

Uses IndexFlatIP (inner product) over L2-normalized vectors, which is
mathematically equivalent to exact cosine similarity search. An approximate
index (IVF/HNSW) would trade recall for speed - a trade not worth making at
this corpus scale (low hundreds of pages -> a few thousand chunks at most),
where an exact flat index is already sub-millisecond per query. Approximate
indexing is the right call once the corpus is large enough that flat search
becomes the bottleneck, which this one isn't.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from hybridrag.embeddings import SentenceEmbedder
from hybridrag.types import Chunk, ScoredChunk


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a zero vector for empty/whitespace text
    return vectors / norms


class DenseIndex:
    """FAISS-backed dense retriever over a fixed set of Chunks.

    Chunks are stored in insertion order and addressed by their position in
    the FAISS index; `chunk_id` -> position is kept in `_id_to_pos` so
    `search` can return real Chunk objects rather than bare integer ids.
    """

    def __init__(self, embedder: SentenceEmbedder) -> None:
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self._id_to_pos: dict[str, int] = {}

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")

        embeddings = self.embedder.embed([c.text for c in chunks])
        embeddings = _normalize(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = list(chunks)
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded yet")

        query_vec = _normalize(self.embedder.embed([query]))
        top_k = min(top_k, len(self.chunks))
        scores, positions = self.index.search(query_vec, top_k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            results.append(ScoredChunk(chunk=self.chunks[pos], score=float(score)))
        return results

    def save(self, dir_path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Nothing to save: index has not been built")
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        chunks_payload = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (out_dir / "chunks.json").write_text(json.dumps(chunks_payload, indent=2))

    @classmethod
    def load(cls, dir_path: str | Path, embedder: SentenceEmbedder) -> DenseIndex:
        in_dir = Path(dir_path)
        instance = cls(embedder)
        instance.index = faiss.read_index(str(in_dir / "index.faiss"))
        payload = json.loads((in_dir / "chunks.json").read_text())
        instance.chunks = [Chunk(**item) for item in payload]
        instance._id_to_pos = {c.chunk_id: i for i, c in enumerate(instance.chunks)}
        return instance

"""Sparse (lexical) retrieval via BM25.

BM25 scores a chunk by exact term overlap with the query (weighted by term
rarity across the corpus and normalized for chunk length), which is the
complement of what dense embeddings are weak at: a chunk that literally
contains the query's acronym or identifier is ranked highly regardless of
whether the embedding model happened to place that acronym's vector near the
query's - see hybrid.py for the concrete case this fixes in this corpus.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from hybridrag.types import Chunk, ScoredChunk

# Alphanumeric runs only, lowercased. This deliberately keeps "BFS" -> "bfs"
# and splits "O(log" -> ["o", "log"] rather than trying to preserve
# programming-notation tokens intact (e.g. via a code-aware tokenizer) -
# that precision isn't worth a custom tokenizer for a notes corpus where
# most exact-match value comes from whole-word acronyms and identifiers.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """BM25-backed sparse retriever over a fixed set of Chunks.

    Mirrors DenseIndex's build/search/save/load interface so the pipeline
    and the eval harness can treat both retrievers uniformly.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("Index has not been built or loaded yet")

        scores = self._bm25.get_scores(_tokenize(query))
        top_k = min(top_k, len(self.chunks))
        # argpartition + sort is O(n + k log k) rather than a full O(n log n)
        # sort; matters little at this corpus size but is the correct habit.
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                # A zero BM25 score means no query term appears in this chunk
                # at all - it wasn't "retrieved," it's just what's left after
                # sorting a mostly-zero array. Excluding it keeps RRF fusion
                # honest about what each retriever actually found.
                continue
            results.append(ScoredChunk(chunk=self.chunks[i], score=score))
        return results

    def save(self, dir_path: str | Path) -> None:
        if self._bm25 is None:
            raise RuntimeError("Nothing to save: index has not been built")
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
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
    def load(cls, dir_path: str | Path) -> BM25Index:
        in_dir = Path(dir_path)
        instance = cls()
        with open(in_dir / "bm25.pkl", "rb") as f:
            instance._bm25 = pickle.load(f)
        payload = json.loads((in_dir / "chunks.json").read_text())
        instance.chunks = [Chunk(**item) for item in payload]
        return instance

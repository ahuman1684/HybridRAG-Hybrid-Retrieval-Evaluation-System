from __future__ import annotations

from pathlib import Path

import pytest

from hybridrag.retrieval.sparse import BM25Index, _tokenize
from hybridrag.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc.txt", text=text, start_char=0, end_char=len(text), chunk_index=0)


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert _tokenize("O(log n) is BFS-friendly!") == ["o", "log", "n", "is", "bfs", "friendly"]

    def test_empty_string_yields_no_tokens(self):
        assert _tokenize("") == []


class TestBM25Index:
    def test_exact_term_match_outranks_unrelated_chunk(self):
        index = BM25Index()
        # 4 chunks (not 2): with classic BM25 IDF, log((N-n+0.5)/(n+0.5)) hits
        # exactly zero when a term appears in half the corpus, which a 2-doc
        # corpus triggers immediately (see test_idf_can_be_zero_in_tiny_corpora
        # below) and would make this test's "distinguishing" assertion
        # meaningless.
        chunks = [
            _chunk("c_bfs", "Breadth-first search BFS explores level by level using a queue."),
            _chunk("c_unrelated_1", "Hash tables use a hash function to map keys to buckets."),
            _chunk("c_unrelated_2", "Dynamic programming caches overlapping subproblems."),
            _chunk("c_unrelated_3", "Binary search trees keep keys in sorted order."),
        ]
        index.build(chunks)

        results = index.search("What is BFS?", top_k=4)
        assert results[0].chunk.chunk_id == "c_bfs"

    def test_chunk_with_no_overlapping_terms_is_excluded(self):
        index = BM25Index()
        chunks = [
            _chunk("c_a", "apple banana cherry"),
            _chunk("c_b", "completely unrelated words here"),
            _chunk("c_c", "yet another unrelated chunk entirely"),
        ]
        index.build(chunks)

        results = index.search("apple", top_k=3)
        result_ids = {sc.chunk.chunk_id for sc in results}
        assert "c_a" in result_ids
        assert "c_b" not in result_ids

    def test_search_respects_top_k(self):
        index = BM25Index()
        # "target" appears in 3 of 7 chunks (n < N/2), which keeps its IDF
        # positive - see test_idf_can_be_zero_in_tiny_corpora for why a term
        # appearing in *most* or *all* chunks would instead get a zero or
        # negative IDF and be filtered out entirely, breaking this test.
        chunks = [_chunk(f"c_{i}", "target term here") for i in range(3)]
        chunks += [_chunk(f"c_other_{i}", "completely unrelated filler text") for i in range(4)]
        index.build(chunks)
        results = index.search("target term", top_k=2)
        assert len(results) == 2

    def test_build_rejects_empty_chunk_list(self):
        with pytest.raises(ValueError):
            BM25Index().build([])

    def test_search_before_build_raises(self):
        with pytest.raises(RuntimeError):
            BM25Index().search("query")

    def test_save_and_load_round_trip(self, tmp_path: Path):
        index = BM25Index()
        chunks = [
            _chunk("c_bfs", "Breadth-first search BFS explores level by level."),
            _chunk("c_dfs", "Depth-first search DFS explores as far as possible."),
            _chunk("c_other", "Hash tables map keys to buckets via a hash function."),
        ]
        index.build(chunks)
        index.save(tmp_path / "bm25")

        loaded = BM25Index.load(tmp_path / "bm25")
        results = loaded.search("DFS", top_k=1)
        assert results[0].chunk.chunk_id == "c_dfs"

    def test_idf_can_be_zero_in_tiny_corpora(self):
        """Documents a real BM25 property, not a bug: classic Robertson-Sparck
        Jones IDF is log((N - n + 0.5) / (n + 0.5)), which is exactly 0 when a
        term appears in precisely half the corpus (e.g. 1 of 2 documents).
        A term that BM25 would treat as informative in a larger corpus can
        score zero relevance in a small one purely from corpus size - worth
        knowing when this pipeline is run on a very small document set.
        """
        index = BM25Index()
        chunks = [_chunk("c_a", "apple"), _chunk("c_b", "banana")]
        index.build(chunks)

        results = index.search("apple", top_k=2)
        assert results == []

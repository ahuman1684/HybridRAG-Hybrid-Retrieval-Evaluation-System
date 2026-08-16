from __future__ import annotations

import pytest

from hybridrag.ingestion.chunking import _split_sentences, chunk_fixed_size, chunk_semantic
from hybridrag.types import Document, SourceType


def _doc(text: str, doc_id: str = "doc.txt") -> Document:
    return Document(doc_id=doc_id, source_path=f"/tmp/{doc_id}", source_type=SourceType.TEXT, text=text)


class TestFixedSizeChunking:
    def test_single_short_document_yields_one_chunk(self):
        doc = _doc("one two three four five")
        chunks = chunk_fixed_size([doc], chunk_size=10, overlap=2)
        assert len(chunks) == 1
        assert chunks[0].text == "one two three four five"
        assert chunks[0].chunk_id == "doc.txt::chunk_0"

    def test_long_document_is_split_into_multiple_chunks(self):
        words = [f"word{i}" for i in range(100)]
        doc = _doc(" ".join(words))
        chunks = chunk_fixed_size([doc], chunk_size=30, overlap=5)
        assert len(chunks) > 1
        # chunk_index should be sequential starting at 0
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_overlap_repeats_words_between_consecutive_chunks(self):
        words = [f"word{i}" for i in range(50)]
        doc = _doc(" ".join(words))
        chunks = chunk_fixed_size([doc], chunk_size=20, overlap=5)

        first_words = chunks[0].text.split()
        second_words = chunks[1].text.split()
        # last `overlap` words of chunk 0 should equal first `overlap` words of chunk 1
        assert first_words[-5:] == second_words[:5]

    def test_char_offsets_are_correct_even_with_repeated_words(self):
        # "the" repeats many times; start_char/end_char must track the actual
        # occurrence at this position, not the first occurrence in the doc.
        doc = _doc("the cat sat on the mat while the dog ran to the store")
        chunks = chunk_fixed_size([doc], chunk_size=4, overlap=1)
        for chunk in chunks:
            assert doc.text[chunk.start_char : chunk.end_char] == chunk.text

    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_fixed_size([_doc("a b c")], chunk_size=5, overlap=5)

    def test_empty_document_yields_no_chunks(self):
        chunks = chunk_fixed_size([_doc("   ")], chunk_size=10, overlap=2)
        assert chunks == []

    def test_chunk_ids_are_stable_and_unique_across_documents(self):
        doc_a = _doc("a b c d e f g h", doc_id="a.txt")
        doc_b = _doc("a b c d e f g h", doc_id="b.txt")
        chunks = chunk_fixed_size([doc_a, doc_b], chunk_size=4, overlap=1)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestSentenceSplitting:
    def test_splits_on_terminal_punctuation(self):
        text = "First sentence. Second sentence! Third sentence?"
        assert _split_sentences(text) == [
            "First sentence.",
            "Second sentence!",
            "Third sentence?",
        ]

    def test_empty_string_yields_no_sentences(self):
        assert _split_sentences("") == []


class TestSemanticChunking:
    def test_groups_similar_sentences_and_splits_on_topic_shift(self, sample_document, fake_embed_fn):
        chunks = chunk_semantic(
            [sample_document],
            embed_fn=fake_embed_fn,
            breakpoint_percentile=50.0,
            min_chunk_chars=0,
            max_chunk_chars=10_000,
        )
        # Three clearly distinct topics (fox / second topic / third topic) should
        # not all be merged into a single chunk.
        assert len(chunks) >= 2

    def test_max_chunk_chars_forces_a_split(self, fake_embed_fn):
        # All sentences map to the same embedding cluster (no natural breakpoint),
        # so only the max-size guard should force chunk boundaries.
        doc = _doc("The fox runs. " * 40)
        chunks = chunk_semantic(
            [doc],
            embed_fn=fake_embed_fn,
            breakpoint_percentile=99.0,
            min_chunk_chars=0,
            max_chunk_chars=100,
        )
        assert len(chunks) > 1
        assert all(len(c.text) <= 150 for c in chunks)  # some slack over the hard cap is fine

    def test_single_sentence_document_yields_one_chunk(self, fake_embed_fn):
        doc = _doc("Just one sentence here.")
        chunks = chunk_semantic([doc], embed_fn=fake_embed_fn)
        assert len(chunks) == 1
        assert chunks[0].text == "Just one sentence here."

    def test_empty_document_yields_no_chunks(self, fake_embed_fn):
        chunks = chunk_semantic([_doc("")], embed_fn=fake_embed_fn)
        assert chunks == []

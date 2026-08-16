from __future__ import annotations

from pathlib import Path

from hybridrag.ingestion.loaders import load_corpus, load_document
from hybridrag.types import SourceType


class TestLoadDocument:
    def test_loads_markdown_file(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("# Heading\n\nSome content.")
        doc = load_document(f, tmp_path)
        assert doc is not None
        assert doc.source_type == SourceType.MARKDOWN
        assert doc.doc_id == "notes.md"
        assert "Some content." in doc.text

    def test_loads_text_file(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("Plain text content.")
        doc = load_document(f, tmp_path)
        assert doc is not None
        assert doc.source_type == SourceType.TEXT

    def test_doc_id_is_relative_to_corpus_root(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        f = subdir / "notes.md"
        f.write_text("content")
        doc = load_document(f, tmp_path)
        assert doc is not None
        assert doc.doc_id == "sub/notes.md"

    def test_empty_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n  ")
        assert load_document(f, tmp_path) is None


class TestLoadCorpus:
    def test_loads_all_supported_file_types(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("markdown content")
        (tmp_path / "b.txt").write_text("text content")
        (tmp_path / "c.ignored").write_text("should be skipped")

        docs = load_corpus(tmp_path)
        doc_ids = {d.doc_id for d in docs}
        assert doc_ids == {"a.md", "b.txt"}

    def test_recurses_into_subdirectories(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.md").write_text("nested content")

        docs = load_corpus(tmp_path)
        assert any(d.doc_id == "sub/nested.md" for d in docs)

    def test_raises_on_missing_directory(self, tmp_path: Path):
        import pytest

        with pytest.raises(NotADirectoryError):
            load_corpus(tmp_path / "does_not_exist")

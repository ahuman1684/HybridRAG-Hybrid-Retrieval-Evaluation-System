"""Load a folder of .md/.pdf/.txt files into Document objects.

Markdown is loaded as raw text rather than rendered/stripped of syntax:
headers, code fences, and bullet markup carry semantic signal (e.g. a
markdown header is a strong hint of topic boundary, useful later for
chunking) and are cheap for the embedding model to see past. Stripping
markdown would be a lossy step done for no measured benefit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from hybridrag.types import Document, SourceType

logger = logging.getLogger(__name__)

_LOADERS_BY_SUFFIX = {".md", ".markdown", ".txt", ".pdf"}


def _source_type_for(suffix: str) -> SourceType:
    if suffix == ".pdf":
        return SourceType.PDF
    if suffix in (".md", ".markdown"):
        return SourceType.MARKDOWN
    return SourceType.TEXT


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
        else:
            logger.warning(
                "No extractable text on page %d of %s (likely a scanned "
                "image page; OCR is out of scope for this pipeline)",
                page_num,
                path,
            )
    return "\n\n".join(pages)


def load_document(path: Path, corpus_root: Path) -> Document | None:
    """Load a single file into a Document, or None if it yielded no text.

    `doc_id` is the path relative to the corpus root (not a hash or UUID) so
    that citations in generated answers and eval labels stay human-readable
    and stable across re-ingestion as long as filenames don't change.
    """
    suffix = path.suffix.lower()
    source_type = _source_type_for(suffix)

    if suffix == ".pdf":
        text = _read_pdf_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    text = text.strip()
    if not text:
        logger.warning("Skipping %s: no extractable text", path)
        return None

    doc_id = str(path.relative_to(corpus_root))
    return Document(
        doc_id=doc_id,
        source_path=str(path),
        source_type=source_type,
        text=text,
        metadata={"filename": path.name},
    )


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Recursively load every supported file under `corpus_dir`.

    Assumes nothing about corpus size beyond "fits in memory as text", which
    holds comfortably for a low-hundreds-of-pages corpus.
    """
    corpus_root = Path(corpus_dir)
    if not corpus_root.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {corpus_root}")

    documents: list[Document] = []
    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _LOADERS_BY_SUFFIX:
            continue
        doc = load_document(path, corpus_root)
        if doc is not None:
            documents.append(doc)

    if not documents:
        logger.warning("No documents loaded from %s", corpus_root)
    return documents

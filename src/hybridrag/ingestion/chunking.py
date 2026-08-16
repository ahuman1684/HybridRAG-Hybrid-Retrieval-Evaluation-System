"""Two chunking strategies: fixed-size with overlap, and semantic (embedding-
similarity-based) chunking.

Both take a list of Documents and an embedder (semantic chunking needs it to
score sentence-boundary similarity; fixed-size ignores it) and return a flat
list of Chunks with stable, citable chunk_ids.

Why implement both instead of picking one:
  Fixed-size is the simplest possible baseline - fast, deterministic, no
  dependency on embedding quality - but it cuts chunks mid-thought whenever a
  concept boundary doesn't land on a word-count multiple. Semantic chunking
  respects the document's actual topic boundaries, at the cost of embedding
  every sentence up front and a less predictable chunk-size distribution.
  The eval harness in Phase 4/5 measures whether that cost is worth it for
  *this* corpus rather than assuming the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from hybridrag.types import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    """Regex sentence splitter.

    A full sentence tokenizer (e.g. nltk's punkt) would handle abbreviations
    and edge cases more precisely, but pulling in a large NLP dependency and
    its model download just to split sentences is disproportionate here -
    the semantic chunker only needs sentence-*ish* boundaries to compute
    similarity shifts, not linguistically perfect ones.
    """
    text = text.strip()
    if not text:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_fixed_size(
    documents: list[Document],
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Split each document into overlapping windows of `chunk_size` words.

    Overlap exists so a fact stated right at a chunk boundary is still fully
    contained in at least one chunk - without it, a sentence split across
    chunk N and N+1 can end up unretrievable by either.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        # Track (word, start_char, end_char) via regex rather than str.split()
        # + str.find(), since find() would match the *first* occurrence of a
        # repeated word/token (e.g. "the", "O(n)") instead of the one at this
        # position - silently wrong start_char/end_char in any text with
        # repeated tokens, which is most technical writing.
        words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", doc.text)]
        if not words:
            continue

        step = chunk_size - overlap
        chunk_index = 0
        for start_word in range(0, len(words), step):
            end_word = min(start_word + chunk_size, len(words))
            chunk_words = words[start_word:end_word]
            start_char = chunk_words[0][1]
            end_char = chunk_words[-1][2]
            chunk_text = doc.text[start_char:end_char]

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk_{chunk_index}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_index,
                    metadata={"strategy": "fixed", "num_words": len(chunk_words)},
                )
            )
            chunk_index += 1

            if end_word == len(words):
                break

    return chunks


@dataclass
class _Sentence:
    text: str
    start_char: int
    end_char: int


def _locate_sentences(doc_text: str, sentences: list[str]) -> list[_Sentence]:
    """Recover character offsets for each sentence within the original text."""
    located = []
    cursor = 0
    for sent in sentences:
        idx = doc_text.find(sent, cursor)
        if idx == -1:
            idx = cursor
        located.append(_Sentence(text=sent, start_char=idx, end_char=idx + len(sent)))
        cursor = idx + len(sent)
    return located


def chunk_semantic(
    documents: list[Document],
    embed_fn,
    breakpoint_percentile: float = 90.0,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 2000,
) -> list[Chunk]:
    """Group consecutive sentences into chunks, splitting where embedding
    similarity between neighboring sentences drops sharply.

    Method (per document, following Kamradt's semantic chunking approach):
      1. Split into sentences and embed each one.
      2. Compute cosine distance between every pair of consecutive sentences.
      3. Take the `breakpoint_percentile`-th percentile of those distances as
         a per-document threshold - a fixed absolute threshold doesn't
         transfer across documents with different writing styles/topic
         density, but a percentile adapts to each document's own
         distribution of "how much does topic drift sentence-to-sentence".
      4. Start a new chunk wherever a distance exceeds the threshold.
      5. Enforce min/max chunk sizes so a document with uniformly similar
         sentences doesn't collapse into one giant chunk, and a document
         with uniformly dissimilar sentences doesn't fragment into
         one-sentence chunks.

    `embed_fn` is injected (rather than importing SentenceEmbedder directly)
    so this module - and its tests - don't need to load a real model.
    """
    chunks: list[Chunk] = []

    for doc in documents:
        sentences_text = _split_sentences(doc.text)
        if not sentences_text:
            continue
        if len(sentences_text) == 1:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk_0",
                    doc_id=doc.doc_id,
                    text=sentences_text[0],
                    start_char=0,
                    end_char=len(doc.text),
                    chunk_index=0,
                    metadata={"strategy": "semantic", "num_sentences": 1},
                )
            )
            continue

        sentences = _locate_sentences(doc.text, sentences_text)
        embeddings = embed_fn([s.text for s in sentences])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        distances = []
        for i in range(len(embeddings) - 1):
            cosine_sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            distances.append(1.0 - cosine_sim)

        threshold = float(np.percentile(distances, breakpoint_percentile))
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}

        groups: list[list[_Sentence]] = []
        current: list[_Sentence] = [sentences[0]]
        for i in range(1, len(sentences)):
            group_char_len = sentences[i].end_char - current[0].start_char
            should_break = (i - 1) in breakpoints and group_char_len >= min_chunk_chars
            should_force_break = group_char_len >= max_chunk_chars
            if should_break or should_force_break:
                groups.append(current)
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            groups.append(current)

        for chunk_index, group in enumerate(groups):
            chunk_text = " ".join(s.text for s in group)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk_{chunk_index}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=group[0].start_char,
                    end_char=group[-1].end_char,
                    chunk_index=chunk_index,
                    metadata={"strategy": "semantic", "num_sentences": len(group)},
                )
            )

    return chunks

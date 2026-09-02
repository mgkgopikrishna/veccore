"""Splitting documents into chunks.

Chunking is part of the embedding space, not a detail beside it: the same model over
differently chunked text produces vectors that are not comparable to each other. That
is why `EmbeddingSpace.preprocessor_version` exists, and why changing chunk size is a
migration, not a config tweak. Getting this wrong is the second most common way a RAG
index silently rots.
"""

from __future__ import annotations

import re

from .models import Chunk, Document

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_document(
    doc: Document, *, max_chars: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Split on sentence boundaries, packing up to `max_chars`, with a character overlap.

    Overlap exists so a fact that straddles a boundary is retrievable from both sides.
    It costs storage proportional to overlap/max_chars -- 15% here.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    text = doc.text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    chunks: list[Chunk] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if not buf.strip():
            return
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc.id}::{ordinal}",
                doc_id=doc.id,
                ordinal=ordinal,
                text=buf.strip(),
                metadata=dict(doc.metadata),
            )
        )
        buf = buf[-overlap:] if overlap else ""

    for sentence in sentences:
        # A single sentence longer than the budget gets hard-split rather than dropped.
        while len(sentence) > max_chars:
            flush()
            head, sentence = sentence[:max_chars], sentence[max_chars:]
            buf = head
            flush()
        if len(buf) + len(sentence) + 1 > max_chars:
            flush()
        buf = f"{buf} {sentence}".strip()

    buf_final, buf = buf, buf
    if buf_final.strip():
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc.id}::{ordinal}",
                doc_id=doc.id,
                ordinal=ordinal,
                text=buf_final.strip(),
                metadata=dict(doc.metadata),
            )
        )
    return chunks

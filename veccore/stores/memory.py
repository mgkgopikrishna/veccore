"""In-memory reference store.

Deliberately simple and deliberately correct. Every guarantee VecCore makes about space
isolation is enforced here, so a persistent backend only has to reproduce this
behaviour to be a valid implementation.
"""

from __future__ import annotations

from ..models import Chunk, Document, Vector


class MemoryStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        # (space_id, chunk_id) -> values.  The composite key IS the isolation.
        self._vectors: dict[tuple[str, str], list[float]] = {}

    # --- corpus ----------------------------------------------------------------- #
    def put_document(self, doc: Document) -> None:
        self._docs[doc.id] = doc

    def get_document(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def put_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunk_ids(self) -> list[str]:
        return list(self._chunks)

    def document_count(self) -> int:
        return len(self._docs)

    def chunk_count(self) -> int:
        return len(self._chunks)

    # --- vectors ---------------------------------------------------------------- #
    def put_vectors(self, vectors: list[Vector]) -> None:
        for v in vectors:
            self._vectors[(v.space_id, v.chunk_id)] = v.values

    def vector_count(self, space_id: str) -> int:
        return sum(1 for s, _ in self._vectors if s == space_id)

    def candidates(self, space_id: str) -> list[tuple[str, list[float]]]:
        return [(cid, vals) for (s, cid), vals in self._vectors.items() if s == space_id]

    def chunk_ids_missing_in(self, space_id: str, limit: int | None = None) -> list[str]:
        """Chunks that exist in the corpus but have no vector in this space.

        This is what makes a backfill resumable and what tells the registry when a
        space is complete enough to be promoted out of BUILDING.
        """
        present = {cid for (s, cid) in self._vectors if s == space_id}
        missing = [cid for cid in self._chunks if cid not in present]
        return missing[:limit] if limit is not None else missing

    def drop_space(self, space_id: str) -> int:
        keys = [k for k in self._vectors if k[0] == space_id]
        for k in keys:
            del self._vectors[k]
        return len(keys)

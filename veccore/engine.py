"""The thing you actually use: ingest documents, query the active space, run migrations."""

from __future__ import annotations

from .chunking import chunk_document
from .embedders import Embedder
from .migration import Migration
from .models import Document, SearchHit, SpaceState, Vector
from .registry import RegistryError, SpaceRegistry
from .similarity import top_k
from .stores import MemoryStore, VectorStore


class VecCore:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store: VectorStore = store or MemoryStore()
        self.registry = SpaceRegistry()
        self._embedders: dict[str, Embedder] = {}

    # --- spaces ------------------------------------------------------------------- #
    def add_space(self, embedder: Embedder, *, activate: bool = False):
        status = self.registry.register(embedder.space, activate=activate)
        self._embedders[embedder.space.id] = embedder
        return status

    def embedder(self, space_id: str) -> Embedder:
        try:
            return self._embedders[space_id]
        except KeyError:
            raise RegistryError(f"no embedder registered for space {space_id!r}") from None

    def spaces(self):
        out = []
        for status in self.registry.all():
            status.vector_count = self.store.vector_count(status.space.id)
            out.append(status)
        return out

    # --- ingest ------------------------------------------------------------------- #
    def ingest(self, doc: Document, *, max_chars: int = 800, overlap: int = 120) -> int:
        """Store a document and embed it into every space that is currently live.

        "Live" means ACTIVE plus SHADOW plus BUILDING -- this is the dual-write. A new
        document must not be missing from a candidate space, or the shadow comparison
        would measure staleness instead of model behaviour.
        """
        self.registry.require_active()
        self.store.put_document(doc)
        chunks = chunk_document(doc, max_chars=max_chars, overlap=overlap)
        if not chunks:
            return 0
        self.store.put_chunks(chunks)

        texts = [c.text for c in chunks]
        for status in self.registry.all():
            if status.state is SpaceState.RETIRED:
                continue
            emb = self._embedders.get(status.space.id)
            if emb is None:
                continue
            vectors = emb.embed(texts)
            self.store.put_vectors(
                [
                    Vector(chunk_id=c.id, space_id=status.space.id, values=v)
                    for c, v in zip(chunks, vectors, strict=True)
                ]
            )
        return len(chunks)

    # --- query -------------------------------------------------------------------- #
    def query(self, text: str, k: int = 5, *, space_id: str | None = None) -> list[SearchHit]:
        """Search. Defaults to the ACTIVE space; naming a space is opt-in and explicit."""
        if space_id is None:
            target = self.registry.require_active()
        else:
            target = self.registry.get(space_id)
            if target.state is SpaceState.BUILDING:
                raise RegistryError(
                    f"space {space_id!r} is still building and would return partial "
                    "results; finish the backfill first"
                )

        emb = self.embedder(target.space.id)
        qv = emb.embed([text])[0]
        pool = self.store.candidates(target.space.id)
        hits = []
        for chunk_id, score in top_k(qv, pool, k):
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:  # pragma: no cover
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    score=round(score, 6),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return hits

    # --- migrate ------------------------------------------------------------------ #
    def rollback_to(self, space_id: str):
        """Restore a retired space to ACTIVE.

        Deliberately not routed through `migrate_to`: after a cutover the source space
        is RETIRED and the target is ACTIVE, so recomputing a Migration would derive
        the source from whatever is active now -- which is the space you are trying to
        roll back *away* from. Rollback is a registry transition, not a migration.
        """
        return self.registry.rollback(space_id)

    def migrate_to(self, target_space_id: str) -> Migration:
        source = self.registry.require_active()
        return Migration(
            store=self.store,
            registry=self.registry,
            source=self.embedder(source.space.id),
            target=self.embedder(target_space_id),
        )

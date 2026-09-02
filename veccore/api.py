"""HTTP API.

Requires the `server` extra (`pip install 'veccore[server]'`). The core engine does not
import this module, so `veccore` itself stays dependency-free.

Endpoint design follows the same rule as the library: a query never silently crosses an
embedding space. `/query` serves the ACTIVE space unless you name another one, and
naming a BUILDING space is a 409 rather than a partial result.

Note the request models live at module level, not inside `create_app`. With
`from __future__ import annotations` every annotation is a string, and FastAPI resolves
those against the module's globals -- function-local models are invisible to it and get
silently misread as query parameters.
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .engine import VecCore
from .models import Document
from .registry import RegistryError


class IngestRequest(BaseModel):
    id: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MigrateRequest(BaseModel):
    target_space: str
    batch_size: int = 64


class CompareRequest(BaseModel):
    target_space: str
    queries: list[str]
    k: int = 5


class CutoverRequest(BaseModel):
    target_space: str
    min_overlap: float | None = None
    queries: list[str] | None = None
    k: int = 5


def _guard(fn):
    """Registry violations are conflicts, not server errors."""
    try:
        return fn()
    except RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_app(core: VecCore) -> FastAPI:
    app = FastAPI(
        title="VecCore",
        version="0.1.0",
        description="A RAG server where embedding spaces are versioned and migrations are safe.",
    )

    @app.get("/health")
    def health() -> dict:
        active = core.registry.active
        return {
            "status": "ok",
            "active_space": active.space.id if active else None,
            "documents": core.store.document_count(),
            "chunks": core.store.chunk_count(),
        }

    @app.get("/spaces")
    def list_spaces() -> dict:
        return {"spaces": [s.to_dict() for s in core.spaces()]}

    @app.post("/documents")
    def ingest(body: IngestRequest) -> dict:
        chunks = _guard(
            lambda: core.ingest(Document(id=body.id, text=body.text, metadata=body.metadata))
        )
        return {"document_id": body.id, "chunks_indexed": chunks}

    @app.get("/query")
    def query(q: str = Query(min_length=1), k: int = 5, space: str | None = None) -> dict:
        hits = _guard(lambda: core.query(q, k=k, space_id=space))
        active = core.registry.active
        return {
            "query": q,
            "space": space or (active.space.id if active else None),
            "hits": [h.to_dict() for h in hits],
        }

    @app.post("/migrations/backfill")
    def backfill(body: MigrateRequest) -> dict:
        return _guard(
            lambda: core.migrate_to(body.target_space).backfill(batch_size=body.batch_size)
        ).to_dict()

    @app.post("/migrations/compare")
    def compare(body: CompareRequest) -> dict:
        report = _guard(
            lambda: core.migrate_to(body.target_space).compare(body.queries, k=body.k)
        )
        return {**report.to_dict(), "per_query": report.per_query}

    @app.post("/migrations/cutover")
    def cutover(body: CutoverRequest) -> dict:
        def run():
            m = core.migrate_to(body.target_space)
            report = m.compare(body.queries, k=body.k) if body.queries else None
            new, old = m.cutover(min_overlap=body.min_overlap, report=report)
            return {
                "active_space": new.space.id,
                "retired_space": old.space.id if old else None,
                "divergence": report.to_dict() if report else None,
            }

        return _guard(run)

    @app.post("/migrations/rollback")
    def rollback(target_space: str = Body(embed=True)) -> dict:
        def run():
            restored, demoted = core.rollback_to(target_space)
            return {"active_space": restored.space.id, "demoted_space": demoted.space.id}

        return _guard(run)

    return app

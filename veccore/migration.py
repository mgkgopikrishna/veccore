"""Migrating a corpus from one embedding space to another, without downtime.

The problem
-----------
You want to change embedding model. Vectors from the old model and the new model are
not comparable, so the moment you switch, every stored vector is worthless. The naive
options are all bad: reindex in place and retrieval is broken until it finishes; build
a second system and you have two sources of truth; do it at 3am and hope.

The approach here
-----------------
Four phases, each with a property worth stating out loud:

    1. BACKFILL   Embed the existing corpus into the new space. The old space keeps
                  serving the whole time. Resumable, because the store can always
                  answer "which chunks are missing from this space".

    2. DUAL-WRITE New documents are embedded into both spaces, so the new space never
                  falls behind while you are still deciding.

    3. SHADOW     Run production queries against both and *measure the difference*.
                  This is the phase people skip, and it is the only one that tells you
                  whether the new model is actually better for your corpus rather than
                  better on someone else's benchmark.

    4. CUTOVER    One atomic state transition. The old space is retired, not deleted,
                  so rollback is another state transition rather than a rebuild.

The honest part: no amount of machinery tells you the new embeddings are *better*. It
tells you how differently they behave, and it makes the switch reversible. Deciding
what "better" means still requires labelled queries you care about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .embedders import Embedder
from .models import SpaceState, Vector
from .registry import RegistryError, SpaceRegistry
from .similarity import top_k
from .stores import VectorStore


@dataclass
class BackfillProgress:
    space_id: str
    embedded: int = 0
    remaining: int = 0
    batches: int = 0
    seconds: float = 0.0

    @property
    def complete(self) -> bool:
        return self.remaining == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "embedded": self.embedded,
            "remaining": self.remaining,
            "batches": self.batches,
            "seconds": round(self.seconds, 3),
            "complete": self.complete,
        }


@dataclass
class DivergenceReport:
    """How differently the candidate space retrieves, compared to the live one."""

    queries: int = 0
    k: int = 5
    mean_overlap: float = 0.0        # |same results| / k, averaged
    top1_agreement: float = 0.0      # fraction of queries whose best hit is identical
    mean_rank_shift: float = 0.0     # average |old rank - new rank| for shared results
    per_query: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.mean_overlap >= 0.9:
            return "near-identical -- the new space ranks almost the same documents"
        if self.mean_overlap >= 0.6:
            return "moderate drift -- review a sample before cutting over"
        return "high drift -- these spaces disagree substantially; do not cut over blind"

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "k": self.k,
            "mean_overlap": round(self.mean_overlap, 4),
            "top1_agreement": round(self.top1_agreement, 4),
            "mean_rank_shift": round(self.mean_rank_shift, 4),
            "verdict": self.verdict,
        }


class Migration:
    """Drives one corpus from `source` space to `target` space."""

    def __init__(
        self,
        store: VectorStore,
        registry: SpaceRegistry,
        source: Embedder,
        target: Embedder,
    ) -> None:
        if source.space.compatible_with(target.space):
            raise RegistryError(
                "source and target describe the same embedding space "
                f"(fingerprint {source.space.fingerprint}). There is nothing to "
                "migrate -- reindexing into an identical space is a no-op."
            )
        self.store = store
        self.registry = registry
        self.source = source
        self.target = target

    # --- phase 1: backfill --------------------------------------------------------- #
    def backfill(self, batch_size: int = 64, max_batches: int | None = None) -> BackfillProgress:
        """Embed missing chunks into the target space.

        Safe to call repeatedly. Each call resumes from whatever is still missing, so a
        crashed or deliberately paused migration costs you the current batch and
        nothing more.
        """
        started = time.perf_counter()
        space_id = self.target.space.id
        progress = BackfillProgress(space_id=space_id)

        while max_batches is None or progress.batches < max_batches:
            todo = self.store.chunk_ids_missing_in(space_id, limit=batch_size)
            if not todo:
                break
            chunks = [self.store.get_chunk(cid) for cid in todo]
            texts = [c.text for c in chunks if c is not None]
            vectors = self.target.embed(texts)
            self.store.put_vectors(
                [
                    Vector(chunk_id=c.id, space_id=space_id, values=v)
                    for c, v in zip([c for c in chunks if c is not None], vectors, strict=True)
                ]
            )
            progress.embedded += len(texts)
            progress.batches += 1

        progress.remaining = len(self.store.chunk_ids_missing_in(space_id))
        progress.seconds = time.perf_counter() - started

        if progress.complete:
            self.registry.promote_to_shadow(space_id, missing=0)
        return progress

    # --- phase 3: shadow comparison ------------------------------------------------ #
    def compare(self, queries: list[str], k: int = 5) -> DivergenceReport:
        """Run the same queries through both spaces and measure the disagreement.

        Each query is embedded twice -- once per space -- because a query embedded in
        one space cannot be scored against the other. That is the whole point.
        """
        target_status = self.registry.get(self.target.space.id)
        if target_status.state is SpaceState.BUILDING:
            raise RegistryError(
                f"space {self.target.space.id!r} is still building; comparing against "
                "an incomplete index would report drift that is really just missing data"
            )

        report = DivergenceReport(queries=len(queries), k=k)
        if not queries:
            return report

        src_pool = self.store.candidates(self.source.space.id)
        tgt_pool = self.store.candidates(self.target.space.id)

        overlaps: list[float] = []
        shifts: list[float] = []
        top1_hits = 0

        src_vecs = self.source.embed(queries)
        tgt_vecs = self.target.embed(queries)

        for query, qs, qt in zip(queries, src_vecs, tgt_vecs, strict=True):
            old = top_k(qs, src_pool, k)
            new = top_k(qt, tgt_pool, k)
            old_ids = [cid for cid, _ in old]
            new_ids = [cid for cid, _ in new]

            shared = set(old_ids) & set(new_ids)
            overlap = len(shared) / k if k else 0.0
            overlaps.append(overlap)

            if old_ids and new_ids and old_ids[0] == new_ids[0]:
                top1_hits += 1

            for cid in shared:
                shifts.append(abs(old_ids.index(cid) - new_ids.index(cid)))

            report.per_query.append(
                {
                    "query": query,
                    "overlap": round(overlap, 3),
                    "old_top": old_ids[:3],
                    "new_top": new_ids[:3],
                }
            )

        report.mean_overlap = sum(overlaps) / len(overlaps)
        report.top1_agreement = top1_hits / len(queries)
        report.mean_rank_shift = (sum(shifts) / len(shifts)) if shifts else 0.0
        return report

    # --- phase 4: cutover / rollback ------------------------------------------------ #
    def cutover(self, *, min_overlap: float | None = None, report: DivergenceReport | None = None):
        """Promote the target space to ACTIVE.

        Pass `min_overlap` together with a report to gate the switch on measured
        behaviour rather than on someone's confidence.
        """
        if min_overlap is not None:
            if report is None:
                raise RegistryError("min_overlap requires a DivergenceReport to check against")
            if report.mean_overlap < min_overlap:
                raise RegistryError(
                    f"refusing to cut over: measured overlap {report.mean_overlap:.3f} "
                    f"is below the required {min_overlap:.3f}. {report.verdict}"
                )
        return self.registry.cutover(self.target.space.id)

    def rollback(self):
        """Put the source space back in front. Its vectors were never deleted."""
        return self.registry.rollback(self.source.space.id)

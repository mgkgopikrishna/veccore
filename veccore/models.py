"""Core types.

The central idea of this project lives in `EmbeddingSpace`.

A vector is meaningless without the function that produced it. Two vectors are only
comparable if they came from the same model, at the same dimension, with the same
normalisation and the same text preprocessing. Most RAG systems leave that fact
implicit -- the model name sits in a config file, and nothing stops a query embedded
with model B from being scored against documents embedded with model A. The result is
not an error. It is silently degraded retrieval, which is far worse, because nothing
alerts and the answers just get quietly worse.

So here, the space is a first-class object with a content-addressed fingerprint. Every
stored vector carries its space id, every query carries a space id, and the store
refuses to compare across spaces. Incompatibility becomes an exception at the boundary
instead of a subtle quality regression in production.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SpaceState(str, Enum):
    """Lifecycle of an embedding space.

    BUILDING -> SHADOW -> ACTIVE -> RETIRED is the happy path of a migration.
    Only one space may be ACTIVE at a time; that is the invariant the registry enforces.
    """

    BUILDING = "building"   # backfill in progress; must never serve reads
    SHADOW = "shadow"       # fully backfilled; receives writes and comparison reads
    ACTIVE = "active"       # serves production reads
    RETIRED = "retired"     # kept for rollback, serves nothing


@dataclass(frozen=True)
class EmbeddingSpace:
    """An immutable description of how vectors in this space were produced."""

    id: str
    model_id: str
    dimension: int
    normalize: bool = True
    pooling: str = "mean"
    # Bump when chunking or text cleaning changes: the same model over differently
    # preprocessed text does not produce a comparable space.
    preprocessor_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not self.id or not self.model_id:
            raise ValueError("id and model_id are required")

    @property
    def fingerprint(self) -> str:
        """Content address of the space. Equal fingerprints mean comparable vectors."""
        canonical = json.dumps(
            {
                "model_id": self.model_id,
                "dimension": self.dimension,
                "normalize": self.normalize,
                "pooling": self.pooling,
                "preprocessor_version": self.preprocessor_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def compatible_with(self, other: EmbeddingSpace) -> bool:
        """Two spaces are interchangeable only if their fingerprints match.

        Note this ignores `id`: you can rename a space, or register the same
        configuration twice, and the vectors stay comparable. What you cannot do is
        change the model, the dimension, the normalisation or the preprocessing and
        pretend the old vectors still mean the same thing.
        """
        return self.fingerprint == other.fingerprint

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        return d


class IncompatibleSpaceError(RuntimeError):
    """Raised when a vector meets a space it did not come from."""

    def __init__(self, expected: EmbeddingSpace, got: str) -> None:
        super().__init__(
            f"Vector belongs to space {got!r}, but this operation is scoped to "
            f"{expected.id!r} (fingerprint {expected.fingerprint}). Vectors from "
            "different embedding spaces are not comparable; run a migration instead."
        )
        self.expected = expected
        self.got = got


@dataclass
class Document:
    """A source document, stored once regardless of how many spaces index it."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    doc_id: str
    ordinal: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Vector:
    """An embedding, permanently tagged with the space that produced it."""

    chunk_id: str
    space_id: str
    values: list[float]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("vector has no values")


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpaceStatus:
    space: EmbeddingSpace
    state: SpaceState
    vector_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "space": self.space.to_dict(),
            "state": self.state.value,
            "vector_count": self.vector_count,
            "created_at": self.created_at,
        }

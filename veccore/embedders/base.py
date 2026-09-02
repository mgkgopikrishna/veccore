"""Embedder interface.

An embedder owns exactly one `EmbeddingSpace`. That coupling is intentional: it is
impossible to hold an embedder and not know which space its output belongs to, so the
space tag can never be forgotten at the call site.
"""

from __future__ import annotations

from typing import Protocol

from ..models import EmbeddingSpace


class Embedder(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

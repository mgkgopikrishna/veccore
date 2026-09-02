"""Cosine similarity.

Pure Python by default so the core package has zero dependencies and the tests run
anywhere. A numpy path is used automatically when numpy is present, because the pure
path is O(n*d) in interpreted Python and that is genuinely slow past a few thousand
vectors. This is a reference implementation, not an ANN index -- see the README's
"Limits" section for what that means in practice.
"""

from __future__ import annotations

import math

try:  # pragma: no cover - exercised implicitly by whichever path is installed
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

HAS_NUMPY = _np is not None


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = num_a = num_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        num_a += x * x
        num_b += y * y
    if num_a == 0.0 or num_b == 0.0:
        return 0.0
    return dot / (math.sqrt(num_a) * math.sqrt(num_b))


def top_k(
    query: list[float], candidates: list[tuple[str, list[float]]], k: int
) -> list[tuple[str, float]]:
    """Return the k highest-scoring (id, score) pairs, descending."""
    if not candidates:
        return []
    if _np is not None and len(candidates) > 64:
        ids = [c[0] for c in candidates]
        matrix = _np.asarray([c[1] for c in candidates], dtype=_np.float32)
        q = _np.asarray(query, dtype=_np.float32)
        norms = _np.linalg.norm(matrix, axis=1) * _np.linalg.norm(q)
        norms[norms == 0] = 1e-12
        scores = (matrix @ q) / norms
        order = _np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in order]

    scored = [(cid, cosine(query, vec)) for cid, vec in candidates]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored[:k]


def l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return list(v)
    return [x / n for x in v]

"""A deterministic, dependency-free embedder.

This exists so the test suite and the demo run with nothing installed, on any machine,
in milliseconds -- and still exercise the real code paths end to end.

It is the hashing trick over word n-grams, optionally plus character n-grams. Lexical
rather than semantic, but a genuine vector space with genuine retrieval behaviour. Two
HashEmbedders configured differently really do disagree about which documents are
similar, so the divergence numbers the shadow comparison reports are measurements
rather than a rehearsed script.

The word-only and word-plus-subword configurations make a realistic migration pair:
adding subword features is exactly the kind of change that improves recall on
morphological variants (kill / killed / killing) while quietly reshuffling rankings
everywhere else -- which is the situation a migration is supposed to make visible.
"""

from __future__ import annotations

import hashlib
import re

from ..models import EmbeddingSpace
from ..similarity import l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(
        self,
        space_id: str,
        dimension: int = 256,
        seed: str = "a",
        word_ngram: int = 1,
        char_ngram: int = 0,
    ) -> None:
        parts = [f"words{word_ngram}"]
        if char_ngram:
            parts.append(f"chars{char_ngram}")
        self._space = EmbeddingSpace(
            id=space_id,
            model_id=f"hashing-{seed}-{'-'.join(parts)}",
            dimension=dimension,
            normalize=True,
            pooling="sum",
            metadata={"kind": "deterministic", "seed": seed},
        )
        self._seed = seed
        self._word_ngram = word_ngram
        self._char_ngram = char_ngram

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def _accumulate(self, vec: list[float], feature: str, weight: float) -> None:
        h = hashlib.blake2b(f"{self._seed}:{feature}".encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % self._space.dimension
        sign = 1.0 if h[4] & 1 else -1.0  # signed hashing cancels collision bias
        vec[idx] += sign * weight

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        n, cn = self._word_ngram, self._char_ngram
        for text in texts:
            vec = [0.0] * self._space.dimension
            tokens = _TOKEN.findall(text.lower())

            grams = (
                tokens
                if n == 1
                else [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            )
            for g in grams:
                self._accumulate(vec, f"w:{g}", 1.0)

            if cn:
                # Subword features are weighted down: they add recall on morphological
                # variants without letting a long word outvote a whole sentence.
                for token in tokens:
                    padded = f"<{token}>"
                    for i in range(len(padded) - cn + 1):
                        self._accumulate(vec, f"c:{padded[i : i + cn]}", 0.35)

            out.append(l2_normalize(vec))
        return out

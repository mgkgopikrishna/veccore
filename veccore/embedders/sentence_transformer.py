"""Real embeddings via sentence-transformers.

Imported lazily so the core package stays dependency-free; installing `veccore[embed]`
is what turns this on.
"""

from __future__ import annotations

from ..models import EmbeddingSpace


class SentenceTransformerEmbedder:
    def __init__(
        self,
        space_id: str,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        preprocessor_version: str = "1",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install 'veccore[embed]'`, or use HashEmbedder for a "
                "dependency-free space."
            ) from exc

        self._model = SentenceTransformer(model_id)
        self._normalize = normalize
        self._space = EmbeddingSpace(
            id=space_id,
            model_id=model_id,
            dimension=int(self._model.get_sentence_embedding_dimension()),
            normalize=normalize,
            pooling="mean",
            preprocessor_version=preprocessor_version,
            metadata={"kind": "sentence-transformers"},
        )

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        vecs = self._model.encode(
            texts, normalize_embeddings=self._normalize, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vecs]

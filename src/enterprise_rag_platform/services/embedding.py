
from typing import Any

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate normalized semantic embeddings locally."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    def __init__(self, model: Any | None = None) -> None:
        # Allow model injection for unit tests.
        self._model = model or SentenceTransformer(
            self.MODEL_NAME,
            device="cpu",
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Input texts must not be empty.")

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = vectors.tolist()

        if any(
            len(embedding) != self.EMBEDDING_DIMENSION
            for embedding in embeddings
        ):
            raise ValueError(
                f"Expected {self.EMBEDDING_DIMENSION}-dimensional embeddings."
            )

        return embeddings

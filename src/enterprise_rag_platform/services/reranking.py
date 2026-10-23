
from typing import Any

from sentence_transformers import CrossEncoder

from enterprise_rag_platform.models import DocumentChunk


class RerankingService:
    """Rerank retrieved document chunks by query relevance."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model: Any | None = None) -> None:
        # Inject a fake model in unit tests to avoid model downloads.
        self._model = model or CrossEncoder(
            self.MODEL_NAME,
            device="cpu",
        )

    def rerank_chunks(
        self,
        query: str,
        chunks: list[DocumentChunk],
        *,
        top_k: int | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return chunks ordered by descending relevance score."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(pairs)

        if len(scores) != len(chunks):
            raise ValueError(
                "Reranking model must return one score per chunk."
            )

        ranked_results = sorted(
            (
                (chunk, float(score))
                for chunk, score in zip(chunks, scores, strict=True)
            ),
            key=lambda result: result[1],
            reverse=True,
        )

        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        return ranked_results

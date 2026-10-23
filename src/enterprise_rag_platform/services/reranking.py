
import logging
from time import perf_counter

from enterprise_rag_platform.models import DocumentChunk

logger = logging.getLogger(__name__)


class RerankingService:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model=None) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

        self.model = model

    def rerank_chunks(
        self,
        query: str,
        chunks: list[DocumentChunk],
        *,
        top_k: int | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return chunks ordered by descending cross-encoder score."""
        if not query.strip():
            raise ValueError("Query must not be empty for reranking.")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not chunks:
            logger.info(
                "Reranking skipped because no chunks were supplied",
                extra={
                    "operation": "reranking",
                    "candidate_count": 0,
                    "result_count": 0,
                    "duration_ms": 0.0,
                },
            )
            return []

        pairs = [(query, chunk.content) for chunk in chunks]

        start_time = perf_counter()
        scores = self.model.predict(pairs)

        if len(scores) != len(chunks):
            raise ValueError(
                "Reranking model returned a different number of scores "
                "than input chunks."
            )

        ranked_results = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        results = [
            (chunk, float(score))
            for chunk, score in ranked_results
        ]

        logger.info(
            "Chunks reranked successfully",
            extra={
                "operation": "reranking",
                "duration_ms": round(
                    (perf_counter() - start_time) * 1000,
                    2,
                ),
                "candidate_count": len(chunks),
                "result_count": len(results),
            },
        )

        return results

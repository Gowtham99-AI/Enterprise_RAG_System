
import logging
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from enterprise_rag_platform.config import get_settings
from enterprise_rag_platform.models import Document, DocumentChunk
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.reranking import RerankingService

logger = logging.getLogger(__name__)


def search_chunks_with_distance(
    db: Session,
    query: str,
    *,
    top_k: int = 5,
    embedding_service: EmbeddingService | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """Return relevant, unique chunks with their cosine distances."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")

    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    start_time = perf_counter()

    service = embedding_service or EmbeddingService()

    embedding_start = perf_counter()
    query_embedding = service.embed_texts([query])[0]

    logger.info(
        "Query embedding completed",
        extra={
            "operation": "query_embedding",
            "duration_ms": round(
                (perf_counter() - embedding_start) * 1000,
                2,
            ),
        },
    )

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(DocumentChunk, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.status == "processed",
            DocumentChunk.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(top_k)
    )

    database_start = perf_counter()
    rows = db.execute(statement).all()

    logger.info(
        "Vector database query completed",
        extra={
            "operation": "vector_database_query",
            "duration_ms": round(
                (perf_counter() - database_start) * 1000,
                2,
            ),
            "candidate_count": len(rows),
        },
    )

    max_distance = get_settings().rag_max_distance

    results: list[tuple[DocumentChunk, float]] = []
    seen_content: set[str] = set()

    for chunk, score in rows:
        score = float(score)

        # Exclude results that are too distant from the query.
        if score > max_distance:
            continue

        # Normalize whitespace and casing before checking duplicates.
        normalized_content = " ".join(chunk.content.split()).casefold()

        if normalized_content in seen_content:
            continue

        seen_content.add(normalized_content)
        results.append((chunk, score))

    logger.info(
        "Vector retrieval completed",
        extra={
            "operation": "vector_retrieval",
            "duration_ms": round(
                (perf_counter() - start_time) * 1000,
                2,
            ),
            "candidate_count": len(rows),
            "result_count": len(results),
        },
    )

    return results


def search_chunks(
    db: Session,
    query: str,
    *,
    top_k: int = 5,
    embedding_service: EmbeddingService | None = None,
) -> list[DocumentChunk]:
    """Return matching chunks without exposing their distance scores."""
    matches = search_chunks_with_distance(
        db,
        query,
        top_k=top_k,
        embedding_service=embedding_service,
    )
    return [chunk for chunk, _ in matches]


def search_chunks_with_reranking(
    db: Session,
    query: str,
    *,
    candidate_k: int = 10,
    top_k: int = 5,
    embedding_service: EmbeddingService | None = None,
    reranking_service: RerankingService | None = None,
) -> list[tuple[DocumentChunk, float]]:
    """Retrieve vector-search candidates and rerank by relevance."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")

    if not 1 <= candidate_k <= 50:
        raise ValueError("candidate_k must be between 1 and 50.")

    if not 1 <= top_k <= min(candidate_k, 20):
        raise ValueError("top_k must be between 1 and min(candidate_k, 20).")

    candidates = search_chunks_with_distance(
        db,
        query,
        top_k=candidate_k,
        embedding_service=embedding_service,
    )

    if not candidates:
        logger.info(
            "Reranking skipped because retrieval returned no candidates",
            extra={
                "operation": "retrieval_reranking",
                "candidate_count": 0,
                "result_count": 0,
            },
        )
        return []

    service = reranking_service or RerankingService()

    reranking_start = perf_counter()
    reranked_results = service.rerank_chunks(
        query,
        [chunk for chunk, _ in candidates],
        top_k=top_k,
    )

    logger.info(
    "Reranking completed",
    extra={
        "operation": "reranking",
        "duration_ms": round(
            (perf_counter() - reranking_start) * 1000,
            2,
        ),
        "candidate_count": len(candidates),
        "result_count": len(reranked_results),
    },
    )
    logger.info(
        "Retrieval and reranking completed",
        extra={
            "operation": "retrieval_reranking",
            "candidate_count": len(candidates),
            "result_count": len(reranked_results),
        },
    )

    return reranked_results

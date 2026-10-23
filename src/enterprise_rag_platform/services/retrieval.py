
from sqlalchemy import select
from sqlalchemy.orm import Session

from enterprise_rag_platform.config import get_settings
from enterprise_rag_platform.models import Document, DocumentChunk
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.reranking import RerankingService


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

    service = embedding_service or EmbeddingService()
    query_embedding = service.embed_texts([query])[0]

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

    rows = db.execute(statement).all()
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
        return []

    service = reranking_service or RerankingService()

    return service.rerank_chunks(
        query,
        [chunk for chunk, _ in candidates],
        top_k=top_k,
    )

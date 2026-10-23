
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.reranking import RerankingService
from enterprise_rag_platform.services.retrieval import (
    search_chunks_with_distance,
    search_chunks_with_reranking,
)

router = APIRouter(prefix="/search", tags=["search"])


def get_embedding_service(request: Request) -> EmbeddingService:
    """Return the application-wide embedding service."""
    service = getattr(request.app.state, "embedding_service", None)

    if service is None:
        raise RuntimeError(
            "Embedding service is not initialized. "
            "Ensure the application lifespan has run."
        )

    return service


def get_reranking_service(request: Request) -> RerankingService:
    """Return the application-wide reranking service."""
    service = getattr(request.app.state, "reranking_service", None)

    if service is None:
        raise RuntimeError(
            "Reranking service is not initialized. "
            "Ensure the application lifespan has run."
        )

    return service


@router.post("", response_model=SearchResponse)
def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
) -> SearchResponse:
    """Search processed document chunks using vector search or reranking."""
    try:
        if request.rerank:
            matches = search_chunks_with_reranking(
                db,
                request.query,
                candidate_k=request.candidate_k,
                top_k=request.top_k,
                embedding_service=embedding_service,
                reranking_service=reranking_service,
            )

            results = [
                SearchResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    filename=chunk.document.filename,
                    relevance_score=score,
                    score_type="reranker",
                )
                for chunk, score in matches
            ]
        else:
            matches = search_chunks_with_distance(
                db,
                request.query,
                top_k=request.top_k,
                embedding_service=embedding_service,
            )

            results = [
                SearchResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    filename=chunk.document.filename,
                    distance=distance,
                    score_type="cosine_distance",
                )
                for chunk, distance in matches
            ]

        return SearchResponse(
            query=request.query,
            results=results,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from enterprise_rag_platform.services.retrieval import (
    search_chunks_with_distance,
    search_chunks_with_reranking,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db),
) -> SearchResponse:
    try:
        if request.rerank:
            matches = search_chunks_with_reranking(
                db,
                request.query,
                candidate_k=request.candidate_k,
                top_k=request.top_k,
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

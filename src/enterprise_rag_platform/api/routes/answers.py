from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
)
from enterprise_rag_platform.services.rag import RAGService

router = APIRouter(prefix="/answers", tags=["answers"])


def get_rag_service(request: Request) -> RAGService:
    """Return the application-scoped RAG service."""
    service = getattr(request.app.state, "rag_service", None)

    if service is None:
        service = RAGService()
        request.app.state.rag_service = service

    return service

@router.post("", response_model=AnswerResponse)
def answer_question(
    request: AnswerRequest,
    db: Session = Depends(get_db),
    service: RAGService = Depends(get_rag_service),
) -> AnswerResponse:
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty.",
        )

    try:
        return service.answer_question(
            db,
            request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
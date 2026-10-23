
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
)
from enterprise_rag_platform.services.rag import RAGService

router = APIRouter(prefix="/answers", tags=["answers"])


@router.post("", response_model=AnswerResponse)
def answer_question(
    request: AnswerRequest,
    db: Session = Depends(get_db),
) -> AnswerResponse:
    service = RAGService()

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

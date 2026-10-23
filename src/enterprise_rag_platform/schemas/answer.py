
from uuid import UUID

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerSource(BaseModel):
    document_id: UUID
    chunk_id: UUID
    filename: str
    chunk_index: int
    distance: float
    content: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[AnswerSource]

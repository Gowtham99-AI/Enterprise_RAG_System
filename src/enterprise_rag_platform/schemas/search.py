
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
    candidate_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_candidate_count(self) -> "SearchRequest":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")
        return self


class SearchResult(BaseModel):
    document_id: UUID
    chunk_id: UUID
    chunk_index: int
    content: str
    filename: str
    distance: float | None = None
    relevance_score: float | None = None
    score_type: Literal["cosine_distance", "reranker"]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

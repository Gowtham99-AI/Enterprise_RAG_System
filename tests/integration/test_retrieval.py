
import numpy as np
import pytest
from sqlalchemy.orm import Session

from enterprise_rag_platform.models import Document, DocumentChunk
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.retrieval import search_chunks


class FakeEmbeddingModel:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def test_search_chunks_ranks_by_cosine_distance(
    db_session: Session,
) -> None:
    document = Document(
        filename="retrieval-test.txt",
        content_type="text/plain",
        raw_text="Annual leave and office parking policies.",
        status="processed",
    )
    document.chunks = [
        DocumentChunk(
            chunk_index=0,
            content="Annual leave policy.",
            embedding=[1.0] + [0.0] * 383,
            embedding_model=EmbeddingService.MODEL_NAME,
            chunk_metadata={},
        ),
        DocumentChunk(
            chunk_index=1,
            content="Office parking policy.",
            embedding=[0.0, 1.0] + [0.0] * 382,
            embedding_model=EmbeddingService.MODEL_NAME,
            chunk_metadata={},
        ),
    ]
    db_session.add(document)
    db_session.commit()

    results = search_chunks(
        db_session,
        "How does annual leave work?",
        top_k=1,
        embedding_service=EmbeddingService(model=FakeEmbeddingModel()),
    )

    assert len(results) == 1
    assert results[0].content == "Annual leave policy."


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(
    db_session: Session,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        search_chunks(db_session, "annual leave", top_k=top_k)


def test_search_rejects_blank_query(db_session: Session) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        search_chunks(db_session, "   ")

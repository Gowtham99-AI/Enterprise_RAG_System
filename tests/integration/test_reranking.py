
import numpy as np
from sqlalchemy.orm import Session

from enterprise_rag_platform.models import Document, DocumentChunk
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.reranking import RerankingService
from enterprise_rag_platform.services.retrieval import (
    search_chunks_with_reranking,
)


class FakeEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


class FakeRerankingModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        scores = {
            "Office parking policy": 0.2,
            "Annual leave policy": 0.9,
        }
        return np.array(
            [scores[pair[1]] for pair in pairs],
            dtype=np.float32,
        )


def test_retrieval_returns_reranked_chunks(db_session: Session) -> None:
    document = Document(
        filename="reranking-test.txt",
        content_type="text/plain",
        raw_text="Office parking policy. Annual leave policy.",
        status="processed",
    )
    document.chunks = [
        DocumentChunk(
            chunk_index=0,
            content="Office parking policy",
            embedding=[1.0] + [0.0] * 383,
            embedding_model=EmbeddingService.MODEL_NAME,
            chunk_metadata={},
        ),
        DocumentChunk(
            chunk_index=1,
            content="Annual leave policy",
            embedding=[0.0, 1.0] + [0.0] * 382,
            embedding_model=EmbeddingService.MODEL_NAME,
            chunk_metadata={},
        ),
    ]
    db_session.add(document)
    db_session.commit()

    results = search_chunks_with_reranking(
        db_session,
        "Annual leave",
        candidate_k=2,
        top_k=2,
        embedding_service=EmbeddingService(model=FakeEmbeddingModel()),
        reranking_service=RerankingService(
            model=FakeRerankingModel()
        ),
    )

    assert len(results) == 2
    assert results[0][0].content == "Annual leave policy"
    assert results[0][1] > results[1][1]

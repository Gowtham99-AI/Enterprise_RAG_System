
import numpy as np
from sqlalchemy.orm import Session

from enterprise_rag_platform.models import Document
from enterprise_rag_platform.services.chunk_persistence import (
    persist_document_chunks,
)
from enterprise_rag_platform.services.embedding import EmbeddingService


class FakeEmbeddingModel:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32)


def test_persist_document_chunks_with_embeddings(
    db_session: Session,
) -> None:
    document = Document(
        filename="embedding-test.txt",
        content_type="text/plain",
        raw_text="A" * 900,
        status="pending",
    )
    db_session.add(document)
    db_session.flush()

    embedding_service = EmbeddingService(model=FakeEmbeddingModel())

    chunks = persist_document_chunks(
        db_session,
        document,
        chunk_size=500,
        overlap=100,
        embedding_service=embedding_service,
    )

    assert len(chunks) == 2
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].content[-100:] == chunks[1].content[:100]

    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(len(chunk.embedding) == 384 for chunk in chunks)
    assert all(
        chunk.embedding_model == EmbeddingService.MODEL_NAME
        for chunk in chunks
    )

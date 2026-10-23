
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from enterprise_rag_platform.models import Document, DocumentChunk


def test_document_and_chunks_can_be_persisted(db_session) -> None:
    document = Document(
        filename="employee-handbook.txt",
        content_type="text/plain",
        status="pending",
    )
    document.chunks.extend(
        [
            DocumentChunk(
                chunk_index=0,
                content="Employees receive annual leave.",
                chunk_metadata={"page": 1},
            ),
            DocumentChunk(
                chunk_index=1,
                content="Leave requests require manager approval.",
                chunk_metadata={"page": 2},
            ),
        ]
    )

    db_session.add(document)
    db_session.flush()

    document_id = document.id

    saved_document = db_session.scalar(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == document_id)
    )

    assert saved_document is not None
    assert saved_document.filename == "employee-handbook.txt"
    assert len(saved_document.chunks) == 2
    assert saved_document.chunks[0].content == (
        "Employees receive annual leave."
    )
    assert saved_document.chunks[1].chunk_metadata == {"page": 2}


def test_deleting_document_removes_its_chunks(db_session) -> None:
    document = Document(
        filename="temporary.txt",
        status="pending",
    )
    document.chunks.append(
        DocumentChunk(
            chunk_index=0,
            content="Temporary document content.",
            chunk_metadata={},
        )
    )

    db_session.add(document)
    db_session.flush()

    document_id = document.id

    # Delete without loading the chunks relationship.
    db_session.expire(document, ["chunks"])
    db_session.delete(document)
    db_session.flush()

    remaining_chunks = db_session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    )

    assert remaining_chunks == 0

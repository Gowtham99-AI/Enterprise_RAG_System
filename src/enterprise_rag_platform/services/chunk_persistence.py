
from sqlalchemy.orm import Session

from enterprise_rag_platform.models import Document, DocumentChunk
from enterprise_rag_platform.services.chunking import chunk_text
from enterprise_rag_platform.services.embedding import EmbeddingService


def persist_document_chunks(
    db: Session,
    document: Document,
    *,
    chunk_size: int = 500,
    overlap: int = 100,
    embedding_service: EmbeddingService | None = None,
) -> list[DocumentChunk]:
    """Chunk a document, generate embeddings, and persist its chunks."""
    if document.raw_text is None:
        raise ValueError("Document has no extracted text to chunk.")

    chunks = chunk_text(
        document.raw_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("Document contains no text to chunk.")

    service = embedding_service or EmbeddingService()
    embeddings = service.embed_texts(chunks)

    document.chunks = [
        DocumentChunk(
            chunk_index=index,
            content=content,
            embedding=embeddings[index],
            embedding_model=service.MODEL_NAME,
            chunk_metadata={
                "start_char": index * (chunk_size - overlap),
                "end_char": min(
                    index * (chunk_size - overlap) + len(content),
                    len(document.raw_text),
                ),
            },
        )
        for index, content in enumerate(chunks)
    ]

    db.add(document)
    db.flush()

    return document.chunks


import hashlib
import logging
from time import perf_counter

from sqlalchemy.orm import Session

from enterprise_rag_platform.models import Document

logger = logging.getLogger(__name__)


def calculate_content_hash(text_content: str) -> str:
    """Return the SHA-256 hash of UTF-8 document text."""
    return hashlib.sha256(text_content.encode("utf-8")).hexdigest()


def persist_document_text(
    db: Session,
    *,
    filename: str,
    content_type: str,
    text_content: str,
    content_hash: str,
) -> Document:
    document = Document(
        filename=filename,
        content_type=content_type,
        raw_text=text_content,
        content_hash=content_hash,
        status="pending",
    )

    start_time = perf_counter()

    try:
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        logger.exception(
            "Document persistence failed",
            extra={
                "operation": "document_persistence",
                "duration_ms": round(
                    (perf_counter() - start_time) * 1000,
                    2,
                ),
            },
        )
        raise

    logger.info(
        "Document persisted successfully",
        extra={
            "operation": "document_persistence",
            "duration_ms": round(
                (perf_counter() - start_time) * 1000,
                2,
            ),
            "document_id": document.id,
        },
    )

    return document

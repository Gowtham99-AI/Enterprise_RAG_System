
import hashlib

from sqlalchemy.orm import Session

from enterprise_rag_platform.models import Document


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

    try:
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        raise

    return document

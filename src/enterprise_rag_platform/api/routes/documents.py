
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from enterprise_rag_platform.config import get_settings
from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.models.document import Document
from enterprise_rag_platform.schemas.document import (
    DocumentListResponse,
    DocumentSummary,
)
from enterprise_rag_platform.services.chunk_persistence import (
    persist_document_chunks,
)
from enterprise_rag_platform.services.ingestion import (
    calculate_content_hash,
    persist_document_text,
)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSION = ".txt"
ALLOWED_CONTENT_TYPE = "text/plain"


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, str | int]:
    settings = get_settings()
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]

    if not filename or not filename.lower().endswith(ALLOWED_EXTENSION):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .txt files are supported.",
        )

    if file.content_type not in (
        ALLOWED_CONTENT_TYPE,
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported content type.",
        )

    content = await file.read(settings.max_upload_size_bytes + 1)

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File exceeds the 5 MiB upload limit.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text files must use UTF-8 encoding.",
        ) from exc

    if not text_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contains no readable text.",
        )

    content_hash = calculate_content_hash(text_content)

    # Reject content that has already been uploaded.
    existing_document = db.scalar(
        select(Document).where(Document.content_hash == content_hash)
    )

    if existing_document is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This document has already been uploaded.",
                "document_id": str(existing_document.id),
                "filename": existing_document.filename,
            },
        )

    # Persist the document, handling simultaneous duplicate uploads.
    try:
        document = persist_document_text(
            db,
            filename=filename,
            content_type=ALLOWED_CONTENT_TYPE,
            text_content=text_content,
            content_hash=content_hash,
        )
    except IntegrityError:
        existing_document = db.scalar(
            select(Document).where(Document.content_hash == content_hash)
        )

        if existing_document is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "This document has already been uploaded.",
                    "document_id": str(existing_document.id),
                    "filename": existing_document.filename,
                },
            ) from None

        # Preserve unrelated database integrity errors.
        raise

    # Split the text into chunks and generate embeddings.
    try:
        chunks = persist_document_chunks(db, document)

        document.status = "processed"
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception as exc:
        db.rollback()

        # Record the processing failure without masking the original error.
        failed_document = db.get(Document, document.id)
        if failed_document is not None:
            failed_document.status = "failed"
            db.add(failed_document)
            db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document was uploaded but processing failed.",
        ) from exc

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
        "chunk_count": len(chunks),
        "message": "Document uploaded, chunked, and embedded successfully.",
    }

@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: Session = Depends(get_db),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    total = db.scalar(select(func.count()).select_from(Document)) or 0

    statement = (
        select(Document)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    documents = db.scalars(statement).all()

    return DocumentListResponse(
        items=[DocumentSummary.model_validate(document) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
    )
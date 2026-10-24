
from fastapi.testclient import TestClient

from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.main import app
from unittest.mock import patch

from enterprise_rag_platform.services.chunk_persistence import (
    persist_document_chunks,
)

def test_upload_valid_text_file(db_session) -> None:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        "handbook.txt",
                        b"Employees receive annual leave.",
                        "text/plain",
                    )
                },
            )

        assert response.status_code == 201
        body = response.json()
        assert body["filename"] == "handbook.txt"
        assert body["status"] == "processed"
        assert body["document_id"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_upload_rejects_unsupported_extension(db_session) -> None:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        "malware.exe",
                        b"not a text file",
                        "text/plain",
                    )
                },
            )

        assert response.status_code == 415
    finally:
        app.dependency_overrides.pop(get_db, None)

def test_upload_rejects_duplicate_content(db_session) -> None:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            file_content = b"Employees receive annual leave."

            first_response = client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        "handbook.txt",
                        file_content,
                        "text/plain",
                    )
                },
            )

            second_response = client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        "handbook-copy.txt",
                        file_content,
                        "text/plain",
                    )
                },
            )

        assert first_response.status_code == 201
        assert second_response.status_code == 409
        assert (
            second_response.json()["detail"]["message"]
            == "This document has already been uploaded."
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

def test_upload_marks_document_failed_when_chunking_fails(
    db_session,
) -> None:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with patch(
            "enterprise_rag_platform.api.routes.documents."
            "persist_document_chunks",
            side_effect=RuntimeError("Embedding generation failed"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/documents",
                    files={
                        "file": (
                            "failure-test.txt",
                            b"Employees receive annual leave.",
                            "text/plain",
                        )
                    },
                )

        assert response.status_code == 500

        from sqlalchemy import select

        from enterprise_rag_platform.models import Document

        document = db_session.scalar(
            select(Document).where(
                Document.filename == "failure-test.txt"
            )
        )

        assert document is not None
        assert document.status == "failed"

    finally:
        app.dependency_overrides.pop(get_db, None)
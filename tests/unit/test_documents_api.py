
from fastapi.testclient import TestClient

from enterprise_rag_platform.database import get_db
from enterprise_rag_platform.main import app


def test_list_documents_returns_paginated_metadata(db_session) -> None:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        client = TestClient(app)
        response = client.get("/api/v1/documents?limit=10&offset=0")

        assert response.status_code == 200
        body = response.json()

        assert body["limit"] == 10
        assert body["offset"] == 0
        assert isinstance(body["total"], int)
        assert isinstance(body["items"], list)

        if body["items"]:
            item = body["items"][0]
            assert "id" in item
            assert "filename" in item
            assert "status" in item
            assert "created_at" in item
            assert "updated_at" in item
            assert "raw_text" not in item
            assert "content_hash" not in item
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_list_documents_rejects_invalid_pagination() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/documents?limit=101")

    assert response.status_code == 422
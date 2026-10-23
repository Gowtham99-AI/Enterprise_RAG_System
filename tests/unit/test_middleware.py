from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from enterprise_rag_platform.api.middleware import CorrelationIdMiddleware


def test_generates_correlation_id() -> None:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request) -> dict[str, str]:
        return {"correlation_id": request.state.correlation_id}

    response = TestClient(app).get("/test")

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == response.json()["correlation_id"]


def test_preserves_supplied_correlation_id() -> None:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    async def test_endpoint() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(app).get(
        "/test",
        headers={"X-Correlation-ID": "test-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "test-request-123"


def test_replaces_oversized_correlation_id() -> None:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    async def test_endpoint() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(app).get(
        "/test",
        headers={"X-Correlation-ID": "x" * 129},
    )

    assert response.status_code == 200
    assert len(response.headers["X-Correlation-ID"]) == 36
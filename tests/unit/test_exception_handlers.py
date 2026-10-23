import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_rag_platform.api.exception_handlers import (
    unhandled_exception_handler,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )

    @app.get("/test-error")
    async def raise_error() -> None:
        raise RuntimeError("Sensitive internal database details")

    return TestClient(app, raise_server_exceptions=False)


def test_unhandled_exception_returns_generic_500(
    client: TestClient,
) -> None:
    response = client.get("/test-error")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error"
    }
    assert "Sensitive internal database details" not in response.text


def test_unhandled_exception_is_logged(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        client.get("/test-error")

    assert "Unhandled exception while processing request" in caplog.text
    assert "Sensitive internal database details" in caplog.text
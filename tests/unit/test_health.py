
from fastapi.testclient import TestClient

from enterprise_rag_platform.main import create_app


def test_health_endpoint_returns_healthy_status() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


from fastapi.testclient import TestClient

from enterprise_rag_platform.main import app


def test_answers_endpoint_returns_answer_and_sources(
    monkeypatch,
) -> None:
    from enterprise_rag_platform.api.routes import answers

    class FakeRAGService:
        def answer_question(
            self,
            db,
            question: str,
            *,
            top_k: int = 5,
        ):
            return {
                "question": question,
                "answer": "Employees receive annual leave. [S1]",
                "sources": [],
            }

    monkeypatch.setattr(answers, "RAGService", FakeRAGService)

    client = TestClient(app)
    response = client.post(
        "/api/v1/answers",
        json={
            "question": "What leave do employees receive?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Employees receive annual leave. [S1]"
    assert body["question"] == "What leave do employees receive?"
    assert body["sources"] == []


def test_answers_endpoint_rejects_blank_question() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/answers",
        json={"question": "   ", "top_k": 3},
    )

    assert response.status_code == 422

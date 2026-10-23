
from types import SimpleNamespace
from uuid import uuid4

import pytest

from enterprise_rag_platform.services import rag
from enterprise_rag_platform.services.rag import RAGService


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


class FakeLLMService:
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Employees receive annual leave. [S1]"


def test_rag_returns_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid4()
    chunk_id = uuid4()

    chunk = SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content="Employees receive annual leave.",
        document=SimpleNamespace(filename="leave-policy.txt"),
    )

    monkeypatch.setattr(
        rag,
        "search_chunks_with_distance",
        lambda *args, **kwargs: [(chunk, 0.15)],
    )

    llm = FakeLLMService()
    service = RAGService(
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
        llm_service=llm,  # type: ignore[arg-type]
    )

    result = service.answer_question(
        db=None,  # type: ignore[arg-type]
        question="What leave do employees receive?",
    )

    assert result.answer == "Employees receive annual leave. [S1]"
    assert len(result.sources) == 1
    assert result.sources[0].filename == "leave-policy.txt"
    assert result.sources[0].distance == 0.15
    assert "[S1]" in llm.last_prompt
    assert "Employees receive annual leave." in llm.last_prompt


def test_rag_returns_no_context_message_when_no_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "search_chunks_with_distance",
        lambda *args, **kwargs: [],
    )

    llm = FakeLLMService()
    service = RAGService(
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
        llm_service=llm,  # type: ignore[arg-type]
    )

    result = service.answer_question(
        db=None,  # type: ignore[arg-type]
        question="What is the leave policy?",
    )

    assert result.sources == []
    assert "couldn't find relevant information" in result.answer
    assert llm.last_prompt == ""


def test_rag_rejects_blank_question() -> None:
    service = RAGService(
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
        llm_service=FakeLLMService(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Question must not be empty"):
        service.answer_question(
            db=None,  # type: ignore[arg-type]
            question="   ",
        )


import httpx
import pytest

from enterprise_rag_platform.services.llm import OllamaService


def test_generate_returns_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"response": "Employees receive annual leave."},
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    service = OllamaService()
    result = service.generate("Explain annual leave.")

    assert result == "Employees receive annual leave."


def test_generate_rejects_blank_prompt() -> None:
    service = OllamaService()

    with pytest.raises(ValueError, match="Prompt must not be empty"):
        service.generate("   ")


def test_generate_rejects_empty_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"response": "   "},
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    service = OllamaService()

    with pytest.raises(ValueError, match="empty response"):
        service.generate("Explain annual leave.")

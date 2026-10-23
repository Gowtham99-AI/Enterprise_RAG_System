
import httpx

from enterprise_rag_platform.config import get_settings


class OllamaService:
    """Generate text using a locally running Ollama model."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout_seconds

    def generate(self, prompt: str) -> str:
        """Generate a response from the configured Ollama model."""
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json().get("response", "").strip()
        if not result:
            raise ValueError("Ollama returned an empty response.")

        return result

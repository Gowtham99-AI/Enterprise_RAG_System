from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Enterprise RAG Platform"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    max_upload_size_bytes: int = 5 * 1024 * 1024  # 5 MiB
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: float = 120.0
    rag_top_k: int = 5
    rag_max_distance: float = 0.80
    log_level: str = "INFO"
    database_url: str = (
        "postgresql+psycopg://rag_user:change_me@localhost:5433/enterprise_rag"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
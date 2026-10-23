
import numpy as np
import pytest

from enterprise_rag_platform.services.embedding import EmbeddingService


class FakeEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32)


def test_embedding_returns_384_dimensions() -> None:
    service = EmbeddingService(model=FakeEmbeddingModel())

    embeddings = service.embed_texts(["Annual leave policy"])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


def test_empty_list_returns_empty_list() -> None:
    service = EmbeddingService(model=FakeEmbeddingModel())

    assert service.embed_texts([]) == []


def test_blank_text_is_rejected() -> None:
    service = EmbeddingService(model=FakeEmbeddingModel())

    with pytest.raises(ValueError, match="must not be empty"):
        service.embed_texts(["   "])


def test_wrong_dimension_is_rejected() -> None:
    class InvalidModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            return np.ones((len(texts), 128), dtype=np.float32)

    service = EmbeddingService(model=InvalidModel())

    with pytest.raises(ValueError, match="384-dimensional"):
        service.embed_texts(["Some document text"])

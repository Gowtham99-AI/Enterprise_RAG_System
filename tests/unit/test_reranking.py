
import numpy as np
import pytest

from enterprise_rag_platform.models import DocumentChunk
from enterprise_rag_platform.services.reranking import RerankingService


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array(self.scores, dtype=np.float32)


def make_chunk(index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_index=index,
        content=content,
        chunk_metadata={},
    )


def test_reranking_sorts_by_descending_score() -> None:
    chunks = [
        make_chunk(0, "Office parking policy"),
        make_chunk(1, "Annual leave policy"),
        make_chunk(2, "Company cafeteria menu"),
    ]
    service = RerankingService(
        model=FakeCrossEncoder([0.2, 0.9, 0.1])
    )

    results = service.rerank_chunks("Annual leave", chunks)

    assert [chunk.chunk_index for chunk, _ in results] == [1, 0, 2]
    assert [score for _, score in results] == pytest.approx(
        [0.9, 0.2, 0.1]
    )


def test_reranking_returns_only_top_k() -> None:
    chunks = [
        make_chunk(0, "Parking"),
        make_chunk(1, "Annual leave"),
        make_chunk(2, "Cafeteria"),
    ]
    service = RerankingService(
        model=FakeCrossEncoder([0.2, 0.9, 0.1])
    )

    results = service.rerank_chunks("Annual leave", chunks, top_k=2)

    assert len(results) == 2
    assert results[0][0].chunk_index == 1


def test_empty_chunks_returns_empty_list() -> None:
    service = RerankingService(model=FakeCrossEncoder([]))

    assert service.rerank_chunks("Annual leave", []) == []


def test_blank_query_is_rejected() -> None:
    service = RerankingService(model=FakeCrossEncoder([0.9]))
    chunks = [make_chunk(0, "Annual leave")]

    with pytest.raises(ValueError, match="Query"):
        service.rerank_chunks("   ", chunks)


def test_invalid_top_k_is_rejected() -> None:
    service = RerankingService(model=FakeCrossEncoder([0.9]))
    chunks = [make_chunk(0, "Annual leave")]

    with pytest.raises(ValueError, match="top_k"):
        service.rerank_chunks("Annual leave", chunks, top_k=0)


from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from enterprise_rag_platform.evaluation.performance_runner import (
    benchmark_search,
)


@patch(
    "enterprise_rag_platform.evaluation.performance_runner.urlopen"
)
def test_benchmark_search_measures_both_modes(
    mock_urlopen: MagicMock,
) -> None:
    response = MagicMock()
    response.status = 200
    response.read.return_value = b'{"results": []}'
    mock_urlopen.return_value.__enter__.return_value = response

    report = benchmark_search(
        base_url="http://localhost:8000",
        query="What is the leave policy?",
        warmup_requests=1,
        measured_requests=2,
    )

    results = report["results"]
    assert results["vector_search"]["successful_requests"] == 2
    assert results["vector_search"]["failed_requests"] == 0
    assert results["reranked_search"]["successful_requests"] == 2
    assert results["reranked_search"]["failed_requests"] == 0
    assert "latency" in results["vector_search"]


@patch(
    "enterprise_rag_platform.evaluation.performance_runner.urlopen",
    side_effect=URLError("connection refused"),
)
def test_benchmark_search_records_request_failures(
    mock_urlopen: MagicMock,
) -> None:
    report = benchmark_search(
        base_url="http://localhost:8000",
        query="What is the leave policy?",
        warmup_requests=0,
        measured_requests=1,
    )

    results = report["results"]
    assert results["vector_search"]["failed_requests"] == 1
    assert results["reranked_search"]["failed_requests"] == 1
    assert "latency" not in results["vector_search"]


def test_benchmark_search_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="Query must not be blank"):
        benchmark_search(
            base_url="http://localhost:8000",
            query="   ",
        )


def test_benchmark_search_rejects_zero_measured_requests() -> None:
    with pytest.raises(
        ValueError,
        match="At least one measured request",
    ):
        benchmark_search(
            base_url="http://localhost:8000",
            query="test",
            measured_requests=0,
        )

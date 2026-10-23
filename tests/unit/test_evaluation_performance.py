
import pytest

from enterprise_rag_platform.evaluation.performance import (
    summarize_latency,
)


def test_summarize_latency_calculates_statistics() -> None:
    result = summarize_latency([10, 20, 30, 40, 100])

    assert result == {
        "sample_count": 5.0,
        "average_ms": 40.0,
        "median_ms": 30.0,
        "p95_ms": 100,
        "min_ms": 10,
        "max_ms": 100,
    }


def test_summarize_latency_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one duration"):
        summarize_latency([])


@pytest.mark.parametrize(
    "durations",
    [
        [-1, 10],
        [True, 10],
    ],
)
def test_summarize_latency_rejects_invalid_values(
    durations: list[float],
) -> None:
    with pytest.raises(ValueError, match="non-negative numbers"):
        summarize_latency(durations)

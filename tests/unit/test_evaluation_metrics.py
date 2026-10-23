
import pytest

from enterprise_rag_platform.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
)


def test_hit_rate_at_k() -> None:
    retrieved = [
        ["A", "B", "C"],
        ["D", "A", "B"],
        ["X", "Y", "Z"],
    ]
    relevant = [{"A"}, {"A"}, {"A"}]

    assert hit_rate_at_k(retrieved, relevant, k=2) == pytest.approx(
        2 / 3
    )


def test_mrr_at_k() -> None:
    retrieved = [
        ["A", "B", "C"],
        ["D", "A", "B"],
        ["X", "Y", "A"],
    ]
    relevant = [{"A"}, {"A"}, {"A"}]

    expected = (1.0 + 0.5 + 1 / 3) / 3

    assert mean_reciprocal_rank_at_k(
        retrieved, relevant, k=3
    ) == pytest.approx(expected)


def test_no_relevant_results_returns_zero() -> None:
    retrieved = [["B", "C"], ["D", "E"]]
    relevant = [{"A"}, {"A"}]

    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
    assert mean_reciprocal_rank_at_k(
        retrieved, relevant, k=2
    ) == 0.0


def test_mismatched_query_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        hit_rate_at_k([["A"]], [{"A"}, {"B"}], k=1)


def test_empty_evaluation_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="(?i)at least one"):
        hit_rate_at_k([], [], k=5)


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        hit_rate_at_k([["A"]], [{"A"}], k=0)


def test_empty_relevant_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="relevant ID"):
        hit_rate_at_k([["A"]], [set()], k=1)

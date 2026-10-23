
import pytest

from enterprise_rag_platform.evaluation.quality_gate import (
    evaluate_quality_gate,
)


def test_quality_gate_passes_when_metrics_meet_thresholds() -> None:
    metrics = {
        "reranked_search": {
            "hit_rate_at_5": 1.0,
            "mrr_at_5": 1.0,
        }
    }

    failures = evaluate_quality_gate(metrics)

    assert failures == []


def test_quality_gate_reports_metric_regressions() -> None:
    metrics = {
        "reranked_search": {
            "hit_rate_at_5": 0.9,
            "mrr_at_5": 0.6,
        }
    }

    failures = evaluate_quality_gate(metrics)

    assert len(failures) == 1
    assert "mrr_at_5" in failures[0]
    assert "0.600" in failures[0]
    assert "0.800" in failures[0]


def test_quality_gate_supports_custom_thresholds() -> None:
    metrics = {
        "reranked_search": {
            "hit_rate_at_5": 0.75,
            "mrr_at_5": 0.70,
        }
    }

    failures = evaluate_quality_gate(
        metrics,
        minimums={
            "hit_rate_at_5": 0.70,
            "mrr_at_5": 0.65,
        },
    )

    assert failures == []


def test_quality_gate_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Evaluation method not found"):
        evaluate_quality_gate({})


def test_quality_gate_rejects_invalid_threshold() -> None:
    metrics = {
        "reranked_search": {
            "mrr_at_5": 1.0,
        }
    }

    with pytest.raises(ValueError, match="Threshold"):
        evaluate_quality_gate(
            metrics,
            minimums={"mrr_at_5": 1.5},
        )


def test_quality_gate_rejects_missing_metric() -> None:
    metrics = {"reranked_search": {}}

    with pytest.raises(ValueError, match="Metric not found"):
        evaluate_quality_gate(metrics)


import math


def evaluate_quality_gate(
    metrics: dict[str, dict[str, float]],
    *,
    method: str = "reranked_search",
    minimums: dict[str, float] | None = None,
) -> list[str]:
    """Return quality-gate failures; an empty list means the gate passed."""
    if minimums is None:
        minimums = {
            "hit_rate_at_5": 0.80,
            "mrr_at_5": 0.80,
        }

    if method not in metrics:
        raise ValueError(f"Evaluation method not found: {method}")

    if not minimums:
        raise ValueError("At least one minimum metric threshold is required.")

    failures: list[str] = []
    method_metrics = metrics[method]

    for metric_name, minimum in minimums.items():
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError(
                f"Threshold for {metric_name} must be between 0 and 1."
            )

        if metric_name not in method_metrics:
            raise ValueError(f"Metric not found: {metric_name}")

        score = method_metrics[metric_name]

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Score for {metric_name} must be between 0 and 1."
            )

        if score < minimum:
            failures.append(
                f"{method}.{metric_name} = {score:.3f}; "
                f"minimum required = {minimum:.3f}"
            )

    return failures

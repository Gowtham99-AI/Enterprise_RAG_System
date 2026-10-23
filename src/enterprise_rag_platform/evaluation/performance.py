
"""Utilities for summarizing performance benchmark timings."""

from collections.abc import Sequence
from statistics import fmean, median


def summarize_latency(
    durations_ms: Sequence[float],
) -> dict[str, float]:
    """Summarize request durations in milliseconds."""
    if not durations_ms:
        raise ValueError("At least one duration is required.")

    if any(
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration < 0
        for duration in durations_ms
    ):
        raise ValueError("Durations must be non-negative numbers.")

    sorted_durations = sorted(durations_ms)
    p95_index = max(0, int(len(sorted_durations) * 0.95 + 0.999999) - 1)

    return {
        "sample_count": float(len(sorted_durations)),
        "average_ms": round(fmean(sorted_durations), 2),
        "median_ms": round(median(sorted_durations), 2),
        "p95_ms": round(sorted_durations[p95_index], 2),
        "min_ms": round(sorted_durations[0], 2),
        "max_ms": round(sorted_durations[-1], 2),
    }

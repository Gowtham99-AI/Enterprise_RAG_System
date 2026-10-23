
"""Benchmark search API latency for vector and reranked retrieval."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from enterprise_rag_platform.evaluation.performance import summarize_latency


def benchmark_search(
    *,
    base_url: str,
    query: str,
    warmup_requests: int = 1,
    measured_requests: int = 5,
    timeout_seconds: float = 180.0,
) -> dict[str, object]:
    """Measure API latency for vector and reranked search."""
    if not query.strip():
        raise ValueError("Query must not be blank.")
    if warmup_requests < 0:
        raise ValueError("Warm-up requests cannot be negative.")
    if measured_requests < 1:
        raise ValueError("At least one measured request is required.")
    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive.")

    endpoint = f"{base_url.rstrip('/')}/api/v1/search"

    modes = {
        "vector_search": {
            "rerank": False,
            "top_k": 5,
            "candidate_k": 10,
        },
        "reranked_search": {
            "rerank": True,
            "top_k": 5,
            "candidate_k": 10,
        },
    }

    results: dict[str, object] = {}

    report: dict[str, object] = {
        "endpoint": endpoint,
        "generated_at": datetime.now(UTC).isoformat(),
        "query": query,
        "timeout_seconds": timeout_seconds,
        "warmup_requests_per_mode": warmup_requests,
        "measured_requests_per_mode": measured_requests,
        "configuration": modes,
        "results": results,
    }

    for mode, options in modes.items():
        durations_ms: list[float] = []
        warmup_errors: list[str] = []
        measured_errors: list[str] = []

        total_requests = warmup_requests + measured_requests

        for request_number in range(total_requests):
            payload = json.dumps(
                {"query": query, **options}
            ).encode("utf-8")

            request = Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            started = time.perf_counter()
            error: str | None = None

            try:
                with urlopen(
                    request,
                    timeout=timeout_seconds,
                ) as response:
                    response.read()

                    if not 200 <= response.status < 300:
                        error = f"Unexpected HTTP status: {response.status}"

            except HTTPError as exc:
                error = f"HTTP {exc.code}"
                exc.read()

            except (URLError, TimeoutError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"

            elapsed_ms = (time.perf_counter() - started) * 1000

            is_warmup = request_number < warmup_requests

            if error is not None:
                if is_warmup:
                    warmup_errors.append(error)
                else:
                    measured_errors.append(error)
                continue

            if not is_warmup:
                durations_ms.append(elapsed_ms)

        successful_requests = len(durations_ms)
        failed_requests = len(measured_errors)

        mode_result: dict[str, object] = {
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "warmup_failures": len(warmup_errors),
            "warmup_errors": warmup_errors,
            "errors": measured_errors,
            "complete": successful_requests == measured_requests,
        }

        if durations_ms:
            mode_result["latency"] = summarize_latency(durations_ms)

        results[mode] = mode_result

    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    report_dir = project_root / "reports" / "performance"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = benchmark_search(
        base_url="http://127.0.0.1:8000",
        query="How many annual leave days are provided?",
        warmup_requests=1,
        measured_requests=5,
        timeout_seconds=180.0,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"search_{timestamp}.json"

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

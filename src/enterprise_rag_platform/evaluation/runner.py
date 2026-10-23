
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag_platform.config import get_settings
from enterprise_rag_platform.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
)
from enterprise_rag_platform.evaluation.quality_gate import (
    evaluate_quality_gate,
)
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.reranking import RerankingService
from enterprise_rag_platform.services.retrieval import (
    search_chunks_with_distance,
    search_chunks_with_reranking,
)


def load_evaluation_cases(
    path: Path,
) -> list[dict[str, Any]]:
    """Load and validate evaluation cases from JSONL."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    cases: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}."
                ) from exc

            if not isinstance(case, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            query = case.get("query")
            relevant = case.get("relevant_filenames")
            top_k = case.get("top_k", 5)

            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Line {line_number} has an invalid query."
                )

            if (
                not isinstance(relevant, list)
                or not relevant
                or any(
                    not isinstance(name, str) or not name.strip()
                    for name in relevant
                )
            ):
                raise ValueError(
                    f"Line {line_number} needs relevant_filenames."
                )

            if (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
                or not 1 <= top_k <= 20
            ):
                raise ValueError(
                    f"Line {line_number} has an invalid top_k."
                )

            cases.append(
                {
                    "query": query,
                    "relevant_filenames": set(relevant),
                    "top_k": top_k,
                }
            )

    if not cases:
        raise ValueError("Evaluation dataset contains no cases.")

    return cases

def _deduplicate_preserving_order(
    filenames: Sequence[str],
) -> list[str]:
    """Return unique filenames while preserving retrieval rank."""
    return list(dict.fromkeys(filenames))


def evaluate_retrieval(
    db: Session,
    cases: list[dict[str, Any]],
    *,
    embedding_service: EmbeddingService,
    reranking_service: RerankingService,
) -> dict[str, dict[str, float]]:
    """Compare vector retrieval with reranked retrieval."""
    baseline_results: list[list[str]] = []
    reranked_results: list[list[str]] = []
    relevant_results: list[set[str]] = []
    evaluation_k = 5

    for case in cases:
        query = case["query"]
        top_k = case["top_k"]
        relevant_results.append(case["relevant_filenames"])

        # Baseline: vector search only.
        baseline = search_chunks_with_distance(
            db,
            query,
            top_k=top_k,
            embedding_service=embedding_service,
        )
        baseline_results.append(
             _deduplicate_preserving_order(
                [chunk.document.filename for chunk, _ in baseline]
            )
        )

        # Candidate pool: up to 10 chunks, or top_k if larger.
        candidate_k = max(top_k, 10)

        reranked = search_chunks_with_reranking(
            db,
            query,
            candidate_k=candidate_k,
            top_k=top_k,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
        )
        reranked_results.append(
            _deduplicate_preserving_order(
                [chunk.document.filename for chunk, _ in reranked]
            )
        )

    def calculate_metrics(
        retrieved: list[list[str]],
    ) -> dict[str, float]:
        return {
            "hit_rate_at_5": hit_rate_at_k(
                retrieved,
                relevant_results,
                k=evaluation_k,
            ),
            "mrr_at_5": mean_reciprocal_rank_at_k(
                retrieved,
                relevant_results,
                k=evaluation_k,
            ),
        }

    return {
        "vector_search": calculate_metrics(baseline_results),
        "reranked_search": calculate_metrics(reranked_results),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = (
        project_root / "data" / "evaluation" / "retrieval_cases.jsonl"
    )

    cases = load_evaluation_cases(dataset_path)
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )

    try:
        with Session(engine) as db:
            # Load each model once and reuse it for all evaluation queries.
            embedding_service = EmbeddingService()
            reranking_service = RerankingService()

            metrics = evaluate_retrieval(
                db,
                cases,
                embedding_service=embedding_service,
                reranking_service=reranking_service,
            )
            failures = evaluate_quality_gate(metrics)

            if failures:
                print("\nQUALITY GATE: FAILED")
                for failure in failures:
                    print(f"  - {failure}")
                sys.exit(1)

            print("\nQUALITY GATE: PASSED")

        # Calculate improvements from the actual evaluation results.
        improvements = {
            metric: round(
                metrics["reranked_search"][metric]
                - metrics["vector_search"][metric],
                6,
            )
            for metric in metrics["vector_search"]
        }

        # Build a reproducible, timestamped report.
        generated_at = datetime.now(UTC)
        report = {
            "generated_at": generated_at.isoformat(),
            "evaluation_queries": len(cases),
            "metrics": metrics,
            "improvement": improvements,
        }

        # Persist the report under the project root.
        report_dir = project_root / "reports" / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
        report_path = report_dir / f"retrieval_{timestamp}.json"

        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        # Display the same results in the terminal.
        print(f"Evaluation queries: {len(cases)}")
        print("\nRetrieval Evaluation Report")
        print("-" * 50)

        for method, method_metrics in metrics.items():
            print(f"\n{method}")
            for metric, score in method_metrics.items():
                print(f"  {metric}: {score:.3f}")

        print("\nImprovement (reranked - baseline)")
        for metric, improvement in improvements.items():
            print(f"  {metric}: {improvement:+.3f}")

        print(f"\nJSON report saved to: {report_path}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

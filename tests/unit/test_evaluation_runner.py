
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from enterprise_rag_platform.evaluation.runner import (
    _deduplicate_preserving_order,
    evaluate_retrieval,
    load_evaluation_cases,
)


def test_load_evaluation_cases_returns_valid_cases(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "What is the leave policy?",
                "relevant_filenames": ["leave_policy.txt"],
                "top_k": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_evaluation_cases(dataset)

    assert len(cases) == 1
    assert cases[0]["query"] == "What is the leave policy?"
    assert cases[0]["relevant_filenames"] == {"leave_policy.txt"}
    assert cases[0]["top_k"] == 5


def test_load_evaluation_cases_uses_default_top_k(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "Who approves leave?",
                "relevant_filenames": ["leave_policy.txt"],
            }
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(dataset)

    assert cases[0]["top_k"] == 5


def test_load_evaluation_cases_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON on line 1"):
        load_evaluation_cases(dataset)


def test_load_evaluation_cases_rejects_blank_query(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "   ",
                "relevant_filenames": ["leave_policy.txt"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid query"):
        load_evaluation_cases(dataset)


def test_load_evaluation_cases_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "Who approves leave?",
                "relevant_filenames": ["leave_policy.txt"],
                "top_k": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid top_k"):
        load_evaluation_cases(dataset)


def test_load_evaluation_cases_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n   \n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Evaluation dataset contains no cases",
    ):
        load_evaluation_cases(dataset)


def test_load_evaluation_cases_rejects_missing_file(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Evaluation dataset not found"):
        load_evaluation_cases(dataset)

def test_evaluate_retrieval_compares_vector_and_reranked_results() -> None:
    cases = [
        {
            "query": "What is the leave policy?",
            "relevant_filenames": {"leave_policy.txt"},
            "top_k": 2,
        },
        {
            "query": "Who approves leave?",
            "relevant_filenames": {"approval_policy.txt"},
            "top_k": 2,
        },
    ]

    def make_chunk(filename: str) -> MagicMock:
        chunk = MagicMock()
        chunk.document.filename = filename
        return chunk

    vector_results = [
        (make_chunk("handbook.txt"), 0.1),
        (make_chunk("leave_policy.txt"), 0.2),
    ]
    reranked_results = [
        (make_chunk("leave_policy.txt"), 0.9),
        (make_chunk("handbook.txt"), 0.5),
    ]

    with (
        patch(
            "enterprise_rag_platform.evaluation.runner.search_chunks_with_distance",
            return_value=vector_results,
        ),
        patch(
            "enterprise_rag_platform.evaluation.runner.search_chunks_with_reranking",
            return_value=reranked_results,
        ),
    ):
        report = evaluate_retrieval(
            db=MagicMock(),
            cases=cases,
            embedding_service=MagicMock(),
            reranking_service=MagicMock(),
        )

    assert report["vector_search"]["hit_rate_at_5"] == 0.5
    assert report["reranked_search"]["hit_rate_at_5"] == 0.5
    assert report["vector_search"]["mrr_at_5"] == 0.25
    assert report["reranked_search"]["mrr_at_5"] == 0.5

def test_deduplicate_preserving_order_removes_duplicates() -> None:
    filenames = [
        "handbook.txt",
        "handbook.txt",
        "leave_policy.txt",
        "handbook.txt",
        "approval_policy.txt",
    ]

    result = _deduplicate_preserving_order(filenames)

    assert result == [
        "handbook.txt",
        "leave_policy.txt",
        "approval_policy.txt",
    ]


def test_deduplicate_preserving_order_handles_empty_input() -> None:
    assert _deduplicate_preserving_order([]) == []


def test_deduplicate_preserving_order_preserves_first_rank() -> None:
    filenames = [
        "leave_policy.txt",
        "handbook.txt",
        "leave_policy.txt",
    ]

    assert _deduplicate_preserving_order(filenames) == [
        "leave_policy.txt",
        "handbook.txt",
    ]

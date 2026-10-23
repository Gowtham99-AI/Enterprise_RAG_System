import json
import logging

from enterprise_rag_platform.core.logging import (
    JsonFormatter,
    correlation_id_context,
)


def test_json_formatter_outputs_structured_log() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="enterprise_rag_platform.services.retrieval",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Retrieval completed",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    result = json.loads(formatted)

    assert result["level"] == "INFO"
    assert result["logger"] == (
        "enterprise_rag_platform.services.retrieval"
    )
    assert result["message"] == "Retrieval completed"
    assert result["correlation_id"] is None
    assert "timestamp" in result


def test_json_formatter_includes_correlation_id() -> None:
    formatter = JsonFormatter()
    token = correlation_id_context.set("test-request-123")

    try:
        record = logging.LogRecord(
            name="enterprise_rag_platform.api",
            level=logging.INFO,
            pathname=__file__,
            lineno=20,
            msg="Request processed",
            args=(),
            exc_info=None,
        )

        result = json.loads(formatter.format(record))

        assert result["correlation_id"] == "test-request-123"
    finally:
        correlation_id_context.reset(token)


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="enterprise_rag_platform.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=30,
        msg="Request completed",
        args=(),
        exc_info=None,
    )
    record.method = "POST"
    record.path = "/api/v1/search"
    record.duration_ms = 125.5
    record.operation = "search"

    result = json.loads(formatter.format(record))

    assert result["method"] == "POST"
    assert result["path"] == "/api/v1/search"
    assert result["duration_ms"] == 125.5
    assert result["operation"] == "search"

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

correlation_id_context: ContextVar[str | None] = ContextVar(
    "correlation_id",
    default=None,
)

STRUCTURED_LOG_FIELDS = (
    "method",
    "path",
    "status_code",
    "duration_ms",
    "operation",
    "document_id",
    "candidate_count",
    "result_count",
    "source_count",
    "stage",
)


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": (
                getattr(record, "correlation_id", None)
                or correlation_id_context.get()
            ),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for field in STRUCTURED_LOG_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                log_data[field] = value

        return json.dumps(log_data, ensure_ascii=False)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure consistent JSON logging for the application."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

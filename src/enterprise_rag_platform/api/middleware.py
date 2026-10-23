import logging
from time import perf_counter
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from enterprise_rag_platform.core.logging import correlation_id_context

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        supplied_id = headers.get(b"x-correlation-id", b"").decode(
            "latin-1"
        )

        if supplied_id and len(supplied_id) <= 128:
            correlation_id = supplied_id
        else:
            correlation_id = str(uuid4())

        scope.setdefault("state", {})["correlation_id"] = correlation_id

        start_time = perf_counter()
        status_code = 500

        async def send_with_correlation_id(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = list(message.get("headers", []))
                response_headers.append(
                    (b"x-correlation-id", correlation_id.encode("ascii"))
                )
                message = {**message, "headers": response_headers}

            await send(message)

        token = correlation_id_context.set(correlation_id)

        try:
            await self.app(scope, receive, send_with_correlation_id)
        except Exception:
            logger.exception(
                "Request failed",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": status_code,
                    "duration_ms": round(
                        (perf_counter() - start_time) * 1000,
                        2,
                    ),
                    "correlation_id": correlation_id,
                    "operation": "http_request",
                },
            )
            raise
        else:
            logger.info(
                "Request completed",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": status_code,
                    "duration_ms": round(
                        (perf_counter() - start_time) * 1000,
                        2,
                    ),
                    "correlation_id": correlation_id,
                    "operation": "http_request",
                },
            )
        finally:
            correlation_id_context.reset(token)
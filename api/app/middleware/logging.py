"""Request logging middleware with JSON-formatted log records.

Logs each request with timestamp, path, method, status code, and response time.
Supports configurable log levels and optional file output for Docker volume persistence.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_LOGGER_NAME = "request_logger"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every HTTP request/response as a JSON record.

    Args:
        app: The ASGI application.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR). Default "INFO".
        log_file: Optional path to a log file (e.g. inside a Docker volume).
    """

    def __init__(
        self,
        app,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ) -> None:
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file
        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        """Create (or retrieve) a logger with a stream handler and optional file handler."""
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(self.log_level)

        # Avoid duplicate handlers when middleware is re-created
        if not logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.log_level)
            logger.addHandler(stream_handler)

        if self.log_file:
            # Check if a file handler for this path already exists
            has_file_handler = any(
                isinstance(h, logging.FileHandler) and h.baseFilename.endswith(self.log_file.split("/")[-1])
                for h in logger.handlers
            )
            if not has_file_handler:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(self.log_level)
                logger.addHandler(file_handler)

        return logger

    @staticmethod
    def format_log_record(
        *,
        timestamp: str,
        path: str,
        method: str,
        status_code: int,
        response_time_ms: float,
    ) -> Dict[str, Any]:
        """Build a JSON-serialisable log record dict.

        Useful for testing and for consistent log formatting.
        """
        return {
            "timestamp": timestamp,
            "path": path,
            "method": method,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        timestamp = datetime.now(timezone.utc).isoformat()
        record = self.format_log_record(
            timestamp=timestamp,
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time_ms=elapsed_ms,
        )

        log_line = json.dumps(record)

        # Choose level based on status code range
        status = response.status_code
        if status >= 500:
            self.logger.error(log_line)
        elif status >= 400:
            self.logger.warning(log_line)
        else:
            self.logger.info(log_line)

        return response

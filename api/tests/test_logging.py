"""Unit tests for RequestLoggingMiddleware."""

import json
import logging
import os
import tempfile

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.logging import RequestLoggingMiddleware


# ---------------------------------------------------------------------------
# Tiny Starlette apps used by the tests
# ---------------------------------------------------------------------------

def _ok_endpoint(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _not_found_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({"error": "not found"}, status_code=404)


def _error_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({"error": "boom"}, status_code=500)


def _build_app(log_level: str = "INFO", log_file: str | None = None) -> Starlette:
    app = Starlette(
        routes=[
            Route("/ok", _ok_endpoint),
            Route("/notfound", _not_found_endpoint),
            Route("/error", _error_endpoint),
        ],
    )
    app.add_middleware(RequestLoggingMiddleware, log_level=log_level, log_file=log_file)
    return app


# ---------------------------------------------------------------------------
# format_log_record
# ---------------------------------------------------------------------------

class TestFormatLogRecord:
    def test_contains_all_required_fields(self):
        record = RequestLoggingMiddleware.format_log_record(
            timestamp="2024-01-01T00:00:00+00:00",
            path="/v1/completions",
            method="POST",
            status_code=200,
            response_time_ms=12.345,
        )
        assert record["timestamp"] == "2024-01-01T00:00:00+00:00"
        assert record["path"] == "/v1/completions"
        assert record["method"] == "POST"
        assert record["status_code"] == 200
        assert record["response_time_ms"] == 12.35  # rounded to 2 decimals

    def test_is_json_serialisable(self):
        record = RequestLoggingMiddleware.format_log_record(
            timestamp="2024-06-15T10:30:00+00:00",
            path="/health",
            method="GET",
            status_code=200,
            response_time_ms=0.5,
        )
        serialised = json.dumps(record)
        parsed = json.loads(serialised)
        assert parsed == record


# ---------------------------------------------------------------------------
# Middleware dispatch behaviour
# ---------------------------------------------------------------------------

class TestMiddlewareDispatch:
    def test_successful_request_logged_at_info(self, caplog):
        app = _build_app(log_level="DEBUG")
        client = TestClient(app)
        with caplog.at_level(logging.DEBUG, logger="request_logger"):
            resp = client.get("/ok")
        assert resp.status_code == 200
        # Find the JSON log line in captured records
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        log_data = json.loads(info_records[-1].message)
        assert log_data["path"] == "/ok"
        assert log_data["method"] == "GET"
        assert log_data["status_code"] == 200
        assert "timestamp" in log_data
        assert "response_time_ms" in log_data

    def test_client_error_logged_at_warning(self, caplog):
        app = _build_app(log_level="DEBUG")
        client = TestClient(app)
        with caplog.at_level(logging.DEBUG, logger="request_logger"):
            resp = client.get("/notfound")
        assert resp.status_code == 404
        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) >= 1
        log_data = json.loads(warn_records[-1].message)
        assert log_data["status_code"] == 404

    def test_server_error_logged_at_error(self, caplog):
        app = _build_app(log_level="DEBUG")
        client = TestClient(app)
        with caplog.at_level(logging.DEBUG, logger="request_logger"):
            resp = client.get("/error")
        assert resp.status_code == 500
        err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(err_records) >= 1
        log_data = json.loads(err_records[-1].message)
        assert log_data["status_code"] == 500

    def test_response_time_is_positive(self, caplog):
        app = _build_app(log_level="DEBUG")
        client = TestClient(app)
        with caplog.at_level(logging.DEBUG, logger="request_logger"):
            client.get("/ok")
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        log_data = json.loads(info_records[-1].message)
        assert log_data["response_time_ms"] >= 0


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------

class TestFileLogging:
    def test_writes_to_log_file(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            # Clear any existing handlers from previous tests
            logger = logging.getLogger("request_logger")
            logger.handlers.clear()

            app = _build_app(log_level="DEBUG", log_file=log_path)
            client = TestClient(app)
            client.get("/ok")

            with open(log_path) as fh:
                content = fh.read()
            assert content.strip() != ""
            log_data = json.loads(content.strip())
            assert log_data["path"] == "/ok"
        finally:
            # Cleanup handlers and temp file
            logger = logging.getLogger("request_logger")
            logger.handlers.clear()
            os.unlink(log_path)

# Feature: vllm-docker-inference, Property 6: Log Kaydı Tutarlılığı
"""
Property-based tests for log record consistency.

**Validates: Requirements 6.1, 6.2**

Property 6: For any HTTP request, the generated log record must be a valid
JSON object and must contain the required fields: timestamp, path, method,
status_code, and response_time.
"""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from app.middleware.logging import RequestLoggingMiddleware


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# ISO-format timestamp strings
timestamp_st = st.datetimes().map(lambda dt: dt.isoformat())

# Request paths
path_st = st.sampled_from(["/v1/completions", "/v1/chat/completions", "/health", "/v1/models", "/metrics"]) | st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789/-_.",
    min_size=1,
    max_size=128,
).map(lambda p: "/" + p)

# HTTP methods
method_st = st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"])

# Status codes (valid HTTP range)
status_code_st = st.integers(min_value=100, max_value=599)

# Response times in milliseconds (positive floats)
response_time_st = st.floats(min_value=0.01, max_value=60000.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"timestamp", "path", "method", "status_code", "response_time_ms"}


class TestLogRecordConsistency:
    """Property 6: Log Kaydı Tutarlılığı."""

    @given(
        timestamp=timestamp_st,
        path=path_st,
        method=method_st,
        status_code=status_code_st,
        response_time=response_time_st,
    )
    @settings(max_examples=100)
    def test_log_record_is_valid_dict_with_required_fields(
        self, timestamp, path, method, status_code, response_time
    ):
        """The log record must be a dict containing all required keys.

        **Validates: Requirements 6.1, 6.2**
        """
        record = RequestLoggingMiddleware.format_log_record(
            timestamp=timestamp,
            path=path,
            method=method,
            status_code=status_code,
            response_time_ms=response_time,
        )

        # Must be a dict
        assert isinstance(record, dict)

        # Must contain all required keys
        assert REQUIRED_KEYS.issubset(record.keys()), (
            f"Missing keys: {REQUIRED_KEYS - record.keys()}"
        )

    @given(
        timestamp=timestamp_st,
        path=path_st,
        method=method_st,
        status_code=status_code_st,
        response_time=response_time_st,
    )
    @settings(max_examples=100)
    def test_log_record_is_json_serializable(
        self, timestamp, path, method, status_code, response_time
    ):
        """The log record must be JSON-serializable (json.dumps must not raise).

        **Validates: Requirements 6.2**
        """
        record = RequestLoggingMiddleware.format_log_record(
            timestamp=timestamp,
            path=path,
            method=method,
            status_code=status_code,
            response_time_ms=response_time,
        )

        # Must not raise
        serialized = json.dumps(record)
        assert isinstance(serialized, str)

    @given(
        timestamp=timestamp_st,
        path=path_st,
        method=method_st,
        status_code=status_code_st,
        response_time=response_time_st,
    )
    @settings(max_examples=100)
    def test_log_record_values_match_inputs(
        self, timestamp, path, method, status_code, response_time
    ):
        """The log record values must match the provided inputs.

        **Validates: Requirements 6.1**
        """
        record = RequestLoggingMiddleware.format_log_record(
            timestamp=timestamp,
            path=path,
            method=method,
            status_code=status_code,
            response_time_ms=response_time,
        )

        assert record["timestamp"] == timestamp
        assert record["path"] == path
        assert record["method"] == method
        assert record["status_code"] == status_code
        assert record["response_time_ms"] == round(response_time, 2)

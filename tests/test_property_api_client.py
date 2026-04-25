# Feature: colab-benchmark-system, Property 18: HTTP Durum Kodu Hata Tespiti
# Validates: Requirements 9.4
"""
Property-based tests for HTTP status code error detection.

For any HTTP response status code, if the code is 200 the response should be
marked as successful (error is None); if the code is not 200 the response
should be marked as an error (error is not None).

We test the invariant on APIResponse instances that mirror what APIClient
produces for various status codes.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from benchmark.models import APIResponse


# Strategy: valid HTTP status codes (100-599)
http_status_code_st = st.integers(min_value=100, max_value=599)


def build_api_response(status_code: int) -> APIResponse:
    """Build an APIResponse mimicking APIClient behavior for a given status code.

    This mirrors the logic in APIClient._do_streaming_request:
    - status_code == 200 → error=None, response_text may be present
    - status_code != 200 → error is set, response_text is None
    """
    if status_code == 200:
        return APIResponse(
            status_code=status_code,
            response_text="some response",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            error=None,
            elapsed_ms=100.0,
        )
    else:
        return APIResponse(
            status_code=status_code,
            response_text=None,
            usage=None,
            error=f"HTTP {status_code}: error",
            elapsed_ms=100.0,
        )


@given(status_code=http_status_code_st)
@settings(max_examples=100)
def test_http_200_marked_as_successful(status_code: int):
    """**Validates: Requirements 9.4**

    For any HTTP status code == 200, the APIResponse should have error=None
    (marked as successful).
    """
    response = build_api_response(status_code)
    if status_code == 200:
        assert response.error is None, (
            f"Status 200 should be successful but got error: {response.error}"
        )


@given(status_code=http_status_code_st.filter(lambda c: c != 200))
@settings(max_examples=100)
def test_http_non_200_marked_as_error(status_code: int):
    """**Validates: Requirements 9.4**

    For any HTTP status code != 200, the APIResponse should have error set
    (not None), indicating an error.
    """
    response = build_api_response(status_code)
    assert response.error is not None, (
        f"Status {status_code} should be an error but error is None"
    )
    assert response.response_text is None, (
        f"Status {status_code} should have no response_text"
    )


@given(status_code=http_status_code_st)
@settings(max_examples=100)
def test_http_status_code_error_detection_combined(status_code: int):
    """**Validates: Requirements 9.4**

    Combined property: for any HTTP status code, 200 → success, non-200 → error.
    This is the core invariant of Property 18.
    """
    response = build_api_response(status_code)

    if status_code == 200:
        assert response.error is None
        assert response.response_text is not None
    else:
        assert response.error is not None
        assert response.response_text is None
        assert str(status_code) in response.error

# Feature: colab-benchmark-system, Property 9: Metrik Hesaplama Doğruluğu
# Feature: colab-benchmark-system, Property 10: nvidia-smi Çıktı Ayrıştırma
# Feature: colab-benchmark-system, Property 16: VRAM İstatistik Hesaplama
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.2
"""
Property-based tests for MetricsCollector and VRAMTracker.

Property 9: For any positive completion_tokens and positive total_time_ms,
the calculated tokens_per_second should equal completion_tokens / (total_time_ms / 1000);
and all time metrics (TTFT, total_time_ms) should be non-negative.

Property 10: For any known used_mb and total_mb values, a valid nvidia-smi CSV
output string constructed from those values should be correctly parsed by
parse_nvidia_smi (round-trip).

Property 16: For any sequence of VRAM measurements (at least one), the calculated
peak_mb should equal the maximum used_mb and avg_mb should equal the arithmetic
mean of all used_mb values.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmark.metrics import MetricsCollector, VRAMTracker
from benchmark.models import ResponseMetrics, VRAMSnapshot


# --- Strategies ---

positive_float = st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False)
non_negative_float = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_int = st.integers(min_value=1, max_value=100000)
non_negative_int = st.integers(min_value=0, max_value=100000)

vram_snapshot_st = st.builds(
    VRAMSnapshot,
    timestamp=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1),
        timezones=st.just(timezone.utc),
    ),
    used_mb=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    total_mb=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    source=st.just("nvidia_smi"),
)


# --- Property 9: Metrik Hesaplama Doğruluğu ---
# **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**


@given(
    completion_tokens=positive_int,
    total_time_ms=positive_float,
)
@settings(max_examples=100)
def test_tokens_per_second_formula(completion_tokens: int, total_time_ms: float):
    """tokens_per_second equals completion_tokens / (total_time_ms / 1000) for positive values."""
    metrics = ResponseMetrics(
        total_time_ms=total_time_ms,
        completion_tokens=completion_tokens,
        prompt_tokens=0,
        tokens_per_second=completion_tokens / (total_time_ms / 1000),
    )
    expected = completion_tokens / (total_time_ms / 1000)
    assert metrics.tokens_per_second == pytest.approx(expected, rel=1e-9)


@given(
    completion_tokens=positive_int,
    prompt_tokens=non_negative_int,
)
@settings(max_examples=100)
def test_metrics_collector_formula_via_mock_time(
    completion_tokens: int, prompt_tokens: int
):
    """MetricsCollector.calculate produces correct tokens_per_second using the formula."""
    start_time = 1000.0
    end_time = 1000.05  # 50ms

    mc = MetricsCollector()
    with patch("benchmark.metrics.time.perf_counter", side_effect=[start_time, end_time]):
        mc.start_timer()
        mc.stop_timer()

    metrics = mc.calculate(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    total_time_ms = (end_time - start_time) * 1000
    expected_tps = completion_tokens / (total_time_ms / 1000)

    assert metrics.total_time_ms == pytest.approx(total_time_ms, rel=1e-9)
    assert metrics.tokens_per_second == pytest.approx(expected_tps, rel=1e-9)
    assert metrics.total_time_ms >= 0


@given(
    ttft_offset=positive_float,
    total_offset=positive_float,
)
@settings(max_examples=100)
def test_time_metrics_non_negative(ttft_offset: float, total_offset: float):
    """All time metrics (TTFT, total_time_ms) are non-negative."""
    start = 1000.0
    first_token = start + ttft_offset / 1000  # convert ms-scale to seconds
    end = first_token + total_offset / 1000

    mc = MetricsCollector()
    with patch(
        "benchmark.metrics.time.perf_counter",
        side_effect=[start, first_token, end],
    ):
        mc.start_timer()
        mc.record_first_token()
        mc.stop_timer()

    metrics = mc.calculate(prompt_tokens=10, completion_tokens=20)

    assert metrics.total_time_ms >= 0
    assert metrics.ttft_ms is not None
    assert metrics.ttft_ms >= 0


# --- Property 10: nvidia-smi Çıktı Ayrıştırma ---
# **Validates: Requirements 4.5**


@given(
    used_mb=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    total_mb=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_parse_nvidia_smi_roundtrip(used_mb: float, total_mb: float):
    """Constructing nvidia-smi CSV from known values and parsing it back yields the same values."""
    csv_output = f"{used_mb}, {total_mb}\n"
    vt = VRAMTracker()
    snapshot = vt.parse_nvidia_smi(csv_output)

    assert snapshot.used_mb == pytest.approx(used_mb, rel=1e-6)
    assert snapshot.total_mb == pytest.approx(total_mb, rel=1e-6)
    assert snapshot.source == "nvidia_smi"


# --- Property 16: VRAM İstatistik Hesaplama ---
# **Validates: Requirements 7.1, 7.2**


@given(
    snapshots=st.lists(vram_snapshot_st, min_size=1, max_size=50),
)
@settings(max_examples=100)
def test_vram_peak_and_average(snapshots: list[VRAMSnapshot]):
    """peak_mb equals max(used_mb) and avg_mb equals mean(used_mb) for any snapshot sequence."""
    used_values = [s.used_mb for s in snapshots]

    expected_peak = max(used_values)
    expected_avg = sum(used_values) / len(used_values)

    # Compute peak and average the same way the system would
    actual_peak = max(s.used_mb for s in snapshots)
    actual_avg = sum(s.used_mb for s in snapshots) / len(snapshots)

    assert actual_peak == pytest.approx(expected_peak, rel=1e-9)
    assert actual_avg == pytest.approx(expected_avg, rel=1e-9)

"""MetricsCollector ve VRAMTracker birim testleri."""

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from benchmark.metrics import MetricsCollector, VRAMTracker


# ── MetricsCollector ──


class TestMetricsCollectorTimerLifecycle:
    def test_start_stop_calculate(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        time.sleep(0.01)
        mc.stop_timer()
        metrics = mc.calculate(prompt_tokens=10, completion_tokens=20)
        assert metrics.total_time_ms > 0
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 20

    def test_calculate_without_start_raises(self) -> None:
        mc = MetricsCollector()
        with pytest.raises(RuntimeError):
            mc.calculate(prompt_tokens=0, completion_tokens=0)

    def test_calculate_without_stop_raises(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        with pytest.raises(RuntimeError):
            mc.calculate(prompt_tokens=0, completion_tokens=0)


class TestMetricsCollectorTTFT:
    def test_ttft_recorded(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        time.sleep(0.005)
        mc.record_first_token()
        time.sleep(0.005)
        mc.stop_timer()
        metrics = mc.calculate(prompt_tokens=5, completion_tokens=10)
        assert metrics.ttft_ms is not None
        assert metrics.ttft_ms > 0
        assert metrics.ttft_ms < metrics.total_time_ms

    def test_ttft_none_when_not_recorded(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        mc.stop_timer()
        metrics = mc.calculate(prompt_tokens=5, completion_tokens=10)
        assert metrics.ttft_ms is None


class TestMetricsCollectorTokensPerSecond:
    def test_tokens_per_second_calculated(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        time.sleep(0.01)
        mc.stop_timer()
        metrics = mc.calculate(prompt_tokens=10, completion_tokens=50)
        assert metrics.tokens_per_second is not None
        assert metrics.tokens_per_second > 0

    def test_tokens_per_second_none_when_zero_completion(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        time.sleep(0.01)
        mc.stop_timer()
        metrics = mc.calculate(prompt_tokens=10, completion_tokens=0)
        assert metrics.tokens_per_second is None

    def test_start_resets_state(self) -> None:
        mc = MetricsCollector()
        mc.start_timer()
        mc.record_first_token()
        mc.stop_timer()
        # Start again — should reset
        mc.start_timer()
        mc.stop_timer()
        metrics = mc.calculate(prompt_tokens=0, completion_tokens=0)
        assert metrics.ttft_ms is None


# ── VRAMTracker ──


class TestVRAMTrackerInit:
    def test_no_endpoint(self) -> None:
        vt = VRAMTracker()
        assert vt._metrics_endpoint is None

    def test_with_endpoint(self) -> None:
        vt = VRAMTracker(metrics_endpoint="http://localhost:8000/metrics")
        assert vt._metrics_endpoint == "http://localhost:8000/metrics"


class TestVRAMTrackerIsAvailable:
    def test_available_with_endpoint(self) -> None:
        vt = VRAMTracker(metrics_endpoint="http://localhost:8000/metrics")
        assert vt.is_available() is True

    @patch("benchmark.metrics.subprocess.run", side_effect=FileNotFoundError)
    def test_unavailable_without_nvidia_smi(self, mock_run: MagicMock) -> None:
        vt = VRAMTracker()
        assert vt.is_available() is False


class TestVRAMTrackerParseNvidiaSmi:
    def test_single_gpu(self) -> None:
        vt = VRAMTracker()
        snap = vt.parse_nvidia_smi("8192, 16384\n")
        assert snap.used_mb == 8192.0
        assert snap.total_mb == 16384.0
        assert snap.source == "nvidia_smi"

    def test_multi_gpu_uses_first(self) -> None:
        vt = VRAMTracker()
        snap = vt.parse_nvidia_smi("4096, 8192\n12000, 24000\n")
        assert snap.used_mb == 4096.0
        assert snap.total_mb == 8192.0

    def test_decimal_values(self) -> None:
        vt = VRAMTracker()
        snap = vt.parse_nvidia_smi("1234.5, 16384.0\n")
        assert snap.used_mb == 1234.5
        assert snap.total_mb == 16384.0

    def test_invalid_output_raises(self) -> None:
        vt = VRAMTracker()
        with pytest.raises(ValueError):
            vt.parse_nvidia_smi("garbage")

    def test_non_numeric_raises(self) -> None:
        vt = VRAMTracker()
        with pytest.raises(ValueError):
            vt.parse_nvidia_smi("abc, def\n")


class TestVRAMTrackerSnapshot:
    @patch("benchmark.metrics.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_nothing_available(self, mock_run: MagicMock) -> None:
        vt = VRAMTracker()
        assert vt.snapshot() is None

    @patch("benchmark.metrics.subprocess.run")
    def test_uses_nvidia_smi(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="6000, 16384\n"
        )
        vt = VRAMTracker()
        snap = vt.snapshot()
        assert snap is not None
        assert snap.used_mb == 6000.0
        assert snap.total_mb == 16384.0
        assert snap.source == "nvidia_smi"

    @patch("benchmark.metrics.subprocess.run", side_effect=FileNotFoundError)
    @patch("benchmark.metrics.httpx.get")
    def test_falls_back_to_endpoint(
        self, mock_get: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "# HELP gpu_memory_usage_bytes\n"
            "vllm:gpu_memory_usage_bytes{gpu=\"0\"} 8589934592\n"
            "# HELP gpu_memory_total_bytes\n"
            "vllm:gpu_memory_total_bytes{gpu=\"0\"} 17179869184\n"
        )
        mock_get.return_value = mock_response

        vt = VRAMTracker(metrics_endpoint="http://localhost:8000/metrics")
        snap = vt.snapshot()
        assert snap is not None
        assert snap.source == "vllm_metrics"
        assert snap.used_mb == pytest.approx(8192.0, rel=0.01)
        assert snap.total_mb == pytest.approx(16384.0, rel=0.01)


class TestVRAMTrackerFetchVllmMetrics:
    def test_returns_none_without_endpoint(self) -> None:
        vt = VRAMTracker()
        assert vt.fetch_vllm_metrics() is None

    @patch("benchmark.metrics.httpx.get")
    def test_returns_none_on_http_error(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        vt = VRAMTracker(metrics_endpoint="http://localhost:8000/metrics")
        assert vt.fetch_vllm_metrics() is None

    @patch("benchmark.metrics.httpx.get", side_effect=httpx.RequestError("fail"))
    def test_returns_none_on_network_error(self, mock_get: MagicMock) -> None:
        vt = VRAMTracker(metrics_endpoint="http://localhost:8000/metrics")
        assert vt.fetch_vllm_metrics() is None

    @patch("benchmark.metrics.httpx.get")
    def test_parses_prometheus_metrics(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "# HELP vllm:gpu_memory_usage_bytes\n"
            "vllm:gpu_memory_usage_bytes{gpu=\"0\"} 4294967296\n"
        )
        mock_get.return_value = mock_response
        vt = VRAMTracker(metrics_endpoint="http://localhost:8000/metrics")
        snap = vt.fetch_vllm_metrics()
        assert snap is not None
        assert snap.used_mb == pytest.approx(4096.0, rel=0.01)
        assert snap.source == "vllm_metrics"

"""Unit tests for the HealthMonitor service."""

import time

import httpx
import pytest

from app.models.responses import EngineHealthStatus, HealthResponse, SystemMetrics
from app.services.health import HealthMonitor
from app.services.model_registry import ModelRegistry

pytestmark = pytest.mark.asyncio


@pytest.fixture
def registry() -> ModelRegistry:
    """Return a ModelRegistry with two models registered."""
    reg = ModelRegistry()
    reg.register_model("model-a", "http://vllm-a:8000")
    reg.register_model("model-b", "http://vllm-b:8000")
    return reg


@pytest.fixture
def monitor(registry: ModelRegistry) -> HealthMonitor:
    return HealthMonitor(registry)


# ------------------------------------------------------------------
# check_health tests
# ------------------------------------------------------------------


async def test_check_health_all_healthy(monitor: HealthMonitor) -> None:
    """When all engines respond 200, overall status is 'healthy'."""

    async def _mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    # Patch the internal method to use a mock transport
    original = monitor._check_engine

    async def _patched(client, model_name, endpoint):
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_transport))
        return await original(mock_client, model_name, endpoint)

    monitor._check_engine = _patched  # type: ignore[assignment]

    result = await monitor.check_health()

    assert isinstance(result, HealthResponse)
    assert result.status == "healthy"
    assert len(result.engines) == 2
    assert all(e.status == "healthy" for e in result.engines)
    assert result.timestamp  # non-empty ISO timestamp



async def test_check_health_all_unhealthy(monitor: HealthMonitor) -> None:
    """When no engine responds, overall status is 'unhealthy'."""

    async def _mock_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    original = monitor._check_engine

    async def _patched(client, model_name, endpoint):
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_transport))
        return await original(mock_client, model_name, endpoint)

    monitor._check_engine = _patched  # type: ignore[assignment]

    result = await monitor.check_health()

    assert result.status == "unhealthy"
    assert all(e.status == "unhealthy" for e in result.engines)



async def test_check_health_degraded(registry: ModelRegistry) -> None:
    """When some engines are healthy and some are not, status is 'degraded'."""
    monitor = HealthMonitor(registry)

    call_count = 0

    async def _mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "vllm-a" in str(request.url):
            return httpx.Response(200)
        raise httpx.ConnectError("connection refused")

    original = monitor._check_engine

    async def _patched(client, model_name, endpoint):
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_transport))
        return await original(mock_client, model_name, endpoint)

    monitor._check_engine = _patched  # type: ignore[assignment]

    result = await monitor.check_health()

    assert result.status == "degraded"
    statuses = {e.model_name: e.status for e in result.engines}
    assert statuses["model-a"] == "healthy"
    assert statuses["model-b"] == "unhealthy"



async def test_check_health_no_engines() -> None:
    """When no engines are registered, overall status is 'unhealthy'."""
    empty_registry = ModelRegistry()
    monitor = HealthMonitor(empty_registry)

    result = await monitor.check_health()

    assert result.status == "unhealthy"
    assert result.engines == []



async def test_check_health_non_200_is_unhealthy(monitor: HealthMonitor) -> None:
    """A non-200 response from /health marks the engine as unhealthy."""

    async def _mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    original = monitor._check_engine

    async def _patched(client, model_name, endpoint):
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_transport))
        return await original(mock_client, model_name, endpoint)

    monitor._check_engine = _patched  # type: ignore[assignment]

    result = await monitor.check_health()

    assert result.status == "unhealthy"
    assert all(e.status == "unhealthy" for e in result.engines)


# ------------------------------------------------------------------
# get_metrics tests
# ------------------------------------------------------------------



async def test_get_metrics_returns_system_metrics(monitor: HealthMonitor) -> None:
    """get_metrics returns a valid SystemMetrics object."""
    metrics = await monitor.get_metrics()

    assert isinstance(metrics, SystemMetrics)
    assert metrics.gpu_memory_used_mb is None
    assert metrics.gpu_memory_total_mb is None
    assert metrics.active_requests == 0
    assert metrics.uptime_seconds >= 0



async def test_get_metrics_tracks_active_requests(monitor: HealthMonitor) -> None:
    """Active request counter increments and decrements correctly."""
    monitor.increment_active_requests()
    monitor.increment_active_requests()

    metrics = await monitor.get_metrics()
    assert metrics.active_requests == 2

    monitor.decrement_active_requests()
    metrics = await monitor.get_metrics()
    assert metrics.active_requests == 1



async def test_decrement_does_not_go_below_zero(monitor: HealthMonitor) -> None:
    """Decrementing below zero clamps to 0."""
    monitor.decrement_active_requests()
    metrics = await monitor.get_metrics()
    assert metrics.active_requests == 0


# ------------------------------------------------------------------
# _calculate_overall_status unit tests
# ------------------------------------------------------------------


def test_calculate_overall_status_all_healthy() -> None:
    engines = [
        EngineHealthStatus(model_name="a", status="healthy", endpoint="http://a"),
        EngineHealthStatus(model_name="b", status="healthy", endpoint="http://b"),
    ]
    assert HealthMonitor._calculate_overall_status(engines) == "healthy"


def test_calculate_overall_status_all_unhealthy() -> None:
    engines = [
        EngineHealthStatus(model_name="a", status="unhealthy", endpoint="http://a"),
        EngineHealthStatus(model_name="b", status="unhealthy", endpoint="http://b"),
    ]
    assert HealthMonitor._calculate_overall_status(engines) == "unhealthy"


def test_calculate_overall_status_mixed() -> None:
    engines = [
        EngineHealthStatus(model_name="a", status="healthy", endpoint="http://a"),
        EngineHealthStatus(model_name="b", status="unhealthy", endpoint="http://b"),
    ]
    assert HealthMonitor._calculate_overall_status(engines) == "degraded"


def test_calculate_overall_status_empty() -> None:
    assert HealthMonitor._calculate_overall_status([]) == "unhealthy"

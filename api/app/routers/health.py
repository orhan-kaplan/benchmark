"""Health router — /health and /metrics endpoints.

Reports overall system health and per-engine statuses, plus system
metrics such as GPU memory, active requests, and uptime.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.routers._registry import get_shared_registry
from app.services.health import HealthMonitor

router = APIRouter()

_monitor: HealthMonitor | None = None


def _get_monitor() -> HealthMonitor:
    """Return the module-level HealthMonitor, creating it on first call."""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor(get_shared_registry())
    return _monitor


@router.get("/health")
async def health_check():
    """Return overall system health and per-engine statuses."""
    monitor = _get_monitor()
    health = await monitor.check_health()
    return JSONResponse(status_code=200, content=health.model_dump())


@router.get("/metrics")
async def metrics():
    """Return system metrics (GPU memory, active requests, uptime)."""
    monitor = _get_monitor()
    system_metrics = await monitor.get_metrics()
    return JSONResponse(status_code=200, content=system_metrics.model_dump())

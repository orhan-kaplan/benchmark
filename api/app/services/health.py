"""Health monitor service for checking vLLM engine status and system metrics."""

import time
from datetime import datetime, timezone
from typing import List

import httpx

from app.models.responses import EngineHealthStatus, HealthResponse, SystemMetrics
from app.services.model_registry import ModelRegistry


class HealthMonitor:
    """Monitors health of registered vLLM engine instances and tracks system metrics."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._start_time = time.monotonic()
        self._active_requests = 0

    def increment_active_requests(self) -> None:
        """Increment the active request counter."""
        self._active_requests += 1

    def decrement_active_requests(self) -> None:
        """Decrement the active request counter."""
        self._active_requests = max(0, self._active_requests - 1)

    async def check_health(self) -> HealthResponse:
        """Check reachability of all registered vLLM engine instances.

        Makes an HTTP GET to ``{endpoint}/health`` for each registered model.
        A 200 response marks the engine as healthy; any other outcome marks it
        as unhealthy.

        Overall status logic:
        - All engines healthy → "healthy"
        - Mixed → "degraded"
        - All unhealthy (or no engines) → "unhealthy"

        Returns:
            HealthResponse with overall status, per-engine statuses, and timestamp.
        """
        models = self._registry.list_models()
        engine_statuses: List[EngineHealthStatus] = []

        async with httpx.AsyncClient(timeout=5.0) as client:
            for model_info in models:
                endpoint = self._registry.get_endpoint(model_info.id)
                status = await self._check_engine(client, model_info.id, endpoint)
                engine_statuses.append(status)

        overall = self._calculate_overall_status(engine_statuses)
        timestamp = datetime.now(timezone.utc).isoformat()

        return HealthResponse(
            status=overall,
            engines=engine_statuses,
            timestamp=timestamp,
        )

    async def get_metrics(self) -> SystemMetrics:
        """Return current system metrics.

        Attempts to read GPU memory via nvidia-smi. Falls back to None
        when no GPU information is available.

        Returns:
            SystemMetrics with gpu info (if available), active requests, and uptime.
        """
        import subprocess

        uptime = time.monotonic() - self._start_time

        gpu_used: float | None = None
        gpu_total: float | None = None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpu_used = float(parts[0])
                    gpu_total = float(parts[1])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return SystemMetrics(
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            active_requests=self._active_requests,
            uptime_seconds=round(uptime, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _check_engine(
        client: httpx.AsyncClient, model_name: str, endpoint: str
    ) -> EngineHealthStatus:
        """Probe a single engine's /health endpoint."""
        try:
            response = await client.get(f"{endpoint}/health")
            status = "healthy" if response.status_code == 200 else "unhealthy"
        except (httpx.RequestError, httpx.HTTPStatusError):
            status = "unhealthy"

        return EngineHealthStatus(
            model_name=model_name,
            status=status,
            endpoint=endpoint,
        )

    @staticmethod
    def _calculate_overall_status(
        engine_statuses: List[EngineHealthStatus],
    ) -> str:
        """Derive the overall system status from individual engine statuses.

        Returns "healthy", "degraded", or "unhealthy".
        """
        if not engine_statuses:
            return "unhealthy"

        healthy_count = sum(1 for e in engine_statuses if e.status == "healthy")

        if healthy_count == len(engine_statuses):
            return "healthy"
        if healthy_count == 0:
            return "unhealthy"
        return "degraded"

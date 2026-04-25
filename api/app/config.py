"""Application configuration reader.

Reads configuration from environment variables and provides
singleton-like access via get_config() and get_model_endpoints().
"""

import os
from typing import List

from app.models.config import AppConfig, ModelEndpointConfig, RateLimitConfig


def _parse_bool(value: str, default: bool = True) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.strip().lower() in ("true", "1", "yes")


def _parse_api_keys(value: str) -> List[str]:
    """Parse comma-separated API keys, filtering out empty strings."""
    return [k.strip() for k in value.split(",") if k.strip()]


def _parse_model_endpoints(value: str) -> List[ModelEndpointConfig]:
    """Parse MODEL_ENDPOINTS env var.

    Format: "model-a:http://vllm-a:8000,model-b:http://vllm-b:8000"

    The name and URL are separated by the *first* colon that is NOT
    followed by ``//``.  This avoids splitting on the ``://`` inside
    the URL scheme.
    """
    import re

    endpoints: List[ModelEndpointConfig] = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Match "name:URL" where the separating colon is NOT part of "://"
        match = re.match(r"^([^:]+):(\w+://.+)$", entry)
        if match:
            name = match.group(1).strip()
            endpoint = match.group(2).strip()
            endpoints.append(ModelEndpointConfig(name=name, endpoint=endpoint))
    return endpoints


def get_config() -> AppConfig:
    """Build and return an AppConfig from environment variables."""
    auth_enabled = _parse_bool(
        os.environ.get("AUTH_ENABLED", "true"),
    )
    api_keys = _parse_api_keys(os.environ.get("API_KEYS", ""))
    rpm = int(os.environ.get("RATE_LIMIT_RPM", "60"))
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    test_mode = _parse_bool(
        os.environ.get("TEST_MODE", "false"), default=False
    )

    return AppConfig(
        auth_enabled=auth_enabled,
        api_keys=api_keys,
        rate_limit=RateLimitConfig(requests_per_minute=rpm),
        log_level=log_level,  # type: ignore[arg-type]
        test_mode=test_mode,
    )


def get_model_endpoints() -> List[ModelEndpointConfig]:
    """Build and return a list of ModelEndpointConfig from environment variables."""
    raw = os.environ.get("MODEL_ENDPOINTS", "")
    return _parse_model_endpoints(raw)

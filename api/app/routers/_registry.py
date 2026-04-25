"""Shared ModelRegistry instance for all routers.

Lazily initialised on first access so that environment variables
are already loaded when the registry is populated.
"""

from app.config import get_model_endpoints
from app.services.model_registry import ModelRegistry

_registry: ModelRegistry | None = None


def get_shared_registry() -> ModelRegistry:
    """Return the application-wide ModelRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        for ep in get_model_endpoints():
            _registry.register_model(ep.name, ep.endpoint)
    return _registry

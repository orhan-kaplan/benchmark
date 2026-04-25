"""Unit tests for ModelRegistry service."""

import pytest

from app.models.responses import ModelInfo
from app.services.model_registry import ModelNotFoundError, ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get_endpoint(self) -> None:
        registry = ModelRegistry()
        registry.register_model("llama-7b", "http://vllm-a:8000")
        assert registry.get_endpoint("llama-7b") == "http://vllm-a:8000"

    def test_get_endpoint_unregistered_raises(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get_endpoint("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert exc_info.value.model_name == "nonexistent"
        assert exc_info.value.available_models == []

    def test_get_endpoint_unregistered_lists_available(self) -> None:
        registry = ModelRegistry()
        registry.register_model("model-a", "http://a:8000")
        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get_endpoint("model-b")
        assert "model-a" in exc_info.value.available_models

    def test_list_models_empty(self) -> None:
        registry = ModelRegistry()
        assert registry.list_models() == []

    def test_list_models_returns_model_info(self) -> None:
        registry = ModelRegistry()
        registry.register_model("model-a", "http://a:8000")
        registry.register_model("model-b", "http://b:8000")
        models = registry.list_models()
        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)
        ids = {m.id for m in models}
        assert ids == {"model-a", "model-b"}

    def test_register_overwrites_endpoint(self) -> None:
        registry = ModelRegistry()
        registry.register_model("model-a", "http://old:8000")
        registry.register_model("model-a", "http://new:8000")
        assert registry.get_endpoint("model-a") == "http://new:8000"
        assert len(registry.list_models()) == 1

    def test_model_info_defaults(self) -> None:
        registry = ModelRegistry()
        registry.register_model("test-model", "http://test:8000")
        info = registry.list_models()[0]
        assert info.id == "test-model"
        assert info.object == "model"
        assert info.owned_by == "local"

    def test_model_not_found_error_is_key_error(self) -> None:
        """ModelNotFoundError is a subclass of KeyError for compatibility."""
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get_endpoint("missing")

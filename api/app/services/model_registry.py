"""Model registry service for managing vLLM engine endpoints."""

from typing import Dict, List

from app.models.responses import ModelInfo


class ModelNotFoundError(KeyError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_name: str, available_models: List[str]) -> None:
        self.model_name = model_name
        self.available_models = available_models
        super().__init__(
            f"Model '{model_name}' not found. "
            f"Available models: {available_models}"
        )


class ModelRegistry:
    """Registry that maps model names to their vLLM engine endpoints."""

    def __init__(self) -> None:
        self._models: Dict[str, str] = {}

    def register_model(self, model_name: str, endpoint: str) -> None:
        """Register a model name with its vLLM engine endpoint.

        Args:
            model_name: Unique identifier for the model.
            endpoint: URL of the vLLM engine serving this model.
        """
        self._models[model_name] = endpoint

    def get_endpoint(self, model_name: str) -> str:
        """Return the endpoint URL for a registered model.

        Args:
            model_name: The model to look up.

        Returns:
            The endpoint URL string.

        Raises:
            ModelNotFoundError: If the model is not registered.
        """
        if model_name not in self._models:
            raise ModelNotFoundError(model_name, list(self._models.keys()))
        return self._models[model_name]

    def list_models(self) -> List[ModelInfo]:
        """Return a list of ModelInfo objects for all registered models."""
        return [ModelInfo(id=name) for name in self._models]

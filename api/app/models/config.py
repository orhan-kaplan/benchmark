"""Configuration models for the vLLM inference API."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    burst_size: int = 10


class AppConfig(BaseModel):
    """Application-wide configuration."""

    auth_enabled: bool = True
    api_keys: List[str] = []
    rate_limit: RateLimitConfig = RateLimitConfig()
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "json"
    test_mode: bool = False


class ModelEndpointConfig(BaseModel):
    """Configuration for a single vLLM model endpoint."""

    name: str
    endpoint: str
    gpu_count: int = 1
    max_model_len: int = 4096
    quantization: Optional[str] = None

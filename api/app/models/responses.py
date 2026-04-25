"""Pydantic response models for the vLLM inference API."""

from typing import List, Literal, Optional

from pydantic import BaseModel

from app.models.requests import ChatMessage


class CompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    text: str
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Text completion response model."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Response for listing available models."""

    object: str = "list"
    data: List[ModelInfo]


class EngineHealthStatus(BaseModel):
    """Health status of a single vLLM engine."""

    model_name: str
    status: Literal["healthy", "unhealthy"]
    endpoint: str


class HealthResponse(BaseModel):
    """Overall system health response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    engines: List[EngineHealthStatus]
    timestamp: str


class SystemMetrics(BaseModel):
    """System metrics including GPU and request info."""

    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    active_requests: int
    uptime_seconds: float


class ErrorDetail(BaseModel):
    """Error detail with type and message."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail

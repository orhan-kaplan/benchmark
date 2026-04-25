"""Pydantic request models for the vLLM inference API."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    """Text completion request model."""

    model: str
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Chat message model with role and content."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""

    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

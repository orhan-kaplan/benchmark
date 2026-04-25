"""Completions router — /v1/completions endpoint.

Handles text completion requests by validating input, looking up the
target vLLM engine via ModelRegistry, transforming the request, proxying
to vLLM, and converting the response back.

Supports both synchronous and streaming (SSE) responses, plus a test
mode that returns mock data without hitting a real vLLM backend.
"""

import logging
import time
import uuid
from typing import AsyncIterator

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_config
from app.models.requests import CompletionRequest
from app.models.responses import (
    CompletionChoice,
    CompletionResponse,
    ErrorDetail,
    ErrorResponse,
    UsageInfo,
)
from app.routers._registry import get_shared_registry
from app.services.model_registry import ModelNotFoundError
from app.services.transformer import RequestTransformer, TransformationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

_transformer = RequestTransformer()


def _build_error(status: int, error_type: str, message: str) -> JSONResponse:
    """Build a consistent JSON error response."""
    body = ErrorResponse(error=ErrorDetail(type=error_type, message=message))
    return JSONResponse(status_code=status, content=body.model_dump())


def _mock_completion_response(request: CompletionRequest) -> CompletionResponse:
    """Return a deterministic mock response for test mode."""
    return CompletionResponse(
        id=f"cmpl-test-{uuid.uuid4().hex[:8]}",
        object="text_completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                text="This is a test completion response.",
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


async def _stream_vllm_response(
    vllm_endpoint: str, vllm_payload: dict
) -> AsyncIterator[str]:
    """Stream SSE chunks from the vLLM backend to the client."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{vllm_endpoint}/v1/completions",
            json=vllm_payload,
            timeout=120.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"


@router.post("/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion.

    Validates the request, resolves the model endpoint, transforms the
    payload, proxies to vLLM, and returns the (optionally streamed) result.
    """
    config = get_config()

    # --- Test mode: return mock without calling vLLM ---
    if config.test_mode:
        mock = _mock_completion_response(request)
        return JSONResponse(status_code=200, content=mock.model_dump())

    # --- Resolve model endpoint ---
    registry = get_shared_registry()
    try:
        vllm_endpoint = registry.get_endpoint(request.model)
    except ModelNotFoundError as exc:
        return _build_error(
            404,
            "model_not_found",
            f"Model '{request.model}' not found. "
            f"Available models: {exc.available_models}",
        )

    # --- Transform request ---
    try:
        vllm_payload = _transformer.to_vllm_format(request)
    except TransformationError as exc:
        return _build_error(500, "transformation_error", exc.message)

    # --- Streaming path ---
    if request.stream:
        try:
            return StreamingResponse(
                _stream_vllm_response(vllm_endpoint, vllm_payload),
                media_type="text/event-stream",
            )
        except httpx.ConnectError:
            return _build_error(
                503,
                "engine_unavailable",
                f"Cannot connect to vLLM engine at {vllm_endpoint}",
            )

    # --- Synchronous path ---
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{vllm_endpoint}/v1/completions",
                json=vllm_payload,
                timeout=120.0,
            )

        if resp.status_code != 200:
            return _build_error(
                502,
                "engine_error",
                f"vLLM engine returned status {resp.status_code}: {resp.text}",
            )

        vllm_data = resp.json()
    except httpx.ConnectError:
        return _build_error(
            503,
            "engine_unavailable",
            f"Cannot connect to vLLM engine at {vllm_endpoint}",
        )
    except httpx.TimeoutException:
        return _build_error(
            503,
            "engine_timeout",
            f"Request to vLLM engine at {vllm_endpoint} timed out",
        )

    # --- Transform response ---
    try:
        completion = _transformer.from_vllm_format(vllm_data, request_type="completion")
    except TransformationError as exc:
        return _build_error(500, "transformation_error", exc.message)

    return JSONResponse(status_code=200, content=completion.model_dump())

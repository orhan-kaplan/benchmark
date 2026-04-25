"""Request/response transformer service for converting between custom and vLLM formats."""

from typing import Union

from app.models.requests import ChatCompletionRequest, ChatMessage, CompletionRequest
from app.models.responses import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionResponse,
    UsageInfo,
)


class TransformationError(Exception):
    """Raised when request/response conversion fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class RequestTransformer:
    """Converts between custom request/response formats and vLLM (OpenAI-compatible) formats."""

    def to_vllm_format(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> dict:
        """Convert a custom request to vLLM-compatible dict format.

        Args:
            request: A CompletionRequest or ChatCompletionRequest instance.

        Returns:
            A dict representing the vLLM API request body.

        Raises:
            TransformationError: If the request type is unsupported or conversion fails.
        """
        try:
            if isinstance(request, CompletionRequest):
                return self._completion_to_vllm(request)
            elif isinstance(request, ChatCompletionRequest):
                return self._chat_completion_to_vllm(request)
            else:
                raise TransformationError(
                    f"Unsupported request type: {type(request).__name__}"
                )
        except TransformationError:
            raise
        except Exception as e:
            raise TransformationError(f"Failed to convert request to vLLM format: {e}")

    def from_vllm_format(
        self, response: dict, request_type: str = "completion"
    ) -> Union[CompletionResponse, ChatCompletionResponse]:
        """Convert a vLLM response dict to a custom response model.

        Args:
            response: A dict representing the vLLM API response.
            request_type: Either "completion" or "chat_completion".

        Returns:
            A CompletionResponse or ChatCompletionResponse instance.

        Raises:
            TransformationError: If the response format is invalid or conversion fails.
        """
        try:
            if request_type == "completion":
                return self._vllm_to_completion(response)
            elif request_type == "chat_completion":
                return self._vllm_to_chat_completion(response)
            else:
                raise TransformationError(
                    f"Unsupported request_type: {request_type}"
                )
        except TransformationError:
            raise
        except Exception as e:
            raise TransformationError(
                f"Failed to convert vLLM response to custom format: {e}"
            )

    def _completion_to_vllm(self, request: CompletionRequest) -> dict:
        """Convert CompletionRequest to vLLM dict."""
        result = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        if request.stop is not None:
            result["stop"] = request.stop
        return result

    def _chat_completion_to_vllm(self, request: ChatCompletionRequest) -> dict:
        """Convert ChatCompletionRequest to vLLM dict."""
        result = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        if request.stop is not None:
            result["stop"] = request.stop
        return result

    def _vllm_to_completion(self, response: dict) -> CompletionResponse:
        """Convert vLLM completion response dict to CompletionResponse."""
        if not isinstance(response, dict):
            raise TransformationError("Response must be a dict")

        required_fields = ["id", "model", "choices", "usage"]
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise TransformationError(f"Missing required fields: {missing}")

        choices = [
            CompletionChoice(
                index=c.get("index", i),
                text=c["text"],
                finish_reason=c.get("finish_reason"),
            )
            for i, c in enumerate(response["choices"])
        ]

        usage = UsageInfo(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        return CompletionResponse(
            id=response["id"],
            object=response.get("object", "text_completion"),
            created=response["created"],
            model=response["model"],
            choices=choices,
            usage=usage,
        )

    def _vllm_to_chat_completion(self, response: dict) -> ChatCompletionResponse:
        """Convert vLLM chat completion response dict to ChatCompletionResponse."""
        if not isinstance(response, dict):
            raise TransformationError("Response must be a dict")

        required_fields = ["id", "model", "choices", "usage"]
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise TransformationError(f"Missing required fields: {missing}")

        choices = [
            ChatCompletionChoice(
                index=c.get("index", i),
                message=ChatMessage(
                    role=c["message"]["role"],
                    content=c["message"]["content"],
                ),
                finish_reason=c.get("finish_reason"),
            )
            for i, c in enumerate(response["choices"])
        ]

        usage = UsageInfo(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        return ChatCompletionResponse(
            id=response["id"],
            object=response.get("object", "chat.completion"),
            created=response["created"],
            model=response["model"],
            choices=choices,
            usage=usage,
        )

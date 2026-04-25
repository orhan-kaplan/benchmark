"""OpenAI uyumlu API endpoint'lerine HTTP istekleri gönderen istemci."""

import asyncio
import logging
import time
from typing import Optional

import httpx

from benchmark.models import APIResponse, GenerationParams

logger = logging.getLogger(__name__)


class APIClient:
    """OpenAI uyumlu endpoint'lere istek gönderen async HTTP istemcisi.

    Non-streaming mod ile güvenilir usage bilgisi alır.
    Exponential backoff retry mantığı ve yapılandırılabilir timeout içerir.
    """

    def __init__(self, timeout: float = 120.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def chat_completion(
        self,
        endpoint: str,
        model: str,
        messages: list[dict],
        params: GenerationParams,
    ) -> APIResponse:
        """POST /v1/chat/completions — non-streaming.

        Exponential backoff ile retry uygular.
        Usage bilgisi (prompt_tokens, completion_tokens) her zaman döner.
        """
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
            "stream": False,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            if attempt > 0:
                backoff = 1.0 * (2 ** (attempt - 1))
                logger.info("Retry %d/%d, bekleniyor %.1fs", attempt, self.max_retries, backoff)
                await asyncio.sleep(backoff)

            start_ms = time.monotonic() * 1000
            try:
                resp = await self._client.post(url, json=payload)
                elapsed_ms = time.monotonic() * 1000 - start_ms

                if resp.status_code != 200:
                    error_text = resp.text[:500] if resp.text else str(resp.status_code)
                    return APIResponse(
                        status_code=resp.status_code,
                        response_text=None,
                        usage=None,
                        error=f"HTTP {resp.status_code}: {error_text}",
                        elapsed_ms=elapsed_ms,
                    )

                data = resp.json()

                # Yanıt metnini çıkar
                response_text = None
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    response_text = msg.get("content")

                # Usage bilgisi
                usage = data.get("usage")

                return APIResponse(
                    status_code=resp.status_code,
                    response_text=response_text,
                    usage=usage,
                    error=None,
                    elapsed_ms=elapsed_ms,
                )

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                logger.warning(
                    "API isteği başarısız (deneme %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

        # Tüm denemeler tükendi
        elapsed_ms = time.monotonic() * 1000 - start_ms
        error_msg = f"Tüm denemeler başarısız ({self.max_retries}): {last_error}"
        logger.error(error_msg)
        return APIResponse(
            status_code=0,
            response_text=None,
            usage=None,
            error=error_msg,
            elapsed_ms=elapsed_ms,
        )

    async def health_check(self, endpoint: str) -> bool:
        """Endpoint erişilebilirlik kontrolü. 200 dönerse True."""
        try:
            resp = await self._client.get(endpoint.rstrip("/"))
            return resp.status_code == 200
        except (httpx.RequestError, httpx.StreamError):
            return False

    async def close(self) -> None:
        """httpx client'ı kapat."""
        await self._client.aclose()

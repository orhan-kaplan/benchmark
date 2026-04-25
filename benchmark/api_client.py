"""OpenAI uyumlu API endpoint'lerine HTTP istekleri gönderen istemci."""

import asyncio
import json
import logging
import time
from typing import Optional

import httpx

from benchmark.models import APIResponse, GenerationParams

logger = logging.getLogger(__name__)


class APIClient:
    """OpenAI uyumlu endpoint'lere istek gönderen async HTTP istemcisi.

    Streaming desteği ile TTFT ölçümü, exponential backoff retry mantığı
    ve yapılandırılabilir timeout içerir.
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
        """POST /v1/chat/completions ile streaming yanıt al.

        Exponential backoff ile retry uygular. İlk chunk'ta TTFT kaydeder,
        tüm chunk'ları birleştirerek tam yanıt oluşturur.
        """
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            if attempt > 0:
                backoff = 1.0 * (2 ** (attempt - 1))
                logger.info("Retry %d/%d, bekleniyor %.1fs", attempt, self.max_retries, backoff)
                await asyncio.sleep(backoff)

            start_ms = time.monotonic() * 1000
            try:
                return await self._do_streaming_request(url, payload, start_ms)
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.StreamError) as exc:
                last_error = exc
                elapsed_ms = time.monotonic() * 1000 - start_ms
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

    async def _do_streaming_request(
        self, url: str, payload: dict, start_ms: float
    ) -> APIResponse:
        """Tek bir streaming isteği gönder ve yanıtı parse et."""
        ttft_ms: Optional[float] = None
        chunks: list[str] = []
        usage: Optional[dict] = None

        async with self._client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                await response.aread()
                elapsed_ms = time.monotonic() * 1000 - start_ms
                error_text = response.text if hasattr(response, "text") else str(response.status_code)
                return APIResponse(
                    status_code=response.status_code,
                    response_text=None,
                    usage=None,
                    error=f"HTTP {response.status_code}: {error_text}",
                    elapsed_ms=elapsed_ms,
                )

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # TTFT: ilk içerik chunk'ında kaydet
                if ttft_ms is None:
                    ttft_ms = time.monotonic() * 1000 - start_ms

                # İçerik çıkar
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        chunks.append(content)

                # Usage bilgisi (bazı API'ler son chunk'ta gönderir)
                if "usage" in data and data["usage"]:
                    usage = data["usage"]

        elapsed_ms = time.monotonic() * 1000 - start_ms
        response_text = "".join(chunks) if chunks else None

        return APIResponse(
            status_code=response.status_code,
            response_text=response_text,
            usage=usage,
            error=None,
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
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

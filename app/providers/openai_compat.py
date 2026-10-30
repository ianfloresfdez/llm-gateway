"""Any server that speaks the OpenAI Chat Completions API: Ollama (/v1), OpenAI, vLLM..."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import Provider, ProviderError, classify_status
from app.schemas import ChatRequest, Completion

# Fields we forward as-is; anything else in the request is dropped.
_PASSTHROUGH = ("temperature", "top_p", "stop", "user", "reasoning_effort")


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.name = name
        # Provider-level defaults, e.g. {"reasoning_effort": "none"} so Ollama's thinking
        # models answer directly instead of spending max_tokens on hidden reasoning.
        self.extra_body = extra_body or {}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    def _body(self, req: ChatRequest, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            **self.extra_body,
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        }
        for field in _PASSTHROUGH:
            value = getattr(req, field, None)
            if value is not None:
                body[field] = value
        if req.output_limit:
            body["max_tokens"] = req.output_limit
        return body

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(path, json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: timeout", status=504, retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}", retryable=True) from e
        if r.status_code >= 400:
            raise classify_status(r.status_code, f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
        try:
            return r.json()
        except ValueError as e:  # e.g. an HTML error page from a proxy with status 200
            raise ProviderError(f"{self.name}: invalid JSON response", retryable=True) from e

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        data = await self._post("/chat/completions", self._body(req, model))
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.name}: response without choices", retryable=True) from e
        usage = data.get("usage") or {}
        return Completion(
            content=choice["message"].get("content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason") or "stop",
            provider=self.name,
            upstream_model=model,
        )

    async def stream(self, req: ChatRequest, model: str) -> AsyncIterator[str | Completion]:
        body = self._body(req, model) | {"stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        usage: dict[str, Any] = {}
        finish = "stop"
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    text = (await r.aread()).decode(errors="replace")
                    raise classify_status(r.status_code, f"{self.name}: HTTP {r.status_code} {text[:200]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            parts.append(delta)
                            yield delta
                        finish = choice.get("finish_reason") or finish
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: timeout", status=504, retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}", retryable=True) from e
        yield Completion(
            content="".join(parts),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=finish,
            provider=self.name,
            upstream_model=model,
        )

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = await self._post("/embeddings", {"model": model, "input": texts})
        return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]

    async def aclose(self) -> None:
        await self._client.aclose()

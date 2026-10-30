"""Anthropic Messages API, translated to and from the OpenAI chat shape."""

from __future__ import annotations

from typing import Any

import anthropic

from app.providers.base import Provider, ProviderError, classify_status
from app.schemas import ChatRequest, Completion

_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter"}
DEFAULT_MAX_TOKENS = 1024


def to_messages_params(req: ChatRequest, model: str) -> dict[str, Any]:
    system = "\n\n".join(m.text() for m in req.messages if m.role in ("system", "developer"))
    messages = [
        {"role": m.role, "content": m.text()} for m in req.messages if m.role in ("user", "assistant")
    ]
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": req.output_limit or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if req.stop_list():
        params["stop_sequences"] = req.stop_list()
    # temperature/top_p are not forwarded: current Claude models reject sampling parameters.
    return params


class AnthropicProvider(Provider):
    def __init__(self, name: str, api_key: str, timeout_s: float = 60.0, client: Any = None):
        self.name = name
        # The gateway owns retries and fallback, so the SDK must not retry on its own.
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout_s)

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        if req.uses_tools:
            # Not translated on this path yet: let the router try the next target.
            raise ProviderError(
                f"{self.name}: tool calling is not supported by this provider",
                status=400,
                retryable=False,
                fallback=True,
            )
        try:
            msg = await self._client.messages.create(**to_messages_params(req, model))
        except anthropic.APITimeoutError as e:
            raise ProviderError(f"{self.name}: timeout", status=504, retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"{self.name}: connection error", retryable=True) from e
        except anthropic.APIStatusError as e:
            raise classify_status(e.status_code, f"{self.name}: HTTP {e.status_code}") from e
        text = "".join(block.text for block in msg.content if block.type == "text")
        return Completion(
            content=text,
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            finish_reason=_FINISH.get(msg.stop_reason or "", "stop"),
            provider=self.name,
            upstream_model=model,
        )

    async def aclose(self) -> None:
        await self._client.close()

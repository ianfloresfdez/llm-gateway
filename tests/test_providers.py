from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from app.config import load_config
from app.pricing import actual_cost, avoided_cost
from app.providers.anthropic_provider import AnthropicProvider, to_messages_params
from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas import ChatRequest
from tests.conftest import make_config


def req(**kw) -> ChatRequest:
    base = {
        "model": "x",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    }
    return ChatRequest.model_validate(base | kw)


def openai_provider(handler) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider("ollama", "http://ollama/v1", transport=httpx.MockTransport(handler))


async def test_openai_compatible_complete_parses_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    c = await openai_provider(handler).complete(req(max_completion_tokens=50, temperature=0), "gemma4:12b")
    assert (c.content, c.prompt_tokens, c.completion_tokens) == ("Hello!", 12, 3)
    assert seen["model"] == "gemma4:12b" and seen["max_tokens"] == 50 and seen["temperature"] == 0


async def test_openai_compatible_stream():
    sse = (
        "".join(
            f"data: {json.dumps(chunk)}\n\n"
            for chunk in [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            ]
        )
        + "data: [DONE]\n\n"
    )
    p = openai_provider(lambda r: httpx.Response(200, text=sse))
    items = [i async for i in p.stream(req(), "m")]
    assert items[:2] == ["Hel", "lo"]
    assert items[-1].content == "Hello" and items[-1].completion_tokens == 2


@pytest.mark.parametrize(
    ("status", "retryable", "fallback"),
    [(500, True, True), (503, True, True), (429, False, True), (404, False, True), (400, False, False)],
)
async def test_openai_compatible_error_classification(status, retryable, fallback):
    p = openai_provider(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(ProviderError) as e:
        await p.complete(req(), "m")
    assert (e.value.retryable, e.value.fallback) == (retryable, fallback)


async def test_openai_compatible_connection_error_is_retryable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError) as e:
        await openai_provider(handler).complete(req(), "m")
    assert e.value.retryable


def test_anthropic_translation_moves_system_and_drops_sampling():
    params = to_messages_params(req(temperature=0.7, stop="END"), "claude-haiku-5-5")
    assert params["system"] == "Be brief."
    assert params["messages"] == [{"role": "user", "content": "Hi"}]
    assert params["stop_sequences"] == ["END"]
    assert params["max_tokens"] == 1024
    assert "temperature" not in params


async def test_anthropic_provider_maps_response():
    message = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="Hola")],
        usage=SimpleNamespace(input_tokens=9, output_tokens=4),
        stop_reason="max_tokens",
    )

    class FakeMessages:
        async def create(self, **kwargs):
            return message

    p = AnthropicProvider("anthropic", "k", client=SimpleNamespace(messages=FakeMessages()))
    c = await p.complete(req(), "claude-haiku-5-5")
    assert (c.content, c.prompt_tokens, c.completion_tokens, c.finish_reason) == ("Hola", 9, 4, "length")


def test_pricing():
    cfg = make_config()
    assert actual_cost(cfg, "mock-2", 1_000_000, 1_000_000) == 3.0
    assert actual_cost(cfg, "mock-1", 1000, 1000) == 0.0
    assert avoided_cost(cfg, "mock-1", 1_000_000, 0) == 10.0  # valued at the reference model
    assert avoided_cost(cfg, "unknown", 1000, 1000) == 0.0


def test_shipped_config_loads(monkeypatch):
    monkeypatch.setenv("CACHE_MODE", "exact")
    cfg = load_config("config/gateway.yaml")
    assert cfg.cache.mode == "exact"
    assert cfg.models["local"].targets[0].model == "gemma4:12b"
    for route in cfg.models.values():
        for t in route.targets:
            assert t.provider in cfg.providers

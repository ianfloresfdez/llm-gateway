from __future__ import annotations

import asyncio
import json

from fakeredis import FakeAsyncRedis, FakeServer

from app.providers.base import Provider, ProviderError
from app.router import Router
from app.schemas import ChatRequest
from tests.conftest import chat, make_config, make_providers, metric


def test_openai_response_shape(make_client):
    r = chat(make_client())
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert (
        body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert r.headers["x-gateway-provider"] == "mock"
    assert r.headers["x-gateway-cache"] == "miss"


def test_exact_cache_hit_returns_same_answer_and_counts_savings(make_client):
    client = make_client()
    saved_before = metric("gateway_cache_saved_usd_total", model="mock")
    first = chat(client, "Explain pods")
    second = chat(client, "Explain pods")
    assert second.headers["x-gateway-cache"] == "hit_exact"
    assert second.json()["choices"][0]["message"] == first.json()["choices"][0]["message"]
    # mock-1 is free but valued at the "ref" price (10 / 20 USD per 1M tokens).
    usage = first.json()["usage"]
    expected = (usage["prompt_tokens"] * 10 + usage["completion_tokens"] * 20) / 1e6
    assert abs(metric("gateway_cache_saved_usd_total", model="mock") - saved_before - expected) < 1e-12


def test_semantic_cache_matches_rephrased_prompt(make_client):
    client = make_client()
    chat(client, "What is a Kubernetes pod?")
    assert chat(client, "what is a kubernetes pod").headers["x-gateway-cache"] == "hit_semantic"
    assert chat(client, "How do I bake bread at home?").headers["x-gateway-cache"] == "miss"


def test_cache_is_scoped_by_model_and_params(make_client):
    client = make_client()
    chat(client, "Explain services")
    assert chat(client, "Explain services", temperature=0.9).headers["x-gateway-cache"] == "miss"
    assert chat(client, "Explain services", model="fallback").headers["x-gateway-cache"] == "miss"


def test_cache_bypass_header(make_client):
    client = make_client()
    chat(client, "Explain ingress")
    r = chat(client, "Explain ingress", headers={"Cache-Control": "no-cache"})
    assert r.headers["x-gateway-cache"] == "bypass"


def test_cache_off(make_client):
    cfg = make_config(cache={"mode": "off"})
    client = make_client(cfg)
    chat(client, "hello")
    assert chat(client, "hello").headers["x-gateway-cache"] == "bypass"


def test_fallback_after_retries(make_client):
    client = make_client()
    before = metric("gateway_fallbacks_total", model="fallback", provider="mock")
    r = chat(client, "hi", model="fallback")
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "mock"
    assert r.headers["x-gateway-attempts"] == "3"  # 2 attempts on "broken" + 1 on "mock"
    assert metric("gateway_fallbacks_total", model="fallback", provider="mock") == before + 1
    # Real spend is recorded for the paid model that served it.
    assert float(r.headers["x-gateway-cost-usd"]) > 0


class BadRequestProvider(Provider):
    def __init__(self):
        self.name = "bad"
        self.calls = 0

    async def complete(self, req, model):
        self.calls += 1
        raise ProviderError("bad: HTTP 400", status=400, retryable=False, fallback=False)


def test_client_errors_do_not_fall_back(make_client):
    cfg = make_config(
        providers={"bad": {"type": "mock"}, "mock": {"type": "mock", "latency_ms": 0}},
        models={"m": {"targets": [{"provider": "bad", "model": "x"}, {"provider": "mock", "model": "y"}]}},
    )
    providers = make_providers(cfg) | {"bad": BadRequestProvider()}
    r = chat(make_client(cfg, providers), model="m")
    assert r.status_code == 400
    assert providers["bad"].calls == 1


def test_all_targets_failing_returns_502(make_client):
    cfg = make_config(models={"m": {"targets": [{"provider": "broken", "model": "x"}]}})
    r = chat(make_client(cfg), model="m")
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"


def test_circuit_breaker_skips_open_provider(make_client):
    cfg = make_config(circuit_breaker={"failure_threshold": 2, "cooldown_s": 60})
    client = make_client(cfg)
    chat(client, "a", model="fallback")  # two failures on "broken" -> circuit opens
    assert metric("gateway_circuit_open", provider="broken") == 1
    r = chat(client, "b", model="fallback")
    assert r.headers["x-gateway-attempts"] == "1"  # went straight to "mock"


def test_unknown_model_and_direct_addressing(make_client):
    client = make_client()
    assert chat(client, model="nope").status_code == 404
    r = chat(client, model="mock/custom-model")
    assert r.status_code == 200
    assert "[custom-model]" in r.json()["choices"][0]["message"]["content"]


def test_rate_limit_rpm(make_client):
    cfg = make_config(limits={"default": {"rpm": 2, "tokens_per_day": 10_000}})
    client = make_client(cfg)
    assert chat(client, "1").status_code == 200
    assert chat(client, "2").status_code == 200
    r = chat(client, "3")
    assert r.status_code == 429
    assert 0 < int(r.headers["Retry-After"]) <= 60


def test_rate_limit_daily_tokens(make_client):
    cfg = make_config(limits={"default": {"rpm": 100, "tokens_per_day": 5}})
    client = make_client(cfg)
    assert chat(client, "a prompt long enough to use more than five tokens").status_code == 200
    r = chat(client, "next")
    assert r.status_code == 429
    assert "tokens_per_day" in r.json()["error"]["message"]


def test_auth_and_per_key_limits(make_client):
    cfg = make_config(limits={"default": {"rpm": 100, "tokens_per_day": 10**6}, "demo": {"rpm": 1}})
    client = make_client(cfg, api_keys="demo:sk-demo,team:sk-team")
    assert chat(client).status_code == 401
    assert chat(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
    demo = {"Authorization": "Bearer sk-demo"}
    assert chat(client, "x", headers=demo).status_code == 200
    assert chat(client, "y", headers=demo).status_code == 429
    assert chat(client, "z", headers={"Authorization": "Bearer sk-team"}).status_code == 200


def _sse(r) -> list:
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_streaming_and_cached_streaming(make_client):
    client = make_client()
    r = chat(client, "stream me", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = _sse(r)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "stream me" in text
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    again = chat(client, "stream me", stream=True)
    assert again.headers["x-gateway-cache"] == "hit_exact"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in _sse(again)) == text


def test_streaming_falls_back_before_first_token(make_client):
    r = chat(make_client(), "hi", model="fallback", stream=True)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in _sse(r))
    assert "[mock-2]" in text


def test_metrics_endpoint_and_probes(make_client):
    client = make_client()
    chat(client)
    assert "gateway_requests_total" in client.get("/metrics").text
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["redis"] is True
    assert {m["id"] for m in client.get("/v1/models").json()["data"]} == {"mock", "fallback"}


class EmptyProvider(Provider):
    def __init__(self):
        self.name = "empty"

    async def complete(self, req, model):
        from app.schemas import Completion

        return Completion("", 5, 60, "length", "empty", model)


def test_empty_answers_are_not_cached(make_client):
    cfg = make_config(models={"m": {"targets": [{"provider": "empty", "model": "e"}]}})
    client = make_client(cfg, make_providers(cfg) | {"empty": EmptyProvider()})
    chat(client, "x", model="m")
    assert chat(client, "x", model="m").headers["x-gateway-cache"] == "miss"


def test_redis_outage_fails_open(make_client):
    server = FakeServer()
    server.connected = False  # every Redis command raises ConnectionError
    client = make_client(redis=FakeAsyncRedis(server=server))
    for _ in range(2):
        r = chat(client, "Explain pods")
        assert r.status_code == 200
        assert r.headers["x-gateway-cache"] == "error"
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["redis"] is False


def test_redis_outage_is_skipped_until_cooldown(make_client, caplog):
    # After the first failure the gateway stops calling Redis (each call would wait the timeout)
    server = FakeServer()
    server.connected = False
    client = make_client(redis=FakeAsyncRedis(server=server))
    gw = client.app.state.gw
    for _ in range(4):
        assert chat(client, "Explain pods").status_code == 200
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert [r.getMessage() for r in warnings] == ["redis unavailable (ConnectionError): skipping it for 30 s"]
    assert not gw.limiter.guard.usable and gw.cache.guard is gw.limiter.guard

    # Back after the cooldown
    gw.limiter.guard.down_until = 0
    server.connected = True
    chat(client, "Explain pods")
    assert chat(client, "Explain pods").headers["x-gateway-cache"] == "hit_exact"


def test_reasoning_effort_splits_the_cache(make_client):
    client = make_client()
    chat(client, "Explain jobs")
    assert chat(client, "Explain jobs", reasoning_effort="high").headers["x-gateway-cache"] == "miss"
    assert chat(client, "Explain jobs", reasoning_effort="high").headers["x-gateway-cache"] == "hit_exact"


async def test_no_backoff_after_the_last_attempt_on_a_target(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        if s:  # the mock providers also sleep(0) for their latency
            sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    cfg = make_config(retries={"max_attempts": 3, "backoff_base_s": 1.0})
    router = Router(cfg, make_providers(cfg))
    req = ChatRequest.model_validate({"model": "fallback", "messages": [{"role": "user", "content": "x"}]})
    _, attempts = await router.complete(req)
    assert attempts == 4  # 3 on "broken" + 1 on "mock"
    assert sleeps == [1.0, 2.0]  # between attempts only, not before falling back

from __future__ import annotations

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.config import GatewayConfig
from app.main import build_gateway, create_app
from app.providers.mock import MockProvider


def make_config(**overrides) -> GatewayConfig:
    raw = {
        "providers": {
            "mock": {"type": "mock", "latency_ms": 0},
            "broken": {"type": "mock", "latency_ms": 0, "failure_rate": 1.0},
        },
        "models": {
            "mock": {"targets": [{"provider": "mock", "model": "mock-1"}]},
            "fallback": {
                "targets": [{"provider": "broken", "model": "b-1"}, {"provider": "mock", "model": "mock-2"}]
            },
        },
        "pricing": {
            "mock-1": {"input": 0, "output": 0, "reference": "ref"},
            "mock-2": {"input": 1.0, "output": 2.0},
            "ref": {"input": 10.0, "output": 20.0},
        },
        "cache": {"mode": "semantic", "semantic": {"embedding_provider": "mock", "threshold": 0.95}},
        "retries": {"max_attempts": 2, "backoff_base_s": 0},
        "circuit_breaker": {"failure_threshold": 100, "cooldown_s": 30},
        "limits": {"default": {"rpm": 1000, "tokens_per_day": 10_000_000}},
    }
    for key, value in overrides.items():
        raw[key] = value
    return GatewayConfig.model_validate(raw)


def make_providers(cfg: GatewayConfig) -> dict:
    return {
        name: MockProvider(name, p.latency_ms, p.failure_rate, seed=1) for name, p in cfg.providers.items()
    }


@pytest.fixture
def make_client(monkeypatch):
    clients = []

    def _make(cfg: GatewayConfig | None = None, providers: dict | None = None, api_keys: str = ""):
        monkeypatch.setenv("GATEWAY_API_KEYS", api_keys)
        cfg = cfg or make_config()
        gateway = build_gateway(cfg, FakeAsyncRedis(), providers or make_providers(cfg))
        client = TestClient(create_app(gateway))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def chat(client: TestClient, content: str = "What is Kubernetes?", model: str = "mock", **extra):
    headers = extra.pop("headers", {})
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers)


def metric(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0

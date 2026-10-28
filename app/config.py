"""Gateway configuration: a YAML file with ${ENV_VAR:-default} interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ProviderConfig(BaseModel):
    type: Literal["openai_compatible", "anthropic", "mock"]
    base_url: str | None = None
    # Name of the env var holding the API key. If set but the variable is empty,
    # the provider is disabled and the router skips it.
    api_key_env: str | None = None
    timeout_s: float = 60.0
    # Merged into every upstream request body (openai_compatible only).
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # mock-only knobs
    latency_ms: int = 200
    failure_rate: float = 0.0


class Target(BaseModel):
    provider: str
    model: str


class ModelRoute(BaseModel):
    description: str = ""
    targets: list[Target]


class Price(BaseModel):
    """USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0
    # For local models (price 0) we still want to report what a cache hit saved:
    # the savings are computed at the price of this hosted reference model.
    reference: str | None = None


class SemanticCacheConfig(BaseModel):
    embedding_provider: str = "embeddings"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    threshold: float = 0.96
    max_entries: int = 1000


class CacheConfig(BaseModel):
    mode: Literal["off", "exact", "semantic"] = "exact"
    ttl_s: int = 3600
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)


class RetryConfig(BaseModel):
    max_attempts: int = 2  # per target, before falling back to the next one
    backoff_base_s: float = 0.2


class BreakerConfig(BaseModel):
    failure_threshold: int = 5
    cooldown_s: float = 30.0


class KeyLimits(BaseModel):
    rpm: int = 60
    tokens_per_day: int = 1_000_000


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelRoute]
    pricing: dict[str, Price] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    # Limits per key name; "default" applies to keys without an explicit entry.
    limits: dict[str, KeyLimits] = Field(default_factory=lambda: {"default": KeyLimits()})

    def limits_for(self, key_name: str) -> KeyLimits:
        return self.limits.get(key_name) or self.limits.get("default") or KeyLimits()


def _interpolate(text: str) -> str:
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), text)


def load_config(path: str | Path | None = None) -> GatewayConfig:
    path = Path(path or os.environ.get("GATEWAY_CONFIG", "config/gateway.yaml"))
    raw = yaml.safe_load(_interpolate(path.read_text(encoding="utf-8")))
    return GatewayConfig.model_validate(raw)


def load_api_keys() -> dict[str, str]:
    """GATEWAY_API_KEYS="team-a:sk-123,team-b:sk-456" -> {api_key: key_name}.

    An empty value disables auth (local development): every caller is "anonymous".
    """
    keys: dict[str, str] = {}
    for item in os.environ.get("GATEWAY_API_KEYS", "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, key = item.partition(":")
        if not key:
            raise ValueError(f"GATEWAY_API_KEYS entry {name!r} must be name:key")
        keys[key] = name
    return keys

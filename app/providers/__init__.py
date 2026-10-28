from __future__ import annotations

import logging
import os

from app.config import GatewayConfig
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import Provider, ProviderError
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatibleProvider

__all__ = ["Provider", "ProviderError", "build_providers"]

log = logging.getLogger(__name__)


def build_providers(cfg: GatewayConfig) -> dict[str, Provider]:
    """Instantiate every configured provider; the ones missing their API key are disabled."""
    providers: dict[str, Provider] = {}
    for name, p in cfg.providers.items():
        api_key = os.environ.get(p.api_key_env, "") if p.api_key_env else None
        if p.api_key_env and not api_key:
            log.info("provider %s disabled: %s is not set", name, p.api_key_env)
            continue
        if p.type == "openai_compatible":
            if not p.base_url:
                raise ValueError(f"provider {name} needs base_url")
            providers[name] = OpenAICompatibleProvider(name, p.base_url, api_key, p.timeout_s, p.extra_body)
        elif p.type == "anthropic":
            providers[name] = AnthropicProvider(name, api_key or "", p.timeout_s)
        elif p.type == "mock":
            providers[name] = MockProvider(name, p.latency_ms, p.failure_rate)
    return providers

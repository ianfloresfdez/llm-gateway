"""Model routing with per-target retries, fallback across providers and a circuit breaker."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app import metrics
from app.config import BreakerConfig, GatewayConfig, Target
from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, Completion


class UnknownModel(Exception):
    pass


class AllTargetsFailed(Exception):
    def __init__(self, errors: list[str], status: int = 502):
        super().__init__("; ".join(errors) or "no provider available")
        self.errors = errors
        self.status = status


@dataclass
class CircuitBreaker:
    """Opens after N consecutive failures; lets one trial call through after the cooldown."""

    cfg: BreakerConfig
    failures: dict[str, int] = field(default_factory=dict)
    opened_at: dict[str, float] = field(default_factory=dict)

    def allow(self, provider: str) -> bool:
        opened = self.opened_at.get(provider)
        return opened is None or time.monotonic() - opened >= self.cfg.cooldown_s

    def success(self, provider: str) -> None:
        self.failures[provider] = 0
        if self.opened_at.pop(provider, None) is not None:
            metrics.CIRCUIT_OPEN.labels(provider).set(0)

    def failure(self, provider: str) -> None:
        self.failures[provider] = self.failures.get(provider, 0) + 1
        if self.failures[provider] >= self.cfg.failure_threshold:
            self.opened_at[provider] = time.monotonic()
            metrics.CIRCUIT_OPEN.labels(provider).set(1)


class Router:
    def __init__(self, cfg: GatewayConfig, providers: dict[str, Provider]):
        self.cfg = cfg
        self.providers = providers
        self.breaker = CircuitBreaker(cfg.circuit_breaker)
        for name in providers:  # export 0 from the start so dashboards show closed breakers
            metrics.CIRCUIT_OPEN.labels(name).set(0)

    def targets(self, model: str) -> list[Target]:
        route = self.cfg.models.get(model)
        if route:
            return route.targets
        # Direct addressing: "ollama/gemma4:12b"
        provider, sep, upstream = model.partition("/")
        if sep and provider in self.cfg.providers:
            return [Target(provider=provider, model=upstream)]
        raise UnknownModel(model)

    def _usable(self, target: Target) -> Provider | None:
        provider = self.providers.get(target.provider)
        if provider is None:
            metrics.UPSTREAM_ATTEMPTS.labels(target.provider, "skipped_disabled").inc()
            return None
        if not self.breaker.allow(target.provider):
            metrics.UPSTREAM_ATTEMPTS.labels(target.provider, "skipped_breaker").inc()
            return None
        return provider

    def _record_failure(self, target: Target, e: ProviderError) -> None:
        self.breaker.failure(target.provider)
        outcome = "retryable_error" if e.retryable else "fatal_error"
        metrics.UPSTREAM_ATTEMPTS.labels(target.provider, outcome).inc()

    async def complete(self, req: ChatRequest) -> tuple[Completion, int]:
        """Returns the completion and the number of upstream attempts it took."""
        errors: list[str] = []
        attempts = 0
        for i, target in enumerate(self.targets(req.model)):
            provider = self._usable(target)
            if provider is None:
                continue
            for attempt in range(self.cfg.retries.max_attempts):
                attempts += 1
                start = time.perf_counter()
                try:
                    result = await provider.complete(req, target.model)
                except ProviderError as e:
                    self._record_failure(target, e)
                    errors.append(str(e))
                    if not e.fallback:
                        raise AllTargetsFailed(errors, status=e.status) from e
                    if not e.retryable:
                        break
                    await asyncio.sleep(self.cfg.retries.backoff_base_s * 2**attempt)
                    continue
                metrics.UPSTREAM_LATENCY.labels(target.provider, target.model).observe(
                    time.perf_counter() - start
                )
                metrics.UPSTREAM_ATTEMPTS.labels(target.provider, "success").inc()
                self.breaker.success(target.provider)
                if i > 0:
                    metrics.FALLBACKS.labels(req.model, target.provider).inc()
                return result, attempts
        raise AllTargetsFailed(errors)

    async def stream(self, req: ChatRequest) -> AsyncIterator[str | Completion]:
        """Fallback only happens before the first token; after that errors propagate."""
        errors: list[str] = []
        for i, target in enumerate(self.targets(req.model)):
            provider = self._usable(target)
            if provider is None:
                continue
            started = False
            try:
                async for item in provider.stream(req, target.model):
                    started = True
                    yield item
                metrics.UPSTREAM_ATTEMPTS.labels(target.provider, "success").inc()
                self.breaker.success(target.provider)
                if i > 0:
                    metrics.FALLBACKS.labels(req.model, target.provider).inc()
                return
            except ProviderError as e:
                self._record_failure(target, e)
                errors.append(str(e))
                if started or not e.fallback:
                    raise AllTargetsFailed(errors, status=e.status) from e
        raise AllTargetsFailed(errors)

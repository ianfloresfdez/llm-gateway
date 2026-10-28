"""FastAPI app: an OpenAI-compatible /v1/chat/completions in front of several providers."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from app import metrics
from app.cache import ResponseCache
from app.config import GatewayConfig, load_api_keys, load_config
from app.pricing import actual_cost, avoided_cost
from app.providers import Provider, build_providers
from app.ratelimit import RateLimiter
from app.router import AllTargetsFailed, Router, UnknownModel
from app.schemas import ChatRequest, Completion, new_completion_id, to_openai_chunk, to_openai_response

log = logging.getLogger("llm_gateway")


@dataclass
class Gateway:
    cfg: GatewayConfig
    providers: dict[str, Provider]
    router: Router
    cache: ResponseCache
    limiter: RateLimiter
    redis: Redis
    api_keys: dict[str, str]


def build_gateway(cfg: GatewayConfig, redis: Redis, providers: dict[str, Provider] | None = None) -> Gateway:
    providers = build_providers(cfg) if providers is None else providers
    embedder = providers.get(cfg.cache.semantic.embedding_provider)
    return Gateway(
        cfg=cfg,
        providers=providers,
        router=Router(cfg, providers),
        cache=ResponseCache(redis, cfg.cache, embedder),
        limiter=RateLimiter(redis),
        redis=redis,
        api_keys=load_api_keys(),
    )


def create_app(gateway: Gateway | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if gateway is None:
            redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
            app.state.gw = build_gateway(load_config(), redis)
        else:
            app.state.gw = gateway
        yield
        for p in app.state.gw.providers.values():
            await p.aclose()
        await app.state.gw.redis.aclose()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)

    def gw(request: Request) -> Gateway:
        return request.app.state.gw

    def authenticate(g: Gateway, request: Request) -> str:
        if not g.api_keys:
            return "anonymous"
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip()
        name = g.api_keys.get(token)
        if not name:
            raise HTTPException(401, detail={"error": {"message": "invalid API key", "type": "auth_error"}})
        return name

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request):
        # Readiness must not depend on Redis: cache and limits fail open, so a Redis
        # outage should not pull every replica out of the Service at once.
        g = gw(request)
        try:
            redis_ok = bool(await g.redis.ping())
        except Exception:
            redis_ok = False
        return {"status": "ready", "redis": redis_ok, "providers": sorted(g.providers)}

    @app.get("/metrics")
    async def prometheus_metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models(request: Request):
        g = gw(request)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "llm-gateway"} for name in g.cfg.models],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        g = gw(request)
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        key_name = authenticate(g, request)

        decision = await g.limiter.check(key_name, g.cfg.limits_for(key_name))
        if not decision.allowed:
            metrics.RATE_LIMITED.labels(key_name, decision.reason).inc()
            return JSONResponse(
                {"error": {"message": f"rate limit exceeded: {decision.reason}", "type": "rate_limit_error"}},
                status_code=429,
                headers={"Retry-After": str(decision.retry_after_s), "x-request-id": request_id},
            )

        try:
            g.router.targets(body.model)
        except UnknownModel:
            return JSONResponse(
                {"error": {"message": f"unknown model {body.model!r}", "type": "invalid_request_error"}},
                status_code=404,
            )

        bypass = "no-cache" in request.headers.get("cache-control", "")
        if bypass:
            cache_result, cached, query_vec = "bypass", None, None
        else:
            cache_result, cached, query_vec = await g.cache.get(body)
        metrics.CACHE_LOOKUPS.labels(cache_result).inc()

        ctx = RequestContext(g, body, key_name, request_id, start, cache_result)
        if body.stream:
            return StreamingResponse(
                ctx.stream(cached, query_vec, bypass),
                media_type="text/event-stream",
                headers=ctx.headers(cached),
            )

        if cached is not None:
            return ctx.finish(cached, attempts=0)
        try:
            completion, attempts = await g.router.complete(body)
        except AllTargetsFailed as e:
            return ctx.fail(e)
        if not bypass:
            await g.cache.set(body, completion, query_vec)
        await g.limiter.record_tokens(key_name, completion.total_tokens)
        return ctx.finish(completion, attempts)

    return app


class RequestContext:
    """Accounting shared by the streaming and non-streaming paths."""

    def __init__(
        self, g: Gateway, req: ChatRequest, key_name: str, request_id: str, start: float, cache: str
    ):
        self.g, self.req, self.key_name, self.request_id = g, req, key_name, request_id
        self.start, self.cache = start, cache
        self.completion_id = new_completion_id()

    @property
    def hit(self) -> bool:
        return self.cache.startswith("hit")

    def headers(self, c: Completion | None) -> dict[str, str]:
        h = {"x-request-id": self.request_id, "x-gateway-cache": self.cache}
        if c is not None:
            h["x-gateway-provider"] = c.provider
        return h

    def account(self, c: Completion, attempts: int, status: str = "ok") -> float:
        """Record metrics and the structured log line; returns the cost in USD."""
        latency = time.perf_counter() - self.start
        cost = 0.0
        if self.hit:
            saved = avoided_cost(self.g.cfg, c.upstream_model, c.prompt_tokens, c.completion_tokens)
            metrics.CACHE_SAVED_USD.labels(self.req.model).inc(saved)
            metrics.CACHE_SAVED_TOKENS.labels("prompt").inc(c.prompt_tokens)
            metrics.CACHE_SAVED_TOKENS.labels("completion").inc(c.completion_tokens)
        else:
            cost = actual_cost(self.g.cfg, c.upstream_model, c.prompt_tokens, c.completion_tokens)
            metrics.COST.labels(c.provider, c.upstream_model).inc(cost)
            metrics.TOKENS.labels(c.provider, c.upstream_model, "prompt").inc(c.prompt_tokens)
            metrics.TOKENS.labels(c.provider, c.upstream_model, "completion").inc(c.completion_tokens)
        cache_label = "hit" if self.hit else "miss"
        metrics.REQUESTS.labels(self.req.model, c.provider, status, cache_label).inc()
        metrics.REQUEST_LATENCY.labels(self.req.model, cache_label).observe(latency)
        log.info(
            json.dumps(
                {
                    "event": "chat_completion",
                    "request_id": self.request_id,
                    "key": self.key_name,
                    "model": self.req.model,
                    "provider": c.provider,
                    "upstream_model": c.upstream_model,
                    "cache": self.cache,
                    "attempts": attempts,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "cost_usd": round(cost, 8),
                    "latency_ms": round(latency * 1000, 1),
                    "stream": self.req.stream,
                }
            )
        )
        return cost

    def finish(self, c: Completion, attempts: int) -> JSONResponse:
        cost = self.account(c, attempts)
        return JSONResponse(
            to_openai_response(c, self.req.model, self.completion_id),
            headers=self.headers(c)
            | {
                "x-gateway-cost-usd": f"{cost:.8f}",
                "x-gateway-attempts": str(attempts),
                "x-gateway-latency-ms": f"{(time.perf_counter() - self.start) * 1000:.1f}",
            },
        )

    def fail(self, e: AllTargetsFailed) -> JSONResponse:
        metrics.REQUESTS.labels(self.req.model, "none", "error", "miss").inc()
        metrics.REQUEST_LATENCY.labels(self.req.model, "miss").observe(time.perf_counter() - self.start)
        log.warning(
            json.dumps({"event": "upstream_failed", "request_id": self.request_id, "errors": e.errors})
        )
        status = e.status if 400 <= e.status < 500 else 502
        return JSONResponse(
            {"error": {"message": str(e), "type": "upstream_error"}},
            status_code=status,
            headers={"x-request-id": self.request_id},
        )

    async def stream(self, cached: Completion | None, query_vec, bypass: bool):
        def sse(obj) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        yield sse(to_openai_chunk(self.req.model, self.completion_id, {"role": "assistant"}))
        if cached is not None:
            yield sse(to_openai_chunk(self.req.model, self.completion_id, {"content": cached.content}))
            final, attempts = cached, 0
        else:
            final, attempts = None, 1
            try:
                async for item in self.g.router.stream(self.req):
                    if isinstance(item, Completion):
                        final = item
                    else:
                        yield sse(to_openai_chunk(self.req.model, self.completion_id, {"content": item}))
            except AllTargetsFailed as e:
                self.fail(e)
                yield sse({"error": {"message": str(e), "type": "upstream_error"}})
                yield "data: [DONE]\n\n"
                return
            if final is not None and not bypass:
                await self.g.cache.set(self.req, final, query_vec)
        if final is not None:
            self.account(final, attempts)
            await self.g.limiter.record_tokens(self.key_name, 0 if self.hit else final.total_tokens)
        yield sse(
            to_openai_chunk(self.req.model, self.completion_id, {}, final.finish_reason if final else "stop")
        )
        yield "data: [DONE]\n\n"


def _configure_logging() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


_configure_logging()
app = create_app()

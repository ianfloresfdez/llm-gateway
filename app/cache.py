"""Response cache in Redis.

exact:    key = sha256(model + messages + sampling params). Cheap and safe.
semantic: on an exact miss, embed the prompt and look for a previous prompt with
          cosine similarity >= threshold, within the same model + params scope.
          Vectors are brute-force compared in numpy; plain Redis has no vector
          index, and up to a few thousand entries per scope this stays in ~ms.

The cache fails open: if Redis or the embedding model is down, requests still go
upstream, they just are not cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from typing import Literal

import numpy as np
from redis.asyncio import Redis

from app.config import CacheConfig
from app.guard import UNREACHABLE, Guard
from app.providers.base import Provider
from app.schemas import ChatRequest, Completion

log = logging.getLogger(__name__)


class EmbedderDown(Exception):
    """The embedding server failed; already recorded in embed_guard."""


CacheResult = Literal["hit_exact", "hit_semantic", "miss", "bypass", "error"]
PREFIX = "llmgw"


def _params(req: ChatRequest) -> dict:
    return {
        "model": req.model,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.output_limit,
        "stop": req.stop_list(),
        # Forwarded upstream and changes the answer, so it must split the cache too.
        "reasoning_effort": getattr(req, "reasoning_effort", None),
    }


def exact_key(req: ChatRequest) -> str:
    payload = _params(req) | {"messages": [m.model_dump(exclude_none=True) for m in req.messages]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def semantic_scope(req: ChatRequest) -> str:
    # Only compare prompts that share model, sampling params and system prompt.
    system = [m.text() for m in req.messages if m.role == "system"]
    payload = _params(req) | {"system": system}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def semantic_text(req: ChatRequest) -> str:
    return "\n".join(f"{m.role}: {m.text()}" for m in req.messages if m.role != "system")


class ResponseCache:
    def __init__(
        self, redis: Redis, cfg: CacheConfig, embedder: Provider | None = None, guard: Guard | None = None
    ):
        self.redis = redis
        self.cfg = cfg
        self.embedder = embedder
        self.guard = guard or Guard("redis")
        self.embed_guard = Guard("embedding server")

    @property
    def semantic_enabled(self) -> bool:
        return self.cfg.mode == "semantic" and self.embedder is not None and self.embed_guard.usable

    async def _embed(self, req: ChatRequest) -> np.ndarray:
        assert self.embedder is not None
        try:
            [vec] = await self.embedder.embed([semantic_text(req)], self.cfg.semantic.embedding_model)
        except Exception as e:
            self.embed_guard.failed(e)
            raise EmbedderDown from e
        v = np.asarray(vec, dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    async def get(self, req: ChatRequest) -> tuple[CacheResult, Completion | None, np.ndarray | None]:
        """Returns (result, cached completion, query embedding to reuse on set())."""
        if self.cfg.mode == "off":
            return "bypass", None, None
        if not self.guard.usable:
            return "error", None, None
        key = exact_key(req)
        try:
            raw = await self.redis.get(f"{PREFIX}:exact:{key}")
            if raw:
                return "hit_exact", Completion(**json.loads(raw)), None
            if not self.semantic_enabled:
                return "miss", None, None
            query = await self._embed(req)
            hit = await self._nearest(semantic_scope(req), query)
            if hit:
                raw = await self.redis.get(f"{PREFIX}:exact:{hit}")
                if raw:
                    return "hit_semantic", Completion(**json.loads(raw)), query
            return "miss", None, query
        except EmbedderDown:
            return "miss", None, None
        except UNREACHABLE as e:
            self.guard.failed(e)
            return "error", None, None
        except Exception:
            log.warning("cache lookup failed", exc_info=True)
            return "error", None, None

    async def _nearest(self, scope: str, query: np.ndarray) -> str | None:
        index = f"{PREFIX}:semidx:{scope}"
        keys = await self.redis.zrevrange(index, 0, self.cfg.semantic.max_entries - 1)
        if not keys:
            return None
        blobs = await self.redis.mget([f"{PREFIX}:semvec:{k.decode()}" for k in keys])
        live = [(k.decode(), b) for k, b in zip(keys, blobs, strict=True) if b is not None]
        expired = [k for k, b in zip(keys, blobs, strict=True) if b is None]
        if expired:
            await self.redis.zrem(index, *expired)
        if not live:
            return None
        matrix = np.stack([np.frombuffer(b, dtype=np.float32) for _, b in live])
        if matrix.shape[1] != query.shape[0]:
            return None
        scores = matrix @ query
        best = int(np.argmax(scores))
        return live[best][0] if scores[best] >= self.cfg.semantic.threshold else None

    async def set(self, req: ChatRequest, completion: Completion, query: np.ndarray | None = None) -> None:
        if self.cfg.mode == "off" or not completion.content or not self.guard.usable:
            return  # never cache an empty answer (e.g. output budget spent on reasoning)
        key = exact_key(req)
        ttl = self.cfg.ttl_s
        try:
            await self.redis.set(f"{PREFIX}:exact:{key}", json.dumps(asdict(completion)), ex=ttl)
            if not self.semantic_enabled:
                return
            if query is None:
                query = await self._embed(req)
            index = f"{PREFIX}:semidx:{semantic_scope(req)}"
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.set(f"{PREFIX}:semvec:{key}", query.astype(np.float32).tobytes(), ex=ttl)
                pipe.zadd(index, {key: time.time()})
                pipe.zremrangebyrank(index, 0, -(self.cfg.semantic.max_entries + 1))
                pipe.expire(index, ttl)
                await pipe.execute()
        except EmbedderDown:
            pass  # the exact entry is stored; only the semantic index misses it
        except UNREACHABLE as e:
            self.guard.failed(e)
        except Exception:
            log.warning("cache store failed", exc_info=True)

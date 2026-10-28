"""Deterministic fake provider: used by tests, CI's kind smoke test and fallback demos."""

from __future__ import annotations

import asyncio
import hashlib
import random

from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, Completion


def _count_tokens(text: str) -> int:
    # Rough approximation (~0.75 words per token) - good enough for a fake.
    return max(1, round(len(text.split()) / 0.75))


class MockProvider(Provider):
    def __init__(self, name: str, latency_ms: int = 200, failure_rate: float = 0.0, seed: int | None = None):
        self.name = name
        self.latency_s = latency_ms / 1000
        self.failure_rate = failure_rate
        self._rng = random.Random(seed)

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        await asyncio.sleep(self.latency_s)
        if self._rng.random() < self.failure_rate:
            raise ProviderError(f"{self.name}: injected failure", status=503, retryable=True)
        prompt = " ".join(m.text() for m in req.messages)
        last = req.messages[-1].text() if req.messages else ""
        digest = hashlib.sha256(last.encode()).hexdigest()[:8]
        content = f"[{model}] mock answer {digest} to: {last[:80]}"
        return Completion(
            content=content,
            prompt_tokens=_count_tokens(prompt),
            completion_tokens=_count_tokens(content),
            finish_reason="stop",
            provider=self.name,
            upstream_model=model,
        )

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        # Bag-of-words hashed into 64 dims: similar wording -> similar vectors.
        vectors = []
        for text in texts:
            v = [0.0] * 64
            for word in text.lower().split():
                v[int(hashlib.md5(word.strip("?.,!").encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(v)
        return vectors

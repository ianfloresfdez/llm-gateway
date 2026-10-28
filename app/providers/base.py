from __future__ import annotations

from collections.abc import AsyncIterator

from app.schemas import ChatRequest, Completion


class ProviderError(Exception):
    """An upstream failure, classified so the router knows what to do next.

    retryable: worth retrying on the same provider (timeouts, 5xx, connection errors).
    fallback:  worth trying the next provider (everything except client errors such
               as a malformed request, which would fail the same way everywhere).
    """

    def __init__(self, message: str, *, status: int = 502, retryable: bool, fallback: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.fallback = fallback


def classify_status(status: int, message: str) -> ProviderError:
    if status == 429:
        # Rate limited upstream: retrying immediately rarely helps; move on.
        return ProviderError(message, status=status, retryable=False, fallback=True)
    if status in (401, 403, 404):
        # Bad key or unknown model on this provider: another provider may work.
        return ProviderError(message, status=status, retryable=False, fallback=True)
    if 400 <= status < 500:
        return ProviderError(message, status=status, retryable=False, fallback=False)
    return ProviderError(message, status=status, retryable=True, fallback=True)


class Provider:
    name: str
    enabled: bool = True

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        raise NotImplementedError

    async def stream(self, req: ChatRequest, model: str) -> AsyncIterator[str | Completion]:
        """Yield text deltas, then exactly one final Completion with the usage.

        Default: no native streaming, so emit the whole answer as one delta.
        """
        c = await self.complete(req, model)
        yield c.content
        yield c

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise ProviderError(f"{self.name} does not support embeddings", retryable=False, fallback=False)

    async def aclose(self) -> None:
        return None

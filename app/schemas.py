"""The subset of the OpenAI Chat Completions schema the gateway understands."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if not self.content:
            return ""
        return "".join(p.get("text", "") for p in self.content if p.get("type") == "text")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    user: str | None = None

    @property
    def output_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    def stop_list(self) -> list[str] | None:
        if self.stop is None:
            return None
        return [self.stop] if isinstance(self.stop, str) else self.stop


@dataclass
class Completion:
    """Provider-neutral result; also what the cache stores."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    provider: str
    upstream_model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def to_openai_response(c: Completion, model: str, completion_id: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": c.content},
                "finish_reason": c.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "total_tokens": c.total_tokens,
        },
    }


def to_openai_chunk(
    model: str, completion_id: str, delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

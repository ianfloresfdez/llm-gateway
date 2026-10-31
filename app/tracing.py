"""Optional traces of every completion to Langfuse (self-hosted or cloud), as OpenTelemetry (OTLP/HTTP JSON) spans, without an SDK.

Off unless LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Each completion becomes a
Langfuse "generation" (model, prompt tail, answer or tool calls, tokens, latency, cost); clients group them with
headers: X-Trace-Session (e.g. a jack run id), X-Trace-Name (e.g. "implement · SHOP-103") and X-Trace-Tags
(comma-separated). Sending happens in the background and never fails or slows down a request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
import time
from typing import Any

import httpx

from app.schemas import ChatRequest, Completion

log = logging.getLogger("gateway.tracing")

MAX_SYSTEM_CHARS = 4_000
TAIL_MESSAGES = 4
MAX_MESSAGE_CHARS = 8_000

_pending: set[asyncio.Task] = set()


def _config() -> tuple[str, str, str] | None:
    base = os.environ.get("LANGFUSE_BASE_URL", "").strip().rstrip("/")
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return (base, pk, sk) if base and pk and sk else None


def enabled() -> bool:
    return _config() is not None


def meta_from(headers: Any) -> dict[str, Any]:
    """Grouping headers sent by the client (Claude Code forwards them with ANTHROPIC_CUSTOM_HEADERS)."""
    tags = [t.strip() for t in headers.get("x-trace-tags", "").split(",") if t.strip()]
    return {"session": headers.get("x-trace-session", "").strip(), "name": headers.get("x-trace-name", "").strip(), "tags": tags}


def _clip(text: Any, limit: int) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + f"… [{len(text) - limit} chars more]"
    return text


def prompt_tail(req: ChatRequest) -> list[dict[str, Any]]:
    """System prompt head plus the last few messages: enough to see why a model loops, without storing the
    whole growing conversation on every turn."""
    msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    system = [m for m in msgs if m.get("role") == "system"][:1]
    rest = [m for m in msgs if m.get("role") != "system"]
    tail = rest[-TAIL_MESSAGES:]
    out = [{**m, "content": _clip(m.get("content"), MAX_SYSTEM_CHARS)} for m in system]
    if len(rest) > len(tail):
        out.append({"role": "note", "content": f"{len(rest) - len(tail)} earlier messages omitted"})
    out += [{**m, "content": _clip(m.get("content"), MAX_MESSAGE_CHARS)} for m in tail]
    return out


def _attr(key: str, value: Any) -> dict[str, Any] | None:
    if value is None or value == [] or value == "":
        return None
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return {"key": key, "value": {"arrayValue": {"values": [{"stringValue": v} for v in value]}}}
    if not isinstance(value, str):
        value = json.dumps(value, default=str, ensure_ascii=False)
    return {"key": key, "value": {"stringValue": value}}


def span(meta: dict[str, Any], request_id: str, req: ChatRequest, c: Completion | None, latency_s: float,
         cost: float, error: str | None = None) -> dict[str, Any]:
    """One completion as an OTLP span that Langfuse reads as a "generation" (langfuse.* / gen_ai.* attributes)."""
    end = time.time_ns()
    start = end - int(latency_s * 1e9)
    session = meta.get("session") or None
    name = meta.get("name") or req.model
    # One trace per (session, name): every turn of a jack stage lands in the same trace
    trace_id = hashlib.sha1(f"{session}|{name}".encode()).hexdigest()[:32] if session else uuid.uuid4().hex
    output: Any = None
    if c is not None:
        output = {"content": c.content, "tool_calls": c.tool_calls} if c.tool_calls else c.content
    attrs = [
        _attr("langfuse.observation.type", "generation"),
        _attr("langfuse.trace.name", name),
        _attr("langfuse.session.id", session),
        _attr("langfuse.trace.tags", meta.get("tags") or None),
        _attr("langfuse.observation.input", prompt_tail(req)),
        _attr("langfuse.observation.output", output),
        _attr("langfuse.observation.level", "ERROR" if error else None),
        _attr("langfuse.observation.status_message", error),
        _attr("langfuse.observation.metadata.request_id", request_id),
        _attr("langfuse.observation.metadata.alias", req.model),
        _attr("langfuse.observation.metadata.provider", c.provider if c else None),
        _attr("gen_ai.request.model", c.upstream_model if c else req.model),
        _attr("gen_ai.usage.input_tokens", c.prompt_tokens if c else None),
        _attr("gen_ai.usage.output_tokens", c.completion_tokens if c else None),
        _attr("gen_ai.usage.cost", cost if c else None),
    ]
    return {
        "traceId": trace_id, "spanId": uuid.uuid4().hex[:16], "name": name, "kind": 3,
        "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
        "attributes": [a for a in attrs if a],
        "status": {"code": 2, "message": error} if error else {"code": 1},
    }


def otlp(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", "llm-gateway")]},
        "scopeSpans": [{"scope": {"name": "llm-gateway"}, "spans": spans}],
    }]}


async def _send(payload: dict[str, Any]) -> None:
    cfg = _config()
    if not cfg:
        return
    base, pk, sk = cfg
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{base}/api/public/otel/v1/traces", auth=(pk, sk), content=json.dumps(payload, default=str),
                                  headers={"Content-Type": "application/json"})
            if r.status_code >= 300:
                log.warning("langfuse otel → %s: %s", r.status_code, r.text[:200])
    except httpx.HTTPError as e:
        log.warning("langfuse unreachable: %s", e)


def record(meta: dict[str, Any], request_id: str, req: ChatRequest, c: Completion | None, latency_s: float,
           cost: float = 0.0, error: str | None = None) -> None:
    """Fire and forget: schedules the upload on the running loop."""
    if not enabled():
        return
    try:
        task = asyncio.get_running_loop().create_task(_send(otlp([span(meta, request_id, req, c, latency_s, cost, error)])))
    except RuntimeError:  # no running loop (sync tests): skip
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)

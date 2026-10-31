"""Anthropic Messages API (/v1/messages) on top of the OpenAI-shaped router.

This lets clients that only speak Anthropic's wire format - the Claude Agent SDK, Claude Code,
jack - use any provider behind the gateway, including a local Ollama model. Tool use is
translated both ways: Anthropic ``tool_use`` / ``tool_result`` blocks become OpenAI
``tool_calls`` / ``role: tool`` messages and back.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.schemas import ChatRequest, Completion

_STOP = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn", "content_filter": "refusal"}


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _text(content: Any) -> str:
    """Text of a string or a list of content blocks (images are dropped with a marker)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "image":
            parts.append("[image omitted]")
    return "\n".join(p for p in parts if p)


def _messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in raw_messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks = content or []
        if role == "assistant":
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})},
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        # user: tool results must come right after the assistant turn that asked for them
        for b in blocks:
            if b.get("type") == "tool_result":
                result = _text(b.get("content"))
                if b.get("is_error"):
                    result = f"Error: {result}"
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": result or "(empty)"})
        text = _text([b for b in blocks if b.get("type") != "tool_result"])
        if text:
            out.append({"role": "user", "content": text})
    return out


def _tools(raw_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in raw_tools or []
        if "input_schema" in t  # server-side tools (web search...) have no schema to forward
    ]
    return tools or None


def _tool_choice(choice: dict[str, Any] | None) -> Any:
    if not choice:
        return None
    kind = choice.get("type")
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    return {"auto": "auto", "none": "none"}.get(kind)


def to_chat_request(raw: dict[str, Any], model: str) -> ChatRequest:
    """Anthropic request body -> the gateway's OpenAI-shaped request, routed to ``model``."""
    if not isinstance(raw.get("messages"), list):
        raise ValueError("messages must be a list")
    messages = []
    system = _text(raw.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    messages += _messages(raw["messages"])
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": raw.get("max_tokens"),
        "temperature": raw.get("temperature"),
        "top_p": raw.get("top_p"),
        "stop": raw.get("stop_sequences"),
        "tools": _tools(raw.get("tools")),
        "tool_choice": _tool_choice(raw.get("tool_choice")),
    }
    return ChatRequest.model_validate({k: v for k, v in body.items() if v is not None})


def _content_blocks(c: Completion) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if c.content:
        blocks.append({"type": "text", "text": c.content})
    for call in c.tool_calls:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw_arguments": fn.get("arguments")}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "input": args if isinstance(args, dict) else {"value": args},
            }
        )
    return blocks or [{"type": "text", "text": ""}]


def stop_reason(c: Completion) -> str:
    if c.tool_calls:
        return "tool_use"
    return _STOP.get(c.finish_reason, "end_turn")


def to_message(c: Completion, model: str, message_id: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _content_blocks(c),
        "stop_reason": stop_reason(c),
        "stop_sequence": None,
        "usage": {"input_tokens": c.prompt_tokens, "output_tokens": c.completion_tokens},
    }


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def message_start(model: str, message_id: str) -> str:
    return sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )


def message_events(c: Completion) -> list[str]:
    """Everything after message_start for a finished completion: blocks, deltas and the stop."""
    events = []
    for i, block in enumerate(_content_blocks(c)):
        if block["type"] == "text":
            events.append(
                sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            events.append(
                sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
            )
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            events.append(
                sse(
                    "content_block_start", {"type": "content_block_start", "index": i, "content_block": start}
                )
            )
            events.append(
                sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
                    },
                )
            )
        events.append(sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason(c), "stop_sequence": None},
                "usage": {"input_tokens": c.prompt_tokens, "output_tokens": c.completion_tokens},
            },
        )
    )
    events.append(sse("message_stop", {"type": "message_stop"}))
    return events


def error_body(message: str, kind: str = "api_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def estimate_tokens(raw: dict[str, Any]) -> int:
    size = len(json.dumps(raw.get("messages", []))) + len(_text(raw.get("system")))
    size += len(json.dumps(raw.get("tools", [])))
    return max(1, size // 4)

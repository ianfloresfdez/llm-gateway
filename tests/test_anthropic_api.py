"""Anthropic /v1/messages: translation both ways, tool use, streaming and model fallback."""

from __future__ import annotations

import json

import httpx
import pytest

from app import anthropic_api
from app.providers.base import Provider
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas import ChatRequest, Completion
from tests.conftest import make_config

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {"type": "web_search_20250305", "name": "web_search"},  # server tool: not forwarded
]


def agent_conversation() -> dict:
    return {
        "model": "claude-opus-5-5",
        "max_tokens": 512,
        "system": [
            {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}
        ],
        "tools": TOOLS,
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": "What does README say?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read it."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "README.md"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "# demo"}],
                    },
                    {"type": "text", "text": "Summarise it."},
                ],
            },
        ],
    }


def test_request_translation_keeps_tools_and_tool_results():
    req = anthropic_api.to_chat_request(agent_conversation(), "agent")
    roles = [m.role for m in req.messages]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assistant, tool = req.messages[2].model_dump(), req.messages[3].model_dump()
    assert assistant["tool_calls"][0]["function"] == {"name": "Read", "arguments": '{"path": "README.md"}'}
    assert tool == {"role": "tool", "content": "# demo", "tool_call_id": "toolu_1"}
    assert [t["function"]["name"] for t in req.tools] == ["Read"]
    assert req.tool_choice == "auto" and req.output_limit == 512 and req.uses_tools


def test_tool_choice_mapping():
    assert anthropic_api._tool_choice({"type": "any"}) == "required"
    assert anthropic_api._tool_choice({"type": "tool", "name": "Read"}) == {
        "type": "function",
        "function": {"name": "Read"},
    }


def test_response_translation_with_tool_call():
    c = Completion(
        content="",
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="tool_calls",
        provider="p",
        upstream_model="m",
        tool_calls=[
            {"id": "call_9", "type": "function", "function": {"name": "Read", "arguments": '{"path": "a"}'}}
        ],
    )
    msg = anthropic_api.to_message(c, "claude-opus-5-5", "msg_1")
    assert msg["stop_reason"] == "tool_use"
    assert msg["content"] == [{"type": "tool_use", "id": "call_9", "name": "Read", "input": {"path": "a"}}]
    assert msg["usage"] == {"input_tokens": 10, "output_tokens": 5}


class ToolCallingProvider(Provider):
    """Asks for Read the first time, answers once it gets the tool result."""

    def __init__(self, name: str = "agentish"):
        self.name = name
        self.requests: list[ChatRequest] = []

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        self.requests.append(req)
        if req.messages[-1].role == "tool" or "Summarise" in req.messages[-1].text():
            return Completion("README is a demo.", 20, 5, "stop", self.name, model)
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Read", "arguments": '{"path": "README.md"}'},
        }
        return Completion("", 15, 4, "tool_calls", self.name, model, tool_calls=[call])


@pytest.fixture
def agent_client(make_client):
    cfg = make_config(
        providers={"agentish": {"type": "mock", "latency_ms": 0}},
        models={"agent": {"targets": [{"provider": "agentish", "model": "local-agent"}]}},
    )
    provider = ToolCallingProvider()
    client = make_client(cfg=cfg, providers={"agentish": provider})
    return client, provider


def test_messages_endpoint_tool_round_trip(agent_client):
    client, provider = agent_client
    first = {
        "model": "claude-opus-5-5",
        "max_tokens": 100,
        "tools": TOOLS[:1],
        "messages": [{"role": "user", "content": "What does README say?"}],
    }
    r = client.post("/v1/messages", json=first)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "claude-opus-5-5"  # the client sees the name it asked for
    assert body["stop_reason"] == "tool_use" and body["content"][0]["name"] == "Read"
    assert provider.requests[0].model == "agent"  # unknown claude-* names go to the agent alias

    r2 = client.post("/v1/messages", json=agent_conversation())
    assert r2.json()["content"] == [{"type": "text", "text": "README is a demo."}]
    assert r2.json()["stop_reason"] == "end_turn"


def test_tool_requests_are_not_cached(agent_client):
    client, provider = agent_client
    body = {
        "model": "agent",
        "max_tokens": 50,
        "tools": TOOLS[:1],
        "messages": [{"role": "user", "content": "hi"}],
    }
    client.post("/v1/messages", json=body)
    client.post("/v1/messages", json=body)
    assert len(provider.requests) == 2


def test_messages_streaming_events(agent_client):
    client, _ = agent_client
    body = agent_conversation() | {"stream": True}
    with client.stream("POST", "/v1/messages", json=body) as r:
        text = "".join(r.iter_text())
    events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert "content_block_delta" in events and "message_delta" in events
    deltas = [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and "text_delta" in line
    ]
    assert deltas[0]["delta"]["text"] == "README is a demo."


def test_unknown_model_without_agent_alias(make_client):
    client = make_client()  # default test config has no "agent" alias
    r = client.post("/v1/messages", json={"model": "claude-x", "max_tokens": 5, "messages": []})
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


def test_x_api_key_auth(agent_client, make_client):
    cfg = make_config(
        providers={"agentish": {"type": "mock", "latency_ms": 0}},
        models={"agent": {"targets": [{"provider": "agentish", "model": "m"}]}},
    )
    client = make_client(cfg=cfg, providers={"agentish": ToolCallingProvider()}, api_keys="jack:sk-local")
    body = {"model": "agent", "max_tokens": 5, "messages": [{"role": "user", "content": "Summarise"}]}
    assert client.post("/v1/messages", json=body).status_code == 401
    assert client.post("/v1/messages", json=body, headers={"x-api-key": "sk-local"}).status_code == 200


def test_count_tokens(agent_client):
    client, _ = agent_client
    r = client.post("/v1/messages/count_tokens", json=agent_conversation())
    assert r.json()["input_tokens"] > 50


async def test_openai_provider_forwards_tools_and_parses_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        call = {"id": "call_7", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    p = OpenAICompatibleProvider("o", "http://x/v1", transport=httpx.MockTransport(handler))
    req = anthropic_api.to_chat_request(agent_conversation(), "agent")
    c = await p.complete(req, "gemma")
    assert seen["tools"][0]["function"]["name"] == "Read" and seen["tool_choice"] == "auto"
    assert seen["messages"][3] == {"role": "tool", "content": "# demo", "tool_call_id": "toolu_1"}
    assert c.tool_calls[0]["id"] == "call_7" and c.content == ""
    await p.aclose()

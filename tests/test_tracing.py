"""Langfuse traces: OTLP spans grouped by client headers, a trimmed prompt, and nothing when off."""

from app import tracing
from app.schemas import ChatRequest, Completion


def req(n=10):
    msgs = [{"role": "system", "content": "S" * 10_000}] + [
        {"role": "user" if i % 2 else "assistant", "content": f"turn {i}"} for i in range(n)
    ]
    return ChatRequest(model="agent", messages=msgs)


def attrs(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


def test_one_trace_per_session_and_stage_as_a_generation():
    meta = tracing.meta_from(
        {"x-trace-session": "run1", "x-trace-name": "implement · SHOP-1", "x-trace-tags": "jack, bench"}
    )
    c = Completion(
        content="",
        prompt_tokens=100,
        completion_tokens=5,
        finish_reason="tool_calls",
        provider="ollama",
        upstream_model="qwen3-coder-64k",
        tool_calls=[{"function": {"name": "Read"}}],
    )
    a = tracing.span(meta, "r1", req(), c, 2.0, 0.0)
    b = tracing.span(meta, "r2", req(), c, 2.0, 0.0)
    assert a["traceId"] == b["traceId"] and len(a["traceId"]) == 32 and a["spanId"] != b["spanId"]
    at = attrs(a)
    assert at["langfuse.observation.type"] == "generation" and at["langfuse.session.id"] == "run1"
    assert at["gen_ai.request.model"] == "qwen3-coder-64k" and at["gen_ai.usage.input_tokens"] == "100"
    assert "tool_calls" in at["langfuse.observation.output"]
    assert tracing.otlp([a])["resourceSpans"][0]["scopeSpans"][0]["spans"] == [a]


def test_prompt_is_trimmed_to_the_system_head_and_the_last_turns():
    prompt = tracing.prompt_tail(req())
    assert prompt[0]["role"] == "system" and len(prompt[0]["content"]) < 5_000
    assert prompt[1]["role"] == "note" and len(prompt) == 2 + tracing.TAIL_MESSAGES


def test_failures_are_errors_and_requests_without_headers_get_their_own_trace():
    s = tracing.span({}, "r9", req(2), None, 1.0, 0.0, error="all targets failed")
    assert attrs(s)["langfuse.observation.level"] == "ERROR" and s["status"]["code"] == 2
    assert tracing.span({}, "r10", req(2), None, 1.0, 0.0)["traceId"] != s["traceId"]


def test_off_without_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    assert not tracing.enabled()
    tracing.record({}, "r", req(1), None, 0.1)  # no loop, no config: must not raise

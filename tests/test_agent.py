"""Agent loop tests with fake LLM clients - no network, no API keys."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from eeg_explorer import tools
from eeg_explorer.agent import AnthropicAgent, OpenAICompatibleAgent, make_agent
from eeg_explorer.data import make_demo_recording
from eeg_explorer.pipeline import AnalysisState


@pytest.fixture
def state() -> AnalysisState:
    s = AnalysisState()
    make_demo_recording(s, n_trials=5)
    return s


# --------------------------------------------------------------------------- #
# tools registry
# --------------------------------------------------------------------------- #
def test_tool_schemas_are_well_formed():
    names = [t.name for t in tools.TOOLS]
    assert len(names) == len(set(names))
    for t in tools.TOOLS:
        assert t.input_schema["type"] == "object"
        assert t.input_schema["additionalProperties"] is False
        assert t.description


def test_run_tool_errors_are_reported_not_raised(state):
    out, err = tools.run_tool(state, "bogus", {})
    assert err and "Unknown tool" in out
    out, err = tools.run_tool(state, "describe_recording", {"nope": 1})
    assert err and "Unknown arguments" in out
    out, err = tools.run_tool(state, "set_viewer", {"time_s": 0.3})
    assert err and "localize_sources" in out
    out, err = tools.run_tool(state, "describe_recording", {})
    assert not err and json.loads(out)["n_eeg_channels"] == 32


def test_compare_with_ground_truth_on_real_data():
    s = AnalysisState()
    make_demo_recording(s, n_trials=5)
    s.truth = None
    out, err = tools.run_tool(s, "compare_with_ground_truth", {})
    assert not err and json.loads(out)["available"] is False


# --------------------------------------------------------------------------- #
# Anthropic loop
# --------------------------------------------------------------------------- #
def _block(**kw):
    return SimpleNamespace(**kw)


class FakeAnthropicStream:
    """Scripted responses: first a tool call, then a final answer."""

    def __init__(self, scripted):
        self.scripted = scripted
        self.calls = []

    @contextmanager
    def stream(self, **kwargs):
        self.calls.append(kwargs)
        response = self.scripted.pop(0)
        yield SimpleNamespace(get_final_message=lambda: response)


def _anthropic_response(content, stop_reason):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, stop_details=None)


def test_anthropic_agent_runs_tools_and_returns_text(state, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = AnthropicAgent(state, on_event=lambda ev: None)
    fake = FakeAnthropicStream([
        _anthropic_response([
            _block(type="thinking", thinking="Need to look at the data first."),
            _block(type="tool_use", id="t1", name="describe_recording", input={}),
            _block(type="tool_use", id="t2", name="preprocess_recording", input={"notch_hz": None}),
        ], "tool_use"),
        _anthropic_response([_block(type="text", text="32 channels, T8 was bad and is now interpolated.")],
                            "end_turn"),
    ])
    agent.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))

    turn = agent.chat("What's in this recording?")

    assert turn.text.startswith("32 channels")
    assert [e.name for e in turn.events] == ["describe_recording", "preprocess_recording"]
    assert turn.thinking == ["Need to look at the data first."]
    assert turn.usage["input_tokens"] == 200
    # Both tool results went back in one user message, and the request used the right model/tools.
    results_msg = agent.history[2]
    assert results_msg["role"] == "user" and len(results_msg["content"]) == 2
    assert {r["tool_use_id"] for r in results_msg["content"]} == {"t1", "t2"}
    assert fake.calls[0]["model"] == "claude-opus-5-5"
    assert {t["name"] for t in fake.calls[0]["tools"]} == {t.name for t in tools.TOOLS}
    assert fake.calls[0]["fallbacks"] == "default"
    assert state.log[-1]["step"] == "preprocess"


def test_anthropic_agent_surfaces_refusal(state, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = AnthropicAgent(state)
    resp = _anthropic_response([], "refusal")
    resp.stop_details = SimpleNamespace(category="general_harms", explanation="nope")
    agent.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeAnthropicStream([resp])))
    turn = agent.chat("...")
    assert turn.stop_reason == "refusal" and "declined" in turn.text


# --------------------------------------------------------------------------- #
# OpenAI-compatible loop
# --------------------------------------------------------------------------- #
class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls
        self.reasoning_content = None

    def model_dump(self, **kw):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls]
        return d


def _oa_call(id_, name, arguments):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def _oa_response(message, finish_reason):
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10))


def test_openai_compatible_agent_loop(state):
    agent = OpenAICompatibleAgent(state, model="fake", base_url="http://localhost:1/v1")
    scripted = [
        _oa_response(FakeMessage(tool_calls=[_oa_call("c1", "describe_recording", "{}")]), "tool_calls"),
        _oa_response(FakeMessage(tool_calls=[_oa_call("c2", "localize_sources", "{not json")]), "tool_calls"),
        _oa_response(FakeMessage(content="Done."), "stop"),
    ]
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return scripted.pop(0)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                   base_url="http://localhost:1/v1")
    turn = agent.chat("hello")
    assert turn.text == "Done."
    assert [e.name for e in turn.events] == ["describe_recording", "localize_sources"]
    assert turn.events[1].is_error and "Invalid JSON" in turn.events[1].result
    assert agent.history[0]["role"] == "system"
    assert calls[0]["tools"][0]["type"] == "function"
    assert agent.history[-2]["role"] == "tool" and agent.history[-2]["tool_call_id"] == "c2"


def test_make_agent_factory(state, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert make_agent("anthropic", state).provider == "anthropic"
    assert make_agent("openai_compatible", state, model="llama3.1").model == "llama3.1"
    with pytest.raises(ValueError):
        make_agent("nope", state)

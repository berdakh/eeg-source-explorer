"""The AI analyst: an agent that drives the pipeline through tool calls.

Two interchangeable backends:

* ``AnthropicAgent`` - Claude via the official Anthropic SDK (default model
  ``claude-opus-5-5``, adaptive thinking, server-side refusal fallbacks).
* ``OpenAICompatibleAgent`` - any open-weight model served through an
  OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, llama.cpp server, ...)
  that supports function calling, e.g. Llama 3.x, Qwen 2.5/3, Mistral, Gemma 3.

Both run the same manual tool loop over the tools in ``tools.py`` and return
an ``AgentTurn`` with the final answer plus every tool call that happened, so
the UI can show the agent's work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .pipeline import AnalysisState
from .tools import TOOLS, run_tool

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_OPEN_MODEL = "qwen2.5:14b"
DEFAULT_OPEN_BASE_URL = "http://localhost:11434/v1"  # Ollama
MAX_TOOL_ROUNDS = 12

SYSTEM_PROMPT = """You are the analyst inside EEG Source Explorer, an app that performs EEG source imaging with MNE-Python. You help neuroscientists, students and clinicians understand where in the brain the activity in their recording comes from.

You work by calling tools that run the real analysis pipeline on the loaded recording; the user sees the results update live in a 3D brain viewer next to this chat. Standard workflow for "where does this activity come from?":
1. describe_recording (if you don't already know the state)
2. preprocess_recording (1-40 Hz band-pass, bad-channel detection, average reference) unless already done
3. epoch_recording if the recording has events; skip for continuous/resting data
4. localize_sources, then find_peak_sources over the window of interest, then set_viewer so the user sees it
Skip steps that the analysis log shows were already run with suitable parameters, and never repeat a tool call with identical arguments.

Interpretation guidance:
- Report concrete numbers (times in ms, positions in mm, relative strengths) and name regions, but be honest about precision: the spherical head model with a 10 mm source grid is accurate to roughly 2 cm for superficial sources and cannot resolve deep structures; the region labels for the sphere model are coarse lobe estimates. fsaverage gives anatomical (Desikan-Killiany) labels but is still a template, not the subject's own anatomy.
- eLORETA has the lowest localisation error; dSPM and sLORETA are noise-normalised and dSPM is biased towards deep/central sources; MNE shows raw current amplitude with a bias towards superficial sources. Mention method choice when it matters.
- Distributed inverse solutions are blurry by construction: a single focal generator appears as a blob several cm wide. Do not over-interpret small differences.
- For the synthetic demo recording you may call compare_with_ground_truth and report true localisation error; for real data say that ground truth is unknown.
- If a tool returns an error, read it, adjust the parameters once or twice, and if it still fails explain the problem to the user instead of guessing results.

Style: concise and direct, like a colleague at the workstation. Lead with the finding, then how you got it, then caveats. Use short paragraphs or compact bullet lists; no headings for short answers. Do not describe tool mechanics ("I will now call...") - just do the work and report."""


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False


@dataclass
class AgentTurn:
    text: str
    events: list[ToolEvent] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


EventCallback = Callable[[ToolEvent], None]


class BaseAgent:
    provider = "base"

    def __init__(self, state: AnalysisState, model: str, on_event: EventCallback | None = None):
        self.state = state
        self.model = model
        self.on_event = on_event
        self.history: list[Any] = []

    def reset(self) -> None:
        self.history = []

    def _execute(self, name: str, arguments: dict[str, Any]) -> ToolEvent:
        result, is_error = run_tool(self.state, name, arguments)
        event = ToolEvent(name=name, arguments=arguments, result=result, is_error=is_error)
        if self.on_event:
            self.on_event(event)
        return event

    def chat(self, user_text: str) -> AgentTurn:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #
class AnthropicAgent(BaseAgent):
    provider = "anthropic"

    def __init__(self, state: AnalysisState, model: str = DEFAULT_CLAUDE_MODEL,
                 api_key: str | None = None, effort: str = "medium",
                 on_event: EventCallback | None = None):
        super().__init__(state, model, on_event)
        import anthropic

        self._anthropic = anthropic
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an `ant auth login` profile.
        self.client = anthropic.Anthropic(api_key=api_key or None)
        self.effort = effort
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                      for t in TOOLS]

    def chat(self, user_text: str) -> AgentTurn:
        anthropic = self._anthropic
        self.history.append({"role": "user", "content": user_text})
        turn = AgentTurn(text="")
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=16000,
                    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    messages=self.history,
                    tools=self.tools,
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": self.effort},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    response = stream.get_final_message()
            except anthropic.AuthenticationError as exc:
                raise RuntimeError("Anthropic API key missing or invalid. Set ANTHROPIC_API_KEY or "
                                   "paste a key in the sidebar.") from exc
            except anthropic.RateLimitError as exc:
                raise RuntimeError("Anthropic rate limit hit - wait a moment and try again.") from exc
            except anthropic.APIStatusError as exc:
                raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
            except anthropic.APIConnectionError as exc:
                raise RuntimeError(f"Could not reach the Anthropic API: {exc}") from exc

            # Pass content back unchanged (thinking blocks included).
            self.history.append({"role": "assistant", "content": response.content})
            turn.stop_reason = response.stop_reason
            usage = response.usage
            turn.usage = {"input_tokens": turn.usage.get("input_tokens", 0) + usage.input_tokens,
                          "output_tokens": turn.usage.get("output_tokens", 0) + usage.output_tokens,
                          "cache_read_input_tokens": turn.usage.get("cache_read_input_tokens", 0)
                          + (usage.cache_read_input_tokens or 0)}

            text_parts, tool_uses = [], []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking" and block.thinking:
                    turn.thinking.append(block.thinking)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                why = f" ({details.category}: {details.explanation})" if details else ""
                turn.text = "".join(text_parts) or f"The model declined this request{why}."
                return turn
            if response.stop_reason == "max_tokens":
                turn.text = "".join(text_parts) + "\n\n*(response cut off at the token limit)*"
                return turn
            if response.stop_reason != "tool_use" or not tool_uses:
                turn.text = "".join(text_parts)
                return turn

            # Run every requested tool and return all results in ONE user message.
            results = []
            for block in tool_uses:
                event = self._execute(block.name, dict(block.input))
                turn.events.append(event)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": event.result, "is_error": event.is_error})
            self.history.append({"role": "user", "content": results})

        turn.text = "I reached the tool-call limit for a single question. Here is where things stand - " \
                    "ask a follow-up to continue."
        return turn


# --------------------------------------------------------------------------- #
# Open-weight models (OpenAI-compatible endpoints)
# --------------------------------------------------------------------------- #
class OpenAICompatibleAgent(BaseAgent):
    provider = "openai_compatible"

    def __init__(self, state: AnalysisState, model: str = DEFAULT_OPEN_MODEL,
                 base_url: str = DEFAULT_OPEN_BASE_URL, api_key: str | None = None,
                 temperature: float = 0.2, on_event: EventCallback | None = None):
        super().__init__(state, model, on_event)
        import openai

        self._openai = openai
        # Local servers ignore the key but the client requires a non-empty string.
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY") or "local")
        self.temperature = temperature
        self.tools = [{"type": "function",
                       "function": {"name": t.name, "description": t.description,
                                    "parameters": t.input_schema}} for t in TOOLS]

    def chat(self, user_text: str) -> AgentTurn:
        openai = self._openai
        if not self.history:
            self.history.append({"role": "system", "content": SYSTEM_PROMPT})
        self.history.append({"role": "user", "content": user_text})
        turn = AgentTurn(text="")
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=self.history, tools=self.tools,
                    tool_choice="auto", temperature=self.temperature)
            except openai.AuthenticationError as exc:
                raise RuntimeError("The endpoint rejected the API key.") from exc
            except openai.NotFoundError as exc:
                raise RuntimeError(f"Model '{self.model}' not found on the server - pull it first "
                                   f"(e.g. `ollama pull {self.model}`).") from exc
            except openai.APIStatusError as exc:
                raise RuntimeError(f"Endpoint error {exc.status_code}: {exc.message}") from exc
            except openai.APIConnectionError as exc:
                raise RuntimeError(f"Could not reach {self.client.base_url} - is the model server running?") from exc

            choice = response.choices[0]
            message = choice.message
            turn.stop_reason = choice.finish_reason
            if response.usage:
                turn.usage = {"input_tokens": turn.usage.get("input_tokens", 0) + (response.usage.prompt_tokens or 0),
                              "output_tokens": turn.usage.get("output_tokens", 0) + (response.usage.completion_tokens or 0)}
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            if reasoning:
                turn.thinking.append(str(reasoning))

            self.history.append(message.model_dump(exclude_none=True, exclude={"reasoning_content", "reasoning"}))
            if not message.tool_calls:
                turn.text = message.content or ""
                return turn

            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    event = ToolEvent(call.function.name, {}, json.dumps({"error": f"Invalid JSON arguments: {exc}"}), True)
                else:
                    event = self._execute(call.function.name, arguments)
                turn.events.append(event)
                self.history.append({"role": "tool", "tool_call_id": call.id, "content": event.result})

        turn.text = "I reached the tool-call limit for a single question - ask a follow-up to continue."
        return turn


# --------------------------------------------------------------------------- #
def make_agent(provider: str, state: AnalysisState, model: str | None = None,
               api_key: str | None = None, base_url: str | None = None,
               on_event: EventCallback | None = None, **kwargs: Any) -> BaseAgent:
    """Factory used by the UI. ``provider`` is 'anthropic' or 'openai_compatible'."""
    if provider == "anthropic":
        return AnthropicAgent(state, model or DEFAULT_CLAUDE_MODEL, api_key=api_key,
                              on_event=on_event, **kwargs)
    if provider == "openai_compatible":
        return OpenAICompatibleAgent(state, model or DEFAULT_OPEN_MODEL,
                                     base_url=base_url or DEFAULT_OPEN_BASE_URL,
                                     api_key=api_key, on_event=on_event, **kwargs)
    raise ValueError(f"Unknown provider '{provider}'")

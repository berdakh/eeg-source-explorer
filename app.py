"""EEG Source Explorer - Streamlit app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from eeg_explorer import pipeline as pl
from eeg_explorer import viz
from eeg_explorer.agent import (DEFAULT_CLAUDE_MODEL, DEFAULT_OPEN_BASE_URL,
                                DEFAULT_OPEN_MODEL, make_agent)
from eeg_explorer.data import SUPPORTED_SUFFIXES, load_recording, make_demo_recording
from eeg_explorer.pipeline import AnalysisState
from eeg_explorer.tools import to_json

load_dotenv()  # ANTHROPIC_API_KEY / OPEN_BASE_URL / OPEN_MODEL from a local .env
st.set_page_config(page_title="EEG Source Explorer", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

PROVIDERS = {
    "Claude (Anthropic API)": "anthropic",
    "Open-weight model (OpenAI-compatible endpoint)": "openai_compatible",
}
SUGGESTIONS = [
    "Where in the brain does the evoked response come from?",
    "Check the data quality and fix any bad channels.",
    "Which region generates the alpha rhythm in this recording?",
    "Compare eLORETA and dSPM for the peak of the evoked response.",
]


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init() -> None:
    ss = st.session_state
    ss.setdefault("state", AnalysisState())
    ss.setdefault("chat", [])  # list of dicts: role, content, events, thinking
    ss.setdefault("agent", None)
    ss.setdefault("agent_key", None)
    ss.setdefault("pending_prompt", None)


def _reset_agent() -> None:
    st.session_state.agent = None
    st.session_state.agent_key = None


def _get_agent(provider: str, model: str, api_key: str, base_url: str):
    key = (provider, model, bool(api_key), base_url)
    ss = st.session_state
    if ss.agent is None or ss.agent_key != key:
        ss.agent = make_agent(provider, ss.state, model=model or None, api_key=api_key or None,
                              base_url=base_url or None)
        ss.agent_key = key
    return ss.agent


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    ss = st.session_state
    state: AnalysisState = ss.state
    with st.sidebar:
        st.title("🧠 EEG Source Explorer")
        st.caption("Interactive EEG source imaging with an AI analyst, built on MNE-Python.")

        st.subheader("Recording")
        if st.button("Generate synthetic demo", use_container_width=True,
                     help="32-channel visuo-motor task with known sources - great for trying the agent."):
            with st.spinner("Simulating recording..."):
                make_demo_recording(state)
            ss.chat = []
            _reset_agent()
            st.rerun()
        upload = st.file_uploader("...or upload your own",
                                  type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
                                  help="EEG channel names must follow the 10-20/10-05 convention "
                                       "(or carry digitised positions) so they can be placed on the head.")
        if upload is not None and ss.get("last_upload") != upload.name:
            try:
                with st.spinner(f"Loading {upload.name}..."):
                    report = load_recording(state, data=upload.getvalue(), filename=upload.name)
                ss.last_upload = upload.name
                ss.chat = []
                _reset_agent()
                if report["dropped_unplaceable"]:
                    st.warning(f"Dropped channels without template positions: "
                               f"{', '.join(report['dropped_unplaceable'])}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load file: {exc}")

        if state.has_raw:
            d = pl.describe_recording(state)
            st.success(f"**{d['source']}**  \n{d['n_eeg_channels']} EEG channels · "
                       f"{d['sfreq_hz']:.0f} Hz · {d['duration_s']:.0f} s · {d['n_events']} events")

        st.subheader("AI analyst")
        provider_label = st.selectbox("Provider", list(PROVIDERS), on_change=_reset_agent)
        provider = PROVIDERS[provider_label]
        if provider == "anthropic":
            model = st.text_input("Model", DEFAULT_CLAUDE_MODEL)
            api_key = st.text_input("Anthropic API key", type="password",
                                    value=os.getenv("ANTHROPIC_API_KEY", ""),
                                    help="Or set ANTHROPIC_API_KEY in the environment / .env file.")
            base_url = ""
        else:
            model = st.text_input("Model", os.getenv("OPEN_MODEL", DEFAULT_OPEN_MODEL),
                                  help="Any tool-calling model on your server, e.g. qwen2.5:14b, "
                                       "llama3.1:8b, mistral-nemo.")
            base_url = st.text_input("Endpoint", os.getenv("OPEN_BASE_URL", DEFAULT_OPEN_BASE_URL),
                                     help="Ollama: http://localhost:11434/v1 · vLLM: http://host:8000/v1")
            api_key = st.text_input("API key (if required)", type="password",
                                    value=os.getenv("OPEN_API_KEY", ""))

        with st.expander("Manual controls (no AI needed)"):
            manual_controls(state)

        st.caption("Head model: spherical (offline) by default; ask the analyst for `fsaverage` "
                   "to use a template cortex (downloads ~300 MB).")
    return {"provider": provider, "model": model, "api_key": api_key, "base_url": base_url}


def manual_controls(state: AnalysisState) -> None:
    disabled = not state.has_raw
    c1, c2 = st.columns(2)
    l_freq = c1.number_input("High-pass (Hz)", 0.0, 20.0, 1.0, 0.5, disabled=disabled)
    h_freq = c2.number_input("Low-pass (Hz)", 5.0, 120.0, 40.0, 1.0, disabled=disabled)
    if st.button("Preprocess", disabled=disabled, use_container_width=True):
        with st.spinner("Filtering and checking channels..."):
            pl.preprocess(state, l_freq=l_freq or None, h_freq=h_freq)
        st.rerun()
    if st.button("Epoch around events", disabled=disabled, use_container_width=True):
        try:
            with st.spinner("Epoching..."):
                pl.make_epochs(state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    method = st.selectbox("Inverse method", pl.INVERSE_METHODS, disabled=disabled)
    head_model = st.selectbox("Head model", pl.HEAD_MODELS, disabled=disabled)
    if st.button("Localize sources", disabled=disabled, type="primary", use_container_width=True):
        try:
            with st.spinner(f"Running {method} on the {head_model} model..."):
                pl.localize(state, method=method, head_model=head_model)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.button("Band power", disabled=disabled, use_container_width=True):
        with st.spinner("Computing spectra..."):
            pl.band_power(state)
        st.rerun()


# --------------------------------------------------------------------------- #
# Main panels
# --------------------------------------------------------------------------- #
def viewer_panel(state: AnalysisState) -> None:
    if state.has_stc:
        stc = state.stc
        t0, t1 = float(stc.times[0]), float(stc.times[-1])
        mode_default = 1 if state.viewer_window else 0
        c1, c2 = st.columns([1, 3])
        mode = c1.radio("Show", ["instant", "window mean"], index=mode_default, horizontal=True,
                        label_visibility="collapsed")
        if mode == "instant":
            default = state.viewer_time if state.viewer_time is not None else float(stc.times[int(abs(stc.data).mean(0).argmax())])
            t = c2.slider("time (s)", t0, t1, float(min(max(default, t0), t1)),
                          step=float(stc.tstep), format="%.3f", label_visibility="collapsed")
            state.viewer_time, state.viewer_window = t, None
            fig = viz.source_map_figure(state, time=t)
            marker = t
        else:
            default = state.viewer_window or (t0, t1)
            w = c2.slider("window (s)", t0, t1, (float(max(default[0], t0)), float(min(default[1], t1))),
                          step=float(stc.tstep), format="%.3f", label_visibility="collapsed")
            state.viewer_window, state.viewer_time = w, None
            fig = viz.source_map_figure(state, window=w)
            marker = (w[0] + w[1]) / 2
    else:
        fig, marker = viz.source_map_figure(state), None
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    tabs = st.tabs(["Sensors", "Source time courses", "Spectrum", "Analysis log"]
                   + (["Ground truth"] if state.truth else []))
    with tabs[0]:
        if state.evoked is not None:
            c1, c2 = st.columns([3, 1])
            c1.plotly_chart(viz.sensor_butterfly_figure(state.evoked, marker), use_container_width=True,
                            config={"displaylogo": False})
            if marker is not None:
                c2.pyplot(viz.topomap_figure(state.evoked, marker), use_container_width=True)
        else:
            st.info("Epoch the recording (or localise a raw window) to see sensor-space data.")
    with tabs[1]:
        if state.has_stc:
            win = state.viewer_window
            peaks = pl.peak_sources(state, *(win or (None, None)), n=4, update_viewer=False)["peaks"]
            st.plotly_chart(viz.source_time_course_figure(state, peaks, marker), use_container_width=True,
                            config={"displaylogo": False})
            st.dataframe([{k: v for k, v in p.items() if k != "source_index"} for p in peaks],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No source estimate yet.")
    with tabs[2]:
        if state.psd is not None:
            st.plotly_chart(viz.psd_figure(state), use_container_width=True, config={"displaylogo": False})
            st.plotly_chart(viz.band_power_figure(state), use_container_width=True, config={"displaylogo": False})
        else:
            st.info("Ask the analyst about rhythms/band power, or use the manual controls.")
    with tabs[3]:
        if state.log:
            for entry in state.log:
                with st.expander(f"{entry['step']}  ·  {entry['elapsed_s']} s"):
                    st.json({"params": entry["params"], "summary": entry["summary"]})
        else:
            st.info("Nothing run yet.")
    if state.truth:
        with tabs[4]:
            st.markdown(state.truth["description"])
            st.json({"sources": state.truth["sources"], "bad_channels": state.truth["bad_channels"]})
            cmp = pl.compare_to_truth(state)
            if cmp:
                st.dataframe(cmp["errors"], use_container_width=True, hide_index=True)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        for th in msg.get("thinking") or []:
            with st.expander("💭 reasoning", expanded=False):
                st.markdown(th)
        for ev in msg.get("events") or []:
            icon = "⚠️" if ev["is_error"] else "🔧"
            args = ", ".join(f"{k}={v!r}" for k, v in ev["arguments"].items())
            with st.expander(f"{icon} {ev['name']}({args})", expanded=False):
                try:
                    st.json(json.loads(ev["result"]), expanded=False)
                except Exception:
                    st.code(ev["result"])
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("usage"):
            u = msg["usage"]
            st.caption(f"{u.get('input_tokens', 0):,} in · {u.get('output_tokens', 0):,} out tokens")


def chat_panel(state: AnalysisState, cfg: dict) -> None:
    ss = st.session_state
    st.markdown("#### AI analyst")
    if not state.has_raw:
        st.info("Load a recording first - the synthetic demo is a one-click start.")
        return
    if not ss.chat:
        st.caption("Try asking:")
        for s in SUGGESTIONS:
            if st.button(s, key=f"sugg-{s}", use_container_width=True):
                ss.pending_prompt = s
                st.rerun()
    for msg in ss.chat:
        render_message(msg)

    prompt = st.chat_input("Ask about the recording...") or ss.pending_prompt
    ss.pending_prompt = None
    if not prompt:
        return
    ss.chat.append({"role": "user", "content": prompt})
    render_message(ss.chat[-1])
    with st.chat_message("assistant"):
        status = st.status("Analysing...", expanded=True)

        def on_event(ev):
            status.write(f"{'⚠️' if ev.is_error else '🔧'} `{ev.name}` {to_json(ev.arguments)}")

        try:
            agent = _get_agent(cfg["provider"], cfg["model"], cfg["api_key"], cfg["base_url"])
            agent.on_event = on_event
            turn = agent.chat(prompt)
            status.update(label=f"Done - {len(turn.events)} tool call(s)", state="complete", expanded=False)
            ss.chat.append({"role": "assistant", "content": turn.text, "thinking": turn.thinking,
                            "usage": turn.usage,
                            "events": [{"name": e.name, "arguments": e.arguments, "result": e.result,
                                        "is_error": e.is_error} for e in turn.events]})
        except Exception as exc:
            status.update(label="Failed", state="error")
            ss.chat.append({"role": "assistant", "content": f"**Error:** {exc}"})
    st.rerun()


# --------------------------------------------------------------------------- #
def main() -> None:
    _init()
    cfg = sidebar()
    state: AnalysisState = st.session_state.state
    left, right = st.columns([3, 2], gap="large")
    with left:
        viewer_panel(state)
    with right:
        chat_panel(state, cfg)


if __name__ == "__main__" or True:  # streamlit runs the module directly
    main()

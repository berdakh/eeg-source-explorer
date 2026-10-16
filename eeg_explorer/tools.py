"""The analysis tools exposed to the AI agent.

Each tool is a plain function of ``(state, **kwargs)`` returning a JSON-able
dict, plus a JSON schema describing its inputs. The agent backends
(``agent.py``) turn these into provider-specific tool definitions, so the same
tools work with Claude and with open-weight models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import pipeline as pl
from .pipeline import AnalysisState


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [],
            "additionalProperties": False}


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def describe_recording(state: AnalysisState) -> dict[str, Any]:
    out = pl.describe_recording(state)
    if state.truth:
        out["demo_scenario"] = state.truth["description"]
    return out


def preprocess_recording(state: AnalysisState, l_freq: float | None = 1.0, h_freq: float | None = 40.0,
                         notch_hz: float | None = None, detect_bads: bool = True,
                         interpolate: bool = True, reference: str = "average") -> dict[str, Any]:
    return pl.preprocess(state, l_freq=l_freq, h_freq=h_freq, notch_hz=notch_hz,
                         detect_bads=detect_bads, interpolate=interpolate, reference=reference)


def epoch_recording(state: AnalysisState, tmin: float = -0.2, tmax: float = 0.8,
                    reject_uv: float | None = 150.0, event_codes: list[int] | None = None) -> dict[str, Any]:
    return pl.make_epochs(state, tmin=tmin, tmax=tmax, reject_uv=reject_uv, event_codes=event_codes)


def localize_sources(state: AnalysisState, method: str = "eLORETA", snr: float = 3.0,
                     head_model: str | None = None, tmin: float | None = None,
                     tmax: float | None = None) -> dict[str, Any]:
    return pl.localize(state, method=method, snr=snr, head_model=head_model, tmin=tmin, tmax=tmax)


def find_peak_sources(state: AnalysisState, tmin: float | None = None, tmax: float | None = None,
                      n: int = 5) -> dict[str, Any]:
    return pl.peak_sources(state, tmin=tmin, tmax=tmax, n=n)


def compute_band_power(state: AnalysisState) -> dict[str, Any]:
    return pl.band_power(state)


def set_viewer(state: AnalysisState, time_s: float | None = None, window_start_s: float | None = None,
               window_end_s: float | None = None) -> dict[str, Any]:
    if state.stc is None:
        raise ValueError("Nothing to show yet - run localize_sources first.")
    if window_start_s is not None and window_end_s is not None:
        state.viewer_window, state.viewer_time = (float(window_start_s), float(window_end_s)), None
        return {"viewer": f"mean activity {window_start_s}-{window_end_s} s"}
    if time_s is None:
        raise ValueError("Provide time_s or a window.")
    state.viewer_time, state.viewer_window = float(time_s), None
    return {"viewer": f"activity at {time_s} s"}


def compare_with_ground_truth(state: AnalysisState) -> dict[str, Any]:
    if state.truth is None:
        return {"available": False, "reason": "This is a real recording - the true sources are unknown."}
    result = pl.compare_to_truth(state)
    if result is None:
        raise ValueError("Run localize_sources first.")
    return {"available": True, "true_bad_channels": state.truth["bad_channels"],
            "detected_bad_channels": state.bads_detected, **result}


def get_analysis_log(state: AnalysisState) -> dict[str, Any]:
    return {"steps": state.log}


def generate_report(state: AnalysisState, findings: str, title: str | None = None,
                    author: str | None = None) -> dict[str, Any]:
    from .report import build_report  # heavy imports (reportlab, matplotlib) only when needed

    info = build_report(state, findings=findings, title=title, author=author)
    return {**info, "note": "The user can download the PDF from the Reports tab / chat."}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TOOLS: list[ToolSpec] = [
    ToolSpec(
        "describe_recording",
        "Summarise the loaded EEG recording: channels, sampling rate, duration, filter settings, "
        "events, amplitude statistics and which analysis steps have already been run. "
        "Call this first whenever you are unsure about the current state.",
        _schema({}), describe_recording),
    ToolSpec(
        "preprocess_recording",
        "Band-pass filter, detect and interpolate bad channels, and re-reference the recording. "
        "Modifies the data in place and invalidates earlier epochs/source estimates. "
        "Typical evoked-response settings: l_freq=1, h_freq=40. Notch at 50 or 60 Hz removes line noise.",
        _schema({
            "l_freq": {"type": ["number", "null"], "description": "High-pass cutoff in Hz (null = none)"},
            "h_freq": {"type": ["number", "null"], "description": "Low-pass cutoff in Hz (null = none)"},
            "notch_hz": {"type": ["number", "null"], "description": "Line-noise frequency to notch (50/60) or null"},
            "detect_bads": {"type": "boolean", "description": "Run automatic bad-channel detection"},
            "interpolate": {"type": "boolean", "description": "Interpolate the bad channels"},
            "reference": {"type": "string", "description": "'average' or comma-separated channel names"},
        }), preprocess_recording),
    ToolSpec(
        "epoch_recording",
        "Cut epochs around the stimulus events (from the stim channel or annotations), reject noisy "
        "epochs above reject_uv peak-to-peak and average them into an evoked response. Required before "
        "localising an evoked response; skip it for continuous/resting data.",
        _schema({
            "tmin": {"type": "number", "description": "Epoch start relative to event (s), e.g. -0.2"},
            "tmax": {"type": "number", "description": "Epoch end relative to event (s), e.g. 0.8"},
            "reject_uv": {"type": ["number", "null"], "description": "Peak-to-peak rejection threshold in µV"},
            "event_codes": {"type": ["array", "null"], "items": {"type": "integer"},
                            "description": "Only use these event codes (null = all)"},
        }), epoch_recording),
    ToolSpec(
        "localize_sources",
        "Build the head model and forward solution if needed, then compute a distributed inverse "
        "solution for the evoked response (or for a [tmin, tmax] window of continuous data when there "
        "are no epochs). Methods: eLORETA (best localisation, default), sLORETA, dSPM (noise-normalised, "
        "biased towards deep sources), MNE (raw current amplitude). head_model 'sphere' is instant and "
        "offline (~2 cm accuracy); 'fsaverage' uses a template cortex with anatomical labels but needs a "
        "~300 MB download on first use.",
        _schema({
            "method": {"type": "string", "enum": list(pl.INVERSE_METHODS)},
            "snr": {"type": "number", "description": "Assumed SNR for regularisation (3 for evoked, 1 for raw)"},
            "head_model": {"type": ["string", "null"], "enum": [*pl.HEAD_MODELS, None]},
            "tmin": {"type": ["number", "null"], "description": "Crop/analysis window start (s)"},
            "tmax": {"type": ["number", "null"], "description": "Crop/analysis window end (s)"},
        }), localize_sources),
    ToolSpec(
        "find_peak_sources",
        "Rank the strongest, spatially separated sources over a time window of the current source "
        "estimate, with coordinates (mm) and a region label. Also points the 3D viewer at that window.",
        _schema({
            "tmin": {"type": ["number", "null"], "description": "Window start (s); null = whole estimate"},
            "tmax": {"type": ["number", "null"], "description": "Window end (s)"},
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
        }), find_peak_sources),
    ToolSpec(
        "compute_band_power",
        "Welch power spectrum of the (preprocessed) recording with relative power per canonical band "
        "(delta, theta, alpha, beta, gamma), the strongest channel per band and the alpha peak frequency.",
        _schema({}), compute_band_power),
    ToolSpec(
        "set_viewer",
        "Point the 3D source viewer at a specific time (time_s) or at the mean over a window "
        "(window_start_s, window_end_s) so the user sees what you are describing.",
        _schema({
            "time_s": {"type": ["number", "null"]},
            "window_start_s": {"type": ["number", "null"]},
            "window_end_s": {"type": ["number", "null"]},
        }), set_viewer),
    ToolSpec(
        "compare_with_ground_truth",
        "For the synthetic demo recording only: report the localisation error (mm) of each simulated "
        "source and whether the planted bad channel was detected. Returns available=false for real data.",
        _schema({}), compare_with_ground_truth),
    ToolSpec(
        "get_analysis_log",
        "Return every analysis step run so far with its parameters, results and run time.",
        _schema({}), get_analysis_log),
    ToolSpec(
        "generate_report",
        "Write a PDF report of the analysis so far: your narrative of the findings, recording and processing "
        "details, source-localisation figures (projections, time courses, sensor data, topography), peak "
        "table, spectra, ground-truth validation (demo only) and the full analysis log. Run the analysis "
        "steps first; then write `findings` as a complete, well-structured summary (Markdown headings, "
        "bullets and **bold** are supported) with concrete numbers, method choices and caveats - it is the "
        "part of the report the reader will actually read.",
        _schema({
            "findings": {"type": "string", "description": "Narrative summary of the findings (Markdown)"},
            "title": {"type": ["string", "null"], "description": "Report title (null = default)"},
            "author": {"type": ["string", "null"], "description": "Author / lab name to print (null = none)"},
        }, required=["findings"]), generate_report),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


def run_tool(state: AnalysisState, name: str, arguments: dict[str, Any] | None) -> tuple[str, bool]:
    """Execute a tool by name. Returns ``(json_result, is_error)``."""
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return to_json({"error": f"Unknown tool '{name}'"}), True
    args = dict(arguments or {})
    allowed = set(spec.input_schema["properties"])
    unknown = set(args) - allowed
    if unknown:
        return to_json({"error": f"Unknown arguments {sorted(unknown)}; allowed: {sorted(allowed)}"}), True
    try:
        return to_json(spec.fn(state, **args)), False
    except Exception as exc:  # the model should see the failure and adapt
        return to_json({"error": f"{type(exc).__name__}: {exc}"}), True

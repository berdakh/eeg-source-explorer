"""Analysis pipeline: preprocessing, head modelling, inverse solutions.

Everything here is plain MNE-Python with no UI or LLM dependencies, so it can
be unit-tested and driven either from the Streamlit app or from the agent's
tools. The ``AnalysisState`` object carries the data through the pipeline and
keeps a log of every step that was run (the agent reads this log).

Two head models are supported:

* ``sphere`` - a 3-layer spherical conductor fitted to the electrode positions
  with a regular volume source grid. Fully offline, builds in ~1 s, and is
  accurate to a couple of centimetres for superficial sources. This is the
  default.
* ``fsaverage`` - MNE's template cortical surface with a 3-layer BEM. More
  anatomically meaningful (sources on the cortex, labelled by Desikan-Killiany
  parcellation) but requires downloading ~300 MB on first use.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

HEAD_MODELS = ("sphere", "fsaverage")


def _pick_montage_name() -> str:
    # MNE 1.13 renamed the colin27-based template; keep working on older versions.
    names = mne.channels.get_builtin_montages()
    return "colin27_1005" if "colin27_1005" in names else "standard_1005"


MONTAGE_NAME = _pick_montage_name()
INVERSE_METHODS = ("eLORETA", "sLORETA", "dSPM", "MNE")

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisState:
    """Everything the pipeline knows about the current recording."""

    raw: mne.io.BaseRaw | None = None
    source_name: str = ""
    truth: dict[str, Any] | None = None  # ground truth for the synthetic demo

    head_model: str = "sphere"
    sphere: Any = None
    src: Any = None
    fwd: Any = None
    subjects_dir: Path | None = None
    labels: list | None = None  # fsaverage parcellation labels

    events: np.ndarray | None = None
    event_id: dict[str, int] | None = None
    epochs: mne.Epochs | None = None
    evoked: mne.Evoked | None = None
    cov: Any = None
    inv: Any = None
    stc: Any = None
    method: str | None = None
    lambda2: float | None = None

    bads_detected: list[str] = field(default_factory=list)
    psd: dict[str, Any] | None = None

    viewer_time: float | None = None
    viewer_window: tuple[float, float] | None = None

    log: list[dict[str, Any]] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)  # generated PDF reports

    # -- helpers ---------------------------------------------------------- #
    def record(self, step: str, params: dict[str, Any], summary: dict[str, Any], t0: float) -> dict:
        entry = {
            "step": step,
            "params": params,
            "summary": summary,
            "elapsed_s": round(time.time() - t0, 2),
        }
        self.log.append(entry)
        return entry

    def reset_derived(self) -> None:
        """Drop everything derived from the raw data (after reloading)."""
        for name in ("sphere", "src", "fwd", "labels", "events", "event_id", "epochs",
                     "evoked", "cov", "inv", "stc", "method", "lambda2", "psd",
                     "viewer_time", "viewer_window"):
            setattr(self, name, None)
        self.bads_detected = []
        self.log = []
        self.reports = []

    @property
    def has_raw(self) -> bool:
        return self.raw is not None

    @property
    def has_stc(self) -> bool:
        return self.stc is not None


# --------------------------------------------------------------------------- #
# Montage / channel helpers
# --------------------------------------------------------------------------- #
def ensure_montage(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Make sure the EEG channels have 3D positions.

    Files from many systems carry channel names but no digitised positions.
    We match names against the standard 10-05 template (case-insensitively)
    and drop EEG channels we cannot place, since the forward model needs
    positions for every channel.
    """
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    template = {name.lower(): name for name in montage.ch_names}

    renames = {}
    for ch in raw.ch_names:
        key = ch.lower().replace("eeg ", "").replace("eeg", "").strip()
        if key in template and template[key] != ch:
            renames[ch] = template[key]
    if renames:
        raw.rename_channels(renames)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=())
    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    unplaceable = [ch for ch in eeg_names if ch.lower() not in template]
    if unplaceable:
        raw.drop_channels(unplaceable)

    # Only (re)apply the template montage if the file carried no positions.
    has_positions = raw.info.get("dig") is not None and any(
        np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3] != 0)
        for ch in raw.info["chs"] if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
    )
    if not has_positions:
        raw.set_montage(montage, on_missing="ignore")

    return {"renamed": renames, "dropped_unplaceable": unplaceable,
            "used_template_positions": not has_positions}


def describe_recording(state: AnalysisState) -> dict[str, Any]:
    """Compact description of the loaded recording for humans and the agent."""
    if state.raw is None:
        return {"loaded": False}
    raw = state.raw
    info = raw.info
    eeg = mne.pick_types(info, eeg=True, exclude=())
    n_events = 0
    event_codes: list[int] = []
    try:
        events, _ = find_events(raw)
        n_events = len(events)
        event_codes = sorted({int(c) for c in events[:, 2]})
    except Exception:
        pass
    data = raw.get_data(picks=eeg, start=0, stop=min(len(raw.times), int(10 * info["sfreq"])))
    return {
        "loaded": True,
        "source": state.source_name,
        "n_eeg_channels": int(len(eeg)),
        "channel_names": [raw.ch_names[i] for i in eeg],
        "sfreq_hz": float(info["sfreq"]),
        "duration_s": round(float(raw.times[-1]), 2),
        "highpass_hz": float(info["highpass"]),
        "lowpass_hz": float(info["lowpass"]),
        "bad_channels": list(info["bads"]),
        "n_events": int(n_events),
        "event_codes": event_codes,
        "first_10s_amplitude_uv": {
            "median_std": round(float(np.median(data.std(axis=1)) * 1e6), 2),
            "max_std": round(float(data.std(axis=1).max() * 1e6), 2),
        },
        "is_synthetic_demo": state.truth is not None,
        "steps_run": [e["step"] for e in state.log],
    }


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def detect_bad_channels(raw: mne.io.BaseRaw, z_thresh: float = 4.0,
                        corr_frac: float = 0.4) -> dict[str, list[str]]:
    """Flag flat, excessively noisy and poorly correlated EEG channels.

    A deliberately simple, transparent heuristic so the agent can explain *why*
    a channel was flagged: robust z-score of the log variance after 1-45 Hz
    filtering, plus each channel's median correlation with the others compared
    to the typical value across channels (``corr_frac`` of the median).
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    names = [raw.ch_names[i] for i in picks]
    tmax = min(raw.times[-1], 60.0)
    seg = raw.copy().crop(tmax=tmax).pick(names)
    seg.filter(1.0, 45.0, verbose=False)
    data = seg.get_data()
    std = data.std(axis=1)

    flat = [n for n, s in zip(names, std) if s < 1e-7]
    log_var = np.log(np.maximum(std, 1e-12) ** 2)
    med = np.median(log_var)
    mad = np.median(np.abs(log_var - med)) * 1.4826 + 1e-12
    z = (log_var - med) / mad
    noisy = [n for n, zi in zip(names, z) if zi > z_thresh and n not in flat]

    corr = np.corrcoef(data)
    np.fill_diagonal(corr, np.nan)
    med_corr = np.nanmedian(np.abs(corr), axis=1)
    corr_floor = corr_frac * float(np.nanmedian(med_corr))
    uncorrelated = [n for n, c in zip(names, med_corr)
                    if c < corr_floor and n not in flat and n not in noisy]

    return {"flat": flat, "noisy": noisy, "uncorrelated": uncorrelated,
            "all": sorted(set(flat) | set(noisy) | set(uncorrelated))}


def preprocess(state: AnalysisState, l_freq: float | None = 1.0, h_freq: float | None = 40.0,
               notch_hz: float | None = None, detect_bads: bool = True,
               interpolate: bool = True, reference: str = "average") -> dict[str, Any]:
    """Filter, flag/interpolate bad channels and re-reference in place."""
    if state.raw is None:
        raise ValueError("No recording loaded.")
    t0 = time.time()
    raw = state.raw
    params = dict(l_freq=l_freq, h_freq=h_freq, notch_hz=notch_hz, detect_bads=detect_bads,
                  interpolate=interpolate, reference=reference)

    if notch_hz:
        raw.notch_filter(np.arange(notch_hz, raw.info["sfreq"] / 2 - 1, notch_hz), verbose=False)
    if l_freq or h_freq:
        raw.filter(l_freq, h_freq, verbose=False)

    bad_report: dict[str, list[str]] = {}
    if detect_bads:
        bad_report = detect_bad_channels(raw)
        state.bads_detected = bad_report["all"]
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bad_report["all"]))
    interpolated: list[str] = []
    if interpolate and raw.info["bads"]:
        interpolated = list(raw.info["bads"])
        raw.interpolate_bads(reset_bads=True, verbose=False)
    if reference == "average":
        raw.set_eeg_reference("average", projection=True, verbose=False)
    elif reference:
        raw.set_eeg_reference(reference.split(","), verbose=False)

    # Derived objects are stale now.
    for name in ("epochs", "evoked", "cov", "inv", "stc", "psd"):
        setattr(state, name, None)

    summary = {
        "bad_channels_detected": bad_report,
        "interpolated": interpolated,
        "reference": reference,
        "band_hz": [l_freq, h_freq],
    }
    state.record("preprocess", params, summary, t0)
    return summary


# --------------------------------------------------------------------------- #
# Events / epochs / evoked
# --------------------------------------------------------------------------- #
def find_events(raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict[str, int]]:
    stim = mne.pick_types(raw.info, stim=True, exclude=())
    if len(stim):
        events = mne.find_events(raw, stim_channel=raw.ch_names[stim[0]], verbose=False)
        event_id = {str(c): int(c) for c in sorted({int(c) for c in events[:, 2]})}
        return events, event_id
    if len(raw.annotations):
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        return events, event_id
    raise ValueError("No stim channel or annotations found in this recording.")


def make_epochs(state: AnalysisState, tmin: float = -0.2, tmax: float = 0.8,
                baseline: tuple[float | None, float | None] | None = (None, 0.0),
                reject_uv: float | None = 150.0,
                event_codes: list[int] | None = None) -> dict[str, Any]:
    """Cut epochs around events and average them into an evoked response."""
    if state.raw is None:
        raise ValueError("No recording loaded.")
    t0 = time.time()
    events, event_id = find_events(state.raw)
    if event_codes:
        event_id = {k: v for k, v in event_id.items() if v in event_codes}
        if not event_id:
            raise ValueError(f"None of the event codes {event_codes} exist. "
                             f"Available: {sorted(set(events[:, 2].tolist()))}")
    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None
    epochs = mne.Epochs(state.raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, reject=reject, preload=True, verbose=False,
                        picks="eeg")
    if len(epochs) == 0:
        raise ValueError("All epochs were rejected - try a higher reject_uv threshold.")
    state.events, state.event_id, state.epochs = events, event_id, epochs
    state.evoked = epochs.average()
    state.cov = state.inv = state.stc = None

    params = dict(tmin=tmin, tmax=tmax, baseline=baseline, reject_uv=reject_uv,
                  event_codes=event_codes)
    gfp = state.evoked.data.std(axis=0)
    summary = {
        "n_events_found": int(len(events)),
        "n_epochs_kept": int(len(epochs)),
        "n_epochs_rejected": int(len(events) - len(epochs)) if not event_codes else None,
        "event_id": event_id,
        "evoked_peak_time_s": round(float(state.evoked.times[int(np.argmax(gfp))]), 3),
        "evoked_peak_gfp_uv": round(float(gfp.max() * 1e6), 2),
    }
    state.record("make_epochs", params, summary, t0)
    return summary


def _evoked_from_raw(state: AnalysisState, tmin: float, tmax: float) -> mne.Evoked:
    raw = state.raw
    picks = mne.pick_types(raw.info, eeg=True)
    start, stop = raw.time_as_index([tmin, tmax])
    data = raw.get_data(picks=picks, start=start, stop=stop)
    info = mne.pick_info(raw.info, picks)
    return mne.EvokedArray(data, info, tmin=tmin, comment="raw segment")


# --------------------------------------------------------------------------- #
# Head model / forward
# --------------------------------------------------------------------------- #
def build_forward(state: AnalysisState, head_model: str = "sphere",
                  grid_mm: float = 10.0) -> dict[str, Any]:
    """Build (or reuse) the forward solution for the requested head model."""
    if state.raw is None:
        raise ValueError("No recording loaded.")
    if head_model not in HEAD_MODELS:
        raise ValueError(f"head_model must be one of {HEAD_MODELS}")
    if state.fwd is not None and state.head_model == head_model:
        return {"reused": True, "head_model": head_model,
                "n_sources": int(state.fwd["nsource"])}

    t0 = time.time()
    info = mne.pick_info(state.raw.info, mne.pick_types(state.raw.info, eeg=True))
    if head_model == "sphere":
        sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info, verbose=False)
        src = mne.setup_volume_source_space(sphere=sphere, pos=grid_mm, mindist=5.0,
                                            exclude=20.0, verbose=False)
        fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                        eeg=True, meg=False, verbose=False)
        state.sphere, state.labels = sphere, None
        extra = {"sphere_center_mm": np.round(sphere["r0"] * 1000, 1).tolist(),
                 "head_radius_mm": round(float(sphere["layers"][-1]["rad"] * 1000), 1)}
    else:
        try:
            fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
        except Exception as exc:  # network blocked, disk full, ...
            raise RuntimeError(
                "Could not fetch the fsaverage template (needs ~300 MB download). "
                f"Use head_model='sphere' instead. Underlying error: {exc}") from exc
        subjects_dir = fs_dir.parent
        src = mne.read_source_spaces(fs_dir / "bem" / "fsaverage-ico-5-src.fif", verbose=False)
        bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
        fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=bem,
                                        eeg=True, meg=False, mindist=5.0, verbose=False)
        state.sphere, state.subjects_dir = None, subjects_dir
        try:
            state.labels = mne.read_labels_from_annot("fsaverage", "aparc",
                                                      subjects_dir=subjects_dir, verbose=False)
        except Exception:
            state.labels = None
        extra = {"template": "fsaverage", "parcellation": "aparc" if state.labels else None}

    state.head_model, state.src, state.fwd = head_model, src, fwd
    state.inv = state.stc = None
    summary = {"head_model": head_model, "n_sources": int(fwd["nsource"]),
               "n_channels": int(fwd["nchan"]), **extra}
    state.record("build_forward", {"head_model": head_model, "grid_mm": grid_mm}, summary, t0)
    return summary


def source_positions(state: AnalysisState) -> np.ndarray:
    """(n_sources, 3) positions in metres, in the order of ``stc.data`` rows."""
    if state.src is None:
        raise ValueError("Forward model not built yet.")
    return np.concatenate([s["rr"][s["vertno"]] for s in state.src])


def source_mesh(state: AnalysisState) -> tuple[np.ndarray, np.ndarray] | None:
    """Cortical mesh (vertices, triangles) indexed like ``stc.data`` - fsaverage only."""
    if state.src is None or state.src.kind != "surface":
        return None
    verts, tris, offset = [], [], 0
    for s in state.src:
        remap = np.full(len(s["rr"]), -1)
        remap[s["vertno"]] = np.arange(len(s["vertno"])) + offset
        t = remap[s["use_tris"]]
        tris.append(t[(t >= 0).all(axis=1)])
        verts.append(s["rr"][s["vertno"]])
        offset += len(s["vertno"])
    return np.concatenate(verts), np.concatenate(tris)


def electrode_positions(state: AnalysisState) -> tuple[list[str], np.ndarray]:
    picks = mne.pick_types(state.raw.info, eeg=True)
    names = [state.raw.ch_names[i] for i in picks]
    pos = np.array([state.raw.info["chs"][i]["loc"][:3] for i in picks])
    return names, pos


# --------------------------------------------------------------------------- #
# Inverse
# --------------------------------------------------------------------------- #
def localize(state: AnalysisState, method: str = "eLORETA", snr: float = 3.0,
             head_model: str | None = None, tmin: float | None = None,
             tmax: float | None = None, depth: float = 0.8) -> dict[str, Any]:
    """Compute the inverse solution and a source estimate for the evoked data.

    If no epochs exist (continuous data without events) the window
    ``[tmin, tmax]`` of the raw recording is localised instead.
    """
    if state.raw is None:
        raise ValueError("No recording loaded.")
    if method not in INVERSE_METHODS:
        raise ValueError(f"method must be one of {INVERSE_METHODS}")
    t0 = time.time()
    build_forward(state, head_model or state.head_model)

    if state.evoked is not None and state.epochs is not None:
        evoked = state.evoked
        cov = mne.compute_covariance(state.epochs, tmax=0.0, method="empirical", verbose=False)
        cov = mne.cov.regularize(cov, evoked.info, eeg=0.1, verbose=False)
        data_desc = f"evoked response ({len(state.epochs)} epochs)"
    else:
        tmin = 0.0 if tmin is None else tmin
        tmax = min(state.raw.times[-1], tmin + 2.0) if tmax is None else tmax
        evoked = _evoked_from_raw(state, tmin, tmax)
        cov = mne.compute_raw_covariance(state.raw, method="empirical", verbose=False)
        cov = mne.cov.regularize(cov, evoked.info, eeg=0.1, verbose=False)
        data_desc = f"continuous segment {tmin:.2f}-{tmax:.2f} s"
        state.evoked = evoked

    loose = 1.0 if state.src.kind == "volume" else 0.2
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, state.fwd, cov, loose=loose,
                                                 depth=depth, verbose=False)
    lambda2 = 1.0 / snr**2
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=lambda2, method=method,
                                         pick_ori=None, verbose=False)
    if tmin is not None and tmax is not None and state.epochs is not None:
        stc = stc.copy().crop(tmin, tmax)

    state.cov, state.inv, state.stc = cov, inv, stc
    state.method, state.lambda2 = method, lambda2
    peak_idx, peak_time = stc.get_peak()
    pos = source_positions(state)
    peak_pos = pos[list(np.concatenate(stc.vertices)).index(peak_idx)]
    state.viewer_time = float(peak_time)
    state.viewer_window = None

    params = dict(method=method, snr=snr, head_model=state.head_model, tmin=tmin, tmax=tmax,
                  depth=depth)
    summary = {
        "data": data_desc,
        "method": method,
        "head_model": state.head_model,
        "n_sources": int(stc.data.shape[0]),
        "time_range_s": [round(float(stc.times[0]), 3), round(float(stc.times[-1]), 3)],
        "peak_time_s": round(float(peak_time), 3),
        "peak_position_mm": np.round(peak_pos * 1000, 1).tolist(),
        "peak_region": anatomical_label(state, peak_pos, peak_idx),
        "peak_value": round(float(stc.data.max() * activity_unit(method)[1]), 3),
        "value_unit": activity_unit(method)[0],
    }
    state.record("localize", params, summary, t0)
    return summary


# --------------------------------------------------------------------------- #
# Interpretation helpers
# --------------------------------------------------------------------------- #
def activity_unit(method: str | None) -> tuple[str, float]:
    """Display unit and scale factor for source-estimate values of ``method``.

    MNE and eLORETA return current amplitudes in Am (shown in nAm); dSPM and
    sLORETA are noise-normalised and unitless.
    """
    if method in ("MNE", "eLORETA"):
        return "nAm", 1e9
    return "noise-normalised (unitless)", 1.0


def display_values(state: AnalysisState, values: np.ndarray) -> np.ndarray:
    return values * activity_unit(state.method)[1]

def anatomical_label(state: AnalysisState, pos: np.ndarray, vertex: int | None = None) -> str:
    """Human-readable region name for a source position.

    fsaverage: the Desikan-Killiany label containing the vertex.
    sphere: a coarse lobe estimate from the position relative to the head centre.
    """
    if state.labels and vertex is not None and state.stc is not None:
        n_lh = len(state.stc.vertices[0])
        all_verts = np.concatenate(state.stc.vertices)
        row = list(all_verts).index(vertex) if vertex in all_verts else None
        hemi = "lh" if row is not None and row < n_lh else "rh"
        for label in state.labels:
            if label.hemi == hemi and vertex in label.vertices:
                return label.name.replace("-lh", " (left)").replace("-rh", " (right)")
    # Coarse lobe estimate: position relative to the head centre, in units of
    # the head radius so it works for any head size.
    if state.sphere is not None:
        center = np.asarray(state.sphere["r0"])
        radius = float(state.sphere["layers"][-1]["rad"])
    else:
        center, radius = np.zeros(3), 0.09
    x, y, z = (pos - center) / radius
    r = float(np.linalg.norm([x, y, z]))
    elevation = z / r if r > 0 else 1.0
    hemi = "left" if x < -0.1 else "right" if x > 0.1 else "midline"
    if y > 0.3:
        lobe = "frontal"
    elif y < -0.45:
        lobe = "occipital"
    elif abs(x) > 0.5 and elevation < 0.35:
        lobe = "temporal"
    elif y > 0.0:
        lobe = "fronto-central"
    else:
        lobe = "parietal" if elevation > 0.3 else "temporo-parietal"
    depth = "deep" if r < 0.55 else "superficial"
    return f"{hemi} {lobe} ({depth})"


def peak_sources(state: AnalysisState, tmin: float | None = None, tmax: float | None = None,
                 n: int = 5, min_separation_mm: float = 25.0,
                 update_viewer: bool = True) -> dict[str, Any]:
    """Top-n spatially separated sources (mean activity over a time window)."""
    if state.stc is None:
        raise ValueError("Run localize() first.")
    stc = state.stc
    tmin = stc.times[0] if tmin is None else tmin
    tmax = stc.times[-1] if tmax is None else tmax
    mask = (stc.times >= tmin) & (stc.times <= tmax)
    if not mask.any():
        raise ValueError(f"Window {tmin}-{tmax} s is outside {stc.times[0]:.2f}-{stc.times[-1]:.2f} s")
    activity = np.abs(stc.data[:, mask]).mean(axis=1)
    pos = source_positions(state)
    verts = np.concatenate(stc.vertices)

    chosen: list[int] = []
    for idx in np.argsort(activity)[::-1]:
        if all(np.linalg.norm(pos[idx] - pos[j]) * 1000 >= min_separation_mm for j in chosen):
            chosen.append(int(idx))
        if len(chosen) == n:
            break

    unit, scale = activity_unit(state.method)
    peaks = []
    for rank, idx in enumerate(chosen, 1):
        tc = np.abs(stc.data[idx, mask])
        peaks.append({
            "rank": rank,
            "source_index": idx,
            "position_mm": np.round(pos[idx] * 1000, 1).tolist(),
            "region": anatomical_label(state, pos[idx], int(verts[idx])),
            "mean_activity": round(float(activity[idx] * scale), 3),
            "relative_to_max": round(float(activity[idx] / activity.max()), 3),
            "peak_time_s": round(float(stc.times[mask][int(np.argmax(tc))]), 3),
        })
    if update_viewer:
        state.viewer_window = (float(tmin), float(tmax))
        state.viewer_time = None
    return {"window_s": [round(float(tmin), 3), round(float(tmax), 3)], "method": state.method,
            "activity_unit": unit, "head_model": state.head_model, "peaks": peaks}


def band_power(state: AnalysisState, bands: dict[str, tuple[float, float]] | None = None,
               fmin: float = 1.0, fmax: float = 45.0) -> dict[str, Any]:
    """Sensor-level spectral power per frequency band (Welch PSD)."""
    if state.raw is None:
        raise ValueError("No recording loaded.")
    t0 = time.time()
    bands = bands or DEFAULT_BANDS
    raw = state.raw
    picks = mne.pick_types(raw.info, eeg=True)
    psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, picks=picks,
                          n_fft=int(2 * raw.info["sfreq"]), verbose=False)
    freqs, power = psd.freqs, psd.get_data()  # V^2/Hz
    names = [raw.ch_names[i] for i in picks]
    total = power.sum(axis=1)

    result = {}
    for band, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        bp = power[:, m].sum(axis=1)
        top = int(np.argmax(bp))
        result[band] = {
            "range_hz": [lo, hi],
            "mean_power_db": round(float(10 * np.log10(bp.mean() + 1e-30)), 2),
            "relative_power": round(float(bp.sum() / total.sum()), 3),
            "strongest_channel": names[top],
            "strongest_channel_relative": round(float(bp[top] / total[top]), 3),
        }
    mean_psd = power.mean(axis=0)
    alpha = (freqs >= 7) & (freqs <= 14)
    alpha_peak = float(freqs[alpha][int(np.argmax(mean_psd[alpha]))])
    state.psd = {"freqs": freqs, "power": power, "ch_names": names, "bands": result}
    summary = {"bands": result, "alpha_peak_frequency_hz": round(alpha_peak, 2),
               "channels": len(names)}
    state.record("band_power", {"bands": bands, "fmin": fmin, "fmax": fmax}, summary, t0)
    return summary


def source_time_course(state: AnalysisState, source_index: int) -> tuple[np.ndarray, np.ndarray]:
    if state.stc is None:
        raise ValueError("Run localize() first.")
    return state.stc.times, state.stc.data[source_index]


def compare_to_truth(state: AnalysisState) -> dict[str, Any] | None:
    """For the synthetic demo: localisation error of each simulated source."""
    if state.truth is None or state.stc is None:
        return None
    pos = source_positions(state)
    out = []
    for src in state.truth["sources"]:
        true_pos = np.array(src["position_mm"]) / 1000
        t_idx = np.argmin(np.abs(state.stc.times - src["peak_time_s"]))
        lo, hi = max(0, t_idx - 5), min(len(state.stc.times), t_idx + 6)
        act = np.abs(state.stc.data[:, lo:hi]).mean(axis=1)
        est = pos[int(np.argmax(act))]
        entry = {"source": src["name"], "kind": src.get("kind", "evoked"),
                 "true_position_mm": src["position_mm"],
                 "estimated_position_mm": np.round(est * 1000, 1).tolist(),
                 "error_mm": round(float(np.linalg.norm(est - true_pos) * 1000), 1)}
        if entry["kind"] == "induced" and state.epochs is not None:
            entry["note"] = ("Induced (non-phase-locked) activity largely cancels in the evoked average, "
                             "so a large error here is expected; it is recovered by band power / "
                             "single-trial analyses, not by evoked-response localisation.")
        out.append(entry)
    return {"method": state.method, "errors": out}

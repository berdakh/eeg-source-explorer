"""Plotly figures for the app.

Colour roles follow one small palette: a single blue ramp for magnitude
(source activity, power), a fixed categorical order for the few multi-series
charts, and muted greys for chrome so the data stands out.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
import plotly.graph_objects as go

from .pipeline import (AnalysisState, activity_unit, display_values,
                       electrode_positions, source_mesh, source_positions)

# Sequential blue ramp (steps 100 -> 700) for magnitude.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_SCALE = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]
# Fixed categorical order - never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
ACCENT = SERIES[1]  # orange: the one highlight colour (peaks, cursors)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, title: str | None = None, height: int = 360, **kw: Any) -> go.Figure:
    layout = dict(
        title=dict(text=title, font=dict(size=14, color=INK["secondary"])) if title else None,
        font=dict(family=FONT, size=12, color=INK["muted"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=16, t=40 if title else 16, b=40), height=height,
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor=INK["grid"], gridwidth=1, zeroline=False,
                     linecolor=INK["axis"], ticks="outside", tickcolor=INK["axis"])
    fig.update_yaxes(showgrid=True, gridcolor=INK["grid"], gridwidth=1, zeroline=False,
                     linecolor=INK["axis"], ticks="outside", tickcolor=INK["axis"])
    return fig


# --------------------------------------------------------------------------- #
# 3D source map
# --------------------------------------------------------------------------- #
def _activity_at(state: AnalysisState, time: float | None,
                 window: tuple[float, float] | None) -> tuple[np.ndarray, str]:
    stc = state.stc
    if window is not None:
        mask = (stc.times >= window[0]) & (stc.times <= window[1])
        if not mask.any():
            mask[np.argmin(np.abs(stc.times - window[0]))] = True
        values = np.abs(stc.data[:, mask]).mean(axis=1)
        return display_values(state, values), f"mean {window[0]*1000:.0f}-{window[1]*1000:.0f} ms"
    t = stc.times[np.argmax(np.abs(stc.data).mean(axis=0))] if time is None else time
    idx = int(np.argmin(np.abs(stc.times - t)))
    return display_values(state, np.abs(stc.data[:, idx])), f"t = {stc.times[idx]*1000:.0f} ms"


def _fibonacci_sphere(n: int = 400) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def source_map_figure(state: AnalysisState, time: float | None = None,
                      window: tuple[float, float] | None = None, threshold: float = 0.35,
                      show_electrode_labels: bool = True, height: int = 560) -> go.Figure:
    """3D view of estimated source activity with the electrodes overlaid."""
    fig = go.Figure()
    if state.raw is None:
        return _layout(fig, "Load a recording to begin", height=height)

    names, elec = electrode_positions(state)
    # Electrodes are in head coordinates; put them in the source frame.
    if state.fwd is not None:
        elec = mne.transforms.apply_trans(mne.transforms.invert_transform(state.fwd["mri_head_t"]), elec)
    elec_mm = elec * 1000

    if state.stc is None:
        fig.add_trace(go.Scatter3d(x=elec_mm[:, 0], y=elec_mm[:, 1], z=elec_mm[:, 2], mode="markers+text",
                                   text=names, textposition="top center",
                                   textfont=dict(size=9, color=INK["muted"]),
                                   marker=dict(size=5, color=SERIES[0]), name="electrodes",
                                   hovertemplate="%{text}<extra></extra>"))
        title = "Sensor layout - run source localisation to see activity"
    else:
        values, when = _activity_at(state, time, window)
        vmax = float(values.max()) or 1.0
        pos_mm = source_positions(state) * 1000
        mesh = source_mesh(state)
        if mesh is not None:  # cortical surface (fsaverage)
            verts, tris = mesh
            v = verts * 1000
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                                    intensity=values, colorscale=SEQ_SCALE, cmin=0, cmax=vmax,
                                    showscale=True, colorbar=_colorbar(state), name="cortex",
                                    lighting=dict(ambient=0.55, diffuse=0.6, specular=0.1, roughness=0.9),
                                    lightposition=dict(x=200, y=200, z=400), hoverinfo="skip"))
        else:  # volume grid (sphere model)
            active = values >= threshold * vmax
            fig.add_trace(go.Scatter3d(x=pos_mm[~active, 0], y=pos_mm[~active, 1], z=pos_mm[~active, 2],
                                       mode="markers", marker=dict(size=1.6, color=INK["grid"]),
                                       name="source grid", hoverinfo="skip"))
            size = 4 + 9 * (values[active] / vmax)
            fig.add_trace(go.Scatter3d(
                x=pos_mm[active, 0], y=pos_mm[active, 1], z=pos_mm[active, 2], mode="markers",
                marker=dict(size=size, color=values[active], colorscale=SEQ_SCALE, cmin=0, cmax=vmax,
                            opacity=0.9, showscale=True, colorbar=_colorbar(state)),
                name="active sources", customdata=values[active],
                hovertemplate="(%{x:.0f}, %{y:.0f}, %{z:.0f}) mm<br>activity %{customdata:.2f}<extra></extra>"))
            if state.sphere is not None:
                r = state.sphere["layers"][-1]["rad"] * 1000
                sph = _fibonacci_sphere() * r + np.asarray(state.sphere["r0"]) * 1000
                fig.add_trace(go.Mesh3d(x=sph[:, 0], y=sph[:, 1], z=sph[:, 2], alphahull=0, opacity=0.06,
                                        color=INK["muted"], name="scalp", hoverinfo="skip"))
        peak = int(np.argmax(values))
        fig.add_trace(go.Scatter3d(x=[pos_mm[peak, 0]], y=[pos_mm[peak, 1]], z=[pos_mm[peak, 2]],
                                   mode="markers", marker=dict(size=16, color=ACCENT, symbol="circle-open",
                                                               line=dict(width=3, color=ACCENT)),
                                   name="peak", hovertemplate="peak<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=elec_mm[:, 0], y=elec_mm[:, 1], z=elec_mm[:, 2],
                                   mode="markers+text" if show_electrode_labels else "markers",
                                   text=names, textposition="top center",
                                   textfont=dict(size=8, color=INK["muted"]),
                                   marker=dict(size=3, color=INK["secondary"]), name="electrodes",
                                   hovertemplate="%{text}<extra></extra>"))
        title = f"{state.method} source activity - {when}"

    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False,
                title="", showspikes=False)
    _layout(fig, title, height=height,
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                       camera=dict(eye=dict(x=-1.3, y=-1.6, z=1.0), up=dict(x=0, y=0, z=1))),
            legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)),
            margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _colorbar(state: AnalysisState) -> dict:
    unit = activity_unit(state.method)[0].replace(" (unitless)", "")
    return dict(title=dict(text=unit, font=dict(size=11)), thickness=10, len=0.5, x=1.0,
                tickfont=dict(size=10), exponentformat="none")


# --------------------------------------------------------------------------- #
# Sensor-space
# --------------------------------------------------------------------------- #
def sensor_butterfly_figure(evoked: mne.Evoked, marker_time: float | None = None,
                            height: int = 300) -> go.Figure:
    """All channels overlaid (thin, grey) with the global field power on top."""
    fig = go.Figure()
    t_ms = evoked.times * 1000
    data_uv = evoked.data * 1e6
    for name, row in zip(evoked.ch_names, data_uv):
        fig.add_trace(go.Scatter(x=t_ms, y=row, mode="lines", name=name,
                                 line=dict(width=1, color="rgba(137,135,129,0.45)"),
                                 showlegend=False, hovertemplate=f"{name}: %{{y:.2f}} µV<extra></extra>"))
    gfp = data_uv.std(axis=0)
    fig.add_trace(go.Scatter(x=t_ms, y=gfp, mode="lines", name="global field power",
                             line=dict(width=2.5, color=SERIES[0]),
                             hovertemplate="GFP %{y:.2f} µV<extra></extra>"))
    if marker_time is not None:
        fig.add_vline(x=marker_time * 1000, line=dict(color=ACCENT, width=1.5, dash="dot"))
    fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    _layout(fig, f"Sensor-space {evoked.comment or 'evoked response'} ({len(evoked.ch_names)} channels)",
            height=height, hovermode="x", showlegend=True,
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=11)))
    fig.update_xaxes(title_text="time (ms)")
    fig.update_yaxes(title_text="amplitude (µV)")
    return fig


def source_time_course_figure(state: AnalysisState, peaks: list[dict[str, Any]],
                              marker_time: float | None = None, height: int = 300) -> go.Figure:
    """Time courses of the top sources returned by ``peak_sources``."""
    fig = go.Figure()
    if state.stc is None:
        return _layout(fig, "No source estimate yet", height=height)
    t_ms = state.stc.times * 1000
    unit = activity_unit(state.method)[0].replace(" (unitless)", "")
    for k, pk in enumerate(peaks[:4]):
        tc = display_values(state, np.abs(state.stc.data[pk["source_index"]]))
        label = f"#{pk['rank']} {pk['region']}"
        fig.add_trace(go.Scatter(x=t_ms, y=tc, mode="lines", name=label,
                                 line=dict(width=2, color=SERIES[k]),
                                 hovertemplate=f"{label}<br>%{{y:.2f}}<extra></extra>"))
    if marker_time is not None:
        fig.add_vline(x=marker_time * 1000, line=dict(color=ACCENT, width=1.5, dash="dot"))
    _layout(fig, f"Source time courses ({state.method})", height=height, hovermode="x",
            legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)))
    fig.update_xaxes(title_text="time (ms)")
    fig.update_yaxes(title_text=f"|activity| ({unit})", exponentformat="none")
    return fig


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
def psd_figure(state: AnalysisState, height: int = 300) -> go.Figure:
    fig = go.Figure()
    if state.psd is None:
        return _layout(fig, "Run band-power analysis to see the spectrum", height=height)
    freqs, power = state.psd["freqs"], state.psd["power"]
    db = 10 * np.log10(power * 1e12 + 1e-30)  # µV²/Hz
    lo, hi = np.percentile(db, [10, 90], axis=0)
    fig.add_trace(go.Scatter(x=np.r_[freqs, freqs[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="10th-90th percentile of channels", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=freqs, y=db.mean(axis=0), mode="lines", name="channel mean",
                             line=dict(width=2, color=SERIES[0]),
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    for k, (band, (f0, f1)) in enumerate(_bands(state).items()):
        fig.add_vrect(x0=f0, x1=f1, fillcolor=INK["grid"] if k % 2 else "rgba(0,0,0,0)",
                      opacity=0.35, line_width=0, annotation_text=band,
                      annotation_position="top left", annotation_font=dict(size=10, color=INK["muted"]))
    _layout(fig, "Power spectral density (Welch)", height=height, hovermode="x",
            legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)))
    fig.update_xaxes(title_text="frequency (Hz)")
    fig.update_yaxes(title_text="power (dB µV²/Hz)")
    return fig


def band_power_figure(state: AnalysisState, height: int = 260) -> go.Figure:
    fig = go.Figure()
    if state.psd is None:
        return _layout(fig, "No band power yet", height=height)
    bands = state.psd["bands"]
    names = list(bands)
    rel = [bands[b]["relative_power"] * 100 for b in names]
    strongest = [bands[b]["strongest_channel"] for b in names]
    fig.add_trace(go.Bar(x=rel, y=names, orientation="h", marker=dict(color=SERIES[0]),
                         text=[f"{v:.0f}%  ·  peak at {c}" for v, c in zip(rel, strongest)],
                         textposition="outside", textfont=dict(color=INK["secondary"], size=11),
                         customdata=strongest,
                         hovertemplate="%{y}: %{x:.1f}% of total power<br>strongest at %{customdata}<extra></extra>"))
    _layout(fig, "Relative band power (all channels)", height=height, bargap=0.35)
    fig.update_xaxes(title_text="% of total power", range=[0, max(rel) * 1.5])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def _bands(state: AnalysisState) -> dict[str, tuple[float, float]]:
    return {b: tuple(v["range_hz"]) for b, v in state.psd["bands"].items()}


# --------------------------------------------------------------------------- #
# Topomap (matplotlib - MNE's topomap is hard to beat)
# --------------------------------------------------------------------------- #
def topomap_figure(evoked: mne.Evoked, time: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
    idx = int(np.argmin(np.abs(evoked.times - time)))
    data = evoked.data[:, idx] * 1e6
    lim = float(np.abs(data).max()) or 1.0
    mne.viz.plot_topomap(data, evoked.info, axes=ax, show=False, cmap=cmap, vlim=(-lim, lim),
                         contours=4, sensors=True)
    ax.set_title(f"scalp map at {evoked.times[idx]*1000:.0f} ms (±{lim:.1f} µV)",
                 fontsize=9, color=INK["secondary"])
    fig.patch.set_alpha(0)
    return fig

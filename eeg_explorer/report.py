"""PDF report generation.

Builds a self-contained analysis report (ReportLab) from the current
``AnalysisState``: the analyst's narrative, recording and processing details,
source-localisation figures and tables, spectra, ground-truth validation for
the demo, and the full analysis log. Figures are rendered with matplotlib so
no browser or kaleido is needed.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem,  # noqa: E402
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from . import pipeline as pl  # noqa: E402
from .pipeline import AnalysisState  # noqa: E402
from .viz import INK, SEQ_BLUE, SERIES  # noqa: E402

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list("div", [SERIES[0], "#f0efec", SERIES[7]])
PAGE_W = A4[0] - 40 * mm  # usable width with 20 mm margins


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, leading=24, alignment=TA_LEFT,
                                textColor=colors.HexColor(INK["primary"]), spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, leading=13,
                                   textColor=colors.HexColor(INK["secondary"]), spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18, spaceBefore=14,
                             spaceAfter=6, textColor=colors.HexColor(INK["primary"])),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=14, spaceBefore=10,
                             spaceAfter=4, textColor=colors.HexColor(INK["secondary"])),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=13.5, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=8, leading=10.5,
                                textColor=colors.HexColor(INK["secondary"])),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=8, leading=10.5,
                                  textColor=colors.HexColor(INK["muted"]), spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8, leading=10),
    }
    return s


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_md(text: str) -> str:
    """Minimal Markdown -> ReportLab markup: bold, italics, code."""
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def markdown_flowables(text: str, s: dict[str, ParagraphStyle]) -> list:
    """Turn the analyst's Markdown-ish narrative into paragraphs, headings and bullets."""
    out: list = []
    bullets: list[str] = []

    def flush_bullets() -> None:
        if bullets:
            out.append(ListFlowable([ListItem(Paragraph(_inline_md(b), s["body"]), leftIndent=10)
                                     for b in bullets], bulletType="bullet", start="•",
                                    leftIndent=12, bulletFontSize=8))
            bullets.clear()

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            m = re.match(r"^(#{1,3})\s+(.*)", ln)
            if m:
                flush_bullets()
                out.append(Paragraph(_inline_md(m.group(2)), s["h2" if len(m.group(1)) > 1 else "h1"]))
            elif re.match(r"^\s*([-*•]|\d+[.)])\s+", ln):
                bullets.append(re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", ln))
            else:
                flush_bullets()
                out.append(Paragraph(_inline_md(ln), s["body"]))
        flush_bullets()
    return out


def _table(rows: list[list[Any]], s: dict[str, ParagraphStyle], col_widths: list[float] | None = None,
           header: bool = True) -> Table:
    data = [[Paragraph(_escape(str(c)), s["cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor(INK["grid"])),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f2ee")),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor(INK["axis"]))]
    t.setStyle(TableStyle(style))
    return t


def _fig_image(fig: plt.Figure, width: float = PAGE_W) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width, height=width * h / w)


# --------------------------------------------------------------------------- #
# Figures (matplotlib)
# --------------------------------------------------------------------------- #
def source_projections_figure(state: AnalysisState, time: float | None = None,
                              window: tuple[float, float] | None = None) -> tuple[plt.Figure, str]:
    """Axial / coronal / sagittal projections of source activity with electrodes."""
    stc = state.stc
    if window is not None:
        mask = (stc.times >= window[0]) & (stc.times <= window[1])
        values = np.abs(stc.data[:, mask]).mean(axis=1)
        when = f"mean {window[0]*1000:.0f}-{window[1]*1000:.0f} ms"
    else:
        t = time if time is not None else stc.times[int(np.argmax(np.abs(stc.data).mean(axis=0)))]
        idx = int(np.argmin(np.abs(stc.times - t)))
        values = np.abs(stc.data[:, idx])
        when = f"t = {stc.times[idx]*1000:.0f} ms"
    values = pl.display_values(state, values)
    pos = pl.source_positions(state) * 1000
    _, elec = pl.electrode_positions(state)
    elec = mne.transforms.apply_trans(mne.transforms.invert_transform(state.fwd["mri_head_t"]), elec) * 1000
    unit = pl.activity_unit(state.method)[0].replace(" (unitless)", "")
    vmax = float(values.max()) or 1.0
    order = np.argsort(values)  # draw strongest last

    views = [("axial (from above)", 0, 1), ("coronal (from front)", 0, 2), ("sagittal (from left)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (name, i, j) in zip(axes, views):
        ax.scatter(elec[:, i], elec[:, j], s=10, c=INK["secondary"], marker="^", zorder=3, linewidths=0)
        sc = ax.scatter(pos[order, i], pos[order, j], c=values[order], cmap=SEQ_CMAP, vmin=0, vmax=vmax,
                        s=6 + 40 * (values[order] / vmax) ** 2, linewidths=0, zorder=2)
        pk = int(np.argmax(values))
        ax.scatter(pos[pk, i], pos[pk, j], s=160, facecolors="none", edgecolors=SERIES[1], linewidths=1.8, zorder=4)
        ax.set_aspect("equal"); ax.set_title(name, fontsize=9, color=INK["secondary"])
        ax.set_xlabel(["x (mm, left→right)", "y (mm, back→front)", "z (mm)"][i], fontsize=7)
        ax.set_ylabel(["x", "y (mm)", "z (mm)"][j], fontsize=7)
        ax.tick_params(labelsize=7, colors=INK["muted"])
        for sp in ax.spines.values():
            sp.set_color(INK["axis"])
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(unit, fontsize=8); cb.ax.tick_params(labelsize=7)
    return fig, when


def sensor_figure(evoked: mne.Evoked, marker: float | None) -> plt.Figure:
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    t_ms = evoked.times * 1000
    data = evoked.data * 1e6
    ax.plot(t_ms, data.T, color=INK["muted"], lw=0.5, alpha=0.5)
    ax.plot(t_ms, data.std(axis=0), color=SERIES[0], lw=2, label="global field power")
    if marker is not None:
        ax.axvline(marker * 1000, color=SERIES[1], lw=1, ls=":")
    ax.axhline(0, color=INK["axis"], lw=0.8)
    ax.set_xlabel("time (ms)", fontsize=8); ax.set_ylabel("amplitude (µV)", fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=7)
    ax.set_title(f"Sensor space ({len(evoked.ch_names)} channels)", fontsize=9, color=INK["secondary"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    t = marker if marker is not None else evoked.times[int(np.argmax(data.std(axis=0)))]
    idx = int(np.argmin(np.abs(evoked.times - t)))
    lim = float(np.abs(data[:, idx]).max()) or 1.0
    mne.viz.plot_topomap(data[:, idx], evoked.info, axes=ax_t, show=False, cmap=DIV_CMAP,
                         vlim=(-lim, lim), contours=4, sensors=True)
    ax_t.set_title(f"scalp map at {evoked.times[idx]*1000:.0f} ms (±{lim:.1f} µV)", fontsize=8,
                   color=INK["secondary"])
    return fig


def time_courses_figure(state: AnalysisState, peaks: list[dict[str, Any]], marker: float | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    t_ms = state.stc.times * 1000
    unit = pl.activity_unit(state.method)[0].replace(" (unitless)", "")
    for k, pk in enumerate(peaks[:4]):
        tc = pl.display_values(state, np.abs(state.stc.data[pk["source_index"]]))
        ax.plot(t_ms, tc, color=SERIES[k], lw=1.6, label=f"#{pk['rank']} {pk['region']}")
    if marker is not None:
        ax.axvline(marker * 1000, color=SERIES[1], lw=1, ls=":")
    ax.set_xlabel("time (ms)", fontsize=8); ax.set_ylabel(f"|activity| ({unit})", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.tick_params(labelsize=7)
    ax.set_title(f"Time courses of the strongest sources ({state.method})", fontsize=9, color=INK["secondary"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig


def spectrum_figure(state: AnalysisState) -> plt.Figure:
    freqs, power = state.psd["freqs"], state.psd["power"]
    db = 10 * np.log10(power * 1e12 + 1e-30)
    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [2, 1]})
    lo, hi = np.percentile(db, [10, 90], axis=0)
    ax.fill_between(freqs, lo, hi, color=SERIES[0], alpha=0.15, lw=0, label="10th-90th pct of channels")
    ax.plot(freqs, db.mean(axis=0), color=SERIES[0], lw=1.8, label="channel mean")
    for k, (band, v) in enumerate(state.psd["bands"].items()):
        f0, f1 = v["range_hz"]
        if k % 2:
            ax.axvspan(f0, f1, color=INK["grid"], alpha=0.4, lw=0)
        ax.text((f0 + f1) / 2, ax.get_ylim()[1], band, ha="center", va="top", fontsize=7, color=INK["muted"])
    ax.set_xlabel("frequency (Hz)", fontsize=8); ax.set_ylabel("power (dB µV²/Hz)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False); ax.tick_params(labelsize=7)
    ax.set_title("Power spectral density (Welch)", fontsize=9, color=INK["secondary"])
    names = list(state.psd["bands"])
    rel = [state.psd["bands"][b]["relative_power"] * 100 for b in names]
    ax_b.barh(names[::-1], rel[::-1], color=SERIES[0], height=0.6)
    for y, (v, b) in enumerate(zip(rel[::-1], names[::-1])):
        ax_b.text(v + 1, y, f"{v:.0f}%  ({state.psd['bands'][b]['strongest_channel']})", va="center", fontsize=7)
    ax_b.set_xlim(0, max(rel) * 1.6); ax_b.set_xlabel("% of total power", fontsize=8)
    ax_b.tick_params(labelsize=7); ax_b.set_title("Relative band power", fontsize=9, color=INK["secondary"])
    for a in (ax, ax_b):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    return fig


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    return str(v)


def build_report(state: AnalysisState, findings: str = "", title: str | None = None,
                 author: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Write the PDF and return a summary dict (path, pages, size, sections)."""
    if state.raw is None:
        raise ValueError("No recording loaded - nothing to report.")
    s = _styles()
    desc = pl.describe_recording(state)
    title = title or f"EEG source imaging report - {desc['source']}"
    now = dt.datetime.now()
    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="eeg_report_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eeg_source_report_{now:%Y%m%d_%H%M%S}.pdf"

    story: list = []
    sections: list[str] = []
    marker = state.viewer_time
    window = state.viewer_window

    # -- header ------------------------------------------------------------
    story.append(Paragraph(_escape(title), s["title"]))
    sub = f"Generated {now:%Y-%m-%d %H:%M} by EEG Source Explorer (MNE-Python {mne.__version__})"
    if author:
        sub += f" · {_escape(author)}"
    if state.method:
        sub += f" · {state.method} on the {state.head_model} head model"
    story.append(Paragraph(sub, s["subtitle"]))

    # -- findings ----------------------------------------------------------
    story.append(Paragraph("Summary of findings", s["h1"]))
    if findings.strip():
        story.extend(markdown_flowables(findings, s))
    else:
        story.append(Paragraph("No analyst narrative was supplied; see the quantitative sections below.",
                               s["body"]))
    sections.append("findings")

    # -- recording ---------------------------------------------------------
    story.append(Paragraph("Recording", s["h1"]))
    rows = [["Property", "Value"],
            ["Source", desc["source"]],
            ["EEG channels", f"{desc['n_eeg_channels']}: {', '.join(desc['channel_names'])}"],
            ["Sampling rate", f"{desc['sfreq_hz']:.0f} Hz"],
            ["Duration", f"{desc['duration_s']:.1f} s"],
            ["Current filter", f"{desc['highpass_hz']:.2g}-{desc['lowpass_hz']:.3g} Hz"],
            ["Events", f"{desc['n_events']} (codes {desc['event_codes']})" if desc["n_events"] else "none"]]
    if state.truth:
        rows.append(["Note", "Synthetic recording with known sources. " + state.truth["description"]])
    story.append(_table(rows, s, [35 * mm, PAGE_W - 35 * mm]))
    sections.append("recording")

    # -- processing --------------------------------------------------------
    pre = [e for e in state.log if e["step"] == "preprocess"]
    epo = [e for e in state.log if e["step"] == "make_epochs"]
    if pre or epo:
        story.append(Paragraph("Processing", s["h1"]))
        rows = [["Step", "Parameters", "Result"]]
        for e in pre + epo:
            summ = dict(e["summary"])
            if "bad_channels_detected" in summ:
                b = summ.pop("bad_channels_detected")
                summ["bad channels"] = b.get("all") or "none"
            rows.append([e["step"], _fmt(e["params"]), _fmt(summ)])
        story.append(_table(rows, s, [25 * mm, 60 * mm, PAGE_W - 85 * mm]))
        sections.append("processing")

    # -- source localisation ----------------------------------------------
    if state.stc is not None:
        story.append(Paragraph("Source localisation", s["h1"]))
        loc = [e for e in state.log if e["step"] == "localize"][-1]
        fwd = [e for e in state.log if e["step"] == "build_forward"]
        unit = pl.activity_unit(state.method)[0]
        intro = (f"Inverse method <b>{state.method}</b> (SNR {loc['params']['snr']}, "
                 f"depth weighting {loc['params']['depth']}) on the <b>{state.head_model}</b> head model "
                 f"with {loc['summary']['n_sources']} sources, applied to the {loc['summary']['data']}. ")
        if fwd and state.head_model == "sphere":
            intro += (f"The spherical model was fitted to the electrode positions (head radius "
                      f"{fwd[-1]['summary'].get('head_radius_mm', '?')} mm). ")
        intro += (f"Global peak {loc['summary']['peak_value']} {unit} at {loc['summary']['peak_time_s']*1000:.0f} ms, "
                  f"position {loc['summary']['peak_position_mm']} mm ({loc['summary']['peak_region']}).")
        story.append(Paragraph(intro, s["body"]))

        fig, when = source_projections_figure(state, marker, window)
        story.append(_fig_image(fig))
        story.append(Paragraph(f"Figure 1. Estimated source activity, {when}. Orange ring marks the peak; "
                               f"triangles are electrodes. Distributed estimates are intrinsically blurred.",
                               s["caption"]))

        peaks = pl.peak_sources(state, *(window or (None, None)), n=5, update_viewer=False)
        rows = [["#", "Region", "Position (mm)", f"Mean activity ({peaks['activity_unit']})", "Rel.", "Peak (ms)"]]
        for p in peaks["peaks"]:
            rows.append([p["rank"], p["region"], _fmt(p["position_mm"]), _fmt(p["mean_activity"]),
                         f"{p['relative_to_max']:.2f}", f"{p['peak_time_s']*1000:.0f}"])
        story.append(KeepTogether([
            Paragraph(f"Strongest spatially separated sources, window {peaks['window_s'][0]*1000:.0f}-"
                      f"{peaks['window_s'][1]*1000:.0f} ms", s["h2"]),
            _table(rows, s, [8 * mm, 55 * mm, 40 * mm, 35 * mm, 12 * mm, PAGE_W - 150 * mm])]))
        story.append(_fig_image(time_courses_figure(state, peaks["peaks"], marker)))
        story.append(Paragraph("Figure 2. Time courses of the strongest sources.", s["caption"]))
        if state.evoked is not None:
            story.append(_fig_image(sensor_figure(state.evoked, marker)))
            story.append(Paragraph("Figure 3. Sensor-space data (all channels, global field power in blue) and scalp "
                                   "topography at the marked latency.", s["caption"]))
        sections.append("localisation")

    # -- spectrum ----------------------------------------------------------
    if state.psd is not None:
        story.append(Paragraph("Spectral analysis", s["h1"]))
        bp = [e for e in state.log if e["step"] == "band_power"][-1]["summary"]
        story.append(Paragraph(f"Welch PSD over all EEG channels. Alpha peak frequency "
                               f"{bp['alpha_peak_frequency_hz']} Hz.", s["body"]))
        story.append(_fig_image(spectrum_figure(state)))
        rows = [["Band", "Range (Hz)", "Relative power", "Strongest channel", "Rel. power there"]]
        for b, v in state.psd["bands"].items():
            rows.append([b, _fmt(v["range_hz"]), f"{v['relative_power']*100:.1f}%", v["strongest_channel"],
                         f"{v['strongest_channel_relative']*100:.1f}%"])
        story.append(_table(rows, s))
        sections.append("spectrum")

    # -- ground truth ------------------------------------------------------
    cmp = pl.compare_to_truth(state)
    if cmp:
        story.append(Paragraph("Validation against simulated ground truth", s["h1"]))
        story.append(Paragraph("This recording is synthetic, so the true generators are known. Localisation "
                               "error is the distance between each simulated source and the strongest "
                               "estimated source around its peak latency.", s["body"]))
        rows = [["Simulated source", "True position (mm)", "Estimated (mm)", "Error (mm)"]]
        for e in cmp["errors"]:
            rows.append([e["source"], _fmt(e["true_position_mm"]), _fmt(e["estimated_position_mm"]),
                         f"{e['error_mm']:.0f}" + (" *" if e.get("note") else "")])
        notes = [e["note"] for e in cmp["errors"] if e.get("note")]
        if notes:
            rows.append(["* " + notes[0], "", "", ""])
        rows.append(["Planted bad channels", _fmt(state.truth["bad_channels"]),
                     f"detected: {_fmt(state.bads_detected) or 'none'}", ""])
        story.append(_table(rows, s))
        sections.append("ground_truth")

    # -- log ----------------------------------------------------------------
    if state.log:
        story.append(PageBreak())
        story.append(Paragraph("Analysis log", s["h1"]))
        rows = [["#", "Step", "Parameters", "Time (s)"]]
        for k, e in enumerate(state.log, 1):
            rows.append([k, e["step"], _fmt(e["params"]), e["elapsed_s"]])
        story.append(_table(rows, s, [8 * mm, 28 * mm, PAGE_W - 54 * mm, 18 * mm]))
        sections.append("log")

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Methods note. Source estimates are distributed minimum-norm solutions computed with MNE-Python. "
        "The spherical head model is accurate to roughly 2 cm for superficial sources and cannot resolve deep "
        "structures; the fsaverage template is not the subject's own anatomy. Region labels for the sphere model "
        "are coarse lobe estimates. Results are for research and educational use.", s["small"]))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(INK["muted"]))
        canvas.drawString(20 * mm, 12 * mm, f"EEG Source Explorer · {now:%Y-%m-%d}")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm, title=title,
                            author=author or "EEG Source Explorer", subject="EEG source imaging report")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    info = {"path": str(path), "filename": path.name, "pages": doc.page, "size_kb": round(path.stat().st_size / 1024, 1),
            "sections": sections, "title": title, "created": now.isoformat(timespec="seconds")}
    state.reports.append(info)
    return info

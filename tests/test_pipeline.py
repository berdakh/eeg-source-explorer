import numpy as np
import pytest

from eeg_explorer import pipeline as pl
from eeg_explorer import viz
from eeg_explorer.data import make_demo_recording
from eeg_explorer.pipeline import AnalysisState


@pytest.fixture(scope="module")
def demo_state() -> AnalysisState:
    state = AnalysisState()
    make_demo_recording(state, n_trials=30, seed=1)
    return state


def test_demo_recording_shape(demo_state):
    d = pl.describe_recording(demo_state)
    assert d["n_eeg_channels"] == 32
    assert d["n_events"] == 30
    assert d["is_synthetic_demo"]


def test_preprocess_detects_planted_bad_channel(demo_state):
    summary = pl.preprocess(demo_state)
    assert summary["bad_channels_detected"]["all"] == ["T8"]
    assert summary["interpolated"] == ["T8"]
    assert demo_state.raw.info["bads"] == []  # interpolated and reset


def test_epochs_and_evoked_peak(demo_state):
    summary = pl.make_epochs(demo_state)
    assert summary["n_epochs_kept"] == 30
    assert abs(summary["evoked_peak_time_s"] - 0.30) < 0.03


@pytest.mark.parametrize("method,max_err_mm", [("eLORETA", 35), ("sLORETA", 35), ("dSPM", 60)])
def test_localize_recovers_motor_source(demo_state, method, max_err_mm):
    summary = pl.localize(demo_state, method=method)
    assert summary["head_model"] == "sphere"
    assert summary["n_sources"] > 1000
    errors = {e["source"]: e["error_mm"] for e in pl.compare_to_truth(demo_state)["errors"]}
    assert errors["left sensorimotor (evoked)"] < max_err_mm
    assert "left" in summary["peak_region"]


def test_peak_sources_are_separated(demo_state):
    pl.localize(demo_state, method="eLORETA")
    result = pl.peak_sources(demo_state, 0.25, 0.35, n=4, min_separation_mm=25)
    peaks = result["peaks"]
    assert len(peaks) == 4 and peaks[0]["relative_to_max"] == 1.0
    pos = np.array([p["position_mm"] for p in peaks])
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.linalg.norm(pos[i] - pos[j]) >= 25
    assert demo_state.viewer_window == (0.25, 0.35)


def test_band_power_alpha_is_posterior(demo_state):
    summary = pl.band_power(demo_state)
    alpha = summary["bands"]["alpha"]
    assert alpha["strongest_channel"] in {"O1", "O2", "Oz", "PO9", "PO10", "P7", "P8", "P3", "P4", "Pz"}
    assert 9 <= summary["alpha_peak_frequency_hz"] <= 11


def test_localize_raw_window_without_epochs():
    state = AnalysisState()
    make_demo_recording(state, n_trials=5, seed=2)
    state.raw.drop_channels(["STI 014"])
    pl.preprocess(state)
    summary = pl.localize(state, method="MNE", tmin=1.0, tmax=2.0)
    assert summary["data"].startswith("continuous segment")
    assert state.stc.times[0] >= 1.0 - 1e-6


def test_figures_build(demo_state):
    pl.localize(demo_state, method="eLORETA")
    pl.band_power(demo_state)
    peaks = pl.peak_sources(demo_state, 0.25, 0.35, n=3, update_viewer=False)["peaks"]
    figs = [
        viz.source_map_figure(demo_state, time=0.3),
        viz.source_map_figure(demo_state, window=(0.6, 0.9)),
        viz.sensor_butterfly_figure(demo_state.evoked, 0.3),
        viz.source_time_course_figure(demo_state, peaks, 0.3),
        viz.psd_figure(demo_state),
        viz.band_power_figure(demo_state),
    ]
    assert all(len(f.data) > 0 for f in figs)
    assert viz.topomap_figure(demo_state.evoked, 0.3).axes


def test_anatomical_label_lobes(demo_state):
    pl.build_forward(demo_state, "sphere")
    r0 = np.asarray(demo_state.sphere["r0"])
    R = demo_state.sphere["layers"][-1]["rad"]
    assert "frontal" in pl.anatomical_label(demo_state, r0 + np.array([0, 0.6 * R, 0.3 * R]))
    assert "occipital" in pl.anatomical_label(demo_state, r0 + np.array([0, -0.6 * R, 0.3 * R]))
    assert pl.anatomical_label(demo_state, r0 + np.array([-0.6 * R, 0, 0])).startswith("left temporal")
    assert "deep" in pl.anatomical_label(demo_state, r0 + np.array([0.1 * R, 0, 0.1 * R]))

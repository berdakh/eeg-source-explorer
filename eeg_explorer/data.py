"""Loading recordings and generating the synthetic demo dataset."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np

from .pipeline import MONTAGE_NAME, AnalysisState, build_forward, ensure_montage

SUPPORTED_SUFFIXES = {
    ".fif": "FIF (MNE / Neuromag)",
    ".edf": "European Data Format",
    ".bdf": "BioSemi",
    ".set": "EEGLAB",
    ".vhdr": "BrainVision",
    ".gdf": "GDF",
    ".cnt": "Neuroscan",
}

DEMO_CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF8", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]


def _read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    readers = {
        ".fif": mne.io.read_raw_fif,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".vhdr": mne.io.read_raw_brainvision,
        ".gdf": mne.io.read_raw_gdf,
        ".cnt": mne.io.read_raw_cnt,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported file type '{suffix}'. "
                         f"Supported: {', '.join(sorted(readers))}")
    return readers[suffix](path, preload=True, verbose=False)


def load_recording(state: AnalysisState, path: str | Path | None = None,
                   data: bytes | None = None, filename: str | None = None) -> dict[str, Any]:
    """Load a recording from disk or from uploaded bytes into ``state``."""
    if data is not None:
        if not filename:
            raise ValueError("filename is required when loading from bytes")
        tmp = Path(tempfile.mkdtemp()) / Path(filename).name
        tmp.write_bytes(data)
        path = tmp
    if path is None:
        raise ValueError("Provide either a path or data+filename")
    path = Path(path)
    raw = _read_raw(path)
    raw.pick(["eeg", "stim"], exclude=[])  # misc/EOG etc. are not used
    montage_report = ensure_montage(raw)
    state.reset_derived()
    state.raw, state.source_name, state.truth = raw, path.name, None
    return {"file": path.name, "format": SUPPORTED_SUFFIXES.get(path.suffix.lower(), path.suffix),
            **montage_report}


# --------------------------------------------------------------------------- #
# Synthetic demo
# --------------------------------------------------------------------------- #
def make_demo_recording(state: AnalysisState, n_trials: int = 40, sfreq: float = 250.0,
                        isi_s: float = 1.5, noise_uv: float = 3.0, seed: int = 0) -> dict[str, Any]:
    """Generate a realistic-looking 32-channel EEG recording with known sources.

    The scenario is a simple visuo-motor task: at each trigger a left
    sensorimotor source produces a ~300 ms evoked response and a right
    occipital source emits a 10 Hz alpha burst 0.5-1.0 s later. Background
    posterior alpha and pink noise are added at the sensors, and one channel
    (T8) is made noisy so the bad-channel detector has something to find.

    Because the generating sources are known, the agent can report true
    localisation error - something you can never do with real data.
    """
    rng = np.random.default_rng(seed)
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    info = mne.create_info(DEMO_CHANNELS, sfreq, ch_types="eeg")
    info.set_montage(montage)

    # Forward model (sphere) to project the sources to the sensors.
    tmp = AnalysisState()
    tmp.raw = mne.io.RawArray(np.zeros((len(DEMO_CHANNELS), 10)), info, verbose=False)
    build_forward(tmp, "sphere")
    fwd, src, sphere = tmp.fwd, tmp.src, tmp.sphere
    rr = src[0]["rr"][src[0]["vertno"]]
    r0 = np.asarray(sphere["r0"])
    gain = fwd["sol"]["data"].reshape(len(DEMO_CHANNELS), -1, 3)  # free orientation

    def nearest(offset_m: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(rr - (r0 + offset_m), axis=1)))

    def radial(idx: int) -> np.ndarray:
        v = rr[idx] - r0
        return v / np.linalg.norm(v)

    motor = nearest(np.array([-0.045, -0.015, 0.045]))     # left hand-knob area
    occipital = nearest(np.array([0.025, -0.065, 0.015]))  # right occipital
    sources = {"motor": motor, "occipital": occipital}

    n_pre = int(1.0 * sfreq)
    trial_len = int(isi_s * sfreq)
    n_samples = n_pre + n_trials * trial_len + int(2.0 * sfreq)
    times = np.arange(n_samples) / sfreq
    eeg = np.zeros((len(DEMO_CHANNELS), n_samples))
    stim = np.zeros(n_samples)

    onsets = n_pre + np.arange(n_trials) * trial_len
    t_rel = np.arange(trial_len) / sfreq
    motor_shape = 40e-9 * np.exp(-((t_rel - 0.30) ** 2) / (2 * 0.035**2))
    motor_shape -= 15e-9 * np.exp(-((t_rel - 0.45) ** 2) / (2 * 0.05**2))  # rebound
    alpha_env = 25e-9 * np.exp(-((t_rel - 0.75) ** 2) / (2 * 0.12**2))

    for onset in onsets:
        stim[onset] = 1
        jitter = rng.normal(0, 0.015)
        amp = rng.uniform(0.8, 1.2)
        shape = 40e-9 * amp * np.exp(-((t_rel - 0.30 - jitter) ** 2) / (2 * 0.035**2))
        shape -= 15e-9 * amp * np.exp(-((t_rel - 0.45 - jitter) ** 2) / (2 * 0.05**2))
        eeg[:, onset:onset + trial_len] += np.outer(gain[:, motor] @ radial(motor), shape)
        phase = rng.uniform(0, 2 * np.pi)
        burst = alpha_env * np.sin(2 * np.pi * 10.0 * t_rel + phase)
        eeg[:, onset:onset + trial_len] += np.outer(gain[:, occipital] @ radial(occipital), burst)

    # Background posterior alpha (both occipital sides) with slow amplitude modulation.
    for off in ([-0.02, -0.07, 0.015], [0.02, -0.07, 0.015]):
        idx = nearest(np.array(off))
        env = 8e-9 * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * times + rng.uniform(0, 6)))
        bg = env * np.sin(2 * np.pi * rng.uniform(9.5, 10.5) * times + rng.uniform(0, 6))
        eeg += np.outer(gain[:, idx] @ radial(idx), bg)

    def pink_noise(shape: tuple[int, int]) -> np.ndarray:
        """White noise shaped to a 1/f spectrum, unit variance per row."""
        spec = np.fft.rfft(rng.normal(0, 1, shape), axis=1)
        freqs = np.fft.rfftfreq(shape[1], 1 / sfreq)
        spec[:, 1:] /= np.sqrt(freqs[1:])
        out = np.fft.irfft(spec, n=shape[1], axis=1)
        return out / out.std(axis=1, keepdims=True)

    # "Brain noise": many weak random sources, so sensor noise is spatially
    # correlated like real EEG rather than independent per channel.
    n_bg = 60
    bg_idx = rng.choice(len(rr), n_bg, replace=False)
    bg_tc = pink_noise((n_bg, n_samples)) * 4e-9
    for k, idx in enumerate(bg_idx):
        eeg += np.outer(gain[:, idx] @ radial(idx), bg_tc[k])

    # Plus a little independent sensor/amplifier noise.
    eeg += pink_noise(eeg.shape) * noise_uv * 1e-6 * 0.3

    # One bad channel so there is something to detect.
    bad = DEMO_CHANNELS.index("T8")
    eeg[bad] += rng.normal(0, 40e-6, n_samples)

    full_info = mne.create_info(DEMO_CHANNELS + ["STI 014"], sfreq, ["eeg"] * len(DEMO_CHANNELS) + ["stim"])
    raw = mne.io.RawArray(np.vstack([eeg, stim[None]]), full_info, verbose=False)
    raw.set_montage(montage)

    state.reset_derived()
    state.raw = raw
    state.source_name = "synthetic demo (visuo-motor task)"
    state.truth = {
        "description": ("Synthetic 32-channel EEG. Each trigger evokes a left sensorimotor "
                        "response peaking ~300 ms after the stimulus, followed by a right "
                        "occipital 10 Hz alpha burst around 0.5-1.0 s. Channel T8 is noisy."),
        "n_trials": n_trials,
        "bad_channels": ["T8"],
        "sources": [
            {"name": "left sensorimotor (evoked)", "position_mm": np.round(rr[motor] * 1000, 1).tolist(),
             "peak_time_s": 0.30, "source_index": motor},
            {"name": "right occipital (10 Hz burst)", "position_mm": np.round(rr[occipital] * 1000, 1).tolist(),
             "peak_time_s": 0.75, "source_index": occipital},
        ],
    }
    return {"source": state.source_name, "n_channels": len(DEMO_CHANNELS), "sfreq_hz": sfreq,
            "duration_s": round(float(times[-1]), 1), "n_trials": n_trials,
            "truth": state.truth}

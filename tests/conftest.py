"""Pytest fixtures — synthetic ECG data, no external dataset required."""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_ecg(n_beats: int = 50, fs: int = 360, noise: float = 0.02) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (signal, r_peaks, fs) for a toy ECG-like waveform.

    Each beat is a Gaussian-shaped R spike + a small P/T-like bump. R-peaks
    sit at regular intervals so detection is trivial and feature extraction
    has plenty of context.
    """
    rng = np.random.default_rng(42)
    rr_samples = int(0.85 * fs)  # ~70 BPM
    n = rr_samples * (n_beats + 2)
    t = np.arange(n)
    signal = noise * rng.standard_normal(n).astype(np.float32)

    r_peaks: list[int] = []
    for k in range(1, n_beats + 1):
        centre = k * rr_samples
        r_peaks.append(centre)
        # R: tall narrow Gaussian
        sigma_r = 0.012 * fs
        signal += np.exp(-((t - centre) ** 2) / (2 * sigma_r ** 2)).astype(np.float32)
        # T: small wider bump after R
        sigma_t = 0.03 * fs
        signal += 0.25 * np.exp(-((t - (centre + 0.20 * fs)) ** 2) / (2 * sigma_t ** 2)).astype(np.float32)
        # P: tiny bump before R
        sigma_p = 0.02 * fs
        signal += 0.15 * np.exp(-((t - (centre - 0.15 * fs)) ** 2) / (2 * sigma_p ** 2)).astype(np.float32)

    return signal.astype(np.float32), np.asarray(r_peaks, dtype=np.int64), fs


@pytest.fixture
def synthetic_ecg() -> tuple[np.ndarray, np.ndarray, int]:
    return _synthetic_ecg()


@pytest.fixture
def synthetic_bundle() -> dict:
    """A model-ready bundle of synthetic data: windows + features + labels."""
    from ekg.features.handcrafted import extract_feature_matrix
    from ekg.features.windows import segment_beats, zscore_window

    signal, r_peaks, fs = _synthetic_ecg(n_beats=120)
    windows, kept = segment_beats(signal, r_peaks, pre=90, post=170)
    windows = zscore_window(windows.astype(np.float32))
    features = extract_feature_matrix(windows, r_peaks[kept], r_peaks, fs)

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=windows.shape[0]).astype(np.int64)
    # Tie a couple of feature dims to the label so trivial models can learn.
    features[:, 0] += labels.astype(np.float32) * 0.5
    return {
        "windows": windows,
        "features": features,
        "labels": labels,
        "r_peaks": r_peaks[kept],
    }

"""Feature extraction smoke tests."""

from __future__ import annotations

import numpy as np

from ekg.features.handcrafted import FEATURE_NAMES, extract_feature_matrix, extract_features
from ekg.features.windows import segment_beats, zscore_window


def test_segment_beats_drops_edge_beats(synthetic_ecg):
    signal, r_peaks, _ = synthetic_ecg
    windows, kept = segment_beats(signal, r_peaks, pre=90, post=170)
    assert windows.shape[1] == 260
    assert windows.shape[0] == kept.size
    assert windows.shape[0] <= r_peaks.size
    # No beats at sample 0 / signal end should survive.
    starts = r_peaks[kept] - 90
    ends = r_peaks[kept] + 170
    assert (starts >= 0).all()
    assert (ends <= signal.size).all()


def test_zscore_window_mean_zero():
    rng = np.random.default_rng(0)
    win = rng.standard_normal((20, 260)).astype(np.float32) * 3 + 5
    norm = zscore_window(win)
    assert np.allclose(norm.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(norm.std(axis=-1), 1, atol=1e-3)


def test_extract_features_shape_and_finite(synthetic_ecg):
    signal, r_peaks, fs = synthetic_ecg
    windows, kept = segment_beats(signal, r_peaks, pre=90, post=170)
    windows = zscore_window(windows.astype(np.float32))
    feats = extract_features(
        windows[0], r_peak_idx=int(r_peaks[kept[0]]),
        all_r_peaks=r_peaks, fs=fs,
    )
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_extract_feature_matrix_shape(synthetic_ecg):
    signal, r_peaks, fs = synthetic_ecg
    windows, kept = segment_beats(signal, r_peaks, pre=90, post=170)
    windows = zscore_window(windows.astype(np.float32))
    matrix = extract_feature_matrix(windows, r_peaks[kept], r_peaks, fs)
    assert matrix.shape == (windows.shape[0], len(FEATURE_NAMES))
    assert np.all(np.isfinite(matrix))

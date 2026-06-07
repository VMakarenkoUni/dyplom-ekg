"""Handcrafted per-beat feature extractor.

Features fall into four groups:
  - RR-interval features (pre-RR, post-RR, local-RR ratio, average over ±10 beats)
  - Morphology features computed directly from the beat window
    (QRS width, R amplitude, R-onset/offset slopes, area under QRS, P/T-zone energy)
  - Wavelet features — 4-level DWT (db4) of the window, mean+std+energy per band
  - Statistical features — skewness, kurtosis, zero-crossing rate, signal energy

These are the inputs to the classical models (XGBoost / RandomForest / SVM) and
to the handcrafted stream of the hybrid classifier.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import kurtosis, skew

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYWT_AVAILABLE = False
    pywt = None  # type: ignore[assignment]


_RR_NAMES = ["rr_pre", "rr_post", "rr_ratio", "rr_local_mean", "rr_local_std"]
_MORPH_NAMES = [
    "qrs_width", "r_amp", "qrs_area", "r_slope_pre", "r_slope_post",
    "p_zone_energy", "t_zone_energy", "pt_ratio",
]
_STAT_NAMES = ["mean", "std", "skew", "kurtosis", "energy", "zcr", "ptp"]
_WAVELET_NAMES = [
    f"wv{level}_{stat}"
    for level in range(5)
    for stat in ("mean", "std", "energy")
]

FEATURE_NAMES: list[str] = _RR_NAMES + _MORPH_NAMES + _STAT_NAMES + _WAVELET_NAMES


def _rr_features(
    r_peak_idx: int,
    all_r_peaks: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Five RR-interval descriptors around the i-th beat."""
    pos = int(np.searchsorted(all_r_peaks, r_peak_idx))
    rr_pre = (all_r_peaks[pos] - all_r_peaks[pos - 1]) / fs if pos >= 1 else np.nan
    rr_post = (all_r_peaks[pos + 1] - all_r_peaks[pos]) / fs if pos + 1 < len(all_r_peaks) else np.nan

    lo = max(0, pos - 10)
    hi = min(len(all_r_peaks), pos + 10)
    neighbourhood = np.diff(all_r_peaks[lo:hi]) / fs
    if neighbourhood.size:
        local_mean = float(np.mean(neighbourhood))
        local_std = float(np.std(neighbourhood))
    else:
        local_mean, local_std = np.nan, np.nan

    rr_ratio = (rr_pre / rr_post) if (rr_post and not np.isnan(rr_pre) and not np.isnan(rr_post) and rr_post > 0) else np.nan
    return np.asarray([rr_pre, rr_post, rr_ratio, local_mean, local_std], dtype=np.float32)


def _morphology_features(window: np.ndarray, pre: int, fs: int) -> np.ndarray:
    """Eight simple morphology descriptors on the windowed beat."""
    n = window.shape[0]
    r_idx = int(np.argmax(np.abs(window - np.median(window))))
    r_amp = float(window[r_idx] - np.median(window))

    qrs_half = max(1, int(0.05 * fs))  # ~50 ms either side of R
    qrs_lo = max(0, r_idx - qrs_half)
    qrs_hi = min(n, r_idx + qrs_half)
    qrs_segment = window[qrs_lo:qrs_hi]
    qrs_width = (qrs_hi - qrs_lo) / fs
    qrs_area = float(np.sum(np.abs(qrs_segment - np.median(qrs_segment))) / fs)

    pre_slope = float((window[r_idx] - window[qrs_lo]) / max(1, r_idx - qrs_lo))
    post_slope = float((window[min(qrs_hi - 1, n - 1)] - window[r_idx]) / max(1, qrs_hi - 1 - r_idx))

    p_zone_end = max(0, pre - int(0.04 * fs))
    p_segment = window[:p_zone_end]
    p_energy = float(np.sum(p_segment ** 2)) if p_segment.size else 0.0

    t_zone_start = min(n, r_idx + int(0.08 * fs))
    t_segment = window[t_zone_start:]
    t_energy = float(np.sum(t_segment ** 2)) if t_segment.size else 0.0

    pt_ratio = p_energy / t_energy if t_energy > 0 else 0.0

    return np.asarray([
        qrs_width, r_amp, qrs_area, pre_slope, post_slope,
        p_energy, t_energy, pt_ratio,
    ], dtype=np.float32)


def _statistical_features(window: np.ndarray) -> np.ndarray:
    mean = float(np.mean(window))
    std = float(np.std(window))
    sk = float(skew(window)) if std > 0 else 0.0
    kt = float(kurtosis(window)) if std > 0 else 0.0
    energy = float(np.sum(window ** 2))
    zcr = float(np.mean(np.abs(np.diff(np.sign(window - mean))) > 0))
    ptp = float(np.ptp(window))
    return np.asarray([mean, std, sk, kt, energy, zcr, ptp], dtype=np.float32)


def _wavelet_features(window: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    if not PYWT_AVAILABLE:
        return np.zeros(3 * (level + 1), dtype=np.float32)
    coeffs = pywt.wavedec(window, wavelet=wavelet, level=level)
    feats: list[float] = []
    for c in coeffs:
        c = np.asarray(c, dtype=np.float32)
        feats.extend([float(np.mean(c)), float(np.std(c)), float(np.sum(c ** 2))])
    return np.asarray(feats, dtype=np.float32)


def extract_features(
    window: np.ndarray,
    *,
    r_peak_idx: int,
    all_r_peaks: np.ndarray,
    fs: int,
    pre: int = 90,
) -> np.ndarray:
    """Per-beat feature vector. Order matches `FEATURE_NAMES`."""
    window = np.asarray(window, dtype=np.float32).ravel()
    rr = _rr_features(r_peak_idx, np.asarray(all_r_peaks, dtype=np.int64), fs)
    morph = _morphology_features(window, pre=pre, fs=fs)
    stat = _statistical_features(window)
    wave = _wavelet_features(window)
    vec = np.concatenate([rr, morph, stat, wave])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_feature_matrix(
    windows: np.ndarray,
    r_peaks: Iterable[int],
    all_r_peaks: np.ndarray,
    fs: int,
    *,
    pre: int = 90,
) -> np.ndarray:
    """Stack `extract_features` over many beats. Returns (n_beats, n_features)."""
    r_peaks = list(r_peaks)
    n = len(r_peaks)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    out = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i, rp in enumerate(r_peaks):
        out[i] = extract_features(
            windows[i], r_peak_idx=int(rp),
            all_r_peaks=all_r_peaks, fs=fs, pre=pre,
        )
    return out

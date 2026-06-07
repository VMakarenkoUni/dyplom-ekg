"""Beat-window utilities: segmentation around R-peaks and normalisation."""

from __future__ import annotations

import numpy as np


def zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-beat z-score normalisation. Operates on the last axis."""
    mean = np.mean(window, axis=-1, keepdims=True)
    std = np.std(window, axis=-1, keepdims=True)
    return (window - mean) / (std + eps)


def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    *,
    pre: int = 90,
    post: int = 170,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed-size windows around each R-peak.

    Returns (windows, kept_indices). Windows that would fall off either end
    of the signal are dropped; `kept_indices` indexes into the original
    `r_peaks` array so callers can re-align labels.
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    r_peaks = np.asarray(r_peaks, dtype=np.int64)
    n = signal.shape[0]

    keep = (r_peaks - pre >= 0) & (r_peaks + post <= n)
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size == 0:
        return np.empty((0, pre + post), dtype=np.float32), kept_idx

    starts = r_peaks[kept_idx] - pre
    offsets = np.arange(pre + post)
    windows = signal[starts[:, None] + offsets[None, :]]
    return windows.astype(np.float32), kept_idx

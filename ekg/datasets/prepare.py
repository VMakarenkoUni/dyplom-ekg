"""Build (windows, features, labels) numpy tensors from MIT-BIH records.

This is the bridge from the raw dataset into the model training loops.
For each record:
  - load the signal and annotations (1 or 2 channels)
  - segment beats around AAMI-mapped R-peaks
  - extract the handcrafted feature matrix using the same beat indices

Results are returned as plain arrays, optionally cached to disk so repeated
training runs don't re-decode the WFDB files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from ekg import AAMI_CLASSES
from ekg.datasets.mitbih import (
    DEFAULT_DATA_DIR,
    aami_label,
    load_record,
)
from ekg.features.handcrafted import extract_feature_matrix
from ekg.features.windows import segment_beats, zscore_window

logger = logging.getLogger(__name__)

LABEL_TO_INT = {c: i for i, c in enumerate(AAMI_CLASSES)}
INT_TO_LABEL = {i: c for c, i in LABEL_TO_INT.items()}


def _cache_path(cache_dir: Path, record_id: str, pre: int, post: int,
                channels_key: str) -> Path:
    return cache_dir / f"{record_id}_{pre}_{post}_{channels_key}.npz"


def _normalise_channels(channels: int | list[int]) -> list[int]:
    if isinstance(channels, int):
        return [channels]
    return list(channels)


def build_record(
    record_id: str,
    *,
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    pre: int = 90,
    post: int = 170,
    channel: int | list[int] = 0,
    cache_dir: str | os.PathLike | None = None,
    normalise: bool = True,
) -> dict:
    """Build the per-record tensor bundle.

    `channel` may be an int or a list of int channel indices. With multiple
    channels, windows have shape (n_beats, n_channels, window_len) and the
    feature matrix concatenates per-channel feature vectors.

    Returns a dict with keys:
      - windows: (n_beats, [n_channels,] window_len) float32
      - features: (n_beats, n_features * n_channels) float32
      - labels: (n_beats,) int64 — AAMI class index
      - r_peaks: (n_beats,) int64
      - fs: int
      - record_id: str
      - n_channels: int
    """
    channels = _normalise_channels(channel)
    n_ch = len(channels)
    channels_key = "_".join(str(c) for c in channels)

    cache_dir_p = Path(cache_dir) if cache_dir else None
    if cache_dir_p is not None:
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        cache = _cache_path(cache_dir_p, record_id, pre, post, channels_key)
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            return {
                "windows": data["windows"],
                "features": data["features"],
                "labels": data["labels"],
                "r_peaks": data["r_peaks"],
                "fs": int(data["fs"]),
                "record_id": record_id,
                "n_channels": n_ch,
            }

    sig_list: list[np.ndarray] = []
    fs_value: int | None = None
    ann_samples_master: np.ndarray | None = None
    ann_symbols_master: np.ndarray | None = None
    for ch in channels:
        sig, ann_samples, ann_symbols, fs = load_record(record_id, data_dir, channel=ch)
        sig_list.append(sig)
        if fs_value is None:
            fs_value = fs
            ann_samples_master = ann_samples
            ann_symbols_master = ann_symbols
    assert fs_value is not None
    assert ann_samples_master is not None
    assert ann_symbols_master is not None

    mapped: list[tuple[int, str]] = []
    for idx, sym in zip(ann_samples_master, ann_symbols_master):
        label = aami_label(str(sym))
        if label is None:
            continue
        mapped.append((int(idx), label))
    if not mapped:
        raise RuntimeError(f"record {record_id!r}: no AAMI-mapped annotations")

    r_peaks = np.asarray([m[0] for m in mapped], dtype=np.int64)
    labels_str = [m[1] for m in mapped]

    per_channel_windows: list[np.ndarray] = []
    kept_idx_master: np.ndarray | None = None
    for sig in sig_list:
        windows, kept_idx = segment_beats(sig, r_peaks, pre=pre, post=post)
        if kept_idx_master is None:
            kept_idx_master = kept_idx
        per_channel_windows.append(windows)
    assert kept_idx_master is not None
    kept_idx = kept_idx_master
    r_peaks_kept = r_peaks[kept_idx]
    labels = np.asarray([LABEL_TO_INT[labels_str[i]] for i in kept_idx], dtype=np.int64)

    if normalise:
        per_channel_windows = [zscore_window(w.astype(np.float32))
                               for w in per_channel_windows]

    if n_ch == 1:
        windows_out = per_channel_windows[0].astype(np.float32)
    else:
        # (n_beats, n_channels, window_len)
        windows_out = np.stack(per_channel_windows, axis=1).astype(np.float32)

    feature_blocks: list[np.ndarray] = []
    for ch_idx in range(n_ch):
        block = extract_feature_matrix(
            per_channel_windows[ch_idx], r_peaks_kept,
            ann_samples_master.astype(np.int64), fs_value, pre=pre,
        )
        feature_blocks.append(block)
    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)

    bundle = {
        "windows": windows_out,
        "features": features,
        "labels": labels,
        "r_peaks": r_peaks_kept,
        "fs": fs_value,
        "record_id": record_id,
        "n_channels": n_ch,
    }
    if cache_dir_p is not None:
        np.savez(
            _cache_path(cache_dir_p, record_id, pre, post, channels_key),
            windows=bundle["windows"], features=bundle["features"],
            labels=bundle["labels"], r_peaks=bundle["r_peaks"],
            fs=np.int32(fs_value),
        )
    return bundle


def build_dataset(
    record_ids: Iterable[str],
    *,
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    pre: int = 90,
    post: int = 170,
    channel: int | list[int] = 0,
    cache_dir: str | os.PathLike | None = None,
    normalise: bool = True,
) -> dict:
    """Concatenate `build_record` outputs across many records."""
    windows: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    record_ids = list(record_ids)
    n_ch_seen: set[int] = set()
    for rid in record_ids:
        try:
            b = build_record(
                rid, data_dir=data_dir, pre=pre, post=post,
                channel=channel, cache_dir=cache_dir, normalise=normalise,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping record %s: %s", rid, exc)
            continue
        windows.append(b["windows"])
        features.append(b["features"])
        labels.append(b["labels"])
        n_ch_seen.add(b["n_channels"])
    if not windows:
        raise RuntimeError("no records loaded")
    return {
        "windows": np.concatenate(windows, axis=0),
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "record_ids": tuple(record_ids),
        "n_channels": next(iter(n_ch_seen)) if len(n_ch_seen) == 1 else max(n_ch_seen),
    }

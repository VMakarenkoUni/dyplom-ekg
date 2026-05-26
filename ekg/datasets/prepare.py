"""Build (windows, features, labels) numpy tensors from MIT-BIH records.

This is the bridge from the raw dataset into the model training loops.
For each record:
  - load the signal and annotations
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


def _cache_path(cache_dir: Path, record_id: str, pre: int, post: int) -> Path:
    return cache_dir / f"{record_id}_{pre}_{post}.npz"


def build_record(
    record_id: str,
    *,
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    pre: int = 90,
    post: int = 170,
    channel: int = 0,
    cache_dir: str | os.PathLike | None = None,
    normalise: bool = True,
) -> dict:
    """Build the per-record tensor bundle.

    Returns a dict with keys:
      - windows: (n_beats, pre+post) float32
      - features: (n_beats, n_features) float32
      - labels: (n_beats,) int64 — AAMI class index
      - r_peaks: (n_beats,) int64 — sample indices kept
      - fs: int
      - record_id: str
    """
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = _cache_path(cache_dir, record_id, pre, post)
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            return {
                "windows": data["windows"],
                "features": data["features"],
                "labels": data["labels"],
                "r_peaks": data["r_peaks"],
                "fs": int(data["fs"]),
                "record_id": record_id,
            }

    sig, ann_samples, ann_symbols, fs = load_record(record_id, data_dir, channel=channel)

    mapped: list[tuple[int, str]] = []
    for idx, sym in zip(ann_samples, ann_symbols):
        label = aami_label(str(sym))
        if label is None:
            continue
        mapped.append((int(idx), label))
    if not mapped:
        raise RuntimeError(f"record {record_id!r}: no AAMI-mapped annotations")

    r_peaks = np.asarray([m[0] for m in mapped], dtype=np.int64)
    labels_str = [m[1] for m in mapped]

    windows, kept_idx = segment_beats(sig, r_peaks, pre=pre, post=post)
    r_peaks_kept = r_peaks[kept_idx]
    labels = np.asarray([LABEL_TO_INT[labels_str[i]] for i in kept_idx], dtype=np.int64)

    if normalise:
        windows = zscore_window(windows.astype(np.float32))

    all_peaks_in_record = ann_samples.astype(np.int64)
    features = extract_feature_matrix(
        windows, r_peaks_kept, all_peaks_in_record, fs, pre=pre,
    )

    bundle = {
        "windows": windows.astype(np.float32),
        "features": features.astype(np.float32),
        "labels": labels,
        "r_peaks": r_peaks_kept,
        "fs": fs,
        "record_id": record_id,
    }
    if cache_dir is not None:
        np.savez(
            _cache_path(cache_dir, record_id, pre, post),
            windows=bundle["windows"], features=bundle["features"],
            labels=bundle["labels"], r_peaks=bundle["r_peaks"],
            fs=np.int32(fs),
        )
    return bundle


def build_dataset(
    record_ids: Iterable[str],
    *,
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    pre: int = 90,
    post: int = 170,
    channel: int = 0,
    cache_dir: str | os.PathLike | None = None,
    normalise: bool = True,
) -> dict:
    """Concatenate `build_record` outputs across many records."""
    windows: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    record_ids = list(record_ids)
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
    if not windows:
        raise RuntimeError("no records loaded")
    return {
        "windows": np.concatenate(windows, axis=0),
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "record_ids": tuple(record_ids),
    }

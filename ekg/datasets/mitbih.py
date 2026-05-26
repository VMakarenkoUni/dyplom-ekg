"""MIT-BIH Arrhythmia Database loader and AAMI label mapping.

Beats are read with the `wfdb` library directly from a local copy of the
database. Each annotated beat is mapped to one of the five AAMI super-classes
(N, S, V, F, Q) per ANSI/AAMI EC57.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    WFDB_AVAILABLE = False
    wfdb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MITDB_NAME = "mitdb"
DEFAULT_DATA_DIR = Path(
    os.environ.get("EKG_DATA_DIR", Path(__file__).resolve().parents[2] / "data")
)

# Mapping from MIT-BIH annotation symbols to AAMI super-classes.
# See ANSI/AAMI EC57:1998/(R)2008 and de Chazal et al. 2004 Table I.
_AAMI_MAP: dict[str, str] = {
    # Normal / bundle-branch block
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion of ventricular and normal
    "F": "F",
    # Unknown / paced / unclassifiable
    "P": "Q", "/": "Q", "f": "Q", "Q": "Q", "U": "Q",
}


def aami_label(symbol: str) -> Optional[str]:
    """Return the AAMI class for a MIT-BIH annotation symbol, or None."""
    return _AAMI_MAP.get(symbol)


@dataclass(frozen=True)
class MitBihBeat:
    """One annotated beat from a MIT-BIH record."""
    record_id: str
    sample: int            # R-peak sample index in the record
    aami: str              # one of N/S/V/F/Q
    original_symbol: str
    signal: np.ndarray     # the windowed signal (n_samples, n_leads)
    fs: int                # sampling rate (Hz)


def download_mitbih(target_dir: str | os.PathLike = DEFAULT_DATA_DIR) -> Path:
    """Download MIT-BIH to `target_dir/mitdb/`. Idempotent."""
    if not WFDB_AVAILABLE:
        raise RuntimeError("wfdb is not installed; pip install wfdb")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / MITDB_NAME
    if dest.exists() and any(dest.iterdir()):
        logger.info("MIT-BIH already present at %s", dest)
        return dest
    logger.info("Downloading MIT-BIH to %s ...", dest)
    wfdb.dl_database(MITDB_NAME, str(dest))
    return dest


def _record_path(record_id: str, data_dir: Path) -> str:
    """Return the path stem wfdb expects (no extension)."""
    return str(data_dir / MITDB_NAME / record_id)


def load_record(
    record_id: str,
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    *,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load one MIT-BIH record.

    Returns (signal, ann_samples, ann_symbols, fs):
      signal       — shape (n_samples,) for the requested channel
      ann_samples  — int array of annotation sample indices
      ann_symbols  — object array of MIT-BIH symbol characters
      fs           — sampling frequency in Hz (always 360 for mitdb)
    """
    if not WFDB_AVAILABLE:
        raise RuntimeError("wfdb is not installed")
    path = _record_path(record_id, Path(data_dir))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    sig = rec.p_signal[:, channel].astype(np.float32)
    return sig, np.asarray(ann.sample), np.asarray(ann.symbol), int(rec.fs)


def iter_beats(
    record_ids: Iterable[str],
    *,
    data_dir: str | os.PathLike = DEFAULT_DATA_DIR,
    window: tuple[int, int] = (90, 170),
    channel: int = 0,
    drop_unknown: bool = True,
) -> Iterator[MitBihBeat]:
    """Yield `MitBihBeat`s windowed around every annotated R-peak.

    `window=(pre, post)` is in samples (default 90 pre + 170 post = 260 samples
    @ 360 Hz ≈ 722 ms, the canonical de-Chazal beat window).
    """
    pre, post = window
    for record_id in record_ids:
        sig, samples, symbols, fs = load_record(record_id, data_dir, channel=channel)
        n = sig.shape[0]
        for idx, sym in zip(samples, symbols):
            label = aami_label(sym)
            if label is None:
                if drop_unknown:
                    continue
                label = "Q"
            start, end = int(idx) - pre, int(idx) + post
            if start < 0 or end > n:
                continue
            window_signal = sig[start:end].reshape(-1, 1)
            yield MitBihBeat(
                record_id=record_id,
                sample=int(idx),
                aami=label,
                original_symbol=str(sym),
                signal=window_signal,
                fs=fs,
            )

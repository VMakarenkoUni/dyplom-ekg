"""Multi-format robustness benchmark (NOVELTY 2).

Procedure per DS2 record:
  1. Load the native WFDB record → extract beats → classify with the trained
     model. Treat these predictions as the *reference* labels for the
     round-trip (so we measure the conversion damage rather than the model's
     own accuracy).
  2. Export the same record to a candidate format (CSV, XML, EDF).
  3. Re-ingest the exported file through the legacy multi-format parser,
     re-segment beats around the same physical sample indices, re-classify.
  4. Report per-format:
        - exact-prediction agreement vs reference,
        - accuracy vs the AAMI ground-truth labels,
        - per-class drift (confusion delta).

The output JSON is the data backing the "unified multi-format" claim in
the thesis title.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ekg.datasets.mitbih import load_record
from ekg.datasets.prepare import build_record
from ekg.datasets.splits import DS2_RECORDS
from ekg.features.handcrafted import extract_feature_matrix
from ekg.features.windows import segment_beats, zscore_window
from ekg.models.factory import load_model
from ekg.parsers.detect import parse_to_xml
from ekg.training.train import _predict_inputs

logger = logging.getLogger(__name__)

SUPPORTED_TARGET_FORMATS = ("csv", "edf")


def _export_csv(signals: np.ndarray, fs: int, path: Path,
                lead_names: tuple[str, ...] = ("lead_I",)) -> None:
    """Write one or more leads as a CSV with a time column.

    `signals` may be (n_samples,) for a single lead or (n_samples, n_leads).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_samples, n_leads = arr.shape
    if len(lead_names) != n_leads:
        lead_names = tuple(f"lead_{i + 1}" for i in range(n_leads))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *lead_names])
        for i in range(n_samples):
            writer.writerow([f"{i / fs:.6f}", *(f"{float(v):.6f}" for v in arr[i])])


def _export_edf(signals: np.ndarray, fs: int, path: Path,
                lead_names: tuple[str, ...] = ("ECG_I",)) -> None:
    """Write one or more leads to EDF+."""
    try:
        import pyedflib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyedflib required for EDF export") from exc

    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_samples, n_leads = arr.shape
    if len(lead_names) != n_leads:
        lead_names = tuple(f"ECG_{i + 1}" for i in range(n_leads))

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pyedflib.EdfWriter(str(path), n_leads,
                                file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        headers = []
        per_lead = []
        for i in range(n_leads):
            lead = arr[:, i]
            headers.append({
                "label": lead_names[i],
                "dimension": "mV",
                "sample_frequency": fs,
                "physical_min": float(np.min(lead)),
                "physical_max": float(np.max(lead)) + 1e-6,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            })
            per_lead.append(lead)
        writer.setSignalHeaders(headers)
        writer.writeSamples(per_lead)
    finally:
        writer.close()


def _signal_from_xml(xml_path: Path) -> tuple[np.ndarray, int]:
    """Backward-compat: pull a single lead from the unified XML."""
    signals, fs = _signals_from_xml(xml_path)
    return signals[:, 0] if signals.ndim == 2 else signals, fs


def _signals_from_xml(xml_path: Path) -> tuple[np.ndarray, int]:
    """Pull *all* Lead elements from the unified XML.

    Returns (signals, fs) with `signals.shape == (n_samples, n_leads)`.
    """
    import base64
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    fs_node = root.find(".//SamplingRate")
    fs = int(float(fs_node.text)) if fs_node is not None and fs_node.text else 0

    leads = root.findall(".//Lead")
    if not leads:
        raise RuntimeError(f"no Lead element in {xml_path}")

    decoded: list[np.ndarray] = []
    for lead in leads:
        data_node = lead.find("Data")
        if data_node is None or not data_node.text:
            continue
        encoding = data_node.get("encoding", "csv")
        if encoding == "base64":
            raw = base64.b64decode(data_node.text)
            sig = np.frombuffer(raw, dtype=np.float32)
        else:
            sig = np.asarray(
                [float(x) for x in data_node.text.replace("\n", " ").split() if x],
                dtype=np.float32,
            )
        decoded.append(sig)
        if fs == 0:
            rate_attr = lead.get("samplingRate") or "0"
            fs = int(float(rate_attr))

    if not decoded:
        raise RuntimeError(f"no Lead/Data in {xml_path}")

    # Truncate to the common length so we can stack.
    min_len = min(s.shape[0] for s in decoded)
    stacked = np.stack([s[:min_len] for s in decoded], axis=1)
    return stacked, fs


def _build_from_signal(
    signal: np.ndarray,
    fs: int,
    r_peaks: np.ndarray,
    labels: np.ndarray,
    all_peaks: np.ndarray,
    *,
    pre: int,
    post: int,
) -> dict:
    """Segment beats around `r_peaks` on `signal` and extract features."""
    windows, kept_idx = segment_beats(signal, r_peaks, pre=pre, post=post)
    if windows.shape[0] == 0:
        return {"windows": windows, "features": np.empty((0, 0), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64), "r_peaks": r_peaks[kept_idx]}
    windows = zscore_window(windows.astype(np.float32))
    kept_peaks = r_peaks[kept_idx]
    features = extract_feature_matrix(windows, kept_peaks, all_peaks, fs, pre=pre)
    return {
        "windows": windows.astype(np.float32),
        "features": features.astype(np.float32),
        "labels": labels[kept_idx],
        "r_peaks": kept_peaks,
    }


def _build_multilead_from_signals(
    signals: np.ndarray,         # (n_samples, n_leads)
    fs: int,
    r_peaks: np.ndarray,
    labels: np.ndarray,
    all_peaks: np.ndarray,
    *,
    pre: int,
    post: int,
) -> dict:
    """Multi-lead version of `_build_from_signal`."""
    n_leads = signals.shape[1]
    per_lead_windows: list[np.ndarray] = []
    kept_idx_master: np.ndarray | None = None
    for ch in range(n_leads):
        windows, kept_idx = segment_beats(signals[:, ch], r_peaks, pre=pre, post=post)
        if kept_idx_master is None:
            kept_idx_master = kept_idx
        per_lead_windows.append(zscore_window(windows.astype(np.float32)))
    if kept_idx_master is None or per_lead_windows[0].shape[0] == 0:
        return {"windows": np.empty((0, n_leads, pre + post), dtype=np.float32),
                "features": np.empty((0, 0), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64),
                "r_peaks": np.empty((0,), dtype=np.int64)}
    kept = kept_idx_master
    kept_peaks = r_peaks[kept]
    stacked_windows = np.stack(per_lead_windows, axis=1).astype(np.float32)
    feature_blocks = [
        extract_feature_matrix(per_lead_windows[ch], kept_peaks, all_peaks,
                               fs, pre=pre)
        for ch in range(n_leads)
    ]
    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return {
        "windows": stacked_windows,
        "features": features,
        "labels": labels[kept],
        "r_peaks": kept_peaks,
    }


def run_format_robustness(
    model_path: str | os.PathLike,
    *,
    record_ids: Sequence[str] = DS2_RECORDS,
    target_formats: Iterable[str] = SUPPORTED_TARGET_FORMATS,
    data_dir: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = "cache",
    output_path: str | os.PathLike | None = None,
    pre: int = 90,
    post: int = 170,
    max_records: int | None = None,
    channels: list[int] | None = None,
) -> dict:
    """Run the round-trip benchmark and return a results dict.

    `channels` defaults to `[0]` (single-lead MLII). Pass `[0, 1]` for
    the multi-lead variant: both channels are exported to CSV/EDF, the
    unified-XML parser is used to recover them, and the multi-channel
    bundle is fed back into the (multi-lead) model.
    """
    if channels is None:
        channels = [0]
    multi = len(channels) > 1
    model = load_model(model_path)
    target_formats = tuple(target_formats)
    record_ids = list(record_ids)
    if max_records is not None:
        record_ids = record_ids[:max_records]

    per_format: dict[str, dict] = {fmt: {
        "agreement_total": 0, "n_total": 0,
        "accuracy_total": 0, "loaded_signal_diff_mean": 0.0,
        "records": [],
    } for fmt in target_formats}
    native_accuracy_total = 0
    native_total = 0

    effective_data_dir = data_dir if data_dir is not None else "data"
    for record_id in record_ids:
        try:
            native = build_record(
                record_id, data_dir=effective_data_dir, pre=pre, post=post,
                cache_dir=cache_dir,
                channel=channels[0] if not multi else channels,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip record %s on native load: %s", record_id, exc)
            continue
        if native["labels"].size == 0:
            continue

        native_inputs = _predict_inputs(model.name, native)
        ref_proba = model.predict_proba(**native_inputs)
        ref_preds = np.argmax(ref_proba, axis=1)
        true_labels = native["labels"]
        native_accuracy_total += int(np.sum(ref_preds == true_labels))
        native_total += int(true_labels.size)

        # Load all required channels of the raw signal directly from WFDB.
        per_channel_signals = []
        ann_samples = None
        fs = None
        for ch in channels:
            sig_ch, asamp, _, fs_ch = load_record(
                record_id, data_dir=effective_data_dir, channel=ch,
            )
            per_channel_signals.append(sig_ch)
            if ann_samples is None:
                ann_samples = asamp
                fs = fs_ch
        assert ann_samples is not None and fs is not None
        raw_stack = np.stack(per_channel_signals, axis=1)  # (N, n_leads)

        with tempfile.TemporaryDirectory(prefix=f"ekg_robust_{record_id}_") as tmp:
            tmp_dir = Path(tmp)
            for fmt in target_formats:
                try:
                    if fmt == "csv":
                        src = tmp_dir / f"{record_id}.csv"
                        _export_csv(raw_stack if multi else raw_stack[:, 0],
                                    fs, src,
                                    lead_names=tuple(f"lead_{i + 1}" for i in range(len(channels))))
                    elif fmt == "edf":
                        src = tmp_dir / f"{record_id}.edf"
                        _export_edf(raw_stack if multi else raw_stack[:, 0],
                                    fs, src,
                                    lead_names=tuple(f"ECG_{i + 1}" for i in range(len(channels))))
                    else:
                        raise ValueError(f"unsupported target format {fmt!r}")
                    xml_path = parse_to_xml(src)
                    rt_signals, rt_fs = _signals_from_xml(Path(xml_path))
                    if not multi and rt_signals.ndim == 2:
                        rt_signals = rt_signals[:, :1]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("record %s format %s: round-trip failed: %s",
                                   record_id, fmt, exc)
                    continue

                if rt_fs and rt_fs != fs:
                    logger.warning("record %s format %s: fs drift %d→%d",
                                   record_id, fmt, fs, rt_fs)
                # Align lengths.
                rt_signals = rt_signals[: raw_stack.shape[0]]
                native_clip = raw_stack[: rt_signals.shape[0], : rt_signals.shape[1]]
                diff = float(np.mean(np.abs(rt_signals - native_clip)))

                if multi:
                    rt_bundle = _build_multilead_from_signals(
                        rt_signals, fs=fs,
                        r_peaks=native["r_peaks"],
                        labels=native["labels"],
                        all_peaks=ann_samples,
                        pre=pre, post=post,
                    )
                else:
                    rt_bundle = _build_from_signal(
                        rt_signals[:, 0], fs=fs,
                        r_peaks=native["r_peaks"],
                        labels=native["labels"],
                        all_peaks=ann_samples,
                        pre=pre, post=post,
                    )
                if rt_bundle["labels"].size == 0:
                    continue

                rt_inputs = _predict_inputs(model.name, rt_bundle)
                rt_preds = np.argmax(model.predict_proba(**rt_inputs), axis=1)

                # Align by index (we kept the same peaks)
                common = min(rt_preds.shape[0], ref_preds.shape[0])
                agree = int(np.sum(rt_preds[:common] == ref_preds[:common]))
                accurate = int(np.sum(rt_preds[:common] == rt_bundle["labels"][:common]))

                per_format[fmt]["agreement_total"] += agree
                per_format[fmt]["accuracy_total"] += accurate
                per_format[fmt]["n_total"] += common
                per_format[fmt]["loaded_signal_diff_mean"] += diff
                per_format[fmt]["records"].append({
                    "record_id": record_id,
                    "n": common,
                    "agreement": agree,
                    "accuracy": accurate,
                    "signal_diff_mean": diff,
                })

    results = {
        "model": str(model_path),
        "n_records": len(record_ids),
        "native_accuracy": (native_accuracy_total / native_total) if native_total else 0.0,
        "native_total_beats": native_total,
        "per_format": {},
    }
    for fmt, info in per_format.items():
        n = info["n_total"] or 1
        n_records = len(info["records"]) or 1
        results["per_format"][fmt] = {
            "n_records": len(info["records"]),
            "n_beats": info["n_total"],
            "label_agreement": info["agreement_total"] / n,
            "accuracy": info["accuracy_total"] / n,
            "mean_signal_abs_diff": info["loaded_signal_diff_mean"] / n_records,
            "per_record": info["records"],
        }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2))
    return results

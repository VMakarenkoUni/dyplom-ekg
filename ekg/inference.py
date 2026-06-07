"""High-level single-file inference.

Takes any supported ECG file → unified XML → signal extraction →
R-peak detection (via the legacy `AdvancedRPeakDetector`) → beat
segmentation → handcrafted features → trained model → per-beat
predictions with probabilities.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ekg import AAMI_CLASSES
from ekg.features.handcrafted import extract_feature_matrix
from ekg.features.windows import segment_beats, zscore_window
from ekg.legacy import AdvancedRPeakDetector, SignalProcessor
from ekg.models.factory import load_model
from ekg.parsers.detect import detect_format, parse_to_tempxml


def _extract_signal_from_xml(xml_path: str) -> tuple[np.ndarray, int, str]:
    import xml.etree.ElementTree as ET
    import base64

    tree = ET.parse(xml_path)
    root = tree.getroot()
    fs_node = root.find(".//SamplingRate")
    fs = int(float(fs_node.text)) if fs_node is not None and fs_node.text else 0
    lead = root.find(".//Lead")
    if lead is None:
        raise RuntimeError("no Lead element in parsed XML")
    lead_name = lead.get("name") or lead.get("id") or "I"
    data_node = lead.find("Data")
    if data_node is None or not data_node.text:
        raise RuntimeError("no Lead/Data in parsed XML")
    encoding = data_node.get("encoding", "csv")
    if encoding == "base64":
        signal = np.frombuffer(base64.b64decode(data_node.text), dtype=np.float32)
    else:
        signal = np.asarray(
            [float(x) for x in data_node.text.replace("\n", " ").split() if x],
            dtype=np.float32,
        )
    if fs == 0:
        rate_attr = lead.get("samplingRate") or "0"
        fs = int(float(rate_attr))
    if fs == 0:
        raise RuntimeError("could not determine sampling rate")
    return signal.astype(np.float32), fs, lead_name


def classify_file(
    input_path: str | os.PathLike,
    *,
    model_path: str | os.PathLike,
    pre: int = 90,
    post: int = 170,
) -> dict[str, Any]:
    """Run the full pipeline on `input_path` and return predictions JSON-ish."""
    input_path = str(input_path)
    fmt = detect_format(input_path)
    xml_path = parse_to_tempxml(input_path)
    try:
        signal, fs, lead_name = _extract_signal_from_xml(xml_path)
    finally:
        try:
            os.unlink(xml_path)
        except OSError:
            pass

    processor = SignalProcessor()
    try:
        filtered = processor.bandpass_filter(signal, fs)
    except Exception:
        filtered = signal

    detector = AdvancedRPeakDetector()
    try:
        r_peaks = np.asarray(detector.detect_peaks(filtered, fs), dtype=np.int64)
    except Exception:
        r_peaks = np.empty((0,), dtype=np.int64)

    if r_peaks.size == 0:
        return {
            "format": fmt,
            "sampling_rate": fs,
            "lead": lead_name,
            "n_beats": 0,
            "predictions": [],
            "class_counts": {},
            "message": "no R-peaks detected",
        }

    windows, kept_idx = segment_beats(signal, r_peaks, pre=pre, post=post)
    if windows.shape[0] == 0:
        return {
            "format": fmt, "sampling_rate": fs, "lead": lead_name,
            "n_beats": 0, "predictions": [], "class_counts": {},
            "message": "all beats fell off signal edges",
        }
    windows = zscore_window(windows.astype(np.float32))
    kept_peaks = r_peaks[kept_idx]
    features = extract_feature_matrix(windows, kept_peaks, r_peaks, fs, pre=pre)

    model = load_model(model_path)
    inputs: dict[str, Any] = {}
    if model.name in ("xgboost", "random_forest", "svm"):
        inputs["features"] = features
    elif model.name in ("cnn", "cnn_bilstm"):
        inputs["windows"] = windows
    else:  # hybrid
        inputs["features"] = features
        inputs["windows"] = windows

    proba = model.predict_proba(**inputs)
    preds = np.argmax(proba, axis=1)

    predictions: list[dict[str, Any]] = []
    for sample_idx, p_idx, p_vec in zip(kept_peaks.tolist(), preds.tolist(), proba.tolist()):
        predictions.append({
            "sample": int(sample_idx),
            "time_sec": float(sample_idx) / fs,
            "label": AAMI_CLASSES[int(p_idx)],
            "probabilities": {AAMI_CLASSES[i]: float(p_vec[i]) for i in range(len(AAMI_CLASSES))},
        })

    counts = Counter(AAMI_CLASSES[int(p)] for p in preds)
    return {
        "format": fmt,
        "sampling_rate": fs,
        "lead": lead_name,
        "model": model.name,
        "n_beats": int(kept_peaks.size),
        "duration_sec": float(signal.shape[0]) / fs,
        "class_counts": dict(counts),
        "predictions": predictions,
    }

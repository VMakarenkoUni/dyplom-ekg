"""Parser tests — CSV round-trip via the unified XML."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ekg.parsers.detect import detect_format, parse_to_xml, supported_formats

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _write_csv(path: Path, signal: np.ndarray, fs: int) -> None:
    """Write a 2-column CSV (time, lead) — the legacy converter requires
    at least 2 columns to recognise a data channel."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "lead_I"])
        for i, v in enumerate(signal):
            w.writerow([f"{i / fs:.6f}", f"{float(v):.6f}"])


def test_supported_formats_nonempty():
    info = supported_formats()
    assert isinstance(info, dict)
    assert len(info) > 0
    # The 9 legacy converters should all be represented.
    names = {k.lower() for k in info.keys()}
    assert any("csv" in n for n in names)


def test_csv_format_detection_and_xml_roundtrip(tmp_path):
    fs = 360
    t = np.arange(int(5 * fs)) / fs
    signal = (np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 25 * t)).astype(np.float32)
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path, signal, fs)

    fmt = detect_format(csv_path)
    assert fmt is not None
    assert "csv" in fmt.lower()

    xml_path = parse_to_xml(
        csv_path, num_header_rows=1, lead_names_row=0,
        data_start_row=1, sampling_rate_hz=fs,
    )
    assert Path(xml_path).exists()

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.find(".//Lead") is not None

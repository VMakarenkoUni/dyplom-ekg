"""Thin wrappers around the legacy multi-format converters.

The legacy `parser.py` already handles DICOM, WFDB, EDF/BDF, HL7 aECG,
CSV, SCP-ECG, MIT-BIH, Philips XML and GE Muse XML. We expose a small,
modern API on top of it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from ekg.legacy import (
    convert_ecg_to_xml,
    detect_file_format,
    get_converter_info,
)


def detect_format(path: str | os.PathLike) -> Optional[str]:
    """Identify the ECG file format. Returns None when unknown."""
    return detect_file_format(str(path))


def supported_formats() -> dict:
    """Return the converter registry — name → {extensions, description, ...}."""
    return get_converter_info()


def parse_to_xml(
    input_path: str | os.PathLike,
    output_path: Optional[str | os.PathLike] = None,
    *,
    format_hint: Optional[str] = None,
    **converter_kwargs,
) -> str:
    """Convert any supported ECG format to the unified XML representation.

    Returns the path to the produced XML file. When `output_path` is None,
    the XML is written next to the input with a `.xml` suffix.
    `converter_kwargs` are forwarded to the underlying converter (e.g. CSV
    accepts delimiter, num_header_rows, lead_names_row, sampling_rate_hz, …).
    """
    input_path = str(input_path)
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".xml"))
    output_path = str(output_path)

    ok = convert_ecg_to_xml(
        input_path, output_path, file_format=format_hint, **converter_kwargs,
    )
    if not ok:
        raise RuntimeError(f"failed to convert {input_path!r} to XML")
    return output_path


def parse_to_tempxml(input_path: str | os.PathLike) -> str:
    """Convert to XML in a temp file, return its path. Caller cleans up."""
    fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="ekg_")
    os.close(fd)
    return parse_to_xml(input_path, tmp)

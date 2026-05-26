"""Re-exports from the legacy coursework modules (main.py / parser.py).

The legacy code still owns multi-format ingestion, R-peak detection,
morphology analysis, HRV, and rule-based arrhythmia detection. The new
ML / dataset / API layers import everything they need through this module
so call sites stay stable if the legacy files are eventually broken up.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_main = importlib.import_module("main")
_parser = importlib.import_module("parser")

ECGConfig = _main.ECGConfig
SignalProcessor = _main.SignalProcessor
AdvancedRPeakDetector = _main.AdvancedRPeakDetector
MorphologyAnalyzer = _main.MorphologyAnalyzer
ArrhythmiaDetector = _main.ArrhythmiaDetector
HRVAnalyzer = _main.HRVAnalyzer
ClinicalInterpreter = _main.ClinicalInterpreter

detect_r_peaks_refined = _main.detect_r_peaks_refined
analyze_ecg_xml_enhanced = _main.analyze_ecg_xml_enhanced
analyze_ecg_file_any_format = _main.analyze_ecg_file_any_format

BaseECGConverter = _parser.BaseECGConverter
detect_file_format = _parser.detect_file_format
convert_ecg_to_xml = _parser.convert_ecg_to_xml
get_converter_info = _parser.get_converter_info


def legacy_attr(name: str) -> Any:
    """Fetch an attribute from the legacy main/parser modules by name."""
    for mod in (_main, _parser):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"legacy modules have no attribute {name!r}")


__all__ = [
    "ECGConfig",
    "SignalProcessor",
    "AdvancedRPeakDetector",
    "MorphologyAnalyzer",
    "ArrhythmiaDetector",
    "HRVAnalyzer",
    "ClinicalInterpreter",
    "detect_r_peaks_refined",
    "analyze_ecg_xml_enhanced",
    "analyze_ecg_file_any_format",
    "BaseECGConverter",
    "detect_file_format",
    "convert_ecg_to_xml",
    "get_converter_info",
    "legacy_attr",
]

"""Plot one MIT-BIH segment in three representations to visualise the
multi-format round-trip used by the robustness benchmark:

  1. native    — float32 WFDB samples
  2. CSV→XML   — re-loaded from the unified XML after exporting to CSV
  3. EDF→XML   — re-loaded from the unified XML after exporting to EDF

Two-axis layout: top shows all three overlaid, bottom shows the residuals
vs native. Makes the "EDF is lossless, CSV introduces visible drift"
finding immediately obvious.

Output: runs/figures/roundtrip_waveforms.png
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ekg.benchmark.format_robustness import _export_csv, _export_edf, _signal_from_xml
from ekg.datasets.mitbih import load_record
from ekg.parsers.detect import parse_to_xml

RECORD = "208"          # has obvious arrhythmia segments
SEGMENT_SEC = (10, 12)  # plot a 2-second window
FS = 360
FIG = Path("runs/figures")
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sig, _, _, fs = load_record(RECORD, data_dir="data")
    assert fs == FS
    n0, n1 = SEGMENT_SEC[0] * fs, SEGMENT_SEC[1] * fs
    native = sig[n0:n1]
    t = np.arange(native.size) / fs + SEGMENT_SEC[0]

    with tempfile.TemporaryDirectory(prefix="ekg_rt_demo_") as tmp:
        tmp_dir = Path(tmp)
        csv_path = tmp_dir / "demo.csv"
        edf_path = tmp_dir / "demo.edf"
        _export_csv(sig, fs, csv_path)
        _export_edf(sig, fs, edf_path)
        csv_xml = parse_to_xml(csv_path)
        edf_xml = parse_to_xml(edf_path)
        csv_round, _ = _signal_from_xml(Path(csv_xml))
        edf_round, _ = _signal_from_xml(Path(edf_xml))

    csv_round = csv_round[n0:n1]
    edf_round = edf_round[n0:n1]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6), dpi=140, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(t, native, color="black", linewidth=1.5, label="Native WFDB")
    ax_top.plot(t, csv_round, color="#d62728", linewidth=0.9,
                linestyle="--", label="CSV → unified XML")
    ax_top.plot(t, edf_round, color="#1f77b4", linewidth=0.9,
                linestyle=":", label="EDF → unified XML")
    ax_top.set_ylabel("Amplitude (mV)")
    ax_top.set_title(
        f"MIT-BIH record {RECORD}, lead MLII, {SEGMENT_SEC[0]}–{SEGMENT_SEC[1]} s "
        "— native vs round-trip"
    )
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(True, alpha=0.3)

    csv_res = csv_round - native
    edf_res = edf_round - native
    ax_bot.plot(t, csv_res, color="#d62728", linewidth=0.9,
                label=f"CSV residual (max |Δ| = {np.max(np.abs(csv_res)):.2e})")
    ax_bot.plot(t, edf_res, color="#1f77b4", linewidth=0.9,
                label=f"EDF residual (max |Δ| = {np.max(np.abs(edf_res)):.2e})")
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Δ vs native")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "roundtrip_waveforms.png")
    plt.close(fig)
    print(f"  roundtrip_waveforms.png  (CSV max |Δ| = {np.max(np.abs(csv_res)):.2e},  "
          f"EDF max |Δ| = {np.max(np.abs(edf_res)):.2e})")


if __name__ == "__main__":
    main()

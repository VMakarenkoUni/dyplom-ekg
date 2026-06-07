"""Download the MIT-BIH Arrhythmia Database from PhysioNet.

Usage:
    python -m scripts.download_mitbih [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys

from ekg.datasets.mitbih import DEFAULT_DATA_DIR, download_mitbih

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = p.parse_args(argv)
    dest = download_mitbih(args.data_dir)
    print(f"MIT-BIH ready at: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

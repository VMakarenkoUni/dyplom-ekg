"""Plot example beats per AAMI class with the model's prediction + confidence.

This is the figure that makes the thesis tangible: it shows the actual
waveforms the classifier sees and what it says about them.

Output: runs/figures/example_beats.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ekg import AAMI_CLASSES
from ekg.datasets.prepare import build_dataset
from ekg.datasets.splits import DS2_RECORDS
from ekg.models.factory import load_model
from ekg.training.train import _predict_inputs

RUNS = Path("runs")
FIG = RUNS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def newest_model_under(name: str, prefer_2lead: bool = True) -> Path:
    folder = RUNS / name
    cands = sorted(folder.glob("*/model.joblib"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {folder}")
    if prefer_2lead:
        for c in cands:
            if "2lead" in c.parent.name:
                return c
    return cands[0]


def main() -> None:
    # Build a small evaluation subset (just a few records, all classes).
    ds = build_dataset(DS2_RECORDS, channel=[0, 1], cache_dir="cache")
    windows = ds["windows"]   # (N, 2, L)
    features = ds["features"]
    labels = ds["labels"]

    # Use the 2-lead hybrid — our best model on rare classes.
    model_path = newest_model_under("hybrid", prefer_2lead=True)
    model = load_model(model_path)
    proba = model.predict_proba(
        **{k: v for k, v in _predict_inputs(model.name,
            {"features": features, "windows": windows, "labels": labels}).items()}
    )

    fig, axes = plt.subplots(len(AAMI_CLASSES), 4, figsize=(14, 12), dpi=140,
                              sharey="row")
    for row, cls in enumerate(AAMI_CLASSES):
        cls_idx = AAMI_CLASSES.index(cls)
        mask = labels == cls_idx
        if not mask.any():
            for col in range(4):
                axes[row, col].axis("off")
                axes[row, col].set_title(f"AAMI {cls}: no examples", fontsize=10)
            continue
        cls_indices = np.flatnonzero(mask)
        rng = np.random.default_rng(42 + row)
        sample = rng.choice(cls_indices, size=min(4, cls_indices.size), replace=False)
        for col, idx in enumerate(sample):
            ax = axes[row, col]
            window = windows[idx]
            for lead_idx, lead_name in enumerate(("MLII", "V1/V5")):
                ax.plot(window[lead_idx], linewidth=1.0,
                        label=lead_name, alpha=0.85)
            pred_idx = int(np.argmax(proba[idx]))
            pred_cls = AAMI_CLASSES[pred_idx]
            confidence = proba[idx, pred_idx]
            colour = "green" if pred_idx == cls_idx else "red"
            ax.set_title(
                f"True {cls} → pred {pred_cls} ({confidence:.2f})",
                fontsize=9, color=colour,
            )
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(f"AAMI {cls}\nz-score", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        "Example DS2 beats — true label vs hybrid 2-lead prediction (4 random samples per class)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "example_beats.png")
    plt.close(fig)
    print("  example_beats.png")


if __name__ == "__main__":
    main()

"""Supplementary figures: XGBoost feature importance + ROC curves.

Writes:
  runs/figures/xgb_feature_importance.png
  runs/figures/roc_curves.png

Uses the most recent xgboost checkpoint and the cached DS2 tensors so
no re-training is needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from ekg import AAMI_CLASSES
from ekg.datasets.prepare import build_dataset
from ekg.datasets.splits import DS2_RECORDS
from ekg.features.handcrafted import FEATURE_NAMES
from ekg.models.factory import load_model
from ekg.training.train import _predict_inputs

RUNS = Path("runs")
FIG = RUNS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def newest_model(name: str) -> Path:
    folder = RUNS / name
    cands = sorted(folder.glob("*/model.joblib"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {folder}")
    return cands[0]


def feature_importance() -> None:
    model_path = newest_model("xgboost")
    model = load_model(model_path)
    booster = model.model.get_booster()
    raw_imp = booster.get_score(importance_type="gain")
    # raw keys look like 'f0', 'f1', ... — map back to FEATURE_NAMES.
    importance = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for k, v in raw_imp.items():
        idx = int(k[1:])
        if 0 <= idx < len(FEATURE_NAMES):
            importance[idx] = v
    order = np.argsort(importance)[::-1][:20]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    y = np.arange(len(order))
    ax.barh(y, importance[order], color="#1f77b4")
    ax.set_yticks(y, [FEATURE_NAMES[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Gain (XGBoost importance)")
    ax.set_title("Top 20 handcrafted features by XGBoost gain")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "xgb_feature_importance.png")
    plt.close(fig)
    print("  xgb_feature_importance.png")

    # Also dump as JSON for the thesis text.
    import json
    ranked = [
        {"feature": FEATURE_NAMES[i], "gain": float(importance[i])}
        for i in np.argsort(importance)[::-1]
        if importance[i] > 0
    ]
    (RUNS / "xgb_feature_importance.json").write_text(json.dumps(ranked, indent=2))


def roc_curves() -> None:
    ds2 = build_dataset(DS2_RECORDS, cache_dir="cache")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    cmap = plt.get_cmap("tab10")

    for model_idx, name in enumerate(("xgboost", "hybrid")):
        try:
            model_path = newest_model(name)
        except FileNotFoundError:
            continue
        model = load_model(model_path)
        proba = model.predict_proba(**_predict_inputs(model.name, ds2))

        for cls_idx, cls in enumerate(AAMI_CLASSES):
            y_bin = (ds2["labels"] == cls_idx).astype(int)
            if y_bin.sum() < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, proba[:, cls_idx])
            ax.plot(
                fpr, tpr,
                color=cmap(cls_idx),
                linestyle="-" if name == "xgboost" else "--",
                linewidth=1.5,
                label=f"{name} / {cls}",
            )

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — XGBoost (solid) vs Hybrid (dashed), DS2")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png")
    plt.close(fig)
    print("  roc_curves.png")


def main() -> None:
    feature_importance()
    roc_curves()


if __name__ == "__main__":
    main()

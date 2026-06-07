"""Generate figures for the thesis from the runs/ JSON reports.

Outputs PNGs under runs/figures/:
  - confusion_<model>.png        per-model normalised confusion matrix
  - per_class_f1.png             grouped bar chart of per-class F1 across models
  - accuracy_vs_macrof1.png      scatter plot of the comparison table
  - format_robustness.png        bar chart of label agreement per format per model

Matplotlib only — no seaborn dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("runs")
FIG_DIR = RUNS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AAMI = ("N", "S", "V", "F", "Q")
_MULTILEAD_KEYS = ("2lead", "2-lead", "multilead", "v3-2lead", "v2-2lead")
MODELS = (
    "rule_based", "xgboost", "random_forest", "cnn", "cnn_bilstm", "hybrid",
    "xgboost_2lead", "cnn_2lead", "hybrid_2lead",
)
PRETTY = {
    "rule_based": "Rule-based (legacy)",
    "xgboost": "XGBoost (1-lead)",
    "random_forest": "Random Forest",
    "cnn": "1D-CNN (1-lead)",
    "cnn_bilstm": "CNN-BiLSTM",
    "hybrid": "Hybrid (1-lead)",
    "xgboost_2lead": "XGBoost (2-lead)",
    "cnn_2lead": "1D-CNN (2-lead)",
    "hybrid_2lead": "Hybrid (2-lead)",
}


def newest_eval(model: str) -> dict | None:
    if model == "rule_based":
        f = RUNS / "rule_based.json"
        return json.loads(f.read_text()) if f.exists() else None
    is_multilead = model.endswith("_2lead")
    base = model[: -len("_2lead")] if is_multilead else model
    folder = RUNS / base
    if not folder.exists():
        return None
    cands = sorted(folder.glob("*/evaluation.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if is_multilead and any(k in c.parent.name for k in _MULTILEAD_KEYS):
            return json.loads(c.read_text())
        if not is_multilead and not any(k in c.parent.name for k in _MULTILEAD_KEYS):
            return json.loads(c.read_text())
    return None


def fig_confusion(model: str, report: dict) -> None:
    cm = np.asarray(report["confusion"], dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=140)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(AAMI)), AAMI)
    ax.set_yticks(range(len(AAMI)), AAMI)
    ax.set_xlabel("Predicted (AAMI)")
    ax.set_ylabel("True (AAMI)")
    ax.set_title(f"{PRETTY[model]}\nrow-normalised confusion matrix")
    for i in range(len(AAMI)):
        for j in range(len(AAMI)):
            colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color=colour, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_{model}.png")
    plt.close(fig)


def fig_per_class_f1(reports: dict[str, dict]) -> None:
    models = [m for m in MODELS if m in reports]
    x = np.arange(len(AAMI))
    width = 0.85 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    cmap = plt.get_cmap("tab20")
    for i, model in enumerate(models):
        per = reports[model]["per_class"]
        f1 = [per.get(c, {}).get("f1-score", 0) for c in AAMI]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, f1, width, label=PRETTY[model], color=cmap(i))
    ax.set_xticks(x, AAMI)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("AAMI class")
    ax.set_title("Per-class F1 across models — MIT-BIH inter-patient DS2")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_f1.png")
    plt.close(fig)


def fig_rare_class_recall(reports: dict[str, dict]) -> None:
    """Dedicated chart for the rare-class story (S, V, F) — clinical headline."""
    models = [m for m in MODELS if m in reports]
    classes = ("S", "V", "F")
    x = np.arange(len(classes))
    width = 0.85 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    cmap = plt.get_cmap("tab20")
    for i, model in enumerate(models):
        per = reports[model]["per_class"]
        rec = [per.get(c, {}).get("recall", 0) for c in classes]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, rec, width, label=PRETTY[model], color=cmap(i))
    ax.set_xticks(x, classes)
    ax.set_ylabel("Recall (sensitivity)")
    ax.set_xlabel("AAMI class")
    ax.set_title("Rare-class recall — the clinically important figure of merit")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rare_class_recall.png")
    plt.close(fig)


def fig_acc_vs_macrof1(reports: dict[str, dict]) -> None:
    models = [m for m in MODELS if m in reports]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        r = reports[model]
        ax.scatter(r["accuracy"], r["macro_f1"], s=140, color=cmap(i),
                   edgecolor="black", linewidth=0.8, label=PRETTY[model])
        ax.annotate(PRETTY[model], (r["accuracy"], r["macro_f1"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Overall accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy vs Macro-F1 — MIT-BIH DS2")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.2, 0.55)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_macrof1.png")
    plt.close(fig)


def fig_format_robustness() -> None:
    fmt_xgb = RUNS / "format_robustness_xgb_full.json"
    fmt_hyb = RUNS / "format_robustness_hybrid.json"
    if not (fmt_xgb.exists() and fmt_hyb.exists()):
        return
    xgb = json.loads(fmt_xgb.read_text())
    hyb = json.loads(fmt_hyb.read_text())

    formats = list(xgb["per_format"].keys())
    x = np.arange(len(formats))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140)
    for ax, metric, title in [
        (axes[0], "label_agreement", "Label agreement vs native"),
        (axes[1], "accuracy", "Accuracy on DS2"),
    ]:
        xgb_vals = [xgb["per_format"][f][metric] for f in formats]
        hyb_vals = [hyb["per_format"][f][metric] for f in formats]
        ax.bar(x - width / 2, xgb_vals, width, label="XGBoost", color="#1f77b4")
        ax.bar(x + width / 2, hyb_vals, width, label="Hybrid", color="#ff7f0e")
        if metric == "accuracy":
            ax.axhline(xgb["native_accuracy"], color="#1f77b4", linestyle="--",
                       linewidth=1, label="XGB native")
            ax.axhline(hyb["native_accuracy"], color="#ff7f0e", linestyle="--",
                       linewidth=1, label="Hybrid native")
        ax.set_xticks(x, [f.upper() for f in formats])
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0.7, 1.02)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Format-robustness benchmark (full DS2)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "format_robustness.png")
    plt.close(fig)


def main() -> None:
    reports = {m: r for m in MODELS if (r := newest_eval(m)) is not None}
    for m, r in reports.items():
        fig_confusion(m, r)
        print(f"  confusion_{m}.png")
    fig_per_class_f1(reports)
    print("  per_class_f1.png")
    fig_rare_class_recall(reports)
    print("  rare_class_recall.png")
    fig_acc_vs_macrof1(reports)
    print("  accuracy_vs_macrof1.png")
    fig_format_robustness()
    print("  format_robustness.png")
    print(f"\nfigures in {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()

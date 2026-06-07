"""Evaluation utilities. Reports per-class precision/recall/F1, macro-F1,
confusion matrix, and one-vs-rest ROC-AUC where it can be computed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ekg import AAMI_CLASSES


@dataclass
class EvaluationReport:
    model: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    roc_auc: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def pretty(self) -> str:
        lines = [
            f"model:     {self.model}",
            f"samples:   {self.n_samples}",
            f"accuracy:  {self.accuracy:.4f}",
            f"macro-F1:  {self.macro_f1:.4f}",
            "",
            f"{'class':>6} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}",
        ]
        for cls in AAMI_CLASSES:
            row = self.per_class.get(cls, {})
            lines.append(
                f"{cls:>6} "
                f"{row.get('precision', 0):>10.4f} "
                f"{row.get('recall', 0):>10.4f} "
                f"{row.get('f1-score', 0):>10.4f} "
                f"{int(row.get('support', 0)):>10}"
            )
        if self.roc_auc:
            lines.append("")
            lines.append("ROC-AUC (one-vs-rest):")
            for cls, auc in self.roc_auc.items():
                lines.append(f"  {cls}: {auc:.4f}")
        lines.append("")
        lines.append("confusion (rows=true, cols=pred):")
        header = "    " + " ".join(f"{c:>6}" for c in AAMI_CLASSES)
        lines.append(header)
        for i, cls in enumerate(AAMI_CLASSES):
            row = self.confusion[i] if i < len(self.confusion) else [0] * len(AAMI_CLASSES)
            lines.append(f"{cls:>3} " + " ".join(f"{v:>6d}" for v in row))
        return "\n".join(lines)


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> EvaluationReport:
    labels = list(range(len(AAMI_CLASSES)))
    target_names = list(AAMI_CLASSES)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    per_class = {cls: report[cls] for cls in AAMI_CLASSES if cls in report}
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

    roc_auc: dict[str, float] = {}
    if y_proba is not None:
        for i, cls in enumerate(AAMI_CLASSES):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
                continue
            try:
                roc_auc[cls] = float(roc_auc_score(y_bin, y_proba[:, i]))
            except ValueError:
                continue

    return EvaluationReport(
        model=model_name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=cm,
        roc_auc=roc_auc,
        n_samples=int(y_true.shape[0]),
    )

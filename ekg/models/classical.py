"""Classical ML baselines over handcrafted features.

Three sklearn-style wrappers exposing the unified `BaseClassifier` interface:
XGBoost, RandomForest, SVM-RBF. All operate on the (n_beats, n_features)
matrix produced by `ekg.features.handcrafted.extract_feature_matrix`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from ekg.models.base import BaseClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False
    XGBClassifier = None  # type: ignore[assignment]


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    table = dict(zip(classes, weights))
    return np.asarray([table[int(v)] for v in y], dtype=np.float32)


class XGBoostClassifier(BaseClassifier):
    name = "xgboost"

    def __init__(self, *, n_estimators: int = 400, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42) -> None:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed")
        self._kwargs = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
        )
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, objective="multi:softprob",
            num_class=self.n_classes, eval_metric="mlogloss",
            tree_method="hist", random_state=random_state,
            n_jobs=-1,
        )

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def fit(self, *, features: np.ndarray | None = None,
            windows: np.ndarray | None = None,
            labels: np.ndarray, **kwargs) -> "XGBoostClassifier":
        if features is None:
            raise ValueError("XGBoostClassifier requires `features`")
        sample_weight = _balanced_sample_weight(labels)
        self.model.fit(features, labels, sample_weight=sample_weight)
        return self

    def predict_proba(self, *, features: np.ndarray | None = None,
                      windows: np.ndarray | None = None) -> np.ndarray:
        if features is None:
            raise ValueError("XGBoostClassifier requires `features`")
        return self.model.predict_proba(features)

    def get_state(self) -> dict[str, Any]:
        return {"model": self.model}

    def load_state(self, payload: dict[str, Any]) -> None:
        self.model = payload["model"]


class RandomForestClassifierModel(BaseClassifier):
    name = "random_forest"

    def __init__(self, *, n_estimators: int = 400, max_depth: int | None = None,
                 random_state: int = 42) -> None:
        self._kwargs = dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        )

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def fit(self, *, features=None, windows=None, labels, **kwargs):
        if features is None:
            raise ValueError("RandomForestClassifierModel requires `features`")
        self.model.fit(features, labels)
        return self

    def predict_proba(self, *, features=None, windows=None) -> np.ndarray:
        if features is None:
            raise ValueError("RandomForestClassifierModel requires `features`")
        # sklearn's RF only emits columns for classes it observed in training;
        # pad to a (n, n_classes) matrix.
        proba = self.model.predict_proba(features)
        full = np.zeros((proba.shape[0], self.n_classes), dtype=np.float32)
        for col, cls in enumerate(self.model.classes_):
            full[:, int(cls)] = proba[:, col]
        return full

    def get_state(self) -> dict[str, Any]:
        return {"model": self.model}

    def load_state(self, payload: dict[str, Any]) -> None:
        self.model = payload["model"]


class SVMClassifier(BaseClassifier):
    name = "svm"

    def __init__(self, *, C: float = 1.0, gamma: str | float = "scale",
                 random_state: int = 42) -> None:
        self._kwargs = dict(C=C, gamma=gamma, random_state=random_state)
        self.scaler = StandardScaler()
        self.model = SVC(
            C=C, gamma=gamma, kernel="rbf", probability=True,
            class_weight="balanced", random_state=random_state,
        )

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def fit(self, *, features=None, windows=None, labels, **kwargs):
        if features is None:
            raise ValueError("SVMClassifier requires `features`")
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)
        return self

    def predict_proba(self, *, features=None, windows=None) -> np.ndarray:
        if features is None:
            raise ValueError("SVMClassifier requires `features`")
        scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(scaled)
        full = np.zeros((proba.shape[0], self.n_classes), dtype=np.float32)
        for col, cls in enumerate(self.model.classes_):
            full[:, int(cls)] = proba[:, col]
        return full

    def get_state(self) -> dict[str, Any]:
        return {"model": self.model, "scaler": self.scaler}

    def load_state(self, payload: dict[str, Any]) -> None:
        self.model = payload["model"]
        self.scaler = payload["scaler"]

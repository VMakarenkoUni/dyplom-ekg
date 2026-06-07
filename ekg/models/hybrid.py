"""Dual-stream hybrid classifier (NOVELTY 1).

Stacks the probability outputs of:
  - an XGBoost classifier trained on handcrafted features, and
  - a 1D-CNN classifier trained on raw beat windows.

A small logistic-regression meta-classifier learns the best linear combination
of the two probability vectors on hold-out predictions. This late-fusion design
keeps the two streams' training cheap and independent and lets each stream
contribute where it is strongest:
  - the handcrafted stream excels on rhythm-driven labels (RR features),
  - the CNN stream excels on shape-driven labels (V, F).

Fit protocol:
  1. K-fold split the training data (default K=5).
  2. Train both base learners on each fold's training partition; predict on
     the validation partition. Concatenate validation probabilities across
     folds — these are the meta-features used to train the meta-classifier.
  3. Refit both base learners on the full training set for final inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ekg.models.base import BaseClassifier
from ekg.models.classical import XGBoostClassifier
from ekg.models.cnn import CNN1DClassifier


class HybridDualStreamClassifier(BaseClassifier):
    name = "hybrid"

    def __init__(self, *, n_folds: int = 5,
                 xgb_kwargs: dict | None = None,
                 cnn_kwargs: dict | None = None,
                 meta_C: float = 1.0,
                 random_state: int = 42) -> None:
        self._kwargs = dict(
            n_folds=n_folds, xgb_kwargs=xgb_kwargs or {},
            cnn_kwargs=cnn_kwargs or {}, meta_C=meta_C,
            random_state=random_state,
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.xgb = XGBoostClassifier(**(xgb_kwargs or {}))
        self.cnn = CNN1DClassifier(**(cnn_kwargs or {}))
        self.meta = LogisticRegression(
            C=meta_C, max_iter=1000,
            class_weight="balanced", random_state=random_state,
        )

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def _stream_proba(self, features: np.ndarray, windows: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(features=features)
        p_cnn = self.cnn.predict_proba(windows=windows)
        return np.concatenate([p_xgb, p_cnn], axis=1)

    def fit(self, *, features: np.ndarray, windows: np.ndarray,
            labels: np.ndarray, verbose: bool = True, **kwargs) -> "HybridDualStreamClassifier":
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state,
        )
        meta_X = np.zeros((labels.shape[0], 2 * self.n_classes), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(features, labels)):
            if verbose:
                print(f"[hybrid] OOF fold {fold + 1}/{self.n_folds}")
            xgb_fold = XGBoostClassifier(**self._kwargs["xgb_kwargs"])
            cnn_fold = CNN1DClassifier(**self._kwargs["cnn_kwargs"])
            xgb_fold.fit(features=features[tr_idx], labels=labels[tr_idx])
            cnn_fold.fit(windows=windows[tr_idx], labels=labels[tr_idx], verbose=False)
            meta_X[va_idx, :self.n_classes] = xgb_fold.predict_proba(features=features[va_idx])
            meta_X[va_idx, self.n_classes:] = cnn_fold.predict_proba(windows=windows[va_idx])

        if verbose:
            print("[hybrid] training meta-classifier on OOF probabilities")
        self.meta.fit(meta_X, labels)

        if verbose:
            print("[hybrid] refitting base learners on full data")
        self.xgb.fit(features=features, labels=labels)
        self.cnn.fit(windows=windows, labels=labels, verbose=verbose)
        return self

    def predict_proba(self, *, features: np.ndarray, windows: np.ndarray,
                      **kwargs) -> np.ndarray:
        meta_X = self._stream_proba(features, windows)
        proba = self.meta.predict_proba(meta_X)
        # Pad to (n, n_classes) in case the meta-classifier never saw a class.
        full = np.zeros((proba.shape[0], self.n_classes), dtype=np.float32)
        for col, cls in enumerate(self.meta.classes_):
            full[:, int(cls)] = proba[:, col]
        return full

    def get_state(self) -> dict[str, Any]:
        return {
            "xgb": self.xgb.get_state(),
            "cnn": self.cnn.get_state(),
            "meta": self.meta,
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        self.xgb.load_state(payload["xgb"])
        self.cnn.load_state(payload["cnn"])
        self.meta = payload["meta"]

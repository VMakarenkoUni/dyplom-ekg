"""Common base for all classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ekg import AAMI_CLASSES


class BaseClassifier(ABC):
    """Shared interface. Subclasses pick which input(s) they consume."""

    name: str = "base"
    n_classes: int = len(AAMI_CLASSES)

    @abstractmethod
    def fit(self, *, features: np.ndarray | None = None,
            windows: np.ndarray | None = None,
            labels: np.ndarray, **kwargs) -> "BaseClassifier": ...

    @abstractmethod
    def predict_proba(self, *, features: np.ndarray | None = None,
                      windows: np.ndarray | None = None) -> np.ndarray: ...

    def predict(self, **inputs) -> np.ndarray:
        proba = self.predict_proba(**inputs)
        return np.argmax(proba, axis=1)

    @abstractmethod
    def get_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state(self, payload: dict[str, Any]) -> None: ...

    def init_kwargs(self) -> dict[str, Any]:
        """Override to record constructor kwargs so `load_model` can rebuild."""
        return {}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.get_state()
        payload["__model__"] = self.name
        payload["__init_kwargs__"] = self.init_kwargs()
        joblib.dump(payload, path)

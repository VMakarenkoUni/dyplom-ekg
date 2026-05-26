"""Model factory — string name → instantiated classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_REGISTRY = (
    "xgboost", "random_forest", "svm",
    "cnn", "cnn_bilstm", "hybrid",
)


def build_model(name: str, **kwargs: Any):
    name = name.lower()
    if name == "xgboost":
        from ekg.models.classical import XGBoostClassifier
        return XGBoostClassifier(**kwargs)
    if name == "random_forest":
        from ekg.models.classical import RandomForestClassifierModel
        return RandomForestClassifierModel(**kwargs)
    if name == "svm":
        from ekg.models.classical import SVMClassifier
        return SVMClassifier(**kwargs)
    if name == "cnn":
        from ekg.models.cnn import CNN1DClassifier
        return CNN1DClassifier(**kwargs)
    if name == "cnn_bilstm":
        from ekg.models.cnn_bilstm import CNNBiLSTMClassifier
        return CNNBiLSTMClassifier(**kwargs)
    if name == "hybrid":
        from ekg.models.hybrid import HybridDualStreamClassifier
        return HybridDualStreamClassifier(**kwargs)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_REGISTRY}")


def load_model(path: str | Path):
    """Load a saved model. The save file records its own class name."""
    import joblib
    path = Path(path)
    payload = joblib.load(path)
    name = payload["__model__"]
    model = build_model(name, **payload.get("__init_kwargs__", {}))
    model.load_state(payload)
    return model

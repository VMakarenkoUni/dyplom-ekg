"""Classifiers — classical (sklearn / xgboost), deep (PyTorch), and hybrid.

All classifiers expose a common interface:

    model = build_model(name, ...)
    model.fit(X, y, ...)
    proba = model.predict_proba(X)  # shape (n, 5)
    preds = model.predict(X)        # shape (n,)
    model.save(path)
    model = Model.load(path)

`name` is one of: 'xgboost', 'random_forest', 'svm', 'cnn', 'cnn_bilstm', 'hybrid'.
"""

from ekg.models.factory import (
    MODEL_REGISTRY,
    build_model,
    load_model,
)

__all__ = ["MODEL_REGISTRY", "build_model", "load_model"]

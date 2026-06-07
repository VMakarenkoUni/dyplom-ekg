"""End-to-end smoke tests for every classifier on synthetic data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ekg import AAMI_CLASSES
from ekg.models.factory import build_model, load_model

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_XGB = importlib.util.find_spec("xgboost") is not None


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
def test_xgboost_fit_predict_save_load(synthetic_bundle, tmp_path):
    model = build_model("xgboost", n_estimators=20, max_depth=3)
    model.fit(features=synthetic_bundle["features"], labels=synthetic_bundle["labels"])
    proba = model.predict_proba(features=synthetic_bundle["features"])
    assert proba.shape == (synthetic_bundle["labels"].size, len(AAMI_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    path = tmp_path / "xgb.joblib"
    model.save(path)
    restored = load_model(path)
    proba2 = restored.predict_proba(features=synthetic_bundle["features"])
    assert np.allclose(proba, proba2)


def test_random_forest_fit_predict(synthetic_bundle):
    model = build_model("random_forest", n_estimators=20)
    model.fit(features=synthetic_bundle["features"], labels=synthetic_bundle["labels"])
    preds = model.predict(features=synthetic_bundle["features"])
    assert preds.shape == (synthetic_bundle["labels"].size,)
    assert preds.dtype.kind in "iu"


def test_svm_fit_predict(synthetic_bundle):
    model = build_model("svm", C=0.5)
    model.fit(features=synthetic_bundle["features"], labels=synthetic_bundle["labels"])
    proba = model.predict_proba(features=synthetic_bundle["features"])
    assert proba.shape == (synthetic_bundle["labels"].size, len(AAMI_CLASSES))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_cnn_fit_predict(synthetic_bundle):
    model = build_model("cnn", epochs=1, batch_size=64)
    model.fit(windows=synthetic_bundle["windows"], labels=synthetic_bundle["labels"],
              verbose=False)
    proba = model.predict_proba(windows=synthetic_bundle["windows"])
    assert proba.shape == (synthetic_bundle["labels"].size, len(AAMI_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_cnn_bilstm_fit_predict(synthetic_bundle):
    model = build_model("cnn_bilstm", epochs=1, batch_size=64)
    model.fit(windows=synthetic_bundle["windows"], labels=synthetic_bundle["labels"],
              verbose=False)
    proba = model.predict_proba(windows=synthetic_bundle["windows"])
    assert proba.shape == (synthetic_bundle["labels"].size, len(AAMI_CLASSES))


@pytest.mark.skipif(not (HAS_TORCH and HAS_XGB), reason="hybrid needs torch + xgboost")
def test_hybrid_fit_predict(synthetic_bundle, tmp_path):
    model = build_model(
        "hybrid", n_folds=2,
        xgb_kwargs={"n_estimators": 10, "max_depth": 3},
        cnn_kwargs={"epochs": 1, "batch_size": 64},
    )
    model.fit(
        features=synthetic_bundle["features"],
        windows=synthetic_bundle["windows"],
        labels=synthetic_bundle["labels"],
        verbose=False,
    )
    proba = model.predict_proba(
        features=synthetic_bundle["features"],
        windows=synthetic_bundle["windows"],
    )
    assert proba.shape == (synthetic_bundle["labels"].size, len(AAMI_CLASSES))


def test_baseline_predict(synthetic_bundle):
    from ekg.training.baseline import predict_baseline
    preds = predict_baseline(synthetic_bundle)
    assert preds.shape == (synthetic_bundle["labels"].size,)
    assert preds.min() >= 0 and preds.max() < len(AAMI_CLASSES)


def test_evaluation_report_shape():
    from ekg.training.evaluate import evaluate
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    y_proba = rng.dirichlet(np.ones(5), size=200).astype(np.float32)
    report = evaluate("test_model", y_true, y_pred, y_proba=y_proba)
    assert report.n_samples == 200
    assert 0 <= report.accuracy <= 1
    assert isinstance(report.pretty(), str)
    assert len(report.confusion) == 5

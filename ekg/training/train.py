"""Unified training entrypoint.

Loads MIT-BIH DS1/DS2 splits, fits the requested model, evaluates on DS2,
prints a report, and saves the checkpoint + evaluation JSON under
`runs/<model>/<timestamp>/`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from ekg.datasets.prepare import build_dataset
from ekg.datasets.splits import DS1_RECORDS, DS2_RECORDS
from ekg.models.factory import build_model
from ekg.training.evaluate import evaluate

logger = logging.getLogger(__name__)


def _stream_kwargs(model_name: str, bundle: dict) -> dict:
    """Pick inputs to pass into the model based on what it consumes."""
    inputs: dict = {"labels": bundle["labels"]}
    if model_name in ("xgboost", "random_forest", "svm"):
        inputs["features"] = bundle["features"]
    elif model_name in ("cnn", "cnn_bilstm"):
        inputs["windows"] = bundle["windows"]
    elif model_name == "hybrid":
        inputs["features"] = bundle["features"]
        inputs["windows"] = bundle["windows"]
    else:
        raise ValueError(f"unknown model {model_name!r}")
    return inputs


def _predict_inputs(model_name: str, bundle: dict) -> dict:
    inputs = _stream_kwargs(model_name, bundle)
    inputs.pop("labels", None)
    return inputs


def train(
    model_name: str,
    *,
    data_dir: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    output_root: str | os.PathLike | None = None,
    run_id: str | None = None,
    model_kwargs: dict | None = None,
) -> dict:
    """Train one model and evaluate on DS2.

    Returns a dict with keys: `model_path`, `report_path`, `report`.
    """
    model_kwargs = dict(model_kwargs or {})
    data_dir = Path(data_dir) if data_dir else None
    cache_dir = Path(cache_dir) if cache_dir else Path("cache")
    output_root = Path(output_root) if output_root else Path("runs")
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"cache_dir": cache_dir}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    logger.info("loading DS1 (%d records) ...", len(DS1_RECORDS))
    ds1 = build_dataset(DS1_RECORDS, **kwargs)
    logger.info("loading DS2 (%d records) ...", len(DS2_RECORDS))
    ds2 = build_dataset(DS2_RECORDS, **kwargs)

    logger.info("DS1 beats=%d  DS2 beats=%d", ds1["labels"].size, ds2["labels"].size)
    logger.info("class counts (DS1): %s",
                {int(c): int(n) for c, n in zip(*np.unique(ds1["labels"], return_counts=True))})

    model = build_model(model_name, **model_kwargs)
    model.fit(**_stream_kwargs(model_name, ds1))

    pred_inputs = _predict_inputs(model_name, ds2)
    proba = model.predict_proba(**pred_inputs)
    preds = np.argmax(proba, axis=1)
    report = evaluate(model_name, ds2["labels"], preds, y_proba=proba)

    model_path = run_dir / "model.joblib"
    model.save(model_path)
    report_path = run_dir / "evaluation.json"
    report.save(report_path)

    print(report.pretty())
    print(f"\nsaved model:  {model_path}")
    print(f"saved report: {report_path}")

    return {
        "model_path": str(model_path),
        "report_path": str(report_path),
        "report": report,
    }

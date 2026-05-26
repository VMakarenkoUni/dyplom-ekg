"""FastAPI service.

Endpoints:
  GET  /health                 — liveness
  GET  /models                 — list discovered checkpoints under runs/
  POST /classify?model=<name>  — multipart file upload → JSON predictions

The model directory is discovered from `runs/<model>/<timestamp>/model.joblib`.
Override with the `EKG_RUNS_DIR` environment variable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from ekg import __version__
from ekg.inference import classify_file
from ekg.parsers.detect import supported_formats

RUNS_DIR = Path(os.environ.get("EKG_RUNS_DIR", "runs"))

app = FastAPI(
    title="ECG Unified Classification API",
    version=__version__,
    description=(
        "Upload an ECG recording in any supported format. The service "
        "normalises it via the unified parser, segments beats around "
        "detected R-peaks, and classifies each beat into one of the AAMI "
        "classes (N, S, V, F, Q) using the requested trained model."
    ),
)


def _discover_models() -> dict[str, str]:
    """Map model-name → newest checkpoint path under RUNS_DIR."""
    found: dict[str, str] = {}
    if not RUNS_DIR.exists():
        return found
    for model_dir in sorted(RUNS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        checkpoints = sorted(
            model_dir.glob("*/model.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            found[model_dir.name] = str(checkpoints[0])
    return found


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/models")
def models() -> dict:
    return {"runs_dir": str(RUNS_DIR), "models": _discover_models()}


@app.get("/formats")
def formats() -> dict:
    return {"supported": supported_formats()}


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="model name; default: newest in runs/"),
) -> dict:
    available = _discover_models()
    if not available:
        raise HTTPException(503, "no trained models found; run `ekg train` first")

    if model is not None:
        if model not in available:
            raise HTTPException(404, f"model {model!r} not found; available: {sorted(available)}")
        model_path = available[model]
    else:
        # newest checkpoint across all models
        model_path = max(available.values(), key=lambda p: Path(p).stat().st_mtime)

    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = classify_file(tmp_path, model_path=model_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"classification failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return result

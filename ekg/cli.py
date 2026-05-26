"""Top-level CLI entrypoint (`ekg ...`)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer

from ekg import __version__
from ekg.models.factory import MODEL_REGISTRY

logging.basicConfig(
    level=os.environ.get("EKG_LOG", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

app = typer.Typer(
    add_completion=False,
    help="Unified multi-format ECG processing & classification.",
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"ekg {__version__}")


@app.command()
def parse(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Convert any supported ECG file to the unified XML representation."""
    from ekg.parsers.detect import parse_to_xml
    out = parse_to_xml(input_path, output)
    typer.echo(out)


@app.command(name="train")
def train_cmd(
    model: str = typer.Option(..., "--model", "-m",
                              help=f"one of: {', '.join(MODEL_REGISTRY)}"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
    output_root: Path = typer.Option(Path("runs"), "--output"),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
) -> None:
    """Train one model on MIT-BIH DS1 and evaluate on DS2."""
    from ekg.training.train import train as run_train
    kwargs = {}
    if epochs is not None and model in ("cnn", "cnn_bilstm"):
        kwargs["epochs"] = epochs
    elif epochs is not None and model == "hybrid":
        kwargs["cnn_kwargs"] = {"epochs": epochs}
    run_train(
        model_name=model, data_dir=data_dir, cache_dir=cache_dir,
        output_root=output_root, model_kwargs=kwargs,
    )


@app.command(name="evaluate")
def evaluate_cmd(
    model_path: Path = typer.Argument(..., exists=True, readable=True,
                                      help="path to a saved model.joblib"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Re-evaluate a saved checkpoint on MIT-BIH DS2."""
    import numpy as np

    from ekg.datasets.prepare import build_dataset
    from ekg.datasets.splits import DS2_RECORDS
    from ekg.models.factory import load_model
    from ekg.training.evaluate import evaluate
    from ekg.training.train import _predict_inputs

    bundle_kwargs = {"cache_dir": cache_dir}
    if data_dir is not None:
        bundle_kwargs["data_dir"] = data_dir

    ds2 = build_dataset(DS2_RECORDS, **bundle_kwargs)
    model = load_model(model_path)
    proba = model.predict_proba(**_predict_inputs(model.name, ds2))
    preds = np.argmax(proba, axis=1)
    report = evaluate(model.name, ds2["labels"], preds, y_proba=proba)
    typer.echo(report.pretty())
    if output is not None:
        report.save(output)
        typer.echo(f"\nsaved report: {output}")


@app.command(name="baseline")
def baseline_cmd(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Evaluate the legacy rule-based baseline on MIT-BIH DS2 for comparison."""
    from ekg.datasets.prepare import build_dataset
    from ekg.datasets.splits import DS2_RECORDS
    from ekg.training.baseline import predict_baseline
    from ekg.training.evaluate import evaluate

    bundle_kwargs = {"cache_dir": cache_dir}
    if data_dir is not None:
        bundle_kwargs["data_dir"] = data_dir

    ds2 = build_dataset(DS2_RECORDS, **bundle_kwargs)
    preds = predict_baseline(ds2)
    report = evaluate("rule_based", ds2["labels"], preds)
    typer.echo(report.pretty())
    if output is not None:
        report.save(output)


@app.command()
def classify(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    model_path: Path = typer.Option(..., "--model", "-m",
                                    exists=True, readable=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Classify every beat in one ECG file using a trained model."""
    from ekg.inference import classify_file
    result = classify_file(input_path, model_path=model_path)
    summary = {k: v for k, v in result.items() if k != "predictions"}
    summary["predictions"] = f"<{len(result.get('predictions', []))} beats>"
    typer.echo(json.dumps(summary, indent=2))
    if output is not None:
        output.write_text(json.dumps(result, indent=2))
        typer.echo(f"\nsaved full predictions: {output}")


benchmark_app = typer.Typer(help="Benchmarks.")
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("format-robustness")
def benchmark_format(
    model_path: Path = typer.Argument(..., exists=True, readable=True),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    formats: list[str] = typer.Option(["csv", "edf"], "--format"),
    max_records: Optional[int] = typer.Option(None, "--max-records"),
) -> None:
    """Round-trip MIT-BIH DS2 through each format and measure accuracy drift."""
    from ekg.benchmark.format_robustness import run_format_robustness
    results = run_format_robustness(
        model_path, data_dir=data_dir, cache_dir=cache_dir,
        target_formats=formats, output_path=output, max_records=max_records,
    )
    summary = {
        "native_accuracy": results["native_accuracy"],
        "n_records": results["n_records"],
        "per_format": {
            fmt: {
                "label_agreement": info["label_agreement"],
                "accuracy": info["accuracy"],
                "n_beats": info["n_beats"],
                "mean_signal_abs_diff": info["mean_signal_abs_diff"],
            }
            for fmt, info in results["per_format"].items()
        },
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI service."""
    import uvicorn
    uvicorn.run("ekg.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

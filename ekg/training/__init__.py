"""Training and evaluation entrypoints."""

from ekg.training.evaluate import EvaluationReport, evaluate
from ekg.training.train import train

__all__ = ["EvaluationReport", "evaluate", "train"]

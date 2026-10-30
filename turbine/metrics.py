"""Regression metrics in the target's original units, without scikit-learn."""
import numpy as np


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require matching, nonempty 1D arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    error = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator else None,
    }

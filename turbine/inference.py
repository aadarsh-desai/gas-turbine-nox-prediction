"""Inference only: load weights once, validate inputs, and predict."""
import hashlib

import numpy as np
import pandas as pd
import torch

from .config import ARTIFACT_DIR, FEATURES
from .model import NOxRegressor


def load_model(path=ARTIFACT_DIR / "model.pt"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES or checkpoint["architecture"] != [9, 32, 16, 1]:
        raise ValueError("The saved model does not match the input schema")
    model = NOxRegressor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, checkpoint


def validate_inputs(frame):
    if frame.empty:
        raise ValueError("Enter at least one row")
    if not frame.columns.is_unique:
        raise ValueError("Input columns must have unique names")
    missing = [name for name in FEATURES if name not in frame]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    try:
        values = frame[FEATURES].to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("All nine inputs must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("Inputs must contain finite numbers, with no missing values")
    return values


def predict(frame, model, checkpoint):
    values = validate_inputs(frame)
    scaler = checkpoint["scaler"]
    x = (values - np.array(scaler["x_mean"], dtype=np.float32)) / np.array(
        scaler["x_std"], dtype=np.float32)
    with torch.inference_mode():
        predicted = model(torch.from_numpy(x.astype(np.float32))).numpy()
    # Return the learned output without clipping or changing the evaluation rule.
    return predicted * scaler["y_std"] + scaler["y_mean"]


def outside_training_range(frame, checkpoint):
    values = validate_inputs(frame)
    outside = (values < np.asarray(checkpoint["input_min"])) | (
        values > np.asarray(checkpoint["input_max"]))
    return pd.DataFrame(outside, columns=FEATURES, index=frame.index)

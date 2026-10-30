"""Read yearly data without mixing future observations into the training set."""
import hashlib
import json

import numpy as np
import pandas as pd

from .config import EXPECTED_ROWS, FEATURES, RAW_DIR, TARGET


def load_years(years):
    manifest = json.loads((RAW_DIR / "manifest.json").read_text())
    frames = []
    for year in years:
        path = RAW_DIR / f"gt_{year}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["files_sha256"][path.name]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_ROWS[year]:
            raise ValueError(f"Unexpected row count in {path.name}")
        if not set(FEATURES + [TARGET]).issubset(frame.columns):
            raise ValueError(f"Missing sensor or target columns in {path.name}")
        if not np.isfinite(frame[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite sensor or target value in {path.name}")
        frame["year"] = year
        frame["source_row"] = np.arange(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fit_scaler(frame):
    """Use training rows only. Scale the target to keep gradients manageable."""
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame[TARGET].to_numpy(dtype=np.float32)
    return {
        "x_mean": x.mean(axis=0).tolist(),
        "x_std": np.maximum(x.std(axis=0), 1e-6).tolist(),
        "y_mean": float(y.mean()),
        "y_std": max(float(y.std()), 1e-6),
    }


def transform(frame, scaler):
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    x = (x - np.array(scaler["x_mean"], dtype=np.float32)) / np.array(
        scaler["x_std"], dtype=np.float32)
    y = (frame[TARGET].to_numpy(dtype=np.float32) - scaler["y_mean"]) / scaler["y_std"]
    return x.astype(np.float32), y.astype(np.float32)

"""Evaluate the frozen selected model on 2014 and 2015; never optimize here."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbine.config import ARTIFACT_DIR, FEATURES, TARGET, TEST_YEARS
from turbine.data import load_years
from turbine.inference import load_model, predict
from turbine.metrics import regression_metrics


def main():
    model, checkpoint = load_model()
    test = load_years(TEST_YEARS)
    actual = test[TARGET].to_numpy()
    neural = predict(test, model, checkpoint)
    scaler = checkpoint["scaler"]
    standardized = (test[FEATURES].to_numpy() - scaler["x_mean"]) / scaler["x_std"]
    design = np.column_stack([np.ones(len(test)), standardized])
    predictions = {
        "Training mean": np.full(len(test), checkpoint["baseline_mean"]),
        "Linear regression": design @ np.asarray(checkpoint["linear_weights"]),
        "Neural network": neural,
    }
    report = {
        "test_years": TEST_YEARS, "test_count": len(test), "units": "mg/m³",
        "checkpoint_sha256": hashlib.sha256((ARTIFACT_DIR / "model.pt").read_bytes()).hexdigest(),
        "models": {name: regression_metrics(actual, values) for name, values in predictions.items()},
        "neural_network_by_year": {
            str(year): regression_metrics(actual[test.year == year], neural[test.year == year])
            for year in TEST_YEARS},
    }
    (ARTIFACT_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output = test[["year", "source_row", TARGET]].copy()
    for name, values in predictions.items():
        output[name] = values
    output.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    history = pd.read_csv(ARTIFACT_DIR / checkpoint["selected_trial"]["history_file"])
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(history.epoch, history.train_mse_scaled, label="Training", color="#176b87")
    ax.plot(history.epoch, history.validation_mse_scaled, label="Validation (2013)", color="#c46a35")
    ax.axvline(checkpoint["selected_trial"]["best_epoch"], color="#666666", linestyle="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="MSE (standardized target)", title="Model selection uses validation data")
    ax.legend()
    fig.savefig(ARTIFACT_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    density = ax.hexbin(actual, neural, gridsize=45, mincnt=1, cmap="Blues")
    limits = [min(actual.min(), neural.min()), max(actual.max(), neural.max())]
    ax.plot(limits, limits, "--", color="#c46a35", label="Perfect prediction")
    ax.set(xlabel="Measured NOx (mg/m³)", ylabel="Predicted NOx (mg/m³)", title="Held-out test years: 2014–2015")
    fig.colorbar(density, ax=ax, label="Observations per bin")
    ax.legend()
    fig.savefig(ARTIFACT_DIR / "test_predictions.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    names = list(predictions)
    errors = [report["models"][name]["rmse"] for name in names]
    bars = ax.bar(names, errors, color=["#a7b2bd", "#77a7b5", "#176b87"])
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set(ylabel="Test RMSE (mg/m³)", title="All models use the same untouched test years", ylim=(0, max(errors) * 1.2))
    fig.savefig(ARTIFACT_DIR / "baseline_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

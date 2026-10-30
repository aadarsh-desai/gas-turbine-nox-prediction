"""Select a learning rate and epoch using 2013 validation data only."""
import argparse
import copy
import csv
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from turbine.config import ARTIFACT_DIR, FEATURES, TARGET, TRAIN_YEARS, VALIDATION_YEARS
from turbine.data import fit_scaler, load_years, transform
from turbine.model import NOxRegressor


def train_candidate(x_train, y_train, x_val, y_val, lr, max_epochs, patience, seed):
    # Reset both candidates to the same initialization and batch-order seed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = NOxRegressor()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state = None
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(x_val), y_val).item()
        history.append({"epoch": epoch, "train_mse_scaled": total_loss / len(x_train),
                        "validation_mse_scaled": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"lr={lr:g} epoch={epoch:3d} validation MSE={val_loss:.4f}", flush=True)
        if stale_epochs >= patience:
            break
    return best_state, best_loss, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    # Training never loads the 2014/2015 test files.
    training = load_years(TRAIN_YEARS)
    validation = load_years(VALIDATION_YEARS)
    scaler = fit_scaler(training)
    x_train, y_train = map(torch.from_numpy, transform(training, scaler))
    x_val, y_val = map(torch.from_numpy, transform(validation, scaler))
    trials = []
    best = None
    for index, lr in enumerate([1e-3, 3e-4], start=1):
        state, loss, epoch, history = train_candidate(
            x_train, y_train, x_val, y_val, lr, args.epochs, args.patience, args.seed)
        history_name = f"history_trial_{index}.csv"
        with (ARTIFACT_DIR / history_name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        trial = {"learning_rate": lr, "best_epoch": epoch,
                 "epochs_run": len(history), "validation_rmse": loss ** 0.5 * scaler["y_std"],
                 "history_file": history_name}
        trials.append(trial)
        if best is None or loss < best["loss"]:
            best = {"state": state, "loss": loss, "trial": trial}
    # Baseline 1: training mean. Baseline 2: ordinary linear least squares,
    # fitted on exactly the same standardized training inputs, no hidden layer.
    design = np.column_stack([np.ones(len(training)), x_train.numpy()])
    linear_weights = np.linalg.lstsq(design, training[TARGET].to_numpy(), rcond=None)[0]
    checkpoint = {
        "state_dict": best["state"], "features": FEATURES, "target": TARGET,
        "scaler": scaler, "architecture": [9, 32, 16, 1],
        "input_min": training[FEATURES].min().to_list(),
        "input_max": training[FEATURES].max().to_list(),
        "baseline_mean": scaler["y_mean"], "linear_weights": linear_weights.tolist(),
        "selected_trial": best["trial"], "seed": args.seed,
    }
    torch.save(checkpoint, ARTIFACT_DIR / "model.pt")
    metadata = {
        "seed": args.seed, "train_years": TRAIN_YEARS,
        "validation_years": VALIDATION_YEARS,
        "train_count": len(training), "validation_count": len(validation),
        "max_epochs": args.epochs, "patience": args.patience,
        "batch_size": 256, "weight_decay": 1e-4, "selection_metric": "validation RMSE",
        "trials": trials, "selected_trial": best["trial"],
        "parameter_count": sum(p.numel() for p in NOxRegressor().parameters()),
        "seconds": time.perf_counter() - start, "torch_version": str(torch.__version__),
        "device": "cpu",
    }
    (ARTIFACT_DIR / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Training examples make the offline demo independent of test selection.
    examples = training.iloc[np.linspace(0, len(training) - 1, 8, dtype=int)]
    examples[FEATURES].to_csv(ARTIFACT_DIR / "demo_inputs.csv", index=False)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

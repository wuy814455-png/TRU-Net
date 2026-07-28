#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRU-Net Day-7 model evaluation.

Run after process.py and train.py:
    python evaluate.py --model-module model --model-class ModifiedRes34_Unet
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, Iterable, Type

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from process import (
    INPUT_CHANNELS,
    S2S_LAT,
    S2S_LON,
    TARGET_LAT,
    TARGET_LON,
    interpolate_to_highres,
    load_preprocessed_data,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CHANNELS = 1
RESULT_DIR = Path(config.DATA_DIR) / "results090903_day7"
DEFAULT_CHECKPOINT = Path(config.DATA_DIR) / "best_day7_model.pth"

METRIC_NAMES = ("MSE", "RMSE", "MAE", "PCC", "R2", "Accuracy")


def load_model_class(module_name: str, class_name: str) -> Type[nn.Module]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Cannot import model module '{module_name}'. Put the model file in the same "
            "directory or pass --model-module with its module name."
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' does not define '{class_name}'."
        ) from exc


def build_model(module_name: str, class_name: str, input_channels: int) -> nn.Module:
    model_class = load_model_class(module_name, class_name)
    return model_class(
        inputchannel=input_channels,
        outputchannel=OUTPUT_CHANNELS,
        BN_enable=True,
        resnet_pretrain=True,
    ).to(DEVICE)


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    target_valid = target[valid]
    prediction_valid = prediction[valid]
    if target_valid.size <= 1:
        return {name: np.nan for name in METRIC_NAMES}

    mse = mean_squared_error(target_valid, prediction_valid)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(target_valid, prediction_valid)),
        "PCC": float(pearsonr(target_valid, prediction_valid)[0]),
        "R2": float(r2_score(target_valid, prediction_valid)),
        "Accuracy": float(np.mean(np.abs(prediction_valid - target_valid) < 2.0)),
    }


def average_metrics(metric_rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metric_rows)
    return {
        name: float(np.nanmean([row[name] for row in rows])) if rows else np.nan
        for name in METRIC_NAMES
    }


def load_land_mask(test_year: int) -> np.ndarray:
    target_path = Path(config.DATA_DIR) / f"temperature_2m_north_china_{test_year}_highres.nc"
    with xr.open_dataset(target_path) as ds:
        if "ocean_mask" in ds:
            return (ds["ocean_mask"].values == 1).astype(bool)
    return np.ones((161, 161), dtype=bool)


def collect_model_predictions(
    model: nn.Module,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    test_terrain: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(test_features, test_targets, test_terrain),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, batch_targets, terrain in tqdm(loader, desc="Evaluating model"):
            features = features.to(DEVICE, non_blocking=True)
            terrain = terrain.to(DEVICE, non_blocking=True)
            outputs = model(features, terrain).squeeze(1).cpu().numpy()
            predictions.extend(outputs)
            targets.extend(batch_targets.numpy())
    return np.asarray(predictions), np.asarray(targets)


def group_by_date(
    predictions: np.ndarray,
    targets: np.ndarray,
    original_t2m: torch.Tensor,
    test_dates: list[str],
) -> Dict[str, Dict[str, list[np.ndarray]]]:
    grouped: Dict[str, Dict[str, list[np.ndarray]]] = {}
    for index, date in enumerate(test_dates):
        grouped.setdefault(date, {"predictions": [], "targets": [], "original": []})
        grouped[date]["predictions"].append(predictions[index])
        grouped[date]["targets"].append(targets[index])

        raw_original = original_t2m[index]
        if isinstance(raw_original, torch.Tensor):
            raw_original = raw_original.cpu().numpy()
        grouped[date]["original"].append(
            interpolate_to_highres(raw_original, S2S_LAT, S2S_LON, TARGET_LAT, TARGET_LON)
        )
    return grouped


def evaluate_grouped_data(
    grouped: Dict[str, Dict[str, list[np.ndarray]]],
    land_mask: np.ndarray,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
    model_date_metrics: list[Dict[str, float]] = []
    original_date_metrics: list[Dict[str, float]] = []
    per_date: Dict[str, Dict[str, Dict[str, float]]] = {}

    for date, values in grouped.items():
        target = np.asarray(values["targets"])[0]
        valid_land = land_mask & np.isfinite(target)
        target_land = target[valid_land]

        model_member_metrics = [
            calculate_metrics(target_land, np.asarray(prediction)[valid_land])
            for prediction in values["predictions"]
        ]
        original_member_metrics = [
            calculate_metrics(target_land, np.asarray(original)[valid_land])
            for original in values["original"]
        ]

        model_mean = average_metrics(model_member_metrics)
        original_mean = average_metrics(original_member_metrics)
        model_date_metrics.append(model_mean)
        original_date_metrics.append(original_mean)
        per_date[date] = {"model": model_mean, "original": original_mean}

    return (
        average_metrics(model_date_metrics),
        average_metrics(original_date_metrics),
        per_date,
    )


def write_metrics(
    output_path: Path,
    model_metrics: Dict[str, float],
    original_metrics: Dict[str, float],
    per_date: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("[TRU-Net]\n")
        for name in METRIC_NAMES:
            file.write(f"{name}: {model_metrics[name]:.6f}\n")

        file.write("\n[Original forecast]\n")
        for name in METRIC_NAMES:
            file.write(f"{name}: {original_metrics[name]:.6f}\n")

        file.write("\n[Per-date ensemble-mean metrics]\n")
        for date in sorted(per_date):
            file.write(f"\n{date}\n")
            for source in ("model", "original"):
                values = ", ".join(
                    f"{name}={per_date[date][source][name]:.6f}" for name in METRIC_NAMES
                )
                file.write(f"{source}: {values}\n")


def evaluate(args: argparse.Namespace) -> None:
    test_data = load_preprocessed_data("test")
    metadata = test_data["metadata"]
    input_channels = int(metadata.get("input_channels", INPUT_CHANNELS))

    checkpoint_path = Path(args.checkpoint)
    model_module = args.model_module
    model_class = args.model_class

    # A checkpoint produced by train.py stores the model import information.
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw_checkpoint, dict):
        model_module = raw_checkpoint.get("model_module", model_module)
        model_class = raw_checkpoint.get("model_class", model_class)
        input_channels = int(raw_checkpoint.get("input_channels", input_channels))

    model = build_model(model_module, model_class, input_channels)
    load_checkpoint(model, checkpoint_path)

    predictions, targets = collect_model_predictions(
        model,
        test_data["features"],
        test_data["targets"],
        test_data["terrain"],
        args.batch_size,
        args.num_workers,
    )

    test_years = list(metadata["test_years"])
    test_dates = list(metadata["test_dates"])
    original_t2m = metadata["test_orig_t2m"]
    if not test_years:
        raise RuntimeError("No test metadata found. Run process.py again.")

    grouped = group_by_date(predictions, targets, original_t2m, test_dates)
    land_mask = load_land_mask(test_years[0])
    model_metrics, original_metrics, per_date = evaluate_grouped_data(grouped, land_mask)

    output_path = Path(args.output)
    write_metrics(output_path, model_metrics, original_metrics, per_date)

    print("TRU-Net mean metrics:")
    print(", ".join(f"{name}={model_metrics[name]:.4f}" for name in METRIC_NAMES))
    print("Original forecast mean metrics:")
    print(", ".join(f"{name}={original_metrics[name]:.4f}" for name in METRIC_NAMES))
    print(f"Evaluation results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TRU-Net Day-7 model")
    parser.add_argument("--model-module", default="model")
    parser.add_argument("--model-class", default="ModifiedRes34_Unet")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(RESULT_DIR / "evaluation_metrics.txt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

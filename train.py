#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRU-Net Day-7 model training.

Run after process.py:
    python train.py --model-module model --model-class ModifiedRes34_Unet
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Type

import joblib
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

import config
from process import INPUT_CHANNELS, load_preprocessed_data

matplotlib.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CHANNELS = 1
RESULT_DIR = Path(config.DATA_DIR) / "results090903_day7"
DEFAULT_CHECKPOINT = Path(config.DATA_DIR) / "best_day7_model.pth"


class NonNaNMSELoss(nn.Module):
    """MSE over finite target values only."""

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.isfinite(target)
        if valid_mask.sum() == 0:
            return prediction.sum() * 0.0
        return self.mse_loss(prediction[valid_mask], target[valid_mask]) / valid_mask.sum()


def load_model_class(module_name: str, class_name: str) -> Type[nn.Module]:
    """Dynamically load the model class kept in the repository's model module."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Cannot import model module '{module_name}'. Put the model file in the same "
            "directory or pass --model-module with its module name."
        ) from exc
    try:
        model_class = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' does not define '{class_name}'. "
            "Pass the correct --model-class value."
        ) from exc
    return model_class


def build_model(module_name: str, class_name: str, input_channels: int) -> nn.Module:
    model_class = load_model_class(module_name, class_name)
    return model_class(
        inputchannel=input_channels,
        outputchannel=OUTPUT_CHANNELS,
        BN_enable=True,
        resnet_pretrain=True,
    ).to(DEVICE)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def train_model(args: argparse.Namespace) -> None:
    train_data = load_preprocessed_data("train")
    val_data = load_preprocessed_data("val")

    train_loader = DataLoader(
        TensorDataset(
            train_data["features"],
            train_data["targets"],
            train_data["terrain"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(
            val_data["features"],
            val_data["targets"],
            val_data["terrain"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metadata = train_data["metadata"]
    input_channels = int(metadata.get("input_channels", INPUT_CHANNELS))
    model = build_model(args.model_module, args.model_class, input_channels)

    criterion = NonNaNMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Device: {DEVICE}")
    print(f"Training samples: {len(train_data['features'])}")
    print(f"Validation samples: {len(val_data['features'])}")

    for epoch in trange(args.epochs, desc="Training", leave=True):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch_features, batch_targets, batch_terrain in train_loader:
            batch_features = batch_features.to(DEVICE, non_blocking=True)
            batch_targets = batch_targets.to(DEVICE, non_blocking=True)
            batch_terrain = batch_terrain.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features, batch_terrain)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            if not torch.isfinite(loss) or loss.item() == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            batch_size_actual = batch_features.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_samples += batch_size_actual

        train_loss = train_loss_sum / train_samples if train_samples else float("nan")
        train_losses.append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_features, batch_targets, batch_terrain in val_loader:
                batch_features = batch_features.to(DEVICE, non_blocking=True)
                batch_targets = batch_targets.to(DEVICE, non_blocking=True)
                batch_terrain = batch_terrain.to(DEVICE, non_blocking=True)

                outputs = model(batch_features, batch_terrain)
                loss = criterion(outputs, batch_targets.unsqueeze(1))
                if not torch.isfinite(loss) or loss.item() == 0:
                    continue
                batch_size_actual = batch_features.size(0)
                val_loss_sum += loss.item() * batch_size_actual
                val_samples += batch_size_actual

        val_loss = val_loss_sum / val_samples if val_samples else float("nan")
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, lr={current_lr:.8f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                    "model_module": args.model_module,
                    "model_class": args.model_class,
                    "input_channels": input_channels,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    joblib.dump(history, RESULT_DIR / "training_history.joblib", compress=3)
    plot_loss_curve(train_losses, val_losses, RESULT_DIR)
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRU-Net Day-7 model")
    parser.add_argument("--model-module", default="model")
    parser.add_argument("--model-class", default="ModifiedRes34_Unet")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    train_model(parse_args())


if __name__ == "__main__":
    main()

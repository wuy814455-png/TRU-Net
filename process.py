#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRU-Net Day-7 data preprocessing.

Run this file once before training/evaluation:
    python process.py
    python process.py --force
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import scipy.interpolate
import torch
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm

import config

DATA_DIR = Path(config.DATA_DIR)
TERRAIN_FILE = Path(config.TERRAIN_FILE)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

YEARS = range(2004, 2024)
FORECAST_DATES = [
    "0530", "0603", "0606", "0610", "0613", "0617", "0620", "0624",
    "0627", "0701", "0704", "0708", "0711", "0715", "0718", "0722",
    "0725", "0729", "0801", "0805", "0808", "0812", "0815", "0819",
    "0822", "0826", "0829",
]
NUM_ENSEMBLE = 10
INPUT_CHANNELS = 5
MAX_NAN_RATIO = float(getattr(config, "MAX_NAN_RATIO", 0.10))

TARGET_LAT = np.linspace(47.0, 31.0, 161)
TARGET_LON = np.linspace(107.0, 123.0, 161)
S2S_LAT = np.linspace(47.0, 31.0, 33)
S2S_LON = np.linspace(107.0, 123.0, 33)

PROGRESS_DIR = OUTPUT_DIR / "processing_progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

PATHS: Dict[str, Path] = {
    "train_features": PROGRESS_DIR / "train_features.joblib",
    "val_features": PROGRESS_DIR / "val_features.joblib",
    "test_features": PROGRESS_DIR / "test_features.joblib",
    "train_targets": PROGRESS_DIR / "train_targets.joblib",
    "val_targets": PROGRESS_DIR / "val_targets.joblib",
    "test_targets": PROGRESS_DIR / "test_targets.joblib",
    "feature_stats": PROGRESS_DIR / "feature_stats.joblib",
    "target_stats": PROGRESS_DIR / "target_stats.joblib",
    "train_features_clean": PROGRESS_DIR / "train_features_clean.joblib",
    "val_features_clean": PROGRESS_DIR / "val_features_clean.joblib",
    "test_features_clean": PROGRESS_DIR / "test_features_clean.joblib",
    "train_targets_clean": PROGRESS_DIR / "train_targets_clean.joblib",
    "val_targets_clean": PROGRESS_DIR / "val_targets_clean.joblib",
    "test_targets_clean": PROGRESS_DIR / "test_targets_clean.joblib",
    "terrain_normalized": PROGRESS_DIR / "terrain_normalized.joblib",
    "metadata": PROGRESS_DIR / "preprocess_progress.joblib",
}


def interpolate_to_highres(
    data: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """Interpolate a 2-D field from the source grid to the destination grid."""
    src_grid_lat, src_grid_lon = np.meshgrid(src_lat, src_lon, indexing="ij")
    dst_grid_lat, dst_grid_lon = np.meshgrid(dst_lat, dst_lon, indexing="ij")

    data_flat = np.asarray(data).reshape(-1)
    valid = np.isfinite(data_flat)
    if valid.sum() < 2:
        return np.full((len(dst_lat), len(dst_lon)), np.nan, dtype=np.float32)

    points = np.column_stack(
        (src_grid_lat.reshape(-1)[valid], src_grid_lon.reshape(-1)[valid])
    )
    grid_points = np.column_stack((dst_grid_lat.reshape(-1), dst_grid_lon.reshape(-1)))
    interpolated = scipy.interpolate.griddata(
        points,
        data_flat[valid],
        grid_points,
        method="linear",
    )
    return interpolated.reshape(len(dst_lat), len(dst_lon)).astype(np.float32)


def load_day7_features(file_path: Path, year: int, forecast_date: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Day-7 features for all ensemble members."""
    with xr.open_dataset(file_path) as ds:
        if "latitude" in ds.coords and "longitude" in ds.coords:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})

        base_date = np.datetime64(
            f"{year}-{forecast_date[:2]}-{forecast_date[2:]}T00:00:00"
        )
        times = ds["time"].values
        start_idx = int(np.abs(times - base_date).argmin())
        ds_slice = ds.isel(time=slice(start_idx, start_idx + 7))
        if ds_slice.sizes.get("time", 0) < 7:
            raise ValueError(f"Insufficient time steps near {base_date} in {file_path}")

        variables = ["t2m", "d2m", "v10", "ssrd"]
        features_list = []
        raw_t2m_list = []
        for ens_idx in range(NUM_ENSEMBLE):
            member_features = []
            for variable in variables:
                array = ds_slice[variable].isel(time=6, number=ens_idx).values
                if variable == "ssrd":
                    array = np.clip(array, 0, 1500)
                member_features.append(array)
                if variable == "t2m":
                    raw_t2m_list.append(array)
            features_list.append(np.stack(member_features, axis=0))

    return (
        np.asarray(features_list, dtype=np.float32),
        np.asarray(raw_t2m_list, dtype=np.float32),
    )


def load_day7_target_and_mask(
    target_path: Path,
    year: int,
    target_date_str: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load the high-resolution target and land mask."""
    if not target_path.exists():
        return None, None

    with xr.open_dataset(target_path) as ds:
        if "latitude" in ds.coords and "longitude" in ds.coords:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        try:
            target = ds["temperature_2m"].sel(
                time=f"{year}-{target_date_str[:2]}-{target_date_str[2:]}",
                method="nearest",
                tolerance="3D",
            ).values
        except (KeyError, ValueError):
            return None, None

        if target.shape != (161, 161):
            return None, None
        if "ocean_mask" in ds:
            land_mask = (ds["ocean_mask"].values == 1).astype(np.float32)
        else:
            land_mask = np.ones_like(target, dtype=np.float32)

    return target.astype(np.float32), land_mask


def load_terrain() -> np.ndarray:
    """Load and normalize the 960 x 960 terrain field."""
    with xr.open_dataset(TERRAIN_FILE) as ds:
        terrain = ds["terrain"].values
    if terrain.shape != (960, 960):
        raise ValueError(f"Terrain shape must be (960, 960), got {terrain.shape}")
    terrain = (terrain - np.nanmean(terrain)) / (np.nanstd(terrain) + 1e-6)
    return np.nan_to_num(terrain, nan=0.0).astype(np.float32)


def _all_cache_files_exist() -> bool:
    return all(path.exists() for path in PATHS.values())


def _save(name: str, value: object) -> None:
    joblib.dump(value, PATHS[name], compress=3)


def _split_indices(sample_years: Iterable[int]) -> Tuple[list[int], list[int], list[int]]:
    years_list = list(sample_years)
    train_idx = [i for i, year in enumerate(years_list) if 2004 <= year <= 2017]
    val_idx = [i for i, year in enumerate(years_list) if 2018 <= year <= 2020]
    test_idx = [i for i, year in enumerate(years_list) if 2021 <= year <= 2023]
    return train_idx, val_idx, test_idx


def preprocess_data(force: bool = False, max_nan_ratio: float = MAX_NAN_RATIO) -> None:
    """Create and cache train/validation/test tensors and metadata."""
    if _all_cache_files_exist() and not force:
        print(f"Preprocessed files already exist in: {PROGRESS_DIR}")
        print("Use --force to regenerate them.")
        return

    terrain_data = load_terrain()
    features_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    sample_years: list[int] = []
    sample_dates: list[str] = []
    sample_ensembles: list[int] = []
    test_orig_t2m: list[np.ndarray] = []
    test_r2_orig_list: list[float] = []
    test_pcc_orig_list: list[float] = []

    total = len(list(YEARS)) * len(FORECAST_DATES) * NUM_ENSEMBLE
    with tqdm(total=total, desc="Preprocessing samples") as progress_bar:
        for year in YEARS:
            for forecast_date in FORECAST_DATES:
                forecast_datetime = datetime.strptime(f"{year}-{forecast_date}", "%Y-%m%d")
                target_datetime = forecast_datetime + timedelta(days=6)
                target_date_str = target_datetime.strftime("%m%d")

                forecast_path = DATA_DIR / f"task_merged_{forecast_date}_masked.nc"
                if not forecast_path.exists():
                    progress_bar.update(NUM_ENSEMBLE)
                    continue

                try:
                    features_ensemble, raw_t2m = load_day7_features(
                        forecast_path, year, forecast_date
                    )
                except Exception as exc:
                    print(f"Failed to load {forecast_path}: {exc}")
                    progress_bar.update(NUM_ENSEMBLE)
                    continue

                target_path = DATA_DIR / f"temperature_2m_north_china_{year}_highres.nc"
                target, land_mask = load_day7_target_and_mask(
                    target_path, year, target_date_str
                )
                if target is None or land_mask is None:
                    progress_bar.update(NUM_ENSEMBLE)
                    continue

                if year <= 2020:
                    land_values = target[land_mask == 1]
                    nan_ratio = float(np.isnan(land_values).mean()) if land_values.size else 0.0
                    if nan_ratio > max_nan_ratio:
                        print(
                            f"Skip {year}-{target_date_str}: target NaN ratio "
                            f"{nan_ratio:.4f} > {max_nan_ratio:.4f}"
                        )
                        progress_bar.update(NUM_ENSEMBLE)
                        continue

                if year >= 2021:
                    r2_values: list[float] = []
                    pcc_values: list[float] = []
                    for ens_idx in range(NUM_ENSEMBLE):
                        original = interpolate_to_highres(
                            raw_t2m[ens_idx], S2S_LAT, S2S_LON, TARGET_LAT, TARGET_LON
                        )
                        valid = np.isfinite(target) & np.isfinite(original) & (land_mask == 1)
                        if valid.sum() > 1:
                            r2_values.append(r2_score(target[valid], original[valid]))
                            pcc_values.append(pearsonr(target[valid], original[valid])[0])
                        else:
                            r2_values.append(np.nan)
                            pcc_values.append(np.nan)
                    test_r2_orig_list.append(float(np.nanmean(r2_values)))
                    test_pcc_orig_list.append(float(np.nanmean(pcc_values)))

                mask_channel = interpolate_to_highres(
                    land_mask, TARGET_LAT, TARGET_LON, S2S_LAT, S2S_LON
                )
                mask_channel = (mask_channel >= 0.5).astype(np.float32)

                for ens_idx in range(NUM_ENSEMBLE):
                    features_with_mask = np.concatenate(
                        [features_ensemble[ens_idx], mask_channel[np.newaxis, :, :]],
                        axis=0,
                    )
                    features_all.append(features_with_mask)
                    targets_all.append(target)
                    sample_years.append(year)
                    sample_dates.append(target_datetime.strftime("%Y-%m-%d"))
                    sample_ensembles.append(ens_idx)
                    if year >= 2021:
                        test_orig_t2m.append(raw_t2m[ens_idx])
                    progress_bar.update(1)
                gc.collect()

    if not features_all:
        raise RuntimeError("No valid samples were generated. Check DATA_DIR and input files.")

    features_array = np.asarray(features_all, dtype=np.float32)
    targets_array = np.asarray(targets_all, dtype=np.float32)
    test_orig_t2m_array = np.asarray(test_orig_t2m, dtype=np.float32)

    train_idx, val_idx, test_idx = _split_indices(sample_years)
    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError(
            "At least one data split is empty. Expected train=2004-2017, "
            "validation=2018-2020, test=2021-2023."
        )

    train_features = features_array[train_idx]
    val_features = features_array[val_idx]
    test_features = features_array[test_idx]
    train_targets = targets_array[train_idx]
    val_targets = targets_array[val_idx]
    test_targets = targets_array[test_idx]

    train_feat_mean = np.nanmean(train_features, axis=(0, 2, 3), keepdims=True)
    train_feat_std = np.nanstd(train_features, axis=(0, 2, 3), keepdims=True) + 1e-6
    train_tar_mean = float(np.nanmean(train_targets))
    train_tar_std = float(np.nanstd(train_targets) + 1e-6)

    train_features = (train_features - train_feat_mean) / train_feat_std
    val_features = (val_features - train_feat_mean) / train_feat_std
    test_features = (test_features - train_feat_mean) / train_feat_std

    terrain_mean = np.nanmean(terrain_data)
    terrain_std = np.nanstd(terrain_data) + 1e-6
    terrain_normalized = np.nan_to_num(
        (terrain_data - terrain_mean) / terrain_std,
        nan=0.0,
    ).astype(np.float32)

    train_features_clean = torch.from_numpy(np.nan_to_num(train_features, nan=0.0).astype(np.float32))
    val_features_clean = torch.from_numpy(np.nan_to_num(val_features, nan=0.0).astype(np.float32))
    test_features_clean = torch.from_numpy(np.nan_to_num(test_features, nan=0.0).astype(np.float32))
    train_targets_clean = torch.from_numpy(np.nan_to_num(train_targets, nan=0.0).astype(np.float32))
    val_targets_clean = torch.from_numpy(np.nan_to_num(val_targets, nan=0.0).astype(np.float32))
    test_targets_clean = torch.from_numpy(np.nan_to_num(test_targets, nan=0.0).astype(np.float32))
    test_orig_t2m_tensor = torch.from_numpy(test_orig_t2m_array)

    metadata = {
        "test_years": [sample_years[i] for i in test_idx],
        "test_dates": [sample_dates[i] for i in test_idx],
        "test_ensembles": [sample_ensembles[i] for i in test_idx],
        "test_orig_t2m": test_orig_t2m_tensor,
        "test_r2_orig_list": test_r2_orig_list,
        "test_pcc_orig_list": test_pcc_orig_list,
        "target_lat": TARGET_LAT,
        "target_lon": TARGET_LON,
        "s2s_lat": S2S_LAT,
        "s2s_lon": S2S_LON,
        "input_channels": INPUT_CHANNELS,
        "num_ensemble": NUM_ENSEMBLE,
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
    }

    _save("train_features", train_features)
    _save("val_features", val_features)
    _save("test_features", test_features)
    _save("train_targets", train_targets)
    _save("val_targets", val_targets)
    _save("test_targets", test_targets)
    _save("feature_stats", {"mean": train_feat_mean, "std": train_feat_std})
    _save("target_stats", {"mean": train_tar_mean, "std": train_tar_std})
    _save("terrain_normalized", terrain_normalized)
    _save("metadata", metadata)
    _save("train_features_clean", train_features_clean)
    _save("val_features_clean", val_features_clean)
    _save("test_features_clean", test_features_clean)
    _save("train_targets_clean", train_targets_clean)
    _save("val_targets_clean", val_targets_clean)
    _save("test_targets_clean", test_targets_clean)

    print("Preprocessing completed.")
    print(f"Cache directory: {PROGRESS_DIR}")
    print(f"Split sizes: {metadata['split_sizes']}")
    print(f"Terrain shape: {terrain_normalized.shape}")


def load_preprocessed_data(split: str) -> Dict[str, object]:
    """Load a cached split for train.py or evaluate.py."""
    split_alias = {"val": "val", "validation": "val", "train": "train", "test": "test"}
    if split not in split_alias:
        raise ValueError("split must be one of: train, val, validation, test")
    prefix = split_alias[split]

    required = [
        PATHS[f"{prefix}_features_clean"],
        PATHS[f"{prefix}_targets_clean"],
        PATHS["terrain_normalized"],
        PATHS["feature_stats"],
        PATHS["target_stats"],
        PATHS["metadata"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessed files. Run process.py first. Missing:\n" + "\n".join(missing)
        )

    features = joblib.load(PATHS[f"{prefix}_features_clean"])
    targets = joblib.load(PATHS[f"{prefix}_targets_clean"])
    terrain = joblib.load(PATHS["terrain_normalized"])
    feature_stats = joblib.load(PATHS["feature_stats"])
    target_stats = joblib.load(PATHS["target_stats"])
    metadata = joblib.load(PATHS["metadata"])

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, dtype=torch.float32)
    terrain_tensor = torch.as_tensor(terrain, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    terrain_tensor = terrain_tensor.expand(len(features), 1, 960, 960)

    return {
        "features": features,
        "targets": targets,
        "terrain": terrain_tensor,
        "feature_stats": feature_stats,
        "target_stats": target_stats,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TRU-Net Day-7 data")
    parser.add_argument("--force", action="store_true", help="Regenerate cached data")
    parser.add_argument(
        "--max-nan-ratio",
        type=float,
        default=MAX_NAN_RATIO,
        help="Maximum allowed NaN ratio over land for train/validation targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_data(force=args.force, max_nan_ratio=args.max_nan_ratio)


if __name__ == "__main__":
    main()

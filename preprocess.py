#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import gc
from tqdm import tqdm, trange
import joblib
import sys
import matplotlib
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import re
import random
import scipy.interpolate
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import DATA_DIR, TERRAIN_FILE, OUTPUT_DIR

data_dir = str(DATA_DIR)
target_data_dir = str(DATA_DIR)

terrain_file = str(TERRAIN_FILE)

years = range(2004, 2024)

forecast_dates = [
    "0530", "0603", "0606", "0610",
    "0613", "0617", "0620", "0624",
    "0627", "0701", "0704", "0708",
    "0711", "0715", "0718", "0722",
    "0725", "0729", "0801", "0805",
    "0808", "0812", "0815", "0819",
    "0822", "0826", "0829"
]

num_ensemble = 10

batch_size = 16

input_channels = 5

output_channels = 1

target_lat = np.linspace(47.0,31.0,161)
target_lon = np.linspace(107.0,123.0,161)

s2s_lat = np.linspace(47.0,31.0,33)
s2s_lon = np.linspace(107.0,123.0,33)

progress_dir = OUTPUT_DIR / "processing_progress"
progress_dir.mkdir(exist_ok=True)

progress_file = progress_dir / "preprocess_progress.joblib"

class NonNaNMSELoss(nn.Module):
    def __init__(self):
        super(NonNaNMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        valid_mask = ~torch.isnan(target)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        return self.mse_loss(pred_valid, target_valid) / valid_mask.sum()

def interpolate_to_highres(data, src_lat, src_lon, dst_lat, dst_lon):
    """将33x33数据插值到161x161"""
    src_grid_lat, src_grid_lon = np.meshgrid(src_lat, src_lon, indexing='ij')
    dst_grid_lat, dst_grid_lon = np.meshgrid(dst_lat, dst_lon, indexing='ij')
    data_flat = data.flatten()
    valid = ~np.isnan(data_flat)
    if np.sum(valid) < 2:
        return np.full((len(dst_lat), len(dst_lon)), np.nan)
    points = np.vstack((src_grid_lat.flatten()[valid], src_grid_lon.flatten()[valid])).T
    values = data_flat[valid]
    values = data_flat[valid]
    grid_points = np.vstack((dst_grid_lat.flatten(), dst_grid_lon.flatten())).T
    interpolated = scipy.interpolate.griddata(points, values, grid_points, method='linear')
    return interpolated.reshape(len(dst_lat), len(dst_lon))

def load_day7_features(file_path, year, f_date):
    ds = xr.open_dataset(file_path)
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    base_date = np.datetime64(f'{year}-{f_date[:2]}-{f_date[2:]}T00:00:00')
    times = ds['time'].values
    start_idx = np.abs(times - base_date).argmin()
    ds_slice = ds.isel(time=slice(start_idx, start_idx + 7))
    features_list, raw_t2m_list = [], []
    vars4 = ['t2m', 'd2m', 'v10', 'ssrd']
    for ens in range(num_ensemble):
        feats = []
        for var in vars4:
            arr = ds_slice[var].isel(time=6, number=ens).values
            if var == 'ssrd':
                arr = np.clip(arr, 0, 1500)
            feats.append(arr)
            if var == 't2m':
                raw_t2m_list.append(arr)
                print(f"Ensemble {ens + 1}, t2m min: {np.nanmin(arr):.2f}, max: {np.nanmax(arr):.2f}")
        features_list.append(np.stack(feats, axis=0))
    ds.close()
    return np.array(features_list, dtype=np.float32), np.array(raw_t2m_list, dtype=np.float32)

def load_day7_target_and_mask(target_path, year, target_date_str):
    ds = xr.open_dataset(target_path)
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    try:
        tar = ds['temperature_2m'].sel(time=f'{year}-{target_date_str[:2]}-{target_date_str[2:]}', method='nearest',
                                       tolerance='3D').values
        print(f"Target t2m min: {np.nanmin(tar):.2f}, max: {np.nanmax(tar):.2f}")
    except KeyError:
        ds.close()
        return None, None
    if tar.shape != (161, 161):
        ds.close()
        return None, None
    if 'ocean_mask' in ds:
        land_mask = (ds['ocean_mask'].values == 1).astype(np.float32)
    else:
        land_mask = np.ones_like(tar, dtype=np.float32)
    ds.close()
    return tar.astype(np.float32), land_mask

def load_terrain():
    """加载地形数据（960x960）"""
    ds = xr.open_dataset(terrain_file)
    terrain = ds['terrain'].values
    if terrain.shape != (960, 960):
        raise ValueError(f"地形数据尺寸应为960x960，实际为{terrain.shape}")
    terrain = (terrain - np.nanmean(terrain)) / (np.nanstd(terrain) + 1e-6)
    terrain = np.nan_to_num(terrain, nan=0.0)
    ds.close()
    return terrain.astype(np.float32)

def preprocess_data(load_existing=True, force_clean=False):
    train_feat_path = os.path.join(progress_dir, 'train_features.joblib')
    val_feat_path = os.path.join(progress_dir, 'val_features.joblib')
    test_feat_path = os.path.join(progress_dir, 'test_features.joblib')
    train_tar_path = os.path.join(progress_dir, 'train_targets.joblib')
    val_tar_path = os.path.join(progress_dir, 'val_targets.joblib')
    test_tar_path = os.path.join(progress_dir, 'test_targets.joblib')
    feat_stats_path = os.path.join(progress_dir, 'feature_stats.joblib')
    tar_stats_path = os.path.join(progress_dir, 'target_stats.joblib')
    train_feat_clean_path = os.path.join(progress_dir, 'train_features_clean.joblib')
    val_feat_clean_path = os.path.join(progress_dir, 'val_features_clean.joblib')
    test_feat_clean_path = os.path.join(progress_dir, 'test_features_clean.joblib')
    train_tar_clean_path = os.path.join(progress_dir, 'train_targets_clean.joblib')
    val_tar_clean_path = os.path.join(progress_dir, 'val_targets_clean.joblib')
    test_tar_clean_path = os.path.join(progress_dir, 'test_targets_clean.joblib')
    terrain_normalized_path = os.path.join(progress_dir, 'terrain_normalized.joblib') 

    all_exist = all(os.path.exists(f) for f in [
        train_feat_path, val_feat_path, test_feat_path,
        train_tar_path, val_tar_path, test_tar_path,
        feat_stats_path, tar_stats_path, progress_file,
        terrain_normalized_path
    ])

    if load_existing and all_exist and not force_clean:
        print(f"从 {progress_dir} 加载预处理数据")
        try:
            train_features = joblib.load(train_feat_path)
            val_features = joblib.load(val_feat_path)
            test_features = joblib.load(test_feat_path)
            train_targets = joblib.load(train_tar_path)
            val_targets = joblib.load(val_tar_path)
            test_targets = joblib.load(test_tar_path)
            feature_stats = joblib.load(feat_stats_path)
            target_stats = joblib.load(tar_stats_path)
            terrain_normalized = joblib.load(terrain_normalized_path)
            train_feat_mean, train_feat_std = feature_stats['mean'], feature_stats['std']
            train_tar_mean, train_tar_std = target_stats['mean'], target_stats['std']
            progress_data = joblib.load(progress_file)
            test_years = progress_data['test_years']
            test_dates = progress_data['test_dates']
            test_ensembles = progress_data['test_ensembles']
            test_orig_t2m = progress_data['test_orig_t2m']
            test_r2_orig_list = progress_data['test_r2_orig_list']
            test_pcc_orig_list = progress_data['test_pcc_orig_list']
            train_features_clean = joblib.load(train_feat_clean_path)
            val_features_clean = joblib.load(val_feat_clean_path)
            test_features_clean = joblib.load(test_feat_clean_path)
            train_targets_clean = joblib.load(train_tar_clean_path)
            val_targets_clean = joblib.load(val_tar_clean_path)
            test_targets_clean = joblib.load(test_tar_clean_path)
          
            terrain_tensor = torch.tensor(terrain_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            train_terrain_clean = terrain_tensor.expand(len(train_features), 1, 960, 960)
            val_terrain_clean = terrain_tensor.expand(len(val_features), 1, 960, 960)
            test_terrain_clean = terrain_tensor.expand(len(test_features), 1, 960, 960)
            print(f"加载的 terrain_normalized 形状: {terrain_normalized.shape}")
            print(f"train_terrain_clean 形状: {train_terrain_clean.shape}")
            print(f"val_terrain_clean 形状: {val_terrain_clean.shape}")
            print(f"test_terrain_clean 形状: {test_terrain_clean.shape}")
            return (train_features, val_features, test_features,
                    train_targets, val_targets, test_targets,
                    None, None, None,
                    train_feat_mean, train_feat_std,
                    train_tar_mean, train_tar_std,
                    test_years, test_dates, test_ensembles, target_lat, target_lon, input_channels,
                    test_orig_t2m, test_r2_orig_list, test_pcc_orig_list,
                    train_features_clean, val_features_clean, test_features_clean,
                    train_targets_clean, val_targets_clean, test_targets_clean,
                    train_terrain_clean, val_terrain_clean, test_terrain_clean)
        except Exception as e:
            print(f"加载失败: {e}, 将重新生成数据")
            force_clean = True

    print("开始生成第7天专用预处理数据...")
    terrain_data = load_terrain()  # 加载一次地形数据，形状 (960, 960)
    features_all = []
    targets_all = []
    sample_years, sample_dates, sample_ensembles = [], [], []
    test_orig_t2m, test_r2_orig_list, test_pcc_orig_list = [], [], []
    with tqdm(total=len(years) * len(forecast_dates) * num_ensemble, desc="处理样本") as pbar:
        for year in years:
            for f_date in forecast_dates:
                f_datetime = datetime.strptime(f'{year}-{f_date}', '%Y-%m%d')
                target_date = f_datetime + timedelta(days=6)
                target_date_str = target_date.strftime('%m%d')
                sample_id = f"{year}-{target_date_str}"
                file_path = os.path.join(data_dir, f'task_merged_{f_date}_masked.nc')
                if not os.path.exists(file_path):
                    print(f"文件 {file_path} 不存在，跳过")
                    pbar.update(num_ensemble)
                    continue
                try:
                    features_ensemble, raw_t2m = load_day7_features(file_path, year, f_date)
                except Exception as e:
                    print(f"{file_path} 加载失败: {e}")
                    pbar.update(num_ensemble)
                    continue
                target_path = os.path.join(target_data_dir, f'temperature_2m_north_china_{year}_highres.nc')
                target, land_mask = load_day7_target_and_mask(target_path, year, target_date_str)
                if target is None or target.shape != (161, 161):
                    print(f"目标数据 {target_path} 无效或形状不符，跳过")
                    pbar.update(num_ensemble)
                    continue
                if year <= 2020:
                    nan_ratio = np.isnan(target[land_mask == 1]).mean() if np.sum(land_mask == 1) > 0 else 0
                    if nan_ratio > max_nan_ratio:
                        print(f"目标数据 NaN 比例 {nan_ratio:.4f} 超过阈值，跳过")
                        pbar.update(num_ensemble)
                        continue
                if year >= 2021:
                    r2_values = []
                    pcc_values = []
                    for ens_idx in range(num_ensemble):
                        orig = interpolate_to_highres(raw_t2m[ens_idx], s2s_lat, s2s_lon, target_lat, target_lon)
                        mask = ~np.isnan(target) & ~np.isnan(orig) & (land_mask == 1)
                        tar_valid = target[mask]
                        orig_valid = orig[mask]
                        valid_count = len(tar_valid)
                        if valid_count > 0:
                            r2 = r2_score(tar_valid, orig_valid)
                            pcc, _ = pearsonr(tar_valid, orig_valid) if len(tar_valid) > 1 else (np.nan, None)
                            r2_values.append(r2)
                            pcc_values.append(pcc)
                            print(f"{sample_id} 集合成员 {ens_idx + 1} 有效点: {valid_count}, R²: {r2:.4f}, PCC: {pcc:.4f}")
                        else:
                            r2_values.append(np.nan)
                            pcc_values.append(np.nan)
                            print(f"{sample_id} 集合成员 {ens_idx + 1} 无有效点")
                    test_r2_orig_list.append(np.nanmean(r2_values))
                    test_pcc_orig_list.append(np.nanmean(pcc_values))
                    print(f"{sample_id} 原始R²（集合平均）: {np.nanmean(r2_values):.4f}")
                for ens_idx in range(num_ensemble):
                    feat = features_ensemble[ens_idx]
                    mask_channel = interpolate_to_highres(land_mask, target_lat, target_lon, s2s_lat, s2s_lon)
                    mask_channel = (mask_channel >= 0.5).astype(np.float32)
                    feat_with_mask = np.concatenate([feat, mask_channel[np.newaxis, :, :]], axis=0)
                    features_all.append(feat_with_mask)
                    targets_all.append(target)
                    sample_years.append(year)
                    sample_dates.append(target_date.strftime('%Y-%m-%d'))
                    sample_ensembles.append(ens_idx)
                    if year >= 2021:
                        test_orig_t2m.append(raw_t2m[ens_idx])
                    pbar.update(1)
                gc.collect()
    features_all = np.array(features_all, dtype=np.float32)
    targets_all = np.array(targets_all, dtype=np.float32)
    test_orig_t2m = np.array(test_orig_t2m, dtype=np.float32)

  
    train_idx = [i for i, y in enumerate(sample_years) if 2004 <= y <= 2017]
    val_idx = [i for i, y in enumerate(sample_years) if 2018 <= y <= 2020]
    test_idx = [i for i, y in enumerate(sample_years) if 2021 <= y <= 2023]

   
    train_features = features_all[train_idx]
    val_features = features_all[val_idx]
    test_features = features_all[test_idx]
    train_targets = targets_all[train_idx]
    val_targets = targets_all[val_idx]
    test_targets = targets_all[test_idx]
    test_years = [sample_years[i] for i in test_idx]
    test_dates = [sample_dates[i] for i in test_idx]
    test_ensembles = [sample_ensembles[i] for i in test_idx]

  
    train_feat_mean = np.nanmean(train_features, axis=(0, 2, 3), keepdims=True)
    train_feat_std = np.nanstd(train_features, axis=(0, 2, 3), keepdims=True) + 1e-6
    train_tar_mean = np.nanmean(train_targets)
    train_tar_std = np.nanstd(train_targets) + 1e-6
    train_features = (train_features - train_feat_mean) / train_feat_std
    val_features = (val_features - train_feat_mean) / train_feat_std
    test_features = (test_features - train_feat_mean) / train_feat_std

    
    terrain_mean = np.nanmean(terrain_data)
    terrain_std = np.nanstd(terrain_data) + 1e-6
    terrain_normalized = (terrain_data - terrain_mean) / terrain_std
    terrain_normalized = np.nan_to_num(terrain_normalized, nan=0.0).astype(np.float32)

    
    train_features_clean = np.nan_to_num(train_features, nan=0.0)
    val_features_clean = np.nan_to_num(val_features, nan=0.0)
    test_features_clean = np.nan_to_num(test_features, nan=0.0)
    train_targets_clean = np.nan_to_num(train_targets, nan=0.0)
    val_targets_clean = np.nan_to_num(val_targets, nan=0.0)
    test_targets_clean = np.nan_to_num(test_targets, nan=0.0)
    train_features_clean = torch.tensor(train_features_clean, dtype=torch.float32)
    val_features_clean = torch.tensor(val_features_clean, dtype=torch.float32)
    test_features_clean = torch.tensor(test_features_clean, dtype=torch.float32)
    train_targets_clean = torch.tensor(train_targets_clean, dtype=torch.float32)
    val_targets_clean = torch.tensor(val_targets_clean, dtype=torch.float32)
    test_targets_clean = torch.tensor(test_targets_clean, dtype=torch.float32)
    test_orig_t2m = torch.tensor(test_orig_t2m, dtype=torch.float32)

   
    terrain_tensor = torch.tensor(terrain_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    train_terrain_clean = terrain_tensor.expand(len(train_idx), 1, 960, 960)
    val_terrain_clean = terrain_tensor.expand(len(val_idx), 1, 960, 960)
    test_terrain_clean = terrain_tensor.expand(len(test_idx), 1, 960, 960)

   
    joblib.dump(train_features, train_feat_path, compress=3)
    joblib.dump(val_features, val_feat_path, compress=3)
    joblib.dump(test_features, test_feat_path, compress=3)
    joblib.dump(train_targets, train_tar_path, compress=3)
    joblib.dump(val_targets, val_tar_path, compress=3)
    joblib.dump(test_targets, test_tar_path, compress=3)
    joblib.dump({'mean': train_feat_mean, 'std': train_feat_std}, feat_stats_path, compress=3)
    joblib.dump({'mean': train_tar_mean, 'std': train_tar_std}, tar_stats_path, compress=3)
    joblib.dump(terrain_normalized, terrain_normalized_path, compress=3)
    joblib.dump({
        'test_years': test_years,
        'test_dates': test_dates,
        'test_ensembles': test_ensembles,
        'test_orig_t2m': test_orig_t2m,
        'test_r2_orig_list': test_r2_orig_list,
        'test_pcc_orig_list': test_pcc_orig_list
    }, progress_file, compress=3)
    joblib.dump(train_features_clean, train_feat_clean_path, compress=3)
    joblib.dump(val_features_clean, val_feat_clean_path, compress=3)
    joblib.dump(test_features_clean, test_feat_clean_path, compress=3)
    joblib.dump(train_targets_clean, train_tar_clean_path, compress=3)
    joblib.dump(val_targets_clean, val_tar_clean_path, compress=3)
    joblib.dump(test_targets_clean, test_tar_clean_path, compress=3)

    print(f"\n预处理完成后数据统计：")
    print(f"训练集样本数: {len(train_features)}，验证集: {len(val_features)}，测试集: {len(test_features)}")
    print(f"地形数据形状: {terrain_normalized.shape}")
    print(f"train_terrain_clean 形状: {train_terrain_clean.shape}")
    print(f"val_terrain_clean 形状: {val_terrain_clean.shape}")
    print(f"test_terrain_clean 形状: {test_terrain_clean.shape}")
    return (train_features, val_features, test_features,
            train_targets, val_targets, test_targets,
            None, None, None,
            train_feat_mean, train_feat_std,
            train_tar_mean, train_tar_std,
            test_years, test_dates, test_ensembles, target_lat, target_lon, input_channels,
            test_orig_t2m, test_r2_orig_list, test_pcc_orig_list,
            train_features_clean, val_features_clean, test_features_clean,
            train_targets_clean, val_targets_clean, test_targets_clean,
            train_terrain_clean, val_terrain_clean, test_terrain_clean)

def train_model(train_features, train_targets, val_features, val_targets, train_terrain, val_terrain, input_channels):
    train_dataset = TensorDataset(train_features, train_targets, train_terrain)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_features, val_targets, val_terrain)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = ModifiedRes34_Unet(
        inputchannel=input_channels,
        outputchannel=output_channels,
        BN_enable=True,
        resnet_pretrain=True
    ).to(device)
    criterion = NonNaNMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    epochs = 100
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    for epoch in trange(epochs, desc="模型训练", leave=True):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch_feats, batch_tars, batch_terrain in train_loader:
            batch_feats, batch_tars, batch_terrain = batch_feats.to(device), batch_tars.to(device), batch_terrain.to(device)
            optimizer.zero_grad()
            outputs = model(batch_feats, batch_terrain)
            loss = criterion(outputs, batch_tars.unsqueeze(1))
            if torch.isnan(loss) or loss.item() == 0 or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * batch_feats.size(0)
            train_samples += batch_feats.size(0)
        train_loss_avg = train_loss_sum / train_samples if train_samples > 0 else float('nan')
        train_losses.append(train_loss_avg)
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_feats, batch_tars, batch_terrain in val_loader:
                batch_feats, batch_tars, batch_terrain = batch_feats.to(device), batch_tars.to(device), batch_terrain.to(device)
                outputs = model(batch_feats, batch_terrain)
                batch_loss = criterion(outputs, batch_tars.unsqueeze(1))
                if torch.isnan(batch_loss) or batch_loss.item() == 0 or torch.isinf(batch_loss):
                    continue
                val_loss_sum += batch_loss.item() * batch_feats.size(0)
                val_samples += batch_feats.size(0)
        val_loss_avg = val_loss_sum / val_samples if val_samples > 0 else float('nan')
        val_losses.append(val_loss_avg)
        scheduler.step(val_loss_avg)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"第{epoch + 1}轮: 训练损失={train_loss_avg:.6f}, 验证损失={val_loss_avg:.6f}, 学习率={current_lr:.8f}")
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(target_data_dir, 'best_day7_model.pth'))
            print(f"保存最佳模型（验证损失{best_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发（第{epoch + 1}轮）")
                break
    return model, train_losses, val_losses

def evaluate_model(model, test_feats, test_tars, test_terrain, test_years, test_dates, test_ensembles,
                   lat, lon, tar_mean, tar_std, feat_mean, feat_std,
                   test_orig_t2m=None, test_r2_orig_list=None, test_pcc_orig_list=None):
    model.eval()
    model = model.to(device)
    test_dataset = TensorDataset(test_feats, test_tars, test_terrain)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    predictions = []
    test_tars_np = []
    best_model_path = os.path.join(target_data_dir, 'best_day7_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("加载最佳模型权重")
    with torch.no_grad():
        for batch_feats, batch_tars, batch_terrain in tqdm(test_loader, desc="测试中", leave=False):
            batch_feats, batch_terrain = batch_feats.to(device), batch_terrain.to(device)
            outputs = model(batch_feats, batch_terrain)
            print(f"Test output shape: {outputs.shape}")
            outputs_np = outputs.squeeze(1).cpu().numpy()
            predictions.extend(outputs_np)
            test_tars_np.extend(batch_tars.numpy())
    predictions = np.array(predictions)
    test_tars_np = np.array(test_tars_np)
    date_ensemble_map = {}
    for i, (year, date, ens) in enumerate(zip(test_years, test_dates, test_ensembles)):
        key = date
        if key not in date_ensemble_map:
            date_ensemble_map[key] = {'predictions': [], 'targets': [], 'orig_t2m': []}
        date_ensemble_map[key]['predictions'].append(predictions[i])
        date_ensemble_map[key]['targets'].append(test_tars_np[i])
        if test_orig_t2m is not None:
           
            orig_data = test_orig_t2m[i].cpu().numpy() if isinstance(test_orig_t2m[i], torch.Tensor) else test_orig_t2m[i]
            if orig_data.ndim == 0:  # 如果是标量
                print(f"警告: test_orig_t2m[{i}] 是标量，跳过插值")
                orig_highres = np.full((161, 161), np.nan)
            else:
                orig_highres = interpolate_to_highres(orig_data, s2s_lat, s2s_lon, lat, lon)
            date_ensemble_map[key]['orig_t2m'].append(orig_highres)
    mse_list, rmse_list, mae_list, pcc_list, r2_list, accuracy_list = [], [], [], [], [], []
    mse_orig_list, rmse_orig_list, mae_orig_list, pcc_orig_list, r2_orig_list, accuracy_orig_list = [], [], [], [], [], []
    target_path = os.path.join(target_data_dir, f'temperature_2m_north_china_{test_years[0]}_highres.nc')
    ds = xr.open_dataset(target_path)
    land_mask = (ds['ocean_mask'].values == 1).astype(bool) if 'ocean_mask' in ds else np.ones((161, 161), dtype=bool)
    ds.close()
    mask = land_mask
    for key, data in date_ensemble_map.items():
        preds = np.array(data['predictions'])
        targets = np.array(data['targets'])
        orig_t2m = np.array(data['orig_t2m']) if data['orig_t2m'] else None
        target = targets[0]
        mask_valid = mask & ~np.isnan(target)
        tar_valid = target[mask_valid]
        valid_count = len(tar_valid)
        # 模型预测指标
        r2_values = []
        pcc_values = []
        mse_values = []
        rmse_values = []
        mae_values = []
        acc_values = []
        for i in range(len(preds)):
            pred = preds[i][mask_valid]
            valid_mask = ~np.isnan(pred) & ~np.isnan(tar_valid)
            tar_valid_final = tar_valid[valid_mask]
            pred_valid = pred[valid_mask]
            final_valid_count = len(tar_valid_final)
            if final_valid_count > 1:
                mse = mean_squared_error(tar_valid_final, pred_valid)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(tar_valid_final, pred_valid)
                pcc = pearsonr(tar_valid_final, pred_valid)[0] if len(tar_valid_final) > 1 else np.nan
                r2 = r2_score(tar_valid_final, pred_valid)
                acc = np.mean(np.abs(pred_valid - tar_valid_final) < 2)
                mse_values.append(mse)
                rmse_values.append(rmse)
                mae_values.append(mae)
                pcc_values.append(pcc)
                r2_values.append(r2)
                acc_values.append(acc)
                print(f"{key} 模型预测集合成员 {i + 1} 有效点: {final_valid_count}, R²: {r2:.4f}")
            else:
                mse_values.append(np.nan)
                rmse_values.append(np.nan)
                mae_values.append(np.nan)
                pcc_values.append(np.nan)
                r2_values.append(np.nan)
                acc_values.append(np.nan)
                print(f"{key} 模型预测集合成员 {i + 1} 无足够有效点 ({final_valid_count})")
        mse_list.append(np.nanmean(mse_values))
        rmse_list.append(np.nanmean(rmse_values))
        mae_list.append(np.nanmean(mae_values))
        pcc_list.append(np.nanmean(pcc_values))
        r2_list.append(np.nanmean(r2_values))
        accuracy_list.append(np.nanmean(acc_values))
        print(f"{key} 模型预测R²（集合平均）: {np.nanmean(r2_values):.4f}")
     
        if orig_t2m is not None:
            r2_values_orig = []
            pcc_values_orig = []
            mse_values_orig = []
            rmse_values_orig = []
            mae_values_orig = []
            acc_values_orig = []
            for i in range(len(orig_t2m)):
                orig = orig_t2m[i][mask_valid]
                valid_mask = ~np.isnan(orig) & ~np.isnan(tar_valid)
                tar_valid_final = tar_valid[valid_mask]
                orig_valid = orig[valid_mask]
                final_valid_count = len(tar_valid_final)
                if final_valid_count > 1:
                    mse_o = mean_squared_error(tar_valid_final, orig_valid)
                    rmse_o = np.sqrt(mse_o)
                    mae_o = mean_absolute_error(tar_valid_final, orig_valid)
                    pcc_o = pearsonr(tar_valid_final, orig_valid)[0] if len(tar_valid_final) > 1 else np.nan
                    r2_o = r2_score(tar_valid_final, orig_valid)
                    acc_o = np.mean(np.abs(orig_valid - tar_valid_final) < 2)
                    mse_values_orig.append(mse_o)
                    rmse_values_orig.append(rmse_o)
                    mae_values_orig.append(mae_o)
                    pcc_values_orig.append(pcc_o)
                    r2_values_orig.append(r2_o)
                    acc_values_orig.append(acc_o)
                    print(f"{key} 原始预报集合成员 {i + 1} 有效点: {final_valid_count}, R²: {r2_o:.4f}")
                else:
                    mse_values_orig.append(np.nan)
                    rmse_values_orig.append(np.nan)
                    mae_values_orig.append(np.nan)
                    pcc_values_orig.append(np.nan)
                    r2_values_orig.append(np.nan)
                    acc_values_orig.append(np.nan)
                    print(f"{key} 原始预报集合成员 {i + 1} 无足够有效点 ({final_valid_count})")
            mse_orig_list.append(np.nanmean(mse_values_orig))
            rmse_orig_list.append(np.nanmean(rmse_values_orig))
            mae_orig_list.append(np.nanmean(mae_values_orig))
            pcc_orig_list.append(np.nanmean(pcc_values_orig))
            r2_orig_list.append(np.nanmean(r2_orig_list))
            accuracy_orig_list.append(np.nanmean(acc_values_orig))
            print(f"{key} 原始预报R²（集合平均）: {np.nanmean(r2_values_orig):.4f}")
    print(f"模型评价指标均值:")
    print(f"MSE: {np.nanmean(mse_list):.4f}, RMSE: {np.nanmean(rmse_list):.4f}, MAE: {np.nanmean(mae_list):.4f}")
    print(f"PCC: {np.nanmean(pcc_list):.4f}, R2: {np.nanmean(r2_list):.4f}, Accuracy: {np.nanmean(accuracy_list):.4f}")
    if test_orig_t2m is not None:
        print(f"原始预报指标均值:")
        print(f"MSE: {np.nanmean(mse_orig_list):.4f}, RMSE: {np.nanmean(rmse_orig_list):.4f}, MAE: {np.nanmean(mae_orig_list):.4f}")
        print(f"PCC: {np.nanmean(pcc_orig_list):.4f}, R2: {np.nanmean(r2_orig_list):.4f}, Accuracy: {np.nanmean(accuracy_orig_list):.4f}")
    return (np.nanmean(mse_list), np.nanmean(rmse_list), np.nanmean(accuracy_list)), date_ensemble_map, (np.nanmean(mse_orig_list), np.nanmean(rmse_orig_list), np.nanmean(mae_orig_list), np.nanmean(pcc_orig_list), np.nanmean(r2_orig_list), np.nanmean(accuracy_orig_list))

def evaluate_original_forecast(test_orig_t2m, test_targets_clean, test_years, test_dates, test_ensembles, lat, lon,
                               output_dir):
   
    os.makedirs(output_dir, exist_ok=True)
    date_ensemble_map = {}
    for i, (year, date, ens) in enumerate(zip(test_years, test_dates, test_ensembles)):
        key = date
        if key not in date_ensemble_map:
            date_ensemble_map[key] = {'targets': [], 'orig_t2m': []}
        date_ensemble_map[key]['targets'].append(test_targets_clean[i].numpy())
        orig_highres = interpolate_to_highres(test_orig_t2m[i].numpy(), s2s_lat, s2s_lon, lat, lon)
        date_ensemble_map[key]['orig_t2m'].append(orig_highres)

        target_path = os.path.join(target_data_dir, f'temperature_2m_north_china_{test_years[0]}_highres.nc')
    ds = xr.open_dataset(target_path)
    land_mask = (ds['ocean_mask'].values == 1).astype(bool) if 'ocean_mask' in ds else np.ones((161, 161), dtype=bool)
    ds.close()
    mask = land_mask

    mse_orig_list, rmse_orig_list, mae_orig_list, pcc_orig_list, r2_orig_list, accuracy_orig_list = [], [], [], [], [], []
    for key, data in date_ensemble_map.items():
        targets = np.array(data['targets'])
        orig_t2m = np.array(data['orig_t2m'])
        target = targets[0]
        mask_valid = mask & ~np.isnan(target)
        tar_valid = target[mask_valid]
        valid_count = len(tar_valid)
        r2_values_orig = []
        pcc_values_orig = []
        mse_values_orig = []
        rmse_values_orig = []
        mae_values_orig = []
        acc_values_orig = []
        for i in range(len(orig_t2m)):
            orig = orig_t2m[i][mask_valid]
            valid_mask = ~np.isnan(orig) & ~np.isnan(tar_valid)
            tar_valid_final = tar_valid[valid_mask]
            orig_valid = orig[valid_mask]
            final_valid_count = len(tar_valid_final)
            if final_valid_count > 1:
                mse_o = mean_squared_error(tar_valid_final, orig_valid)
                rmse_o = np.sqrt(mse_o)
                mae_o = mean_absolute_error(tar_valid_final, orig_valid)
                pcc_o = pearsonr(tar_valid_final, orig_valid)[0] if len(tar_valid_final) > 1 else np.nan
                r2_o = r2_score(tar_valid_final, orig_valid)
                acc_o = np.mean(np.abs(orig_valid - tar_valid_final) < 2)
                mse_values_orig.append(mse_o)
                rmse_values_orig.append(rmse_o)
                mae_values_orig.append(mae_o)
                pcc_values_orig.append(pcc_o)
                r2_values_orig.append(r2_o)
                acc_values_orig.append(acc_o)
                print(f"{key} 原始预报集合成员 {i + 1} 有效点: {final_valid_count}, R²: {r2_o:.4f}")
            else:
                mse_values_orig.append(np.nan)
                rmse_values_orig.append(np.nan)
                mae_values_orig.append(np.nan)
                pcc_values_orig.append(np.nan)
                r2_values_orig.append(np.nan)
                acc_values_orig.append(np.nan)
                print(f"{key} 原始预报集合成员 {i + 1} 无足够有效点 ({final_valid_count})")
        mse_orig_list.append(np.nanmean(mse_values_orig))
        rmse_orig_list.append(np.nanmean(rmse_values_orig))
        mae_orig_list.append(np.nanmean(mae_values_orig))
        pcc_orig_list.append(np.nanmean(pcc_values_orig))
        r2_orig_list.append(np.nanmean(r2_values_orig))
        accuracy_orig_list.append(np.nanmean(acc_values_orig))
        print(f"{key} 原始预报R²（集合平均）: {np.nanmean(r2_values_orig):.4f}")
    print(f"原始预报指标均值（训练前）:")
    print(f"MSE: {np.nanmean(mse_orig_list):.4f}, RMSE: {np.nanmean(rmse_orig_list):.4f}, MAE: {np.nanmean(mae_orig_list):.4f}")
    print(f"PCC: {np.nanmean(pcc_orig_list):.4f}, R2: {np.nanmean(r2_orig_list):.4f}, Accuracy: {np.nanmean(accuracy_orig_list):.4f}")
    return (np.nanmean(mse_orig_list), np.nanmean(rmse_orig_list), np.nanmean(mae_orig_list),
            np.nanmean(pcc_orig_list), np.nanmean(r2_orig_list), np.nanmean(accuracy_orig_list))

def plot_loss_curve(train_losses, val_losses, output_dir):
  
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    
    output_dir = os.path.join(target_data_dir, 'results090903_day7')
    os.makedirs(output_dir, exist_ok=True)

    
    (train_features, val_features, test_features,
     train_targets, val_targets, test_targets,
     _, _, _,
     train_feat_mean, train_feat_std,
     train_tar_mean, train_tar_std,
     test_years, test_dates, test_ensembles, lat, lon, input_channels,
     test_orig_t2m, test_r2_orig_list, test_pcc_orig_list,
     train_features_clean, val_features_clean, test_features_clean,
     train_targets_clean, val_targets_clean, test_targets_clean,
     train_terrain_clean, val_terrain_clean, test_terrain_clean) = preprocess_data(load_existing=True)

    
    orig_metrics = evaluate_original_forecast(
        test_orig_t2m, test_targets_clean, test_years, test_dates, test_ensembles, lat, lon, output_dir
    )
    
    model, train_losses, val_losses = train_model(
        train_features_clean, train_targets_clean, val_features_clean, val_targets_clean,
        train_terrain_clean, val_terrain_clean, input_channels
    )

   
    plot_loss_curve(train_losses, val_losses, output_dir)

    
    orig_metrics = evaluate_original_forecast(
        test_orig_t2m, test_targets_clean, test_years, test_dates, test_ensembles, lat, lon, output_dir
    )

   
    model_metrics, date_ensemble_map, orig_metrics_post = evaluate_model(
        model, test_features_clean, test_targets_clean, test_terrain_clean,
        test_years, test_dates, test_ensembles, lat, lon,
        train_tar_mean, train_tar_std, train_feat_mean, train_feat_std,
        test_orig_t2m, test_r2_orig_list, test_pcc_orig_list
    )

   
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"MSE: {model_metrics[0]:.4f}\nRMSE: {model_metrics[1]:.4f}\nMAE: {model_metrics[2]:.4f}\n")
        f.write(f"PCC: {model_metrics[3]:.4f}\nR2: {model_metrics[4]:.4f}\nAccuracy: {model_metrics[5]:.4f}\n")
        f.write(f"MSE: {orig_metrics_post[0]:.4f}\nRMSE: {orig_metrics_post[1]:.4f}\nMAE: {orig_metrics_post[2]:.4f}\n")
        f.write(f"PCC: {orig_metrics_post[3]:.4f}\nR2: {orig_metrics_post[4]:.4f}\nAccuracy: {orig_metrics_post[5]:.4f}\n")
        f.write(f"MSE: {orig_metrics[0]:.4f}\nRMSE: {orig_metrics[1]:.4f}\nMAE: {orig_metrics[2]:.4f}\n")
        f.write(f"PCC: {orig_metrics[3]:.4f}\nR2: {orig_metrics[4]:.4f}\nAccuracy: {orig_metrics[5]:.4f}\n")

    print("训练和评估完成，结果已保存至:", output_dir)

if __name__ == "__main__":
    main()

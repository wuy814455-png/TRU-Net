# TRU-Net
Terrain-aware U-Net for S2S Temperature Bias Correction and Downscaling
# TRU-Net: Terrain-aware Residual U-Net for Subseasonal 2-m Temperature Bias Correction and Downscaling

Official implementation of the manuscript:

> **TRU-Net: Terrain-aware Residual U-Net for Subseasonal 2-m Temperature Bias Correction and Downscaling**

---

# Overview

This repository provides the official implementation of **TRU-Net**, a terrain-aware deep learning framework for subseasonal (day 7–42) 2-m air temperature bias correction and downscaling.

The proposed framework is designed for S2S (Subseasonal-to-Seasonal) temperature prediction over North China. By incorporating multi-scale terrain information into a residual U-Net architecture, TRU-Net improves the representation of terrain-induced spatial heterogeneity and enhances forecast skill over regions with complex topography.

The repository contains the minimum set of source code required to reproduce the main experiments presented in the manuscript, including

- model architectures,
- baseline implementations,
- preprocessing,
- training,
- evaluation.

---

# Repository Structure

```
TRU-Net/
│
├── models/
│   ├── trunet.py
│   └── baselines.py
│
├── preprocess_and_train.py
│
├── requirements.txt
│
└── README.md
```

### Description

### models/

Contains the neural network implementations used in this study.

- **trunet.py**

  Implementation of the proposed TRU-Net.

- **baselines.py**

  Implementations of baseline model, including

  - U-Net

---

### preprocess_and_train.py

Contains the complete workflow including

- data preprocessing
- feature preparation
- model training
- validation
- evaluation

The implementation follows the methodology described in the manuscript.

---

# Requirements

The experiments were conducted under the following environment.

- Python 3.9
- PyTorch 2.5.1
- CUDA 11.8

Required Python packages are listed in

```
requirements.txt
```

Install all dependencies by

```bash
pip install -r requirements.txt
```

---

# Data

The datasets used in this study are publicly available.

## Forecast Data

ECMWF S2S Reforecast Dataset

https://apps.ecmwf.int/datasets

## Target Data

ERA5-Land Reanalysis Dataset

https://cds.climate.copernicus.eu/

Because of the large volume of these datasets, they are **not included** in this repository.

Users should download the original datasets from the official providers.

---

# Data Preparation

Before training, users should preprocess the downloaded datasets following the procedure described in Section XX of the manuscript.

The preprocessing includes

- extracting the study region
- temporal matching
- spatial interpolation
- terrain preparation
- normalization
- generation of training samples

The preprocessing pipeline is implemented in

```
preprocess_and_train.py
```

---

# Training

To train the proposed model,

```bash
python preprocess_and_train.py
```

The script performs

1. preprocessing

2. model training

3. validation

4. evaluation

---

# Baseline Models

The repository includes the implementations of all baseline models used in the paper.

These models are implemented under

```
models/baselines.py
```

and are trained under the same preprocessing and evaluation framework as TRU-Net.

---

# Evaluation

The evaluation follows the experimental protocol described in the manuscript.

Metrics include

- RMSE
- PCC
- Accuracy

The evaluation procedure is integrated into

```
preprocess_and_train.py
```

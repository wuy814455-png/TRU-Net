# TRU-Net

**Terrain-aware Residual U-Net for Subseasonal 2-m Temperature Bias Correction and Downscaling**

This repository contains the implementation of TRU-Net, its ablation variants, the U-Net baseline, and the scripts used for data preprocessing, model training, and evaluation.

---

## Repository Structure

```text
TRU-Net/
├── models/
│   ├── TRU-net.py
│   └── baselines/
│       ├── NoTerrain.py
│       ├── Simple.py
│       ├── Single.py
│       ├── SWINIR.py
│       ├── BiConvLSTM.py
│       ├── ModifiedUNet.py
│       └── unet.py
├── process.py
├── train.py
├── evaluate.py
├── requirement
├── LICENSE
└── README.md
````

### File Description

* `models/TRU-net.py`: implementation of the proposed TRU-Net model.
* `models/baselines/NoTerrain`: TRU-Net variant without terrain information.
* `models/baselines/Simple`: simplified terrain-processing variant.
* `models/baselines/Single`: single-stage terrain-fusion variant.
* `models/baselines/unet.py`: U-Net baseline.
* * `models/baselines/SWINIR.py`: Comparsion model.
  * * `models/baselines/BiConvLSTM.py`: Comparsion model.
    * * `models/baselines/ModifiedUNet.py`: Comparsion model.
* `process.py`: data preprocessing and dataset preparation.
* `train.py`: model training and validation.
* `evaluate.py`: model evaluation and metric calculation.

---

## Requirements

The experiments were conducted using:

* Python 3.9
* PyTorch 2.5.1
* CUDA 11.8

Install the required packages using:

```bash
pip install -r requirement
```

---

## Data

The datasets used in this study are publicly available but are not included in this repository because of their large size.

### Forecast Data

ECMWF S2S reforecast data:

[https://apps.ecmwf.int/datasets](https://apps.ecmwf.int/datasets)

### Target Data

ERA5-Land reanalysis data:

[https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

---

## Usage

The workflow is divided into preprocessing, training, and evaluation.

### 1. Data Preprocessing

Run:

```bash
python process.py
```
The processed training, validation, and test data are saved for use by the training and evaluation scripts.

### 2. Model Training

Run:

```bash
python train.py
```

The model and data paths should be checked in the script before training.

### 3. Model Evaluation

Run:

```bash
python evaluate.py
```

This script loads the processed test data and the saved model checkpoint, calculates the evaluation metrics, and saves the evaluation results.

---

## Included Models

The following models are included in this repository:

### Proposed Model

* TRU-Net

### Ablation Variants

* NoTerrain
* Simple
* Single-stage

### Baseline Model

* U-Net

The included models use the same preprocessing and evaluation procedures.

---

## Additional Literature Baselines

The manuscript also compares TRU-Net with three deep learning methods reported in previous studies.

The corresponding references are provided below.

### Modified U-Net

[https://doi.org/10.1016/j.heliyon.2024.e35933](https://doi.org/10.1016/j.heliyon.2024.e35933)

### BiConvLSTM

[https://doi.org/10.1002/qj.4989](https://doi.org/10.1002/qj.4989)

### SwinIR

[https://doi.org/10.1002/qj.4596](https://doi.org/10.1002/qj.4596)

---

## Evaluation Metrics

The main evaluation metrics are:

* Root Mean Square Error (RMSE)
* Pearson Correlation Coefficient (PCC)
* Accuracy

---

## License

This project is released under the terms specified in the [LICENSE](LICENSE) file.

---

## Citation

Please cite the corresponding manuscript when using this repository.

The complete citation information will be added after publication.

```
```

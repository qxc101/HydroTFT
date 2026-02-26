# HydroTFT: Temporal Fusion Transformer for Multi-Horizon Hydrological Forecasting

This repository extends the [Kratzert et al. (2019) EA-LSTM codebase](https://github.com/kratzert/ealstm_regional_modeling) with a Temporal Fusion Transformer (TFT) for multi-horizon streamflow prediction on the CAMELS US dataset (531 basins).

## Setup

### 1. Clone the base repository

```bash
git clone https://github.com/kratzert/ealstm_regional_modeling.git
cd ealstm_regional_modeling
```

### 2. Apply our modifications

Replace/add the following files from our upload into the cloned repository:

**Replace existing files:**
```
main.py
papercode/tft.py
papercode/evalutils.py
papercode/datasets.py
papercode/datautils.py
papercode/nseloss.py
papercode/plotutils.py
papercode/utils.py
notebooks/performance.ipynb
```

**Add new files:**
```
data/basin_list_quick5.txt
data/basin_list_quick50.txt
```

### 3. Download the CAMELS dataset

Follow the data download instructions in the [original README](https://github.com/kratzert/ealstm_regional_modeling#data-needed). You need:
- CAMELS time series meteorology, observed flow, and meta data
- CAMELS Attributes
- Updated Maurer forcings (with daily min/max temperature)
- CAMELS benchmark model simulations (for Table 1 comparisons)

### 4. Python environment

```bash
conda env create -f environment_gpu.yml   # or environment_cpu.yml
conda activate ealstm
pip install einops  # additional dependency for TFT
```

## Usage

### Train

```bash
# TFT 1-day forecast (vanilla architecture, pred_days=0 nowcast)
python main.py train --camels_root /path/to/CAMELS \
    --model_type tft --pred_days 0 --use_starter_features \
    --seq_length 270 --learning_rate 1e-3 --dropout 0.4 \
    --epochs 30 --seed 456 --cache_data True

# TFT 7-day forecast (v3f architecture)
python main.py train --camels_root /path/to/CAMELS \
    --model_type tft --pred_days 7 --use_starter_features \
    --seq_length 270 --learning_rate 1e-3 --dropout 0.4 \
    --epochs 30 --seed 456 --cache_data True

# EA-LSTM baseline (unchanged from original)
python main.py train --camels_root /path/to/CAMELS --seed 456 --cache_data True

# LSTM baseline with static concatenation
python main.py train --camels_root /path/to/CAMELS \
    --concat_static True --seed 456 --cache_data True
```

### Evaluate

```bash
python main.py evaluate --camels_root /path/to/CAMELS --run_dir runs/<run_folder>

# Optionally evaluate at a specific epoch (instead of the last)
python main.py evaluate --camels_root /path/to/CAMELS --run_dir runs/<run_folder> --eval_epoch 20
```

### Key command-line options

| Flag | Description |
|------|-------------|
| `--model_type tft` | Use TFT (default is EA-LSTM) |
| `--pred_days N` | Forecast horizon in days. 0 = nowcast (vanilla TFT), >=1 = forecast (v3f TFT) |
| `--use_starter_features` | Add 5 engineered features: doy_sin, doy_cos, prcp_sum_90, degday_7, wetdays_7 |
| `--seq_length N` | Input sequence length (default 270) |
| `--eval_epoch N` | Evaluate model checkpoint at a specific epoch |
| `--no_attention True` | TFT ablation: disable self-attention |
| `--no_feature_selection True` | TFT ablation: disable variable selection networks |

## Modified files summary

| File | Changes |
|------|---------|
| `main.py` | TFT training/evaluation loop, multi-step prediction, starter features, `--eval_epoch` |
| `papercode/tft.py` | Merged TFT file containing both vanilla (`VanillaTFT`) and v3f (`TFT`) architectures |
| `papercode/evalutils.py` | Added `eval_tft_models`, `eval_tft_models_all_steps` for per-day evaluation |
| `papercode/datasets.py` | Multi-step target support for TFT |
| `papercode/datautils.py` | Starter feature engineering, multi-step data loading |
| `papercode/nseloss.py` | Multi-step NSE loss |
| `papercode/plotutils.py` | TFT draw style entries for plots |
| `papercode/utils.py` | Minor utilities |
| `notebooks/performance.ipynb` | NSE tables: 1-day (with benchmarks), 7-day and 14-day per-day breakdowns |

## Analysis notebooks and experiment scripts

Shell scripts (`run_experiments.sh`, `run_ablation.sh`, etc.) and additional analysis notebooks will be made available upon publication to preserve anonymity during review.

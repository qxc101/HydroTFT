# HydroTFT: Temporal Fusion Transformer for Multi-Horizon Hydrological Forecasting
## Note
This is the initial upload with the main focus of helping the reviewers to understand our paper and show reproducibility of our project. The current upload is already self-contained and functional but full version will be released upon publication. 

Shell scripts, model checkpointsm, and analysis notebooks will be made available upon publication to preserve anonymity during review.
## Setup

### 1. Clone the baseline repository

```bash
git clone https://github.com/kratzert/ealstm_regional_modeling.git
cd ealstm_regional_modeling
```

### 2. Apply our modifications

Replace/add the following files from our repo into the cloned repository:

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
| `--model_type tft` | Choose the model to use, can be tft, lstm, ealstm |
| `--pred_days N` | Forecast horizon in days. N can be 0 (nowcast), or any positive int. |
| `--use_starter_features True` | Add 5 engineered features as input. |
| `--seq_length N` | Input sequence length (default 270) |


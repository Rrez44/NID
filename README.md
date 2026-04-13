# Intrusion Detection System (IDS) - Thesis Project

A machine learning-based Intrusion Detection System using XGBoost to classify network traffic by attack type (multiclass).

## Project Overview

This project implements an IDS that uses the CICIDS2017 dataset to train an XGBoost classifier for **multiclass** classification: Normal Traffic, DoS, DDoS, Port Scanning, Brute Force, Web Attacks, and Bot. The system includes a leak-safe preprocessing pipeline (deduplication, feature selection fit on training data only, attack grouping) and multiple train/test split strategies to avoid session leakage.

## Project Structure

```
ids-thesis/
├── config.py                       # Path configuration (auto-detects project root)
├── Makefile                        # Pipeline automation (make all, make train, etc.)
├── requirements.txt                # Python dependencies
├── training/
│   ├── common.py                   # sys.path setup for imports
│   ├── merge_data.py               # Merge multiple CSV files
│   ├── preprocess.py               # Leak-safe data preprocessing
│   ├── feature_selection.py        # Post-split feature selector (fit on train only)
│   ├── train_xgb.py                # Model training (XGBoost / Random Forest)
│   └── inference.py                # Batch inference
├── api/
│   ├── __init__.py
│   └── serve.py                    # FastAPI inference endpoint
├── models/                         # Trained models and artifacts
│   ├── xgb_ids_model.pkl
│   ├── xgb_ids_model_label_mapping.pkl
│   ├── xgb_ids_model_feature_names.pkl
│   ├── xgb_ids_model_feature_selector.pkl
│   ├── xgb_ids_model_feature_importance.csv
│   └── experiment_*.json           # Experiment logs
├── data/
│   ├── raw/                        # Raw dataset files
│   │   ├── MachineLearningCVE/     # CIC-IDS2017 per-day CSVs
│   │   └── cicids2018/             # CSE-CIC-IDS2018 per-day CSVs (optional)
│   └── merged/                     # Processed data
│       ├── MachineLearningCSV_merged.csv
│       └── MachineLearningCSV_cleaned.csv
├── README.md
└── LICENSE
```

## Key Design Decisions

### No Data Leakage
Feature selection (zero-variance removal, correlation filtering, irrelevant-feature removal) is performed **after** the train/test split, fitted on training data only. This is handled by `FeatureSelector` in `training/feature_selection.py`. The preprocessing script (`preprocess.py`) only applies structural transforms that don't depend on data statistics.

### Class Imbalance Handling
Inverse-frequency sample weights (`sklearn.utils.class_weight.compute_sample_weight('balanced')`) are applied during training so minority classes (Bot, Web Attacks) contribute proportionally to the loss.

### Session Leakage Prevention
GroupKFold cross-validation uses `Source_File` as the group key, ensuring no flows from the same capture file appear in both train and validation folds.

### Cross-Dataset Generalization
The pipeline supports **two datasets simultaneously** — CIC-IDS2017 (the original) and CSE-CIC-IDS2018. Rows are tagged with a `Dataset` column during merge, which unlocks two extra split strategies in [training/train_xgb.py](training/train_xgb.py):

- `cross_dataset_2017to2018` — train on all 2017, test on all 2018 (zero-shot)
- `cross_dataset_2018to2017` — reverse
- `mixed_holdout` — 70/30 file-holdout split stratified by year so each dataset appears in both train and test

Cross-dataset splits are the honest test of whether the model learned generalizable attack patterns vs. host-specific fingerprints (e.g. `Init_Win_bytes_backward == 235` for a specific Apache build in 2017).

## Installation

```bash
git clone <repository-url>
cd ids-thesis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datasets

### CIC-IDS2017 (required)
Place per-day CSVs under `data/raw/MachineLearningCVE/`. Source: https://www.unb.ca/cic/datasets/ids-2017.html.

### CSE-CIC-IDS2018 (optional, enables cross-dataset evaluation)
Place the 10 `TrafficForML_CICFlowMeter.csv` files under `data/raw/cicids2018/`:

```
data/raw/cicids2018/
├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv   # FTP/SSH Brute Force
├── Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv    # DoS (GoldenEye, Slowloris)
├── Friday-16-02-2018_TrafficForML_CICFlowMeter.csv      # DoS (Hulk, SlowHTTPTest)
├── Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv    # DDoS (LOIC-HTTP) — note misspelled filename
├── Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv   # DDoS (LOIC-UDP, HOIC)
├── Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv    # Web Attacks
├── Friday-23-02-2018_TrafficForML_CICFlowMeter.csv      # Web Attacks
├── Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv   # Infiltration
├── Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv    # Infiltration
└── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv      # Bot
```

Total size: ~6.5 GB. Source: https://www.unb.ca/cic/datasets/ids-2018.html (or `aws s3 sync --no-sign-request s3://cse-cic-ids2018/Processed\ Traffic\ Data\ for\ ML\ Algorithms/ data/raw/cicids2018/`).

Once present, `merge_data.py` picks up both directories automatically; no config changes needed.

## Quick Start (Makefile)

```bash
make all                         # Full pipeline: merge → preprocess → train
make train-rf                    # Random Forest baseline
make train-tune                  # XGBoost with Optuna hyperparameter tuning
make train-ablation              # XGBoost without Destination Port (ablation study)
make train-mixed                 # 2017+2018 mixed file-holdout (requires 2018 data)
make train-cross-2017to2018      # Zero-shot: train on 2017, test on 2018
make train-cross-2018to2017      # Zero-shot: train on 2018, test on 2017
make shap                        # Train + SHAP interpretability analysis
make serve                       # Start FastAPI inference server on port 8000
make help                        # Show all available targets
```

## Usage (Manual)

### 1. Merge Raw Data Files

```bash
# CIC-IDS2017 only (default when cicids2018/ is absent)
python3 training/merge_data.py

# Explicitly merge both 2017 and 2018
python3 training/merge_data.py \
    --input-dir data/raw/MachineLearningCVE \
    --input-dir data/raw/cicids2018 \
    --output data/merged/MachineLearningCSV_merged.csv

# Downsample huge 2018 files during merge (e.g. 500k rows per file)
python3 training/merge_data.py --max-rows-per-file 500000
```

If `data/raw/cicids2018/` exists, `merge_data.py` picks it up automatically and tags rows with a `Dataset` column (`'2017'` / `'2018'`). The script also:

- Canonicalizes 2018 column names to the 2017 schema (e.g. `Dst Port` → `Destination Port`, `Tot Fwd Pkts` → `Total Fwd Packets`).
- Drops 2017's duplicate `Fwd Header Length.1`.
- Drops 2018-only metadata (`Protocol`, `Timestamp`, `Flow ID`, `Src IP`, etc.).
- Filters stray embedded header rows from the Excel-truncated 2018 files.
- Streams in chunks (`--chunksize`) to keep peak memory low — the 4 GB Tuesday 2018 file won't blow up your RAM.

### 2. Preprocess Data

```bash
python3 training/preprocess.py
```

This step: removes identifiers (IP, Flow ID, Timestamp), deduplicates rows and columns, handles infinities/NaN, groups 15 attack labels into 7 semantic categories, drops rare classes (Infiltration, Heartbleed).

**Note:** Zero-variance, correlation, and irrelevant-feature removal are now handled post-split inside the training script to prevent data leakage.

### 3. Train Model

```bash
# XGBoost with file-holdout split (default, recommended)
python3 training/train_xgb.py --split-strategy file_holdout --save-plots

# Random Forest baseline for comparison
python3 training/train_xgb.py --model-type rf --model models/rf_ids_model.pkl

# With Optuna hyperparameter tuning (50 trials)
python3 training/train_xgb.py --tune --tune-method optuna

# With legacy RandomizedSearchCV tuning
python3 training/train_xgb.py --tune --tune-method random

# Destination Port ablation study
python3 training/train_xgb.py --drop-port --model models/xgb_no_port.pkl

# SHAP interpretability analysis
python3 training/train_xgb.py --shap --save-plots

# Disable class weights (for comparison)
python3 training/train_xgb.py --no-class-weights
```

**Split strategies:** `file_holdout` (default), `per_file`, `temporal`, `file`, `random`, `mixed_holdout`, `cross_dataset_2017to2018`, `cross_dataset_2018to2017`. The last three require the `Dataset` column from a two-year merge.

#### Cross-dataset evaluation (CIC-IDS2017 ↔ CSE-CIC-IDS2018)

```bash
# Assumes data/raw/cicids2018/ is populated and you've re-run merge + preprocess.
python3 training/train_xgb.py --split-strategy cross_dataset_2017to2018 --save-plots
python3 training/train_xgb.py --split-strategy cross_dataset_2018to2017 --save-plots
python3 training/train_xgb.py --split-strategy mixed_holdout --save-plots
```

A large gap between in-dataset F1-macro (e.g. 0.95 on `file_holdout`) and cross-dataset F1-macro (e.g. 0.25 on `cross_dataset_2017to2018`) is the quantitative signature of host-specific overfitting: the model relies on fingerprints like `Init_Win_bytes_backward` that are stable within a capture but change across years/tools.

### 4. Run Inference

```bash
python3 training/inference.py \
    --input data/test_data.csv \
    --model models/xgb_ids_model.pkl \
    --output predictions.csv
```

### 5. API Server

```bash
uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `POST /predict` — classify a single flow (pass features as JSON)
- `GET /health` — liveness check
- `GET /classes` — list known attack classes

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Destination Port": 80, "Flow Duration": 1234, ...}}'
```

## Live IDS CLI

The `live_ids` package adds a terminal application that captures live network
traffic (or replays a PCAP), assembles bidirectional flows, transforms each
completed flow into the 40-feature dict the trained model expects, and POSTs
each one to the running FastAPI server. The CLI is a **pure client** of
`POST /predict` — it never loads the model itself.

### Install

```bash
sudo apt install libpcap-dev          # libpcap headers (Debian/Ubuntu)
pip install -r requirements.txt       # adds nfstream, scapy, httpx
```

For live capture without sudo, grant raw-socket capability to the venv
interpreter once:

```bash
sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python))
```

### Quickstart

```bash
# Terminal 1: train (once) and start the API
make train
make serve

# Terminal 2: self-check the feature mapping against the model artifact
python -m live_ids validate

# Terminal 3a: replay a PCAP through the pipeline
python -m live_ids capture --pcap sample.pcap

# Terminal 3b: live capture, only print non-benign predictions
python -m live_ids capture -i eth0 --bpf-filter 'tcp' --only-non-benign
```

`make live-validate` and `make live-capture IFACE=eth0` wrap the above.

### Subcommands

| Command | Purpose |
|---|---|
| `capture` | Capture packets (`-i IFACE`) or replay a PCAP (`--pcap FILE`); POST each completed flow to the API and print predictions |
| `validate` | Assert that `live_ids/features.py:FEATURE_NAMES` matches `models/xgb_ids_model_feature_names.pkl`; smoke-test a synthetic flow; with `--pcap`, also run a real PCAP through the mapping |
| `info` | Print the active capture engine (`nfstream` or `scapy`) and the API `/health` + `/classes` |

Notable `capture` flags: `--engine {auto,nfstream,scapy}`, `--api-url`,
`--idle-timeout`, `--active-timeout`, `--active-gap`, `--bpf-filter`,
`--min-confidence`, `--only-non-benign`, `--json`, `--dry-run`,
`--max-flows`, `--debug`. Run `python -m live_ids capture --help` for the
full list.

### Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `IDS API unreachable` | `make serve` not running | start the API |
| `model not loaded` | server up but `.pkl` artifacts missing | `make train` |
| `permission denied` on live capture | no `CAP_NET_RAW` | use sudo or `setcap` (above) |
| `Missing features: [...]` (HTTP 422) | feature-mapping drift | `python -m live_ids validate` and inspect `FEATURE_SOURCE` |

### Approximations

`live_ids` provides approximate CICFlowMeter compatibility via nfstream plus
an `NFPlugin` hook (or a scapy fallback when nfstream is unavailable).
Features marked `S` (synthesized) in `live_ids/features.py:FEATURE_SOURCE`
are computed from raw per-packet state inside the capture loop. The
`validate` subcommand verifies that all 40 expected feature names are
produced and finite. Active/Idle period statistics are clipped at 8192
packets per flow (`MAX_TS_PER_FLOW` in `live_ids/capture.py`); flows beyond
that cap get approximate Active stats.

## Evaluation Metrics

The training script reports both **weighted** and **macro** averaged metrics:

| Metric | Averaging | Purpose |
|--------|-----------|---------|
| Accuracy | — | Overall correctness |
| Precision | weighted + macro | False positive control |
| Recall | weighted + macro | False negative control |
| F1-Score | weighted + macro | Balanced measure |
| ROC-AUC | macro (OvR) | Ranking quality |

Per-class precision/recall/F1/support is logged and saved in the experiment JSON.

## Output Artifacts

After training, the following files are generated in `models/`:

| File | Description |
|------|-------------|
| `xgb_ids_model.pkl` | Trained model (joblib serialised) |
| `*_label_mapping.pkl` | Index → class-name mapping |
| `*_feature_names.pkl` | Ordered feature list (for inference alignment) |
| `*_feature_selector.pkl` | Fitted FeatureSelector (documents what was dropped) |
| `*_feature_importance.csv` | Feature importance scores |
| `experiment_*.json` | Full experiment record (params, metrics, data hash) |
| `confusion_matrix.png` | Confusion matrix plot (if `--save-plots`) |
| `shap_summary_bar.png` | SHAP bar summary (if `--shap`) |
| `shap_beeswarm_*.png` | Per-class SHAP beeswarm (if `--shap`) |
| `shap_values.csv` | Mean |SHAP| per feature per class (if `--shap`) |

## Experiment Log Schema

Each training run writes a JSON file:

```json
{
  "timestamp": "2026-03-23T12:00:00+00:00",
  "model_type": "xgb",
  "params": {"max_depth": 6, "n_estimators": 100, "...": "..."},
  "split_strategy": "file_holdout",
  "tune_method": null,
  "n_train": 1800000,
  "n_test": 600000,
  "classes": ["Bot", "Brute Force", "DDoS", "DoS", "Normal Traffic", "Port Scanning", "Web Attacks"],
  "metrics": {
    "accuracy": 0.9876,
    "f1_weighted": 0.9870,
    "f1_macro": 0.9234,
    "precision_macro": 0.9345,
    "recall_macro": 0.9123,
    "roc_auc_macro": 0.9876,
    "per_class": {
      "Normal Traffic": {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 480000},
      "Bot": {"precision": 0.85, "recall": 0.78, "f1": 0.81, "support": 1200}
    }
  },
  "cv_f1_macro": 0.9200,
  "feature_selection": {
    "corr_threshold": 0.95,
    "n_features_out": 40,
    "n_correlated": 5,
    "n_zero_var": 2,
    "n_irrelevant": 8
  },
  "dataset_hash": "a1b2c3d4e5f6g7h8",
  "class_weights": true,
  "drop_port": false
}
```

## License

https://github.com/Rrez44/NID/blob/main/LICENSE

## Author

Rrezon Beqiri, University Of Prishtin

## Citation

If you use this code in your research, please cite!

## Acknowledgments

- CICIDS2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Optuna: https://optuna.readthedocs.io/

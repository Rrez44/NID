# Intrusion Detection System (IDS) - Thesis Project

A machine learning-based Intrusion Detection System using XGBoost to classify network traffic as BENIGN or ATTACK.

## Project Overview

This project implements an IDS that uses the CICIDS2017 dataset to train an XGBoost classifier for binary classification of network traffic. The system processes network flow features and predicts whether traffic is benign or malicious.

## Project Structure

```
ids-thesis/
├── config.py                   # Configuration file (auto-detects project root)
├── data/
│   ├── raw/                    # Raw dataset files
│   │   ├── MachineLearningCSV.zip
│   │   └── MachineLearningCVE/ # Individual CSV files
│   └── merged/                  # Processed data
│       ├── MachineLearningCSV_merged.csv
│       └── MachineLearningCSV_cleaned.csv
├── training/                    # Training scripts
│   ├── merge_data.py           # Merge multiple CSV files
│   ├── preprocess.py           # Data preprocessing
│   ├── train_xgb.py            # Model training
│   └── inference.py            # Model inference
├── models/                      # Trained models and outputs
│   ├── xgb_ids_model.pkl
│   ├── xgb_ids_model_label_mapping.pkl
│   └── xgb_ids_model_feature_importance.csv
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

The project uses a `config.py` file that automatically detects the project root directory and provides relative paths for all data and model files. This makes the project portable - you can move it to any location and it will work without modifying any paths.

All scripts use relative paths by default, but you can still override them using command-line arguments if needed.

## Features

- **Data Processing Pipeline**: Automated merging and preprocessing of network traffic data
- **XGBoost Classifier**: High-performance gradient boosting model
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, ROC-AUC
- **Hyperparameter Tuning**: Optional randomized search for optimal parameters
- **Feature Importance Analysis**: Identifies most important features for classification
- **Model Persistence**: Save and load trained models for inference
- **Visualization**: Confusion matrix and ROC curve plots

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ids-thesis
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Merge Raw Data Files

If you have multiple CSV files in the raw data directory, merge them first:

```bash
python training/merge_data.py \
    --input-dir data/raw/MachineLearningCVE \
    --output data/merged/MachineLearningCSV_merged.csv
```

### 2. Preprocess Data

Clean and preprocess the merged data:

```bash
python training/preprocess.py \
    --input data/merged/MachineLearningCSV_merged.csv \
    --output data/merged/MachineLearningCSV_cleaned.csv
```

This step:
- Removes non-generalizable columns (IPs, Flow IDs, Timestamps)
- Handles missing values and infinities
- Converts labels to binary format (BENIGN/ATTACK)

### 3. Train Model

Train the XGBoost classifier:

```bash
# Basic training with default hyperparameters
python training/train_xgb.py \
    --input data/merged/MachineLearningCSV_cleaned.csv \
    --model models/xgb_ids_model.pkl

# With hyperparameter tuning (slower but may improve performance)
python training/train_xgb.py \
    --input data/merged/MachineLearningCSV_cleaned.csv \
    --model models/xgb_ids_model.pkl \
    --tune

# With evaluation plots
python training/train_xgb.py \
    --input data/merged/MachineLearningCSV_cleaned.csv \
    --model models/xgb_ids_model.pkl \
    --save-plots \
    --output-dir models/
```

### 4. Run Inference

Make predictions on new data:

```bash
python training/inference.py \
    --input data/test_data.csv \
    --model models/xgb_ids_model.pkl \
    --output predictions.csv
```

## Model Evaluation Metrics

The training script provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Feature Importance**: Top features contributing to classification

## Data Format

### Input Data Requirements

The input CSV files should contain network flow features. The preprocessing script expects:
- A `Label` column with values that can be converted to BENIGN/ATTACK
- Network flow features (statistical, timing, protocol features, etc.)

### Columns Removed During Preprocessing

The following columns are automatically removed as they are not generalizable:
- `Flow ID`
- `Src IP`
- `Dst IP`
- `Timestamp`

## Model Architecture

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Default Hyperparameters**:
  - `max_depth`: 6
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `eval_metric`: logloss

Hyperparameter tuning can be enabled with the `--tune` flag.

## Output Files

After training, the following files are generated:

- `xgb_ids_model.pkl`: Trained model (serialized)
- `xgb_ids_model_label_mapping.pkl`: Label encoding mapping
- `xgb_ids_model_feature_importance.csv`: Feature importance scores
- `confusion_matrix.png`: Confusion matrix visualization (if `--save-plots` used)
- `roc_curve.png`: ROC curve visualization (if `--save-plots` used)

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure data files exist in the specified paths
2. **Memory Error**: Dataset may be too large - consider sampling or using a machine with more RAM
3. **Label Column Missing**: Ensure your CSV has a 'Label' column
4. **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Data Validation

The scripts include data validation that will raise errors if:
- Input files don't exist
- Required columns are missing
- Data is empty
- Labels are invalid

## Performance Tips

1. **Hyperparameter Tuning**: Use `--tune` for better performance (takes longer)
2. **GPU Acceleration**: Install CUDA-enabled XGBoost for faster training
3. **Data Sampling**: For very large datasets, consider sampling for faster iteration
4. **Cross-Validation**: The training script includes 5-fold CV for robust evaluation

## Future Improvements

Potential enhancements:
- Support for multi-class classification (specific attack types)
- Real-time inference API
- Model versioning and experiment tracking
- Additional ML models (Random Forest, Neural Networks)
- Feature engineering and selection
- Handling class imbalance with SMOTE or class weights

## License

[Add your license here]

## Author

[Add your name/institution here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Acknowledgments

- CICIDS2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- XGBoost: https://xgboost.readthedocs.io/

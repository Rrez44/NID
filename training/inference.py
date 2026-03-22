"""
Inference script for IDS thesis project.

This script loads a trained XGBoost model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import logging
from pathlib import Path

import common  # noqa: F401 - ensures project root on path
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Load trained model, label mapping, and expected feature names.

    Returns:
        tuple: (model, reverse_mapping, feature_names or None)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    label_mapping_path = path.with_name(path.stem + '_label_mapping.pkl')
    feature_names_path = path.with_name(path.stem + '_feature_names.pkl')
    if label_mapping_path.exists():
        mapping = joblib.load(label_mapping_path)
        # Support both old (str->int) and new (int->label or list) formats
        if isinstance(mapping, dict):
            # If keys are strings, assume {label: index} and invert
            if all(isinstance(k, str) for k in mapping.keys()):
                reverse_mapping = {v: k for k, v in mapping.items()}
            else:
                # Assume {index: label}
                reverse_mapping = mapping
        elif isinstance(mapping, (list, tuple, np.ndarray)):
            reverse_mapping = {idx: lbl for idx, lbl in enumerate(mapping)}
        else:
            logger.warning("Unrecognized label mapping format, falling back to binary default")
            reverse_mapping = {0: 'BENIGN', 1: 'ATTACK'}
    else:
        logger.warning("Label mapping file not found, using default mapping")
        reverse_mapping = {0: 'BENIGN', 1: 'ATTACK'}

    feature_names = None
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
        logger.info(f"Loaded expected feature list ({len(feature_names)} features)")

    return model, reverse_mapping, feature_names


def preprocess_input(df, feature_names=None, drop_cols=None):
    """
    Preprocess input to match training format.
    Input should be the cleaned CSV (output of preprocess.py) or have the same feature columns.

    Args:
        df: Input dataframe
        feature_names: Expected feature list (from model). If provided, X is aligned to this order.
        drop_cols: Extra columns to drop (identifiers, labels). Default: Flow ID, Src IP, Dst IP, Timestamp.
    """
    if drop_cols is None:
        drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df = df.copy()
    df.columns = df.columns.str.strip()
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    for c in ('Label', 'Source_File', 'Attack Type'):
        if c in df.columns:
            df = df.drop(columns=[c])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    if feature_names is not None:
        missing = set(feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Input missing required features: {list(missing)}")
        df = df[feature_names]
    return df


def predict(model, X, reverse_mapping, return_proba=False):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained XGBoost model
        X (pd.DataFrame): Feature matrix
        reverse_mapping (dict): Mapping from numeric labels to string labels
        return_proba (bool): Whether to return prediction probabilities
        
    Returns:
        array or tuple: Predictions (and probabilities if return_proba=True)
    """
    logger.info(f"Making predictions on {len(X)} samples...")
    
    # Get predictions
    y_pred_numeric = model.predict(X)
    y_pred = np.array([reverse_mapping[label] for label in y_pred_numeric])
    
    if return_proba:
        y_proba = model.predict_proba(X)
        return y_pred, y_proba
    else:
        return y_pred


def run_inference(input_csv, model_path, output_csv=None):
    """
    Run inference on a CSV file.
    Input CSV should be in cleaned format (output of preprocess.py) with same features as training.
    """
    logger.info(f"Starting inference on {input_csv}")
    model, reverse_mapping, feature_names = load_model(model_path)
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    X = preprocess_input(df, feature_names=feature_names)
    
    # Check if we have the expected features
    # Note: In a production system, you'd want to ensure feature alignment
    # For now, we'll use whatever features are available
    
    # Make predictions
    y_pred, y_proba = predict(model, X, reverse_mapping, return_proba=True)
    
    # Add predictions to dataframe
    df['Prediction'] = y_pred
    # Add per-class probabilities
    n_classes = y_proba.shape[1]
    for class_idx in range(n_classes):
        class_label = reverse_mapping.get(class_idx, str(class_idx))
        safe_label = str(class_label).replace(" ", "_").replace("/", "_")
        df[f'Probability_{safe_label}'] = y_proba[:, class_idx]
    
    # Log summary
    prediction_counts = pd.Series(y_pred).value_counts()
    logger.info(f"\nPrediction Summary:")
    logger.info(f"{prediction_counts}")
    
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to {output_csv}")
    else:
        # Print first few predictions (prediction + top-1 probability)
        logger.info("\nFirst 10 predictions:")
        proba_cols = [c for c in df.columns if c.startswith('Probability_')]
        print(df[['Prediction'] + proba_cols].head(10))
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using trained XGBoost model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file for prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.DEFAULT_MODEL),
        help="Path to trained model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        run_inference(args.input, args.model, args.output)
        logger.info("Inference completed successfully!")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

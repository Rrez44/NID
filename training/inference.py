"""
Inference script for IDS thesis project.

This script loads a trained XGBoost model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import os
from pathlib import Path
import sys

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Load trained model and label mapping.
    
    Args:
        model_path (str): Path to saved model file
        
    Returns:
        tuple: (model, label_mapping)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Try to load label mapping
    label_mapping_path = model_path.replace('.pkl', '_label_mapping.pkl')
    if os.path.exists(label_mapping_path):
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
    
    return model, reverse_mapping


def preprocess_input(df, drop_cols=None):
    """
    Preprocess input data to match training data format.
    
    Args:
        df (pd.DataFrame): Input dataframe
        drop_cols (list, optional): Columns to drop
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    if drop_cols is None:
        drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop non-generalizable columns if they exist
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        logger.info(f"Dropping columns: {existing_drop_cols}")
        df = df.drop(columns=existing_drop_cols)
    
    # Handle infinities and NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
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
    
    Args:
        input_csv (str): Path to input CSV file
        model_path (str): Path to trained model
        output_csv (str, optional): Path to save predictions
    """
    logger.info(f"Starting inference on {input_csv}")
    
    # Load model
    model, reverse_mapping = load_model(model_path)
    
    # Load input data
    logger.info(f"Loading input data from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Preprocess input data
    logger.info("Preprocessing input data...")
    X = preprocess_input(df)
    
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
    
    # Save results if output path provided
    if output_csv:
        logger.info(f"Saving predictions to {output_csv}")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved successfully!")
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

"""
Inference script for IDS thesis project.

Loads a trained model and its companion artifacts (label mapping, feature names)
and makes predictions on new data.  Input should be a cleaned CSV produced by
preprocess.py (or any CSV with matching feature columns).
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import logging
from pathlib import Path

import common  # noqa: F401 - ensures project root on path
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _artifact_path(model_path, suffix):
    """Derive companion artifact path from the model path using Path operations."""
    p = Path(model_path)
    return p.with_name(f'{p.stem}{suffix}')


def load_model(model_path):
    """
    Load trained model, label mapping, and expected feature names.

    Returns:
        tuple: (model, reverse_mapping dict, feature_names list or None)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    # Label mapping
    lm_path = _artifact_path(model_path, '_label_mapping.pkl')
    if lm_path.exists():
        mapping = joblib.load(lm_path)
        if isinstance(mapping, dict):
            if all(isinstance(k, str) for k in mapping.keys()):
                reverse_mapping = {v: k for k, v in mapping.items()}
            else:
                reverse_mapping = mapping
        elif isinstance(mapping, (list, tuple, np.ndarray)):
            reverse_mapping = {idx: lbl for idx, lbl in enumerate(mapping)}
        else:
            logger.warning("Unrecognized label mapping format – using binary default")
            reverse_mapping = {0: 'BENIGN', 1: 'ATTACK'}
    else:
        logger.warning("Label mapping not found at %s – using binary default", lm_path)
        reverse_mapping = {0: 'BENIGN', 1: 'ATTACK'}

    # Feature names
    fn_path = _artifact_path(model_path, '_feature_names.pkl')
    feature_names = None
    if fn_path.exists():
        feature_names = joblib.load(fn_path)
        logger.info("Loaded expected feature list (%d features)", len(feature_names))

    return model, reverse_mapping, feature_names


def preprocess_input(df, feature_names=None, drop_cols=None):
    """
    Align input DataFrame to the feature set expected by the model.

    The saved ``feature_names`` already reflect post-split feature selection,
    so applying them during inference achieves the same column subsetting that
    was used during training – no separate FeatureSelector load required.
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
            raise ValueError(f"Input missing required features: {sorted(missing)}")
        df = df[feature_names]
    return df


def predict(model, X, reverse_mapping, return_proba=False):
    """Make predictions and optionally return class probabilities."""
    logger.info("Predicting on %d samples...", len(X))
    y_pred_numeric = model.predict(X)
    y_pred = np.array([reverse_mapping[int(label)] for label in y_pred_numeric])

    if return_proba:
        return y_pred, model.predict_proba(X)
    return y_pred


def run_inference(input_csv, model_path, output_csv=None):
    """Run batch inference on a CSV file."""
    logger.info("Starting inference on %s", input_csv)
    model, reverse_mapping, feature_names = load_model(model_path)

    df = pd.read_csv(input_csv)
    logger.info("Loaded %d samples with %d columns", len(df), len(df.columns))
    X = preprocess_input(df, feature_names=feature_names)

    y_pred, y_proba = predict(model, X, reverse_mapping, return_proba=True)

    df['Prediction'] = y_pred
    for class_idx in range(y_proba.shape[1]):
        class_label = reverse_mapping.get(class_idx, str(class_idx))
        safe_label = str(class_label).replace(' ', '_').replace('/', '_')
        df[f'Probability_{safe_label}'] = y_proba[:, class_idx]

    prediction_counts = pd.Series(y_pred).value_counts()
    logger.info("\nPrediction Summary:\n%s", prediction_counts)

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("Predictions saved to %s", output_csv)
    else:
        logger.info("\nFirst 10 predictions:")
        proba_cols = [c for c in df.columns if c.startswith('Probability_')]
        print(df[['Prediction'] + proba_cols].head(10))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using trained IDS model"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file for prediction")
    parser.add_argument("--model", type=str, default=str(config.DEFAULT_MODEL),
                        help="Path to trained model file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions (optional)")

    args = parser.parse_args()

    try:
        run_inference(args.input, args.model, args.output)
        logger.info("Inference completed successfully!")
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise

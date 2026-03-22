"""
XGBoost training script for IDS thesis project.

This script trains an XGBoost classifier for intrusion detection with:
- Comprehensive evaluation metrics
- Optional hyperparameter tuning
- Model persistence
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import argparse
import logging
import numpy as np
from pathlib import Path

import common  # noqa: F401 - ensures project root on path
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, plotting will be disabled")


def validate_training_data(X, y):
    """
    Validate training data before model training.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
        
    Raises:
        ValueError: If validation fails
    """
    if X.empty or y.empty:
        raise ValueError("Training data is empty")
    
    if len(X) != len(y):
        raise ValueError(f"Feature and label lengths don't match: {len(X)} vs {len(y)}")
    
    if X.isna().any().any():
        raise ValueError("Features contain NaN values")
    
    if y.isna().any():
        raise ValueError("Labels contain NaN values")
    
    unique_labels = y.unique()
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes, found: {unique_labels}")
    
    logger.info(f"Data validation passed. Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"Label distribution:\n{y.value_counts()}")


def evaluate_model(y_true, y_pred, y_pred_proba=None, save_plots=False, output_dir=None, class_names=None):
    """
    Comprehensive model evaluation with multiple metrics and visualizations.

    Args:
        y_true (array-like): True labels (numeric indices)
        y_pred (array-like): Predicted labels (numeric indices)
        y_pred_proba (array-like, optional): Predicted probabilities, shape (n_samples, n_classes)
        save_plots (bool): Whether to save evaluation plots
        output_dir (str, optional): Directory to save plots
        class_names (list, optional): Human-readable names for each class index (e.g. ['BENIGN', 'DDoS', ...])

    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance...")

    # Use zero_division=0 to avoid warnings when a class has no true or predicted samples
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # ROC-AUC: only when probability shape matches classes present in y_true
    if y_pred_proba is not None:
        try:
            if (np.ndim(y_pred_proba) == 1) or (np.ndim(y_pred_proba) == 2 and y_pred_proba.shape[1] == 1):
                roc_auc = roc_auc_score(y_true, y_pred_proba.ravel())
                metrics['roc_auc'] = roc_auc
                logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
            else:
                # Multiclass: use only classes that appear in y_true so shapes match
                labels_present = np.unique(y_true)
                n_present = len(labels_present)
                if n_present >= 2 and n_present <= y_pred_proba.shape[1]:
                    y_score = y_pred_proba[:, labels_present]
                    roc_auc = roc_auc_score(
                        y_true, y_score, multi_class='ovr', average='macro',
                        labels=labels_present
                    )
                    metrics['roc_auc'] = roc_auc
                    logger.info(f"ROC-AUC Score (macro, %d classes in test): %s", n_present, f"{roc_auc:.4f}")
                else:
                    logger.warning(
                        "Skipping ROC-AUC: need 2+ classes in test set (got %d). "
                        "Test set may contain only one label after dropping unseen classes.",
                        n_present,
                    )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")

    # Print metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    # Classification report (with optional class names and zero_division)
    labels_in_order = sorted(np.unique(y_true))
    target_names = None
    if class_names is not None:
        target_names = [str(class_names[i]) if i < len(class_names) else str(i) for i in labels_in_order]
    logger.info("\nClassification Report:")
    logger.info(
        "\n%s",
        classification_report(
            y_true, y_pred, labels=labels_in_order, target_names=target_names, zero_division=0
        ),
    )
    
    # Save plots if requested
    if save_plots and output_dir:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping plot generation")
            return metrics
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix plot (works for binary and multiclass)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plot_labels = sorted(np.unique(y_true))
        tick_labels = (
            [str(class_names[i]) if class_names and i < len(class_names) else str(i) for i in plot_labels]
            if class_names else [str(i) for i in plot_labels]
        )
        tick_marks = np.arange(len(plot_labels))
        plt.xticks(tick_marks, tick_labels, rotation=45, ha='right')
        plt.yticks(tick_marks, tick_labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'confusion_matrix.png')
        plt.close()
        logger.info(f"Confusion matrix plot saved to {output_dir}")
        
        # ROC curve if probabilities available (only for binary problems)
        if y_pred_proba is not None and (
            (np.ndim(y_pred_proba) == 1) or (np.ndim(y_pred_proba) == 2 and y_pred_proba.shape[1] == 1)
        ):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba.ravel())
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics.get("roc_auc", 0):.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(Path(output_dir) / 'roc_curve.png')
            plt.close()
            logger.info(f"ROC curve plot saved to {output_dir}")
    
    return metrics


def train_xgb(preprocessed_csv, model_path, tune_hyperparameters=False, save_plots=False, output_dir=None, split_strategy='temporal'):
    """
    Train XGBoost classifier for intrusion detection.
    
    Args:
        preprocessed_csv (str): Path to preprocessed CSV file
        model_path (str): Path to save trained model
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning
        save_plots (bool): Whether to save evaluation plots
        output_dir (str, optional): Directory to save plots
        split_strategy (str): 'file_holdout' (realistic: randomly hold out whole files for test; no leakage),
            'per_file' (70%% of each file train, 30%% test; optimistic, all classes in test),
            'temporal' (first 80%% train, last 20%% test by row order),
            'file' (last 20%% of files as test), or 'random' (legacy; can inflate accuracy).
        
    Returns:
        xgb.XGBClassifier: Trained model
    """
    logger.info(f"Starting training with data from {preprocessed_csv}")
    
    if not Path(preprocessed_csv).exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_csv}")
    
    # Load cleaned data
    logger.info("Loading preprocessed data...")
    try:
        df = pd.read_csv(preprocessed_csv)
        logger.info(f"Loaded dataframe shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise
    
    # Split features and labels
    # Support both old ('Label') and new ('Attack Type') column names
    label_col = 'Attack Type' if 'Attack Type' in df.columns else 'Label'
    if label_col not in df.columns:
        raise ValueError("Neither 'Attack Type' nor 'Label' column found in dataframe")
    logger.info(f"Using label column: '{label_col}'")

    # Keep Source_File for file-based split, then drop from features
    source_file = df['Source_File'].copy() if 'Source_File' in df.columns else None
    non_feature_cols = {label_col, 'Source_File', 'Label'}  # drop all non-feature cols
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].copy()
    y_raw = df[label_col].copy()
    
    # Validate training data (on raw string labels)
    validate_training_data(X, y_raw)
    
    # Train/test split (80/20) - use temporal or file-based to avoid session leakage
    logger.info("Splitting data into train/test sets using %s split...", split_strategy)
    if split_strategy == 'file' and source_file is not None:
        # Whole files in train or test: no flow from same capture in both sets (last 20% of files = test)
        unique_files = source_file.dropna().unique()
        unique_files = sorted(unique_files)  # deterministic
        n_test_files = max(1, int(0.2 * len(unique_files)))
        test_files = set(unique_files[-n_test_files:])  # last 20% of files = test (future data)
        test_mask = source_file.isin(test_files)
        train_mask = ~test_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train_raw, y_test_raw = y_raw[train_mask], y_raw[test_mask]
        logger.info("File-based split: test files = %s", test_files)
    elif split_strategy == 'file_holdout' and source_file is not None:
        # Randomly hold out whole files for test → realistic eval with mix of attack types, no within-file leakage
        unique_files = sorted(source_file.dropna().unique())
        n_files = len(unique_files)
        n_test_files = max(2, int(0.25 * n_files))  # at least 2 files so test has diverse labels
        rng = np.random.RandomState(42)
        test_idx = rng.choice(n_files, size=n_test_files, replace=False)
        test_files = set(unique_files[i] for i in test_idx)
        test_mask = source_file.isin(test_files)
        train_mask = ~test_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train_raw, y_test_raw = y_raw[train_mask], y_raw[test_mask]
        logger.info("File-holdout split (realistic): test files = %s", sorted(test_files))
    elif split_strategy in ('file', 'file_holdout') and source_file is None:
        logger.warning("Source_File not in data; falling back to temporal split. Re-run merge and preprocess to add Source_File.")
        n = len(X)
        split_idx = int(0.8 * n)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_raw, y_test_raw = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]
    elif split_strategy == 'temporal':
        # First 80% of rows = train, last 20% = test (data is in merge/time order)
        n = len(X)
        split_idx = int(0.8 * n)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_raw, y_test_raw = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]
    elif split_strategy == 'per_file' and source_file is not None:
        # 70% of each file -> train, 30% of each file -> test (within-file order preserved)
        train_mask = np.zeros(len(X), dtype=bool)
        test_mask = np.zeros(len(X), dtype=bool)
        for f in sorted(source_file.dropna().unique()):
            file_locs = np.where((source_file == f).values)[0]
            n_file = len(file_locs)
            split_idx = int(0.7 * n_file)
            train_mask[file_locs[:split_idx]] = True
            test_mask[file_locs[split_idx:]] = True
        X_train = X.iloc[train_mask]
        X_test = X.iloc[test_mask]
        y_train_raw = y_raw.iloc[train_mask]
        y_test_raw = y_raw.iloc[test_mask]
        logger.info("Per-file split: 70%% train / 30%% test within each of %d files", len(np.unique(source_file.dropna())))
    elif split_strategy == 'per_file' and source_file is None:
        logger.warning("Source_File not in data; falling back to temporal split. Re-run merge and preprocess for per_file.")
        n = len(X)
        split_idx = int(0.8 * n)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_raw, y_test_raw = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]
    else:
        # random (legacy): can cause session leakage and inflated accuracy
        logger.warning("Random split may leak flows from same attack session into train and test.")
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y_raw, test_size=0.2, random_state=42, stratify=y_raw
        )
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Encode labels for multiclass classification based only on training labels
    logger.info("Encoding labels for multiclass classification (based on training set)...")
    label_encoder = LabelEncoder()
    y_train = pd.Series(label_encoder.fit_transform(y_train_raw), index=y_train_raw.index)
    classes = list(label_encoder.classes_)
    n_classes = len(classes)
    logger.info(f"Detected {n_classes} classes in training: {classes}")

    # Handle any labels in test set that were not seen during training
    valid_test_mask = y_test_raw.isin(label_encoder.classes_)
    if not valid_test_mask.all():
        n_dropped = (~valid_test_mask).sum()
        logger.warning(
            "Dropping %d test samples with labels not seen in training: %s",
            n_dropped,
            sorted(y_test_raw[~valid_test_mask].unique().tolist()),
        )
        X_test = X_test[valid_test_mask]
        y_test_raw = y_test_raw[valid_test_mask]

    y_test = pd.Series(label_encoder.transform(y_test_raw), index=y_test_raw.index)
    
    # Hyperparameter tuning or default parameters
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning with cross-validation...")
        from sklearn.model_selection import RandomizedSearchCV
        
        param_distributions = {
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Configure base model for binary or multiclass
        if n_classes > 2:
            base_model = xgb.XGBClassifier(
                n_jobs=-1,
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=n_classes,
                random_state=42
            )
        else:
            base_model = xgb.XGBClassifier(
                n_jobs=-1,
                eval_metric='logloss',
                objective='binary:logistic',
                random_state=42
            )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=20,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
    else:
        # Create XGBoost classifier with default parameters
        logger.info("Using default hyperparameters...")
        if n_classes > 2:
            model = xgb.XGBClassifier(
                max_depth=6,
                n_estimators=100,
                learning_rate=0.1,
                n_jobs=-1,
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=n_classes,
                random_state=42
            )
        else:
            model = xgb.XGBClassifier(
                max_depth=6,
                n_estimators=100,
                learning_rate=0.1,
                n_jobs=-1,
                eval_metric='logloss',
                objective='binary:logistic',
                random_state=42
            )
    
    # Train the model
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info("Training completed!")
    
    # Cross-validation on training set
    logger.info("Performing 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    logger.info(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predict on test set
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    # For binary and multiclass, use full probability matrix
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate model
    metrics = evaluate_model(
        y_test, y_pred, y_pred_proba, save_plots, output_dir, class_names=classes
    )
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    logger.info(f"\n{feature_importance.head(10).to_string(index=False)}")
    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    # Save label mapping (index -> class label for multiclass)
    label_mapping = {idx: cls for idx, cls in enumerate(classes)}
    label_mapping_path = model_path.replace('.pkl', '_label_mapping.pkl')
    joblib.dump(label_mapping, label_mapping_path)
    logger.info(f"Label mapping saved to {label_mapping_path}")

    # Save expected feature names for inference alignment
    feature_names_path = model_path.replace('.pkl', '_feature_names.pkl')
    joblib.dump(list(X.columns), feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")

    # Save feature importance
    importance_path = model_path.replace('.pkl', '_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")
    
    logger.info("Training completed successfully!")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier for intrusion detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.CLEANED_CSV),
        help="Path to preprocessed CSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.DEFAULT_MODEL),
        help="Path to save trained model"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning (slower but may improve performance)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots (confusion matrix, ROC curve)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.MODELS_DIR),
        help="Directory to save plots and additional outputs"
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=('file_holdout', 'per_file', 'temporal', 'file', 'random'),
        default='file_holdout',
        help="Split: file_holdout (realistic, hold out random files), per_file (70/30 within each file), temporal, file, random"
    )
    
    args = parser.parse_args()
    
    try:
        train_xgb(
            args.input,
            args.model,
            tune_hyperparameters=args.tune,
            save_plots=args.save_plots,
            output_dir=args.output_dir,
            split_strategy=args.split_strategy
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

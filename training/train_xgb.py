"""
Training script for IDS thesis project.

Trains an XGBoost classifier (or Random Forest baseline) with:
- Post-split feature selection via FeatureSelector (no data leakage)
- Inverse-frequency sample weights for class imbalance
- GroupKFold cross-validation to prevent session leakage
- Comprehensive evaluation (macro + weighted + per-class metrics)
- Optional SHAP interpretability analysis
- Optional Optuna Bayesian hyperparameter optimisation
- JSON experiment logging for reproducibility
"""

import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold, RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib

import common  # noqa: F401 – ensures project root on path
import config
from feature_selection import FeatureSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

RANDOM_STATE = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def validate_training_data(X, y):
    if X.empty or y.empty:
        raise ValueError("Training data is empty")
    if len(X) != len(y):
        raise ValueError(f"Feature/label length mismatch: {len(X)} vs {len(y)}")
    if X.isna().any().any():
        raise ValueError("Features contain NaN values")
    if y.isna().any():
        raise ValueError("Labels contain NaN values")
    if y.nunique() < 2:
        raise ValueError(f"Need >= 2 classes, found: {y.unique()}")
    logger.info("Validation OK – features: %d, samples: %d", X.shape[1], X.shape[0])
    logger.info("Label distribution:\n%s", y.value_counts().to_string())


def get_sample_weights(y):
    """Inverse-frequency weights so minority classes contribute more to loss."""
    weights = compute_sample_weight('balanced', y)
    logger.info("Sample weights: min=%.4f  max=%.4f", weights.min(), weights.max())
    return weights


def get_cv_splitter(groups=None, n_splits=5):
    """GroupKFold when group metadata is available, otherwise StratifiedKFold."""
    if groups is not None:
        n_groups = len(np.unique(groups))
        effective = min(n_splits, n_groups)
        if effective < n_splits:
            logger.warning("Only %d groups; using %d-fold GroupKFold", n_groups, effective)
        return GroupKFold(n_splits=effective), groups
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None


def manual_cv(model, X, y, cv, groups=None, sample_weight=None):
    """Cross-validate with proper group + sample_weight routing per fold.

    GroupKFold can produce folds where some classes are entirely absent, making
    the encoded labels non-contiguous (e.g. [0,1,2,4]).  XGBoost rejects this.
    We re-encode to 0..n_fold_classes inside each fold and map back for scoring.
    """
    scores = []
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

        # Re-encode labels so they are contiguous 0..k within this fold
        fold_le = LabelEncoder()
        y_t_enc = pd.Series(fold_le.fit_transform(y_t), index=y_t.index)

        # Validation labels: keep only classes seen in training fold
        valid_mask = y_v.isin(fold_le.classes_)
        X_v, y_v = X_v[valid_mask], y_v[valid_mask]
        if len(y_v) == 0:
            continue
        y_v_enc = pd.Series(fold_le.transform(y_v), index=y_v.index)

        fit_kw = {}
        if sample_weight is not None:
            fit_kw['sample_weight'] = sample_weight[train_idx]

        fold_model = clone(model)
        # Override num_class to match this fold's class count
        n_fold_classes = len(fold_le.classes_)
        if hasattr(fold_model, 'num_class'):
            fold_model.set_params(num_class=n_fold_classes)

        fold_model.fit(X_t, y_t_enc, **fit_kw)
        y_pred = fold_model.predict(X_v)
        scores.append(f1_score(y_v_enc, y_pred, average='macro', zero_division=0))
    return np.array(scores)


# ─── Model construction ──────────────────────────────────────────────────────

def build_model(model_type, n_classes, params=None):
    """Return a configured XGBClassifier or RandomForestClassifier."""
    params = params or {}
    if model_type == 'rf':
        defaults = dict(n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if n_classes > 2:
        defaults = dict(
            max_depth=6, n_estimators=100, learning_rate=0.1, n_jobs=-1,
            eval_metric='mlogloss', objective='multi:softprob',
            num_class=n_classes, random_state=RANDOM_STATE,
        )
    else:
        defaults = dict(
            max_depth=6, n_estimators=100, learning_rate=0.1, n_jobs=-1,
            eval_metric='logloss', objective='binary:logistic',
            random_state=RANDOM_STATE,
        )
    defaults.update(params)
    return xgb.XGBClassifier(**defaults)


# ─── Hyperparameter tuning ────────────────────────────────────────────────────

def tune_random(X, y, n_classes, cv, groups=None, sample_weight=None):
    """RandomizedSearchCV with expanded search space.

    Uses StratifiedKFold internally (RandomizedSearchCV cannot re-encode labels
    per fold for GroupKFold), then refits on the full training set with weights.
    """
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }
    base = build_model('xgb', n_classes)
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30, cv=strat_cv, scoring='f1_macro',
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y)
    logger.info("RandomSearch best F1-macro: %.4f", search.best_score_)
    logger.info("RandomSearch best params: %s", search.best_params_)
    best_model = search.best_estimator_
    if sample_weight is not None:
        best_model.fit(X, y, sample_weight=sample_weight)
    return best_model, search.best_params_


def tune_optuna(X, y, n_classes, cv, groups=None, sample_weight=None, n_trials=50):
    """Bayesian optimisation via Optuna with manual CV for proper weight routing."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Install optuna: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'n_estimators':     trial.suggest_int('n_estimators', 50, 500, step=50),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'gamma':            trial.suggest_float('gamma', 1e-3, 5.0, log=True),
        }
        model = build_model('xgb', n_classes, params)
        scores = manual_cv(model, X, y, cv, groups=groups, sample_weight=sample_weight)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna best F1-macro: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)

    best_model = build_model('xgb', n_classes, study.best_params)
    fit_kw = {'sample_weight': sample_weight} if sample_weight is not None else {}
    best_model.fit(X, y, **fit_kw)
    return best_model, study.best_params


# ─── SHAP analysis ────────────────────────────────────────────────────────────

def run_shap_analysis(model, X_test, class_names, output_dir):
    """Generate SHAP summary plots and per-feature CSV."""
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed – skipping SHAP analysis")
        return
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed – skipping SHAP plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values (subsample up to 2 000 rows)...")
    n_sample = min(2000, len(X_test))
    X_shap = X_test.sample(n=n_sample, random_state=RANDOM_STATE) if len(X_test) > n_sample else X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Bar summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type='bar',
                      class_names=class_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary_bar.png', dpi=150)
    plt.close()

    # Per-class beeswarm (multiclass) or single beeswarm
    if isinstance(shap_values, list):
        for i, name in enumerate(class_names):
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[i], X_shap, show=False, max_display=20)
            plt.title(f'SHAP – {name}')
            plt.tight_layout()
            safe = name.replace(' ', '_')
            plt.savefig(output_dir / f'shap_beeswarm_{safe}.png', dpi=150)
            plt.close()
        shap_df = pd.DataFrame(
            {name: np.abs(sv).mean(axis=0) for name, sv in zip(class_names, shap_values)},
            index=X_shap.columns,
        )
    else:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_beeswarm.png', dpi=150)
        plt.close()
        shap_df = pd.DataFrame({'mean_abs_shap': np.abs(shap_values).mean(axis=0)},
                                index=X_shap.columns)

    shap_df.to_csv(output_dir / 'shap_values.csv')
    logger.info("SHAP analysis saved to %s", output_dir)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_pred_proba=None, save_plots=False,
                   output_dir=None, class_names=None):
    """Comprehensive evaluation with both weighted and macro metrics."""
    logger.info("=== Model Evaluation ===")

    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'accuracy':           float(accuracy),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted':    float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_weighted':        float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_macro':    float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro':       float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro':           float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }

    # Per-class breakdown
    labels_sorted = sorted(np.unique(y_true))
    pc_p = precision_score(y_true, y_pred, labels=labels_sorted, average=None, zero_division=0)
    pc_r = recall_score(y_true, y_pred, labels=labels_sorted, average=None, zero_division=0)
    pc_f = f1_score(y_true, y_pred, labels=labels_sorted, average=None, zero_division=0)

    per_class = {}
    for i, lbl in enumerate(labels_sorted):
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        per_class[name] = {
            'precision': float(pc_p[i]),
            'recall':    float(pc_r[i]),
            'f1':        float(pc_f[i]),
            'support':   int((np.asarray(y_true) == lbl).sum()),
        }
    metrics['per_class'] = per_class

    # ROC-AUC (macro, OvR)
    if y_pred_proba is not None:
        try:
            present = np.unique(y_true)
            if len(present) >= 2:
                if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
                    roc = roc_auc_score(y_true, y_pred_proba.ravel())
                else:
                    roc = roc_auc_score(
                        y_true, y_pred_proba[:, present],
                        multi_class='ovr', average='macro', labels=present,
                    )
                metrics['roc_auc_macro'] = float(roc)
        except Exception as e:
            logger.warning("ROC-AUC failed: %s", e)

    # Log headline numbers
    logger.info("Accuracy:            %.4f", metrics['accuracy'])
    logger.info("F1 (weighted):       %.4f", metrics['f1_weighted'])
    logger.info("F1 (macro):          %.4f", metrics['f1_macro'])
    logger.info("Precision (macro):   %.4f", metrics['precision_macro'])
    logger.info("Recall (macro):      %.4f", metrics['recall_macro'])
    if 'roc_auc_macro' in metrics:
        logger.info("ROC-AUC (macro):     %.4f", metrics['roc_auc_macro'])

    # Full classification report
    target_names = None
    if class_names:
        target_names = [str(class_names[i]) if i < len(class_names) else str(i) for i in labels_sorted]
    logger.info("\nClassification Report:\n%s",
                classification_report(y_true, y_pred, labels=labels_sorted,
                                      target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n%s", cm)

    # Plots
    if save_plots and output_dir and MATPLOTLIB_AVAILABLE:
        _save_confusion_matrix_plot(cm, labels_sorted, class_names, output_dir)

    return metrics


def _save_confusion_matrix_plot(cm, labels_sorted, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tick_labels = (
        [str(class_names[i]) if class_names and i < len(class_names) else str(i)
         for i in labels_sorted]
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(labels_sorted))
    plt.xticks(ticks, tick_labels, rotation=45, ha='right')
    plt.yticks(ticks, tick_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    logger.info("Confusion matrix plot saved to %s", output_dir)


# ─── Experiment logging ───────────────────────────────────────────────────────

def _dataset_hash(path, n_bytes=65536):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def log_experiment(output_dir, *, model_type, params, split_strategy, metrics,
                   feature_selector_summary, n_train, n_test, classes,
                   dataset_path, tune_method, cv_f1_macro, extra=None):
    """Write a timestamped JSON record of the experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_params = {}
    for k, v in (params or {}).items():
        try:
            json.dumps(v)
            safe_params[k] = v
        except (TypeError, ValueError):
            safe_params[k] = str(v)

    record = {
        'timestamp':          datetime.now(timezone.utc).isoformat(),
        'model_type':         model_type,
        'params':             safe_params,
        'split_strategy':     split_strategy,
        'tune_method':        tune_method,
        'n_train':            n_train,
        'n_test':             n_test,
        'classes':            classes,
        'metrics':            metrics,
        'cv_f1_macro':        cv_f1_macro,
        'feature_selection':  feature_selector_summary,
        'dataset_hash':       _dataset_hash(dataset_path),
        'dataset_path':       str(dataset_path),
    }
    if extra:
        record.update(extra)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = output_dir / f'experiment_{ts}.json'
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    logger.info("Experiment log → %s", path)
    return path


# ─── Data splitting ───────────────────────────────────────────────────────────

def split_data(X, y, source_file, strategy):
    """Return X_train, X_test, y_train, y_test, groups_train."""
    logger.info("Split strategy: %s", strategy)
    groups_train = None

    if strategy == 'file_holdout' and source_file is not None:
        unique_files = sorted(source_file.dropna().unique())
        n_test = max(2, int(0.25 * len(unique_files)))
        rng = np.random.RandomState(RANDOM_STATE)
        test_idx = rng.choice(len(unique_files), size=n_test, replace=False)
        test_files = {unique_files[i] for i in test_idx}
        mask = source_file.isin(test_files)
        X_train, X_test = X[~mask], X[mask]
        y_train, y_test = y[~mask], y[mask]
        groups_train = source_file[~mask]
        logger.info("file_holdout – test files: %s", sorted(test_files))

    elif strategy == 'per_file' and source_file is not None:
        train_m = np.zeros(len(X), dtype=bool)
        test_m = np.zeros(len(X), dtype=bool)
        for f in sorted(source_file.dropna().unique()):
            locs = np.where(source_file.values == f)[0]
            sp = int(0.7 * len(locs))
            train_m[locs[:sp]] = True
            test_m[locs[sp:]] = True
        X_train, X_test = X[train_m], X[test_m]
        y_train, y_test = y[train_m], y[test_m]
        groups_train = source_file[train_m]
        logger.info("per_file – 70/30 within %d files", source_file.nunique())

    elif strategy == 'temporal':
        sp = int(0.8 * len(X))
        X_train, X_test = X.iloc[:sp], X.iloc[sp:]
        y_train, y_test = y.iloc[:sp], y.iloc[sp:]
        if source_file is not None:
            groups_train = source_file.iloc[:sp]

    elif strategy == 'file' and source_file is not None:
        unique_files = sorted(source_file.dropna().unique())
        n_test = max(1, int(0.2 * len(unique_files)))
        test_files = set(unique_files[-n_test:])
        mask = source_file.isin(test_files)
        X_train, X_test = X[~mask], X[mask]
        y_train, y_test = y[~mask], y[mask]
        groups_train = source_file[~mask]

    elif strategy == 'random':
        logger.warning("Random split – session leakage is possible.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        if source_file is not None:
            groups_train = source_file.loc[X_train.index]

    else:
        if source_file is None and strategy in ('file_holdout', 'per_file', 'file'):
            logger.warning("Source_File unavailable – falling back to temporal split.")
        sp = int(0.8 * len(X))
        X_train, X_test = X.iloc[:sp], X.iloc[sp:]
        y_train, y_test = y.iloc[:sp], y.iloc[sp:]

    logger.info("Train: %d samples  Test: %d samples", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, groups_train


# ─── Main training function ──────────────────────────────────────────────────

def train_model(preprocessed_csv, model_path, *, model_type='xgb',
                split_strategy='file_holdout', tune_method=None,
                corr_threshold=0.95, drop_port=False, use_class_weights=True,
                run_shap_flag=False, save_plots=False, output_dir=None):
    """
    Full training pipeline with post-split feature selection.
    """
    preprocessed_csv = str(preprocessed_csv)
    model_path = str(model_path)
    if not Path(preprocessed_csv).exists():
        raise FileNotFoundError(f"Not found: {preprocessed_csv}")

    logger.info("Loading data from %s", preprocessed_csv)
    df = pd.read_csv(preprocessed_csv)
    logger.info("Loaded shape: %s", df.shape)

    # Identify label + metadata columns
    label_col = 'Attack Type' if 'Attack Type' in df.columns else 'Label'
    if label_col not in df.columns:
        raise ValueError("No label column found")

    source_file = df['Source_File'].copy() if 'Source_File' in df.columns else None
    non_feature = {label_col, 'Source_File', 'Label'}
    feature_cols = [c for c in df.columns if c not in non_feature]
    X = df[feature_cols].copy()
    y_raw = df[label_col].copy()

    validate_training_data(X, y_raw)

    # ── Split (on raw string labels) ─────────────────────────────────────────
    X_train, X_test, y_train_raw, y_test_raw, groups_train = split_data(
        X, y_raw, source_file, split_strategy)

    # ── Post-split feature selection (fit on train ONLY) ─────────────────────
    selector = FeatureSelector(corr_threshold=corr_threshold, drop_port=drop_port)
    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)
    logger.info("Features after selection: %d", X_train.shape[1])

    # ── Encode labels (fit on train only) ────────────────────────────────────
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train_raw), index=y_train_raw.index)
    classes = list(le.classes_)
    n_classes = len(classes)
    logger.info("Classes (%d): %s", n_classes, classes)

    valid_mask = y_test_raw.isin(le.classes_)
    if not valid_mask.all():
        n_drop = (~valid_mask).sum()
        logger.warning("Dropping %d test samples with unseen labels: %s",
                       n_drop, sorted(y_test_raw[~valid_mask].unique()))
        X_test = X_test[valid_mask]
        y_test_raw = y_test_raw[valid_mask]
    y_test = pd.Series(le.transform(y_test_raw), index=y_test_raw.index)

    # ── Sample weights for class imbalance ───────────────────────────────────
    sample_weight = get_sample_weights(y_train) if use_class_weights else None

    # ── CV splitter (GroupKFold when possible) ───────────────────────────────
    cv_groups = groups_train.values if groups_train is not None else None
    cv, cv_groups_for_split = get_cv_splitter(cv_groups)

    # ── Tune or build model ──────────────────────────────────────────────────
    best_params = {}
    if tune_method == 'optuna':
        model, best_params = tune_optuna(
            X_train, y_train, n_classes, cv,
            groups=cv_groups_for_split, sample_weight=sample_weight)
    elif tune_method == 'random':
        model, best_params = tune_random(
            X_train, y_train, n_classes, cv,
            groups=cv_groups_for_split, sample_weight=sample_weight)
    else:
        model = build_model(model_type, n_classes)
        fit_kw = {'sample_weight': sample_weight} if sample_weight is not None else {}
        logger.info("Training %s with default hyperparameters...", model_type.upper())
        model.fit(X_train, y_train, **fit_kw)

    logger.info("Training complete.")

    # ── CV evaluation on training set ────────────────────────────────────────
    logger.info("Running %d-fold CV on training set...", cv.get_n_splits())
    cv_scores = manual_cv(model, X_train, y_train, cv,
                          groups=cv_groups_for_split, sample_weight=sample_weight)
    cv_f1 = float(cv_scores.mean())
    logger.info("CV F1-macro: %.4f (+/- %.4f)", cv_f1, cv_scores.std() * 2)

    # ── Predict on test set ──────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else Path(model_path).parent
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, save_plots,
                             str(out_dir), class_names=classes)

    # ── Feature importance ───────────────────────────────────────────────────
    fi = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    logger.info("\nTop 10 features:\n%s", fi.head(10).to_string(index=False))

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if run_shap_flag:
        run_shap_analysis(model, X_test, classes, str(out_dir))

    # ── Save artifacts ───────────────────────────────────────────────────────
    model_p = Path(model_path)
    model_p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_p)
    logger.info("Model → %s", model_p)

    stem = model_p.stem
    parent = model_p.parent

    joblib.dump({i: c for i, c in enumerate(classes)}, parent / f'{stem}_label_mapping.pkl')
    joblib.dump(list(X_train.columns), parent / f'{stem}_feature_names.pkl')
    selector.save(parent / f'{stem}_feature_selector.pkl')
    fi.to_csv(parent / f'{stem}_feature_importance.csv', index=False)

    # ── Experiment log ───────────────────────────────────────────────────────
    all_params = best_params or {}
    if not all_params and hasattr(model, 'get_params'):
        all_params = model.get_params()

    log_experiment(
        str(out_dir),
        model_type=model_type,
        params=all_params,
        split_strategy=split_strategy,
        metrics=metrics,
        feature_selector_summary=selector.summary(),
        n_train=len(X_train),
        n_test=len(X_test),
        classes=classes,
        dataset_path=preprocessed_csv,
        tune_method=tune_method,
        cv_f1_macro=cv_f1,
        extra={'class_weights': use_class_weights, 'drop_port': drop_port},
    )

    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train IDS classifier")
    parser.add_argument('--input', default=str(config.CLEANED_CSV),
                        help="Path to preprocessed CSV")
    parser.add_argument('--model', default=str(config.DEFAULT_MODEL),
                        help="Path to save trained model (.pkl)")
    parser.add_argument('--model-type', choices=('xgb', 'rf'), default='xgb',
                        help="xgb = XGBoost (default), rf = Random Forest baseline")
    parser.add_argument('--split-strategy',
                        choices=('file_holdout', 'per_file', 'temporal', 'file', 'random'),
                        default='file_holdout')
    parser.add_argument('--tune', action='store_true',
                        help="Enable hyperparameter tuning")
    parser.add_argument('--tune-method', choices=('random', 'optuna'), default='optuna',
                        help="Tuning method (default: optuna)")
    parser.add_argument('--corr-threshold', type=float, default=0.95,
                        help="Pearson |r| threshold for correlated-feature removal")
    parser.add_argument('--drop-port', action='store_true',
                        help="Ablation: drop Destination Port feature")
    parser.add_argument('--no-class-weights', action='store_true',
                        help="Disable inverse-frequency class weights")
    parser.add_argument('--shap', action='store_true',
                        help="Run SHAP interpretability analysis")
    parser.add_argument('--save-plots', action='store_true',
                        help="Save confusion matrix and other plots")
    parser.add_argument('--output-dir', default=str(config.MODELS_DIR),
                        help="Directory for plots, logs, and extra artifacts")

    args = parser.parse_args()
    tune = args.tune_method if args.tune else None

    try:
        train_model(
            args.input, args.model,
            model_type=args.model_type,
            split_strategy=args.split_strategy,
            tune_method=tune,
            corr_threshold=args.corr_threshold,
            drop_port=args.drop_port,
            use_class_weights=not args.no_class_weights,
            run_shap_flag=args.shap,
            save_plots=args.save_plots,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise

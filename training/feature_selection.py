"""
Post-split feature selection for IDS thesis project.

This module provides a FeatureSelector that must be fit() on training data ONLY,
then applied to both train and test sets. This prevents data leakage from computing
correlation / variance statistics on the full dataset before splitting.

Usage:
    selector = FeatureSelector(corr_threshold=0.95)
    X_train = selector.fit_transform(X_train)
    X_test  = selector.transform(X_test)
    selector.save("models/feature_selector.pkl")
"""

import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IRRELEVANT_FEATURES = [
    'ECE Flag Count',
    'RST Flag Count',
    'Fwd URG Flags',
    'Idle Std',
    'Fwd PSH Flags',
    'Active Std',
    'Down/Up Ratio',
    'URG Flag Count',
]


class FeatureSelector:
    """
    Sklearn-style feature selector that removes zero-variance, highly correlated,
    and domain-irrelevant features.  Fit on training data only to prevent leakage.
    """

    def __init__(self, corr_threshold=0.95, irrelevant_features=None, drop_port=False):
        self.corr_threshold = corr_threshold
        self.irrelevant_features = (
            irrelevant_features if irrelevant_features is not None else IRRELEVANT_FEATURES
        )
        self.drop_port = drop_port

        self.zero_var_cols_ = None
        self.correlated_cols_ = None
        self.dropped_irrelevant_ = None
        self.feature_names_out_ = None

    def fit(self, X):
        """Learn which columns to keep from *training* data statistics."""
        drop = set()

        # 1. Zero-variance
        num_cols = [c for c in X.select_dtypes(include=[np.number]).columns]
        self.zero_var_cols_ = [c for c in num_cols if X[c].nunique() <= 1]
        drop.update(self.zero_var_cols_)
        if self.zero_var_cols_:
            logger.info("FeatureSelector – zero-variance (train): %s", self.zero_var_cols_)

        # 2. High absolute Pearson correlation
        remaining = [c for c in num_cols if c not in drop]
        if len(remaining) > 1:
            corr = X[remaining].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            self.correlated_cols_ = [
                col for col in upper.columns if any(upper[col] > self.corr_threshold)
            ]
            drop.update(self.correlated_cols_)
            if self.correlated_cols_:
                logger.info(
                    "FeatureSelector – correlated (r>%.2f): %s",
                    self.corr_threshold, self.correlated_cols_,
                )
        else:
            self.correlated_cols_ = []

        # 3. Domain-irrelevant features (Kruskal-Wallis + RF based)
        self.dropped_irrelevant_ = [f for f in self.irrelevant_features if f in X.columns]
        drop.update(self.dropped_irrelevant_)
        if self.dropped_irrelevant_:
            logger.info("FeatureSelector – domain-irrelevant: %s", self.dropped_irrelevant_)

        # 4. Optional Destination Port ablation
        if self.drop_port and 'Destination Port' in X.columns:
            drop.add('Destination Port')
            logger.info("FeatureSelector – ablation: dropping Destination Port")

        self.feature_names_out_ = [c for c in X.columns if c not in drop]
        logger.info(
            "FeatureSelector: keeping %d / %d features (dropped %d)",
            len(self.feature_names_out_), len(X.columns), len(drop),
        )
        return self

    def transform(self, X):
        """Select the fitted feature subset."""
        if self.feature_names_out_ is None:
            raise RuntimeError("FeatureSelector not fitted – call fit() first")
        missing = set(self.feature_names_out_) - set(X.columns)
        if missing:
            raise ValueError(
                f"Input missing {len(missing)} expected features: {sorted(missing)[:10]}"
            )
        return X[self.feature_names_out_].copy()

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FeatureSelector saved to %s", path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def summary(self):
        """Return a JSON-serialisable dict describing what was dropped."""
        return {
            'corr_threshold': self.corr_threshold,
            'drop_port': self.drop_port,
            'n_zero_var': len(self.zero_var_cols_ or []),
            'n_correlated': len(self.correlated_cols_ or []),
            'n_irrelevant': len(self.dropped_irrelevant_ or []),
            'n_features_out': len(self.feature_names_out_ or []),
            'zero_var_cols': self.zero_var_cols_,
            'correlated_cols': self.correlated_cols_,
            'dropped_irrelevant': self.dropped_irrelevant_,
        }

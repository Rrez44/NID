"""
Data preprocessing script for IDS thesis project.

Implements the full cleaning pipeline based on the CICIDS2017 analysis notebook:
1. Strip whitespace from column names
2. Drop non-generalizable identifiers (IP, Flow ID, Timestamp)
3. Remove duplicate rows
4. Remove perfectly/near-perfectly duplicate columns
5. Replace infinities with NaN then drop rows with NaN (minimal loss)
6. Remove zero-variance columns
7. Group fine-grained attack labels into 7 semantic categories
8. Remove extremely rare attack categories (< min_class_count samples)
9. Remove highly correlated features (threshold configurable, default 0.95)
10. Remove statistically irrelevant features (hard-coded list from Kruskal-Wallis + RF analysis)
11. Keep Source_File column for train/test splitting
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attack type grouping: fine-grained label -> semantic group
ATTACK_GROUP_MAPPING = {
    'BENIGN':                       'Normal Traffic',
    'DoS Hulk':                     'DoS',
    'DoS GoldenEye':                'DoS',
    'DoS slowloris':                'DoS',
    'DoS Slowhttptest':             'DoS',
    'DDoS':                         'DDoS',
    'PortScan':                     'Port Scanning',
    'FTP-Patator':                  'Brute Force',
    'SSH-Patator':                  'Brute Force',
    'Bot':                          'Bot',
    'Web Attack \xef\xbf\xbd Brute Force':  'Web Attacks',
    'Web Attack \xef\xbf\xbd XSS':          'Web Attacks',
    'Web Attack \xef\xbf\xbd Sql Injection': 'Web Attacks',
    # encoded variants (mojibake from Windows-1252 read as UTF-8)
    'Web Attack \x96 Brute Force':  'Web Attacks',
    'Web Attack \x96 XSS':          'Web Attacks',
    'Web Attack \x96 Sql Injection':'Web Attacks',
    # plain ASCII fallback
    'Web Attack - Brute Force':     'Web Attacks',
    'Web Attack - XSS':             'Web Attacks',
    'Web Attack - Sql Injection':   'Web Attacks',
    # replacement-character variants (U+FFFD)
    'Web Attack \ufffd Brute Force':'Web Attacks',
    'Web Attack \ufffd XSS':        'Web Attacks',
    'Web Attack \ufffd Sql Injection':'Web Attacks',
    'Infiltration':                 'Infiltration',
    'Heartbleed':                   'Heartbleed',
}

# Features identified as statistically irrelevant by Kruskal-Wallis + RF analysis
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

# Rare classes to drop entirely (too few samples to learn from reliably)
RARE_CLASSES = {'Infiltration', 'Heartbleed'}


def validate_data(df):
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if 'Label' not in df.columns:
        raise ValueError("'Label' column not found in dataframe")
    logger.info(f"Data validation passed. Shape: {df.shape}, Columns: {len(df.columns)}")


def remove_duplicate_columns(df, skip_cols=('Label', 'Source_File')):
    """Remove columns whose data is identical to another column (keep first occurrence)."""
    cols = [c for c in df.columns if c not in skip_cols]
    to_drop = []
    seen = {}
    for col in cols:
        key = tuple(df[col].values)
        if key in seen:
            to_drop.append(col)
        else:
            seen[key] = col
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} duplicate-value columns: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        logger.info("No duplicate-value columns found.")
    return df


def remove_zero_variance_columns(df, skip_cols=('Label', 'Source_File', 'Attack Type')):
    """Remove columns that contain only a single unique value."""
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip_cols]
    zero_var = [c for c in num_cols if df[c].nunique() <= 1]
    if zero_var:
        logger.info(f"Dropping {len(zero_var)} zero-variance columns: {zero_var}")
        df = df.drop(columns=zero_var)
    else:
        logger.info("No zero-variance columns found.")
    return df


def remove_highly_correlated_features(df, threshold=0.95, skip_cols=('Label', 'Source_File', 'Attack Type')):
    """
    Remove one feature from each pair with absolute correlation above threshold.
    Keeps the first of each correlated pair (alphabetically earlier column).
    """
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip_cols]
    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} highly correlated features (threshold={threshold}): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        logger.info(f"No features with correlation > {threshold} found.")
    return df


def group_attack_labels(df):
    """Map fine-grained attack labels to semantic groups."""
    # Normalise label text first (strip whitespace)
    df['Label'] = df['Label'].astype(str).str.strip()

    # Try direct mapping; fall back to 'Unknown' for anything not in the map
    df['Attack Type'] = df['Label'].map(ATTACK_GROUP_MAPPING)

    unmapped = df[df['Attack Type'].isna()]['Label'].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped labels treated as 'Unknown': {list(unmapped)}")
        df['Attack Type'] = df['Attack Type'].fillna('Unknown')

    logger.info(f"Attack type distribution after grouping:\n{df['Attack Type'].value_counts()}")
    return df


def preprocess_csv(input_csv, output_csv, corr_threshold=0.95, min_class_count=100):
    """
    Full preprocessing pipeline for CICIDS2017.

    Args:
        input_csv (str): Path to merged CSV file (output of merge_data.py)
        output_csv (str): Path to save cleaned CSV
        corr_threshold (float): Correlation threshold for feature removal (default 0.95)
        min_class_count (int): Drop attack groups with fewer than this many samples (default 100)
    """
    logger.info(f"Starting preprocessing: {input_csv}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading CSV file...")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded dataframe shape: {df.shape}")
    validate_data(df)

    # ── Step 1: Clean column names ────────────────────────────────────────────
    logger.info("Step 1: Cleaning column names...")
    df.columns = df.columns.str.strip()

    # ── Step 2: Drop identifiers ──────────────────────────────────────────────
    logger.info("Step 2: Dropping non-generalizable identifier columns...")
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        logger.info(f"  Dropped: {existing_drop}")

    # ── Step 3: Remove duplicate rows ─────────────────────────────────────────
    logger.info("Step 3: Removing duplicate rows...")
    before = len(df)
    df = df.drop_duplicates(keep='first')
    removed = before - len(df)
    logger.info(f"  Removed {removed:,} duplicate rows ({removed/before*100:.1f}%). Shape: {df.shape}")

    # ── Step 4: Remove duplicate columns ──────────────────────────────────────
    logger.info("Step 4: Removing duplicate-value columns...")
    df = remove_duplicate_columns(df)
    logger.info(f"  Shape after: {df.shape}")

    # ── Step 5: Handle infinities → NaN → drop rows ───────────────────────────
    logger.info("Step 5: Handling infinities and missing values...")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning(f"  Found {inf_count} infinite values, replacing with NaN")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_rows = df.isna().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning(f"  Dropping {nan_rows:,} rows with NaN values ({nan_rows/len(df)*100:.2f}%)")
        df = df.dropna()
    logger.info(f"  Shape after: {df.shape}")

    # ── Step 6: Remove zero-variance columns ──────────────────────────────────
    logger.info("Step 6: Removing zero-variance columns...")
    df = remove_zero_variance_columns(df)
    logger.info(f"  Shape after: {df.shape}")

    # ── Step 7: Group attack labels ───────────────────────────────────────────
    logger.info("Step 7: Grouping attack labels into semantic categories...")
    before_counts = df['Label'].value_counts()
    logger.info(f"  Original label distribution:\n{before_counts}")
    df = group_attack_labels(df)

    # ── Step 8: Remove rare attack classes ────────────────────────────────────
    logger.info(f"Step 8: Removing rare attack classes (< {min_class_count} samples or in RARE_CLASSES list)...")
    class_counts = df['Attack Type'].value_counts()
    rare = set(class_counts[class_counts < min_class_count].index) | RARE_CLASSES
    rare = rare & set(df['Attack Type'].unique())  # only what actually exists
    if rare:
        before_rare = len(df)
        df = df[~df['Attack Type'].isin(rare)]
        logger.info(f"  Dropped classes: {rare}")
        logger.info(f"  Removed {before_rare - len(df):,} rows. Shape: {df.shape}")
    else:
        logger.info("  No rare classes to remove.")

    logger.info(f"  Final Attack Type distribution:\n{df['Attack Type'].value_counts()}")

    # ── Step 9: Remove highly correlated features ─────────────────────────────
    logger.info(f"Step 9: Removing highly correlated features (threshold={corr_threshold})...")
    df = remove_highly_correlated_features(df, threshold=corr_threshold)
    logger.info(f"  Shape after: {df.shape}")

    # ── Step 10: Remove statistically irrelevant features ────────────────────
    logger.info("Step 10: Removing statistically irrelevant features...")
    existing_irrelevant = [c for c in IRRELEVANT_FEATURES if c in df.columns]
    if existing_irrelevant:
        df = df.drop(columns=existing_irrelevant)
        logger.info(f"  Dropped: {existing_irrelevant}")
    else:
        logger.info("  None of the irrelevant features were present.")
    logger.info(f"  Shape after: {df.shape}")

    # ── Final validation ──────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Preprocessed dataframe is empty after cleaning")
    if 'Attack Type' not in df.columns:
        raise ValueError("'Attack Type' column missing after preprocessing")

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info(f"Saving preprocessed CSV to {output_csv}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Preprocessing complete! Final shape: {df.shape}")
    logger.info(f"Final columns ({len(df.columns)}): {list(df.columns)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CICIDS2017 CSV data for machine learning"
    )
    parser.add_argument(
        "--input", type=str,
        default=str(config.MERGED_CSV),
        help="Path to merged input CSV file"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(config.CLEANED_CSV),
        help="Path to output preprocessed CSV file"
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=0.95,
        help="Pearson correlation threshold for feature removal (default: 0.95)"
    )
    parser.add_argument(
        "--min-class-count", type=int, default=100,
        help="Drop attack groups with fewer than this many samples (default: 100)"
    )

    args = parser.parse_args()

    try:
        preprocess_csv(
            args.input, args.output,
            corr_threshold=args.corr_threshold,
            min_class_count=args.min_class_count,
        )
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

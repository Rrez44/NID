"""
Data preprocessing script for IDS thesis project.

Implements the *leak-safe* cleaning pipeline for CICIDS2017.
Only structural / domain-knowledge transforms happen here.
All *data-dependent* feature selection (zero-variance, correlation, irrelevant
features) is deferred to post-split via ``feature_selection.FeatureSelector``
in the training script.

Steps performed:
1. Strip whitespace from column names
2. Drop non-generalizable identifiers (IP, Flow ID, Timestamp)
3. Remove duplicate rows
4. Remove perfectly duplicate columns (hash-based, memory-efficient)
5. Replace infinities with NaN then drop rows with NaN (minimal loss)
6. Group fine-grained attack labels into 7 semantic categories
7. Remove extremely rare attack categories (< min_class_count samples)
8. Keep Source_File column for file-based train/test splitting
"""

import hashlib

import pandas as pd
import numpy as np
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

# Attack type grouping: fine-grained label -> semantic group.
# Covers both CIC-IDS2017 and CSE-CIC-IDS2018 label variants.
ATTACK_GROUP_MAPPING = {
    # ── CIC-IDS2017 ──────────────────────────────────────────────────────────
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

    # ── CSE-CIC-IDS2018 ──────────────────────────────────────────────────────
    'Benign':                       'Normal Traffic',
    'DoS attacks-Hulk':             'DoS',
    'DoS attacks-SlowHTTPTest':     'DoS',
    'DoS attacks-GoldenEye':        'DoS',
    'DoS attacks-Slowloris':        'DoS',
    'DDoS attacks-LOIC-HTTP':       'DDoS',
    'DDOS attack-HOIC':             'DDoS',
    'DDOS attack-LOIC-UDP':         'DDoS',
    'FTP-BruteForce':               'Brute Force',
    'SSH-Bruteforce':               'Brute Force',
    # 2018 "Brute Force" web attacks map to the same Web Attacks group as 2017.
    'Brute Force -Web':             'Web Attacks',
    'Brute Force -XSS':             'Web Attacks',
    'SQL Injection':                'Web Attacks',
    # NOTE: 'Infilteration' is the exact spelling in the 2018 CSVs — not a typo
    # in this mapping. Both 'Infiltration' (2017) and 'Infilteration' (2018)
    # land in the RARE_CLASSES set below and are dropped.
    'Infilteration':                'Infiltration',
    # 'Bot' is already mapped above and applies to 2018 as well.
}

# Rare classes to drop entirely (too few samples to learn from reliably).
RARE_CLASSES = {'Infiltration', 'Heartbleed'}


def validate_data(df):
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if 'Label' not in df.columns:
        raise ValueError("'Label' column not found in dataframe")
    logger.info(f"Data validation passed. Shape: {df.shape}, Columns: {len(df.columns)}")


def remove_duplicate_columns(df, skip_cols=('Label', 'Source_File', 'Dataset', 'Attack Type')):
    """Remove columns with identical values using SHA-256 hashing (memory-efficient)."""
    cols = [c for c in df.columns if c not in skip_cols]
    seen_hashes = {}
    to_drop = []
    for col in cols:
        h = hashlib.sha256(df[col].values.tobytes()).hexdigest()
        if h in seen_hashes:
            to_drop.append(col)
        else:
            seen_hashes[h] = col
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} duplicate-value columns: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        logger.info("No duplicate-value columns found.")
    return df


def group_attack_labels(df):
    """Map fine-grained attack labels to semantic groups."""
    df['Label'] = df['Label'].astype(str).str.strip()
    df['Attack Type'] = df['Label'].map(ATTACK_GROUP_MAPPING)

    unmapped = df[df['Attack Type'].isna()]['Label'].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped labels treated as 'Unknown': {list(unmapped)}")
        df['Attack Type'] = df['Attack Type'].fillna('Unknown')

    logger.info(f"Attack type distribution after grouping:\n{df['Attack Type'].value_counts()}")
    return df


def preprocess_csv(input_csv, output_csv, min_class_count=100):
    """
    Leak-safe preprocessing pipeline for CICIDS2017.

    Only applies transforms that do NOT depend on data statistics:
    identifier removal, deduplication, inf/NaN handling, label grouping.
    Data-dependent feature selection is handled post-split by FeatureSelector.

    Args:
        input_csv (str): Path to merged CSV file (output of merge_data.py)
        output_csv (str): Path to save cleaned CSV
        min_class_count (int): Drop attack groups with fewer than this many samples
    """
    logger.info(f"Starting preprocessing: {input_csv}")

    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # ── Load (streaming + float32 + per-chunk dedup to keep peak memory low) ─
    # The combined 2017+2018 merged CSV is large enough that a single-shot
    # pd.read_csv blows past available RAM (float64 default), and a
    # single-shot drop_duplicates on the concatenated frame does the same.
    # We stream in chunks and do as much work per-chunk as possible:
    #   - drop identifier columns early (fewer cols → cheaper dedup)
    #   - defensively coerce object-dtype feature columns to numeric
    #   - downcast float64 → float32 (halves memory)
    #   - inf → NaN → dropna
    #   - per-chunk drop_duplicates (most CIC duplicates are adjacent in
    #     time, so nearly all are intra-chunk)
    # Peak memory is then roughly the size of the final cleaned frame.
    logger.info("Loading CSV in chunks (float32 + per-chunk dedup)...")
    non_feature = {'Label', 'Source_File', 'Dataset', 'Attack Type'}
    drop_identifiers = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp',
                        'Src Port', 'Protocol']

    chunks = []
    raw_rows = 0
    kept_rows = 0
    for i, chunk in enumerate(
        pd.read_csv(input_csv, chunksize=1_000_000, low_memory=False)
    ):
        raw_rows += len(chunk)
        chunk.columns = chunk.columns.str.strip()

        # Drop identifier columns early (reduces dedup column count).
        existing = [c for c in drop_identifiers if c in chunk.columns]
        if existing:
            chunk = chunk.drop(columns=existing)

        # Defensive: coerce object-dtype feature cols to numeric
        # (2018 files had stray strings from Excel-truncated shards).
        for c in chunk.columns:
            if c not in non_feature and chunk[c].dtype == object:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')

        # Downcast float64 → float32 (halves feature-column memory).
        for c in chunk.select_dtypes(include=['float64']).columns:
            chunk[c] = chunk[c].astype('float32')

        # inf → NaN → dropna per chunk to shrink before concat.
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk = chunk.dropna()

        # Per-chunk dedup: cheap (1M-row dedup) and catches most duplicates
        # since CIC datasets' duplicate rows come from the flow exporter
        # re-emitting the same flow on timeout (adjacent in time).
        chunk = chunk.drop_duplicates(keep='first')

        kept_rows += len(chunk)
        chunks.append(chunk)
        logger.info(
            f"  chunk {i+1}: read={len(chunk):,} kept "
            f"(cumulative read={raw_rows:,}, kept={kept_rows:,})"
        )

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    logger.info(f"Streaming load complete. Shape: {df.shape}")
    validate_data(df)

    # ── Step 1: Clean column names ────────────────────────────────────────────
    logger.info("Step 1: Cleaning column names...")
    df.columns = df.columns.str.strip()

    # ── Step 2: Drop identifiers ──────────────────────────────────────────────
    logger.info("Step 2: Dropping non-generalizable identifier columns...")
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp',
                 'Src Port', 'Protocol']  # 2018-only leftovers if merge ran without canonicalization
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        logger.info(f"  Dropped: {existing_drop}")

    # ── Step 2b: Coerce feature columns to numeric ───────────────────────────
    # Some 2018 CSVs have object-dtype feature columns (stray header rows, mixed
    # values). Coerce now so inf/NaN handling and dedup work on clean floats.
    non_feature = {'Label', 'Source_File', 'Dataset', 'Attack Type'}
    feature_cols = [c for c in df.columns if c not in non_feature]
    object_cols = [c for c in feature_cols if df[c].dtype == object]
    if object_cols:
        logger.info(f"Step 2b: Coercing {len(object_cols)} object-dtype feature columns to numeric")
        for c in object_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # ── Step 3: Remove cross-chunk duplicate rows (hash-based) ───────────────
    # Intra-chunk duplicates were already removed per-chunk in the loader.
    # Here we catch cross-chunk duplicates. Hashing each row to a single
    # uint64 cuts memory dramatically vs comparing ~74 float32 columns:
    # row_hash is ~150 MB for 18.9M rows and dedup on a single column
    # fits comfortably in RAM.
    logger.info("Step 3: Removing cross-chunk duplicate rows (hash-based)...")
    before = len(df)
    row_hash = pd.util.hash_pandas_object(df, index=False)
    mask = ~row_hash.duplicated(keep='first')
    del row_hash
    df = df[mask].reset_index(drop=True)
    del mask
    removed = before - len(df)
    logger.info(
        f"  Removed {removed:,} cross-chunk duplicate rows "
        f"({removed/before*100:.2f}%). Shape: {df.shape}"
    )

    # ── Step 4: Remove duplicate columns (hash-based) ────────────────────────
    logger.info("Step 4: Removing duplicate-value columns...")
    df = remove_duplicate_columns(df)
    logger.info(f"  Shape after: {df.shape}")

    # ── Step 5: Handle infinities → NaN → drop rows ──────────────────────────
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

    # ── Step 6: Group attack labels ───────────────────────────────────────────
    logger.info("Step 6: Grouping attack labels into semantic categories...")
    before_counts = df['Label'].value_counts()
    logger.info(f"  Original label distribution:\n{before_counts}")
    df = group_attack_labels(df)

    # ── Step 7: Remove rare attack classes ────────────────────────────────────
    logger.info(f"Step 7: Removing rare attack classes (< {min_class_count} samples or in RARE_CLASSES list)...")
    class_counts = df['Attack Type'].value_counts()
    rare = set(class_counts[class_counts < min_class_count].index) | RARE_CLASSES
    rare = rare & set(df['Attack Type'].unique())
    if rare:
        before_rare = len(df)
        df = df[~df['Attack Type'].isin(rare)]
        logger.info(f"  Dropped classes: {rare}")
        logger.info(f"  Removed {before_rare - len(df):,} rows. Shape: {df.shape}")
    else:
        logger.info("  No rare classes to remove.")

    logger.info(f"  Final Attack Type distribution:\n{df['Attack Type'].value_counts()}")

    # NOTE: Zero-variance removal, correlation filtering, and irrelevant-feature
    # removal have been moved to training/feature_selection.py (post-split) to
    # prevent data leakage.  The FeatureSelector is fit on training data only.

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
        "--min-class-count", type=int, default=100,
        help="Drop attack groups with fewer than this many samples (default: 100)"
    )

    args = parser.parse_args()

    try:
        preprocess_csv(
            args.input, args.output,
            min_class_count=args.min_class_count,
        )
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

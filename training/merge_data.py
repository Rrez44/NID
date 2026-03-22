"""
Data merging script for IDS thesis project.

This script merges multiple CSV files from the CICIDS2017 dataset into a single
consolidated CSV file for preprocessing and training.
"""

import pandas as pd
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


def merge_csv_files(input_dir, output_csv):
    """
    Merge all CSV files from a directory into a single CSV file.
    
    Args:
        input_dir (str): Directory containing CSV files to merge
        output_csv (str): Path to output merged CSV file
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    logger.info(f"Starting merge process from {input_dir}")
    
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all CSV files
    csv_files = list(Path(input_dir).glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files to merge")
    
    # Ensure output directory exists (skip if output_csv has no path component)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read and write in a stream to avoid holding full merged dataset in memory
    merged_columns = None
    total_rows = 0
    label_counts = None

    for i, csv_file in enumerate(sorted(csv_files)):  # deterministic order (e.g. Monday -> Friday)
        logger.info(f"Reading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            df['Source_File'] = csv_file.name  # for session/file-based train-test split
            logger.info(f"  - Shape: {df.shape}, Columns: {len(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading {csv_file.name}: {e}")
            raise

        if merged_columns is not None and list(df.columns) != merged_columns:
            raise ValueError(
                f"Column mismatch in {csv_file.name}: expected {merged_columns}, got {list(df.columns)}"
            )
        merged_columns = list(df.columns)

        # First file: write with header. Rest: append without header
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        df.to_csv(output_csv, index=False, mode=mode, header=header, encoding='utf-8')
        total_rows += len(df)

        if 'Label' in df.columns:
            if label_counts is None:
                label_counts = df['Label'].value_counts()
            else:
                label_counts = label_counts.add(df['Label'].value_counts(), fill_value=0).fillna(0).astype(int)

    logger.info(f"Merged total: {total_rows} rows")
    if label_counts is not None:
        logger.info(f"Label distribution:\n{label_counts}")
    logger.info(f"Merged CSV saved to {output_csv}")

    # Return a minimal dataframe for compatibility (optional callers)
    return pd.DataFrame(columns=merged_columns or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV files into a single file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(config.MACHINE_LEARNING_CVE_DIR),
        help="Directory containing CSV files to merge"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.MERGED_CSV),
        help="Path to output merged CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        merge_csv_files(args.input_dir, args.output)
        logger.info("Merge completed successfully!")
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise

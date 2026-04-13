"""
Data merging script for IDS thesis project.

Streams CSV files from one or more raw-data directories into a single merged
CSV. Handles both CIC-IDS2017 (data/raw/MachineLearningCVE) and the
CSE-CIC-IDS2018 daily files (data/raw/cicids2018):

- Canonicalizes 2018 column names to the 2017 schema (77 shared features).
- Drops the duplicate `Fwd Header Length.1` column from 2017 (it is always a
  byte-for-byte copy of `Fwd Header Length`).
- Drops columns that exist only in 2018 (`Protocol`, `Timestamp`, and the
  5-tuple in the 84-column Tuesday file).
- Filters out stray embedded header rows inside 2018 files (several 2018 CSVs
  are concatenations of Excel-truncated exports with duplicated header lines).
- Tags every row with a `Dataset` column (`'2017'` or `'2018'`) for the
  cross-dataset split strategies in train_xgb.py.
- Streams in chunks to keep peak memory low (2018 Tuesday alone is 4 GB).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

import common  # noqa: F401 - ensures project root on path
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Schema canonicalization ───────────────────────────────────────────────────

# 2018 column name → 2017 column name. Only columns that need renaming are
# listed; unchanged names (e.g. "Flow Duration", "CWE Flag Count") pass through.
CANONICAL_2018_TO_2017 = {
    'Dst Port':          'Destination Port',
    'Tot Fwd Pkts':      'Total Fwd Packets',
    'Tot Bwd Pkts':      'Total Backward Packets',
    'TotLen Fwd Pkts':   'Total Length of Fwd Packets',
    'TotLen Bwd Pkts':   'Total Length of Bwd Packets',
    'Fwd Pkt Len Max':   'Fwd Packet Length Max',
    'Fwd Pkt Len Min':   'Fwd Packet Length Min',
    'Fwd Pkt Len Mean':  'Fwd Packet Length Mean',
    'Fwd Pkt Len Std':   'Fwd Packet Length Std',
    'Bwd Pkt Len Max':   'Bwd Packet Length Max',
    'Bwd Pkt Len Min':   'Bwd Packet Length Min',
    'Bwd Pkt Len Mean':  'Bwd Packet Length Mean',
    'Bwd Pkt Len Std':   'Bwd Packet Length Std',
    'Flow Byts/s':       'Flow Bytes/s',
    'Flow Pkts/s':       'Flow Packets/s',
    'Fwd IAT Tot':       'Fwd IAT Total',
    'Bwd IAT Tot':       'Bwd IAT Total',
    'Fwd Header Len':    'Fwd Header Length',
    'Bwd Header Len':    'Bwd Header Length',
    'Fwd Pkts/s':        'Fwd Packets/s',
    'Bwd Pkts/s':        'Bwd Packets/s',
    'Pkt Len Min':       'Min Packet Length',
    'Pkt Len Max':       'Max Packet Length',
    'Pkt Len Mean':      'Packet Length Mean',
    'Pkt Len Std':       'Packet Length Std',
    'Pkt Len Var':       'Packet Length Variance',
    'FIN Flag Cnt':      'FIN Flag Count',
    'SYN Flag Cnt':      'SYN Flag Count',
    'RST Flag Cnt':      'RST Flag Count',
    'PSH Flag Cnt':      'PSH Flag Count',
    'ACK Flag Cnt':      'ACK Flag Count',
    'URG Flag Cnt':      'URG Flag Count',
    'ECE Flag Cnt':      'ECE Flag Count',
    'Pkt Size Avg':      'Average Packet Size',
    'Fwd Seg Size Avg':  'Avg Fwd Segment Size',
    'Bwd Seg Size Avg':  'Avg Bwd Segment Size',
    'Fwd Byts/b Avg':    'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg':    'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg':  'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg':    'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg':    'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg':  'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts':  'Subflow Fwd Packets',
    'Subflow Fwd Byts':  'Subflow Fwd Bytes',
    'Subflow Bwd Pkts':  'Subflow Bwd Packets',
    'Subflow Bwd Byts':  'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min':  'min_seg_size_forward',
}

# Columns present only in 2018 (no 2017 analogue): dropped at merge time.
DROP_FROM_2018 = {
    'Protocol', 'Timestamp',
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP',
}

# The 2017 dataset ships a `Fwd Header Length.1` duplicate column — always a
# byte-for-byte copy of `Fwd Header Length`. Dropping it during merge is a
# no-op for the model (the preprocess-stage hash dedupe would remove it
# anyway) and simplifies column alignment with 2018.
DROP_FROM_2017 = {'Fwd Header Length.1'}

# Final canonical feature + metadata column order (77 features + label + metas).
CANONICAL_COLUMNS = [
    'Destination Port', 'Flow Duration',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Label', 'Source_File', 'Dataset',
]


def _detect_dataset(header_cols) -> str:
    """Return '2018' if header looks like CSE-CIC-IDS2018, else '2017'."""
    return '2018' if 'Dst Port' in header_cols else '2017'


def _canonicalize(df: pd.DataFrame, dataset: str, source_file: str) -> pd.DataFrame:
    """Rename / drop columns so the frame matches CANONICAL_COLUMNS."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if dataset == '2018':
        drop_cols = [c for c in DROP_FROM_2018 if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        rename = {k: v for k, v in CANONICAL_2018_TO_2017.items() if k in df.columns}
        if rename:
            df = df.rename(columns=rename)
    else:  # 2017
        drop_cols = [c for c in DROP_FROM_2017 if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    df['Dataset'] = dataset
    df['Source_File'] = source_file

    # Drop stray embedded header rows (2018 files have Label == 'Label' rows
    # where the CIC team re-concatenated Excel-truncated shards).
    if 'Label' in df.columns:
        bad = df['Label'].astype(str).str.strip().eq('Label')
        if bad.any():
            df = df[~bad]

    # Reindex to the canonical order. Missing cols become NaN (should only
    # happen if upstream schema drifts — surfaces as NaN rows dropped by
    # preprocess.py rather than silent column misalignment).
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if missing:
        logger.warning("  %s: missing canonical cols: %s", source_file, missing)
    if extra:
        logger.warning("  %s: unexpected extra cols dropped: %s", source_file, extra)
    return df.reindex(columns=CANONICAL_COLUMNS)


def _iter_chunks(csv_path: Path, chunksize: int):
    """Yield pandas DataFrame chunks from a CSV. Uses low_memory=False so mixed
    dtype warnings don't flood logs for the messy 2018 files."""
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        yield chunk


def merge_csv_files(input_dirs, output_csv, chunksize: int = 500_000,
                    max_rows_per_file: int | None = None) -> int:
    """
    Stream-merge every CSV under each path in `input_dirs` into `output_csv`.

    Args:
        input_dirs: Iterable of directory paths OR individual CSV paths.
        output_csv: Destination CSV path.
        chunksize:  Rows per pandas chunk (tune for memory).
        max_rows_per_file: Optional cap on rows emitted per source CSV (useful
                           for downsampling the huge 2018 files).

    Returns:
        Total rows written.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Expand dirs → list of CSV paths, keep individual files as-is.
    csv_files: list[Path] = []
    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            logger.warning("Input path not found, skipping: %s", p)
            continue
        if p.is_dir():
            csv_files.extend(sorted(p.glob('*.csv')))
        elif p.suffix.lower() == '.csv':
            csv_files.append(p)
    if not csv_files:
        raise ValueError(f"No CSV files found under: {list(input_dirs)}")

    logger.info("Merging %d CSV files → %s", len(csv_files), output_path)

    first_write = True
    total_rows = 0
    label_counts: pd.Series | None = None

    for csv_file in csv_files:
        # Peek at header to detect schema (tiny read — single row).
        header = pd.read_csv(csv_file, nrows=0)
        header.columns = header.columns.str.strip()
        dataset = _detect_dataset(header.columns)
        logger.info("[%s] %s  (%s)", dataset, csv_file.name, f"{csv_file.stat().st_size/1e6:.0f} MB")

        rows_from_file = 0
        try:
            for chunk in _iter_chunks(csv_file, chunksize):
                chunk = _canonicalize(chunk, dataset, csv_file.name)

                if max_rows_per_file is not None:
                    remaining = max_rows_per_file - rows_from_file
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining]

                chunk.to_csv(
                    output_path,
                    mode='w' if first_write else 'a',
                    header=first_write,
                    index=False,
                    encoding='utf-8',
                )
                first_write = False
                rows_from_file += len(chunk)
                total_rows += len(chunk)

                if 'Label' in chunk.columns:
                    lc = chunk['Label'].value_counts()
                    label_counts = lc if label_counts is None else label_counts.add(lc, fill_value=0)

                if max_rows_per_file is not None and rows_from_file >= max_rows_per_file:
                    break
        except Exception as e:
            logger.error("Failed on %s: %s", csv_file.name, e)
            raise

        logger.info("  → %s rows written (cumulative: %s)", f"{rows_from_file:,}", f"{total_rows:,}")

    logger.info("Merge complete: %s rows → %s", f"{total_rows:,}", output_path)
    if label_counts is not None:
        label_counts = label_counts.astype(int).sort_values(ascending=False)
        logger.info("Label distribution (raw, pre-preprocess):\n%s", label_counts.to_string())
    return total_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge CIC-IDS2017 and CSE-CIC-IDS2018 CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        action='append',
        default=None,
        help="Directory (or CSV file) to include. Repeat to add more. "
             "Default: both MachineLearningCVE (2017) and cicids2018 (if present).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.MERGED_CSV),
        help="Path to output merged CSV file",
    )
    parser.add_argument(
        "--chunksize", type=int, default=500_000,
        help="pandas read_csv chunksize (rows per chunk; default: 500000).",
    )
    parser.add_argument(
        "--max-rows-per-file", type=int, default=None,
        help="Optional cap on rows emitted per source CSV (for downsampling).",
    )

    args = parser.parse_args()

    if args.input_dir:
        input_dirs = args.input_dir
    else:
        input_dirs = [str(config.MACHINE_LEARNING_CVE_DIR)]
        if config.CICIDS2018_DIR.exists():
            input_dirs.append(str(config.CICIDS2018_DIR))
            logger.info("CIC-IDS2018 directory detected — including both datasets.")
        else:
            logger.info("CIC-IDS2018 directory not found — using 2017 only.")

    try:
        merge_csv_files(
            input_dirs, args.output,
            chunksize=args.chunksize,
            max_rows_per_file=args.max_rows_per_file,
        )
        logger.info("Merge completed successfully!")
    except Exception as e:
        logger.error("Merge failed: %s", e)
        raise

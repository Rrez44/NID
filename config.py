"""
Configuration file for IDS thesis project.

Provides path configuration that automatically detects the project root
and exposes relative paths, making the project portable across systems.
"""

from pathlib import Path


def get_project_root():
    """Detect the project root by checking for repository markers."""
    project_root = Path(__file__).resolve().parent
    markers = ['README.md', 'requirements.txt', '.git']
    if any((project_root / marker).exists() for marker in markers):
        return project_root
    return project_root


PROJECT_ROOT = get_project_root()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MERGED_DATA_DIR = DATA_DIR / "merged"
MACHINE_LEARNING_CVE_DIR = RAW_DATA_DIR / "MachineLearningCVE"  # CIC-IDS2017
CICIDS2018_DIR = RAW_DATA_DIR / "cicids2018"                    # CSE-CIC-IDS2018
MERGED_CSV = MERGED_DATA_DIR / "MachineLearningCSV_merged.csv"
CLEANED_CSV = MERGED_DATA_DIR / "MachineLearningCSV_cleaned.csv"

# Model / artifact paths
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "xgb_ids_model.pkl"
DEFAULT_LABEL_MAPPING = MODELS_DIR / "xgb_ids_model_label_mapping.pkl"
DEFAULT_FEATURE_NAMES = MODELS_DIR / "xgb_ids_model_feature_names.pkl"
DEFAULT_FEATURE_SELECTOR = MODELS_DIR / "xgb_ids_model_feature_selector.pkl"
DEFAULT_FEATURE_IMPORTANCE = MODELS_DIR / "xgb_ids_model_feature_importance.csv"

# Training scripts
TRAINING_DIR = PROJECT_ROOT / "training"

# API
API_DIR = PROJECT_ROOT / "api"

# Reproducibility
RANDOM_STATE = 42


def get_paths():
    """Return all paths as a {name: str} dict."""
    return {
        'project_root':           str(PROJECT_ROOT),
        'data_dir':               str(DATA_DIR),
        'raw_data_dir':           str(RAW_DATA_DIR),
        'merged_data_dir':        str(MERGED_DATA_DIR),
        'machine_learning_cve_dir': str(MACHINE_LEARNING_CVE_DIR),
        'merged_csv':             str(MERGED_CSV),
        'cleaned_csv':            str(CLEANED_CSV),
        'models_dir':             str(MODELS_DIR),
        'default_model':          str(DEFAULT_MODEL),
        'default_label_mapping':  str(DEFAULT_LABEL_MAPPING),
        'default_feature_names':  str(DEFAULT_FEATURE_NAMES),
        'default_feature_selector': str(DEFAULT_FEATURE_SELECTOR),
        'default_feature_importance': str(DEFAULT_FEATURE_IMPORTANCE),
        'training_dir':           str(TRAINING_DIR),
    }


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, MERGED_DATA_DIR, MACHINE_LEARNING_CVE_DIR,
              MODELS_DIR, TRAINING_DIR, API_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project Configuration:")
    print(f"Project Root: {PROJECT_ROOT}")
    print("\nPaths:")
    for key, value in get_paths().items():
        print(f"  {key}: {value}")
    print("\nEnsuring directories exist...")
    ensure_directories()
    print("Done!")

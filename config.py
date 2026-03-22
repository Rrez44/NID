"""
Configuration file for IDS thesis project.

This module provides path configuration that automatically detects the project root
and provides relative paths, making the project portable across different systems.
"""

import os
from pathlib import Path


def get_project_root():
    """
    Automatically detect the project root directory.
    
    Looks for common markers (like .git, README.md, requirements.txt) to identify
    the project root, or uses the directory containing this config.py file.
    
    Returns:
        Path: Path to project root directory
    """
    # Get the directory where this config file is located
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Verify it's the project root by checking for common files
    markers = ['README.md', 'requirements.txt', '.git']
    if any((project_root / marker).exists() for marker in markers):
        return project_root
    
    # If markers not found, assume current directory is project root
    return project_root


# Get project root
PROJECT_ROOT = get_project_root()

# Data paths (relative to project root)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MERGED_DATA_DIR = DATA_DIR / "merged"
MACHINE_LEARNING_CVE_DIR = RAW_DATA_DIR / "MachineLearningCVE"
MERGED_CSV = MERGED_DATA_DIR / "MachineLearningCSV_merged.csv"
CLEANED_CSV = MERGED_DATA_DIR / "MachineLearningCSV_cleaned.csv"

# Model paths (relative to project root)
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "xgb_ids_model.pkl"
DEFAULT_LABEL_MAPPING = MODELS_DIR / "xgb_ids_model_label_mapping.pkl"
DEFAULT_FEATURE_IMPORTANCE = MODELS_DIR / "xgb_ids_model_feature_importance.csv"

# Training paths
TRAINING_DIR = PROJECT_ROOT / "training"

# Convert Path objects to strings for compatibility
def get_paths():
    """
    Get all paths as a dictionary with string values.
    
    Returns:
        dict: Dictionary of path names to string paths
    """
    return {
        'project_root': str(PROJECT_ROOT),
        'data_dir': str(DATA_DIR),
        'raw_data_dir': str(RAW_DATA_DIR),
        'merged_data_dir': str(MERGED_DATA_DIR),
        'machine_learning_cve_dir': str(MACHINE_LEARNING_CVE_DIR),
        'merged_csv': str(MERGED_CSV),
        'cleaned_csv': str(CLEANED_CSV),
        'models_dir': str(MODELS_DIR),
        'default_model': str(DEFAULT_MODEL),
        'default_label_mapping': str(DEFAULT_LABEL_MAPPING),
        'default_feature_importance': str(DEFAULT_FEATURE_IMPORTANCE),
        'training_dir': str(TRAINING_DIR),
    }


# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        MERGED_DATA_DIR,
        MACHINE_LEARNING_CVE_DIR,
        MODELS_DIR,
        TRAINING_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print all paths for debugging
    print("Project Configuration:")
    print(f"Project Root: {PROJECT_ROOT}")
    print("\nPaths:")
    for key, value in get_paths().items():
        print(f"  {key}: {value}")
    
    print("\nEnsuring directories exist...")
    ensure_directories()
    print("Done!")

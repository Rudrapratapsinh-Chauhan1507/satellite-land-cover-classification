
from pathlib import Path

# Project root:
# satellite-land-cover-classification/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
AOI_DIR = DATA_DIR / "aoi"
RAW_DATA_DIR = DATA_DIR / "raw" / "sentinel2"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "sentinel2"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDICES_DIR = OUTPUTS_DIR / "indices"
TRAINING_INDICES_DIR = OUTPUTS_DIR / "training_indices"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"
MODELS_DIR = OUTPUTS_DIR / "models"
CLASSIFICATION_DIR = OUTPUTS_DIR / "classification"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Important files
AOI_FILE = AOI_DIR / "study_area.geojson"
TRAINING_SAMPLES_FILE = AOI_DIR / "training_samples_v2.gpkg"
TRAINING_DATASET_FILE = OUTPUTS_DIR / "training_dataset.csv"
MODEL_FILE = MODELS_DIR / "random_forest_baseline.joblib"
CLASSIFICATION_FILE = CLASSIFICATION_DIR / "land_cover_baseline.tif"


def ensure_output_directories():
    """Create required output directories if they do not exist."""
    for directory in (
        INDICES_DIR,
        TRAINING_INDICES_DIR,
        TRAINING_OUTPUT_DIR,
        MODELS_DIR,
        CLASSIFICATION_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

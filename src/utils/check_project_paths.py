from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    AOI_FILE,
    TRAINING_SAMPLES_FILE,
    TRAINING_DATASET_FILE,
    MODEL_FILE,
    CLASSIFICATION_FILE,
)

print("=" * 60)
print("PROJECT PATH AUDIT")
print("=" * 60)

paths_to_check = {
    "Raw Sentinel-2 data": RAW_DATA_DIR,
    "Processed Sentinel-2 data": PROCESSED_DATA_DIR,
    "Study area": AOI_FILE,
    "Training samples": TRAINING_SAMPLES_FILE,
    "Training dataset": TRAINING_DATASET_FILE,
    "Trained model": MODEL_FILE,
    "Classification raster": CLASSIFICATION_FILE,
}

for name, path in paths_to_check.items():
    exists = path.exists()
    status = "FOUND" if exists else "MISSING"
    print(f"{status:8} | {name}: {path}")

print("\nChecking source scripts for hard-coded paths...")

source_dir = PROJECT_ROOT / "src"

for file in sorted(source_dir.rglob("*.py")):
    if file.name == "check_project_paths.py":
        continue

    try:
        content = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    suspicious_lines = [
        (number, line.strip())
        for number, line in enumerate(content.splitlines(), start=1)
        if (
            "C:\\Users\\" in line
            or "C:/Users/" in line
            or "Desktop\\satellite-land-cover-classification" in line
        )
    ]

    for number, line in suspicious_lines:
        print(f"{file.relative_to(PROJECT_ROOT)}:{number}: {line}")

print("\nPath audit complete.")

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import (
    TRAINING_DATASET_FILE,
    PROCESSED_DATA_DIR,
    TRAINING_INDICES_DIR,
)

# Keep Extra Trees outputs separate from the Random Forest baseline.
MODEL_OUTPUT = ROOT / "outputs/models/extra_trees_candidate.joblib"
RASTER_OUTPUT = ROOT / "outputs/classification/land_cover_extra_trees.tif"

FEATURES = [
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "NDBI",
]

# Each value is a complete path.
BAND_FILES = {
    "B02": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B02_10m_clipped.tif",
    "B03": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B03_10m_clipped.tif",
    "B04": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B04_10m_clipped.tif",
    "B05": PROCESSED_DATA_DIR / "aligned_10m/T42QZL_20261003T053651_B05_10m_aligned.tif",
    "B06": PROCESSED_DATA_DIR / "aligned_10m/T42QZL_20261003T053651_B06_10m_aligned.tif",
    "B07": PROCESSED_DATA_DIR / "aligned_10m/T42QZL_20261003T053651_B07_10m_aligned.tif",
    "B08": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B08_10m_clipped.tif",
    "B8A": PROCESSED_DATA_DIR / "aligned_10m/T42QZL_20261003T053651_B8A_10m_aligned.tif",
    "B11": PROCESSED_DATA_DIR / "aligned_10m/T42QZL_20261003T053651_B11_10m_aligned.tif",
    "B12": PROCESSED_DATA_DIR / "aligned_10m/T42QZL_20261003T053651_B12_10m_aligned.tif",
}

INDEX_FILES = {
    "NDVI": TRAINING_INDICES_DIR / "ndvi.tif",
    "NDWI": TRAINING_INDICES_DIR / "ndwi.tif",
    "NDBI": TRAINING_INDICES_DIR / "ndbi.tif",
}

CLASS_MAPPING = {
    "vegetation": 1,
    "built_up": 2,
    "bare_soil": 3,
    "water": 4,
    "road": 5,
}


def load_aligned_raster(path, name, reference):
    """Load a raster only if its grid matches the B02 reference."""
    if not path.exists():
        raise FileNotFoundError(f"{name} raster not found: {path}")

    with rasterio.open(path) as src:
        if (
            src.width != reference["width"]
            or src.height != reference["height"]
            or src.crs != reference["crs"]
            or src.transform != reference["transform"]
        ):
            raise ValueError(
                f"{name} grid does not match B02. "
                "Check CRS, dimensions and transform."
            )

        array = src.read(1).astype(np.float32)
        nodata = src.nodata

    return array, nodata


def main():
    # 1. Load labelled training samples.
    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATASET_FILE)

    X = df[FEATURES].astype(float)
    y = df["class_name"].astype(str)

    unknown_classes = set(y.unique()) - set(CLASS_MAPPING)
    if unknown_classes:
        raise ValueError(f"Unknown training classes: {unknown_classes}")

    print(f"Training samples: {len(df)}")
    print("Training Extra Trees candidate...")

    # 2. Train the candidate model.
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # 3. Read the reference grid from B02.
    reference_path = BAND_FILES["B02"]
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference B02 raster not found: {reference_path}"
        )

    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        reference = {
            "width": ref.width,
            "height": ref.height,
            "crs": ref.crs,
            "transform": ref.transform,
        }

    height = reference["height"]
    width = reference["width"]

    print(f"Reference grid: {width} x {height}")
    print(f"Reference CRS: {reference['crs']}")

    # 4. Load all spectral bands and indices.
    arrays = {}
    nodata_values = {}

    for name, path in BAND_FILES.items():
        arrays[name], nodata_values[name] = load_aligned_raster(
            path, name, reference
        )
        print(f"Loaded band: {name}")

    for name, path in INDEX_FILES.items():
        arrays[name], nodata_values[name] = load_aligned_raster(
            path, name, reference
        )
        print(f"Loaded index: {name}")

    # 5. Identify pixels with valid values in every feature.
    valid = np.ones((height, width), dtype=bool)

    for name in FEATURES:
        arr = arrays[name]
        valid &= np.isfinite(arr)

        nodata = nodata_values[name]
        if nodata is not None and np.isfinite(nodata):
            valid &= arr != nodata

    valid_count = int(valid.sum())
    print("Valid pixels:", valid_count)

    if valid_count == 0:
        raise ValueError(
            "No valid pixels remain. Check raster NoData values and alignment."
        )

    # 6. Predict in chunks to limit memory usage.
    output = np.zeros((height, width), dtype=np.uint8)
    flat_valid = np.flatnonzero(valid.ravel())
    chunk_size = 20000

    for start in range(0, len(flat_valid), chunk_size):
        pixel_indices = flat_valid[start:start + chunk_size]

        chunk = pd.DataFrame({
            feature: arrays[feature].ravel()[pixel_indices]
            for feature in FEATURES
        })

        predicted_names = model.predict(chunk)
        predicted_ids = np.array(
            [CLASS_MAPPING[name] for name in predicted_names],
            dtype=np.uint8,
        )

        output.ravel()[pixel_indices] = predicted_ids

        completed = min(start + chunk_size, len(flat_valid))
        print(f"Predicted {completed}/{len(flat_valid)} pixels")

    # 7. Save the model and metadata separately from Random Forest.
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RASTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "class_names": list(CLASS_MAPPING.keys()),
            "class_mapping": CLASS_MAPPING,
        },
        MODEL_OUTPUT,
    )
    print("\nSaved model:", MODEL_OUTPUT)

    # 8. Save the classification GeoTIFF.
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(RASTER_OUTPUT, "w", **profile) as dst:
        dst.write(output, 1)

    print("\nExtra Trees prediction complete.")
    print("Saved raster:", RASTER_OUTPUT)
    print("Class mapping:", CLASS_MAPPING)
    print("Raster dimensions:", width, "x", height)
    print("Class pixel counts:")

    for class_name, class_id in CLASS_MAPPING.items():
        count = int(np.count_nonzero(output == class_id))
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()

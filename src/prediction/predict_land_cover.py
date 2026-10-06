import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# 1. Project paths
# Allow importing shared paths when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    PROCESSED_DATA_DIR,
    TRAINING_INDICES_DIR,
    MODEL_FILE,
    CLASSIFICATION_DIR,
    CLASSIFICATION_FILE,
)

# Project paths
BANDS_DIR = PROCESSED_DATA_DIR
INDICES_DIR = TRAINING_INDICES_DIR
MODEL_PATH = MODEL_FILE
OUTPUT_DIR = CLASSIFICATION_DIR
OUTPUT_PATH = CLASSIFICATION_FILE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load trained model

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run src/training/train_model.py first."
    )

saved = joblib.load(MODEL_PATH)

model = saved["model"]
features = list(saved["features"])
class_mapping = saved["class_mapping"]

EXPECTED_FEATURES = [
    "B02", "B03", "B04",
    "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "NDBI",
]

if features != EXPECTED_FEATURES:
    raise ValueError(
        "Feature order mismatch!\n"
        f"Expected: {EXPECTED_FEATURES}\n"
        f"Loaded:   {features}"
    )

print("Feature order verified successfully.")

print("Loaded model:", MODEL_PATH.name)
print("Predictor features:", features)


# --------------------------------------------------
# 3. Locate clipped satellite band files
# --------------------------------------------------

def find_band(band_name):
    matches = list(
        BANDS_DIR.glob(f"*_{band_name}_*_clipped.tif")
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one clipped file for {band_name}, "
            f"but found {len(matches)} in {BANDS_DIR}.\n"
            "Check the filenames in data/processed/sentinel2."
        )

    return matches[0]


# Use the 10 m B02 raster as the output grid.
reference_path = find_band("B02")

with rasterio.open(reference_path) as reference:
    reference_data = reference.read(1).astype(np.float32)
    reference_profile = reference.profile.copy()

    width = reference.width
    height = reference.height
    transform = reference.transform
    crs = reference.crs

print(f"Reference grid: {width} x {height}")
print("Reference CRS:", crs)


# --------------------------------------------------
# 4. Read and align all predictors to the reference
# --------------------------------------------------


def read_aligned(path, resampling=Resampling.bilinear):
    """Read a raster and align it to the B02 reference grid."""

    with rasterio.open(path) as src:
        source = src.read(1).astype(np.float32)

        # Convert the source raster's NoData values to NaN.
        if src.nodata is not None:
            source[source == src.nodata] = np.nan

        same_grid = (
            src.width == width
            and src.height == height
            and src.crs == crs
            and src.transform == transform
        )

        if same_grid:
            return source

        destination = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=resampling,
            init_dest_nodata=True,
        )

        return destination


arrays = {"B02": reference_data}

for band in [
    "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12"
]:
    path = find_band(band)
    arrays[band] = read_aligned(path)
    print("Loaded band:", band)


# Read index rasters. These are the training-specific indices,
# which retain valid index values for SCL class 7.
for index_name in ["NDVI", "NDWI", "NDBI"]:
    path = INDICES_DIR / f"{index_name.lower()}.tif"

    if not path.exists():
        raise FileNotFoundError(
            f"Required index raster not found: {path}"
        )

    arrays[index_name] = read_aligned(path)

    print("Loaded index:", index_name)


# --------------------------------------------------
# 5. Build the valid-pixel mask
# --------------------------------------------------

# All reflectance bands must have positive values.
band_names = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

valid = np.ones((height, width), dtype=bool)

for name in band_names:
    valid &= np.isfinite(arrays[name])
    valid &= arrays[name] > 0

# Require valid values for every index as well.
for name in ["NDVI", "NDWI", "NDBI"]:
    valid &= np.isfinite(arrays[name])

    # Reject implausible normalized-index values.
    valid &= arrays[name] >= -1.001
    valid &= arrays[name] <= 1.001

valid_count = int(valid.sum())

if valid_count == 0:
    raise RuntimeError(
        "No valid pixels found. Check raster alignment, nodata "
        "values, and the training index rasters."
    )

print("Valid pixels to classify:", valid_count)


# --------------------------------------------------
# 6. Prepare model input and predict
# --------------------------------------------------

# Use exactly the same feature order as training.
X = np.column_stack([
    arrays[name][valid]
    for name in features
]).astype(np.float32)

print("Prediction matrix shape:", X.shape)

X_df = pd.DataFrame(X, columns=features)
predicted_names = model.predict(X_df)

# Convert predicted class names to integer class IDs.
name_to_id = {
    str(name): int(class_id)
    for name, class_id in class_mapping.items()
}

predicted_ids = np.array(
    [name_to_id[str(name)] for name in predicted_names],
    dtype=np.uint8
)


# --------------------------------------------------
# 7. Create the classification raster
# --------------------------------------------------

# 0 means nodata. Class IDs 1–5 represent land-cover classes.
classification = np.zeros(
    (height, width),
    dtype=np.uint8
)

classification[valid] = predicted_ids

output_profile = reference_profile.copy()
output_profile.update(
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="uint8",
    nodata=0,
    compress="deflate",
)

with rasterio.open(
    OUTPUT_PATH,
    "w",
    **output_profile
) as destination:
    destination.write(classification, 1)

    destination.set_band_description(
        1,
        "Random Forest land-cover class"
    )


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\nClassification complete.")
print("Saved raster:", OUTPUT_PATH)

print("\nPredicted pixel counts:")
for class_name, class_id in name_to_id.items():
    count = int(np.count_nonzero(classification == class_id))
    print(f"  {class_name} (ID {class_id}): {count:,}")

print("\nClass ID mapping:")
for class_name, class_id in name_to_id.items():
    print(f"  {class_id} = {class_name}")

print("  0 = NoData")

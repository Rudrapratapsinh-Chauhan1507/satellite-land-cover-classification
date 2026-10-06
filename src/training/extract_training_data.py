import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared project paths.
from src.utils.paths import (
    TRAINING_SAMPLES_FILE,
    PROCESSED_DATA_DIR,
    TRAINING_INDICES_DIR,
    OUTPUTS_DIR,
)

# Input and output directories.
TRAINING_FILE = TRAINING_SAMPLES_FILE
RASTER_DIR = PROCESSED_DATA_DIR
INDEX_DIR = TRAINING_INDICES_DIR
OUTPUT_DIR = OUTPUTS_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Input rasters
# --------------------------------------------------

band_files = {
    "B02": RASTER_DIR / "T42QZL_20261003T053651_B02_10m_clipped.tif",
    "B03": RASTER_DIR / "T42QZL_20261003T053651_B03_10m_clipped.tif",
    "B04": RASTER_DIR / "T42QZL_20261003T053651_B04_10m_clipped.tif",
    "B05": RASTER_DIR / "T42QZL_20261003T053651_B05_20m_clipped.tif",
    "B06": RASTER_DIR / "T42QZL_20261003T053651_B06_20m_clipped.tif",
    "B07": RASTER_DIR / "T42QZL_20261003T053651_B07_20m_clipped.tif",
    "B08": RASTER_DIR / "T42QZL_20261003T053651_B08_10m_clipped.tif",
    "B8A": RASTER_DIR / "T42QZL_20261003T053651_B8A_20m_clipped.tif",
    "B11": RASTER_DIR / "T42QZL_20261003T053651_B11_20m_clipped.tif",
    "B12": RASTER_DIR / "T42QZL_20261003T053651_B12_20m_clipped.tif",
}

index_files = {
    "NDVI": INDEX_DIR / "ndvi.tif",
    "NDWI": INDEX_DIR / "ndwi.tif",
    "NDBI": INDEX_DIR / "ndbi.tif",
}


# --------------------------------------------------
# 3. Load training points
# --------------------------------------------------

print("Loading training points...")

if not TRAINING_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAINING_FILE}"
    )

points = gpd.read_file(TRAINING_FILE, layer="training_samples")

required_columns = {"class_name", "class_id", "geometry"}

if not required_columns.issubset(points.columns):
    raise ValueError(
        f"Training layer must contain: {required_columns}"
    )

if points.empty:
    raise ValueError("The training layer contains no features.")

if points.crs is None:
    raise ValueError("The training points have no CRS.")

if points.geometry.isna().any() or points.geometry.is_empty.any():
    raise ValueError("Some training points have empty geometry.")

if not points.geometry.geom_type.eq("Point").all():
    raise ValueError("Every training feature must be a Point.")

points = points.reset_index(drop=True)

points["class_name"] = points["class_name"].astype(str).str.strip()
points["class_id"] = pd.to_numeric(
    points["class_id"], errors="raise"
).astype(int)

expected_classes = {
    "vegetation": 1,
    "built_up": 2,
    "bare_soil": 3,
    "water": 4,
    "road": 5,
}

# Normalize case and detect spelling mistakes.
points["class_name"] = points["class_name"].str.lower()

invalid_classes = points.loc[
    ~points["class_name"].isin(expected_classes)
    | points.apply(
        lambda row: expected_classes.get(row["class_name"])
        != row["class_id"],
        axis=1,
    ),
    ["class_name", "class_id"],
]

if not invalid_classes.empty:
    raise ValueError(
        "Fix these class labels/IDs in QGIS before continuing:\n"
        + invalid_classes.to_string(index=False)
    )

print(f"Training points loaded: {len(points)}")
print("\nSamples per class:")
print(points["class_name"].value_counts())


# --------------------------------------------------
# 4. Sample each raster at every point
# --------------------------------------------------

features = pd.DataFrame(index=points.index)
invalid_rows = set()


def extract_raster_values(raster_path, feature_name):
    """Sample one raster, transforming points to its CRS."""

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    values = np.full(len(points), np.nan, dtype="float64")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        # Each raster can have a different CRS or resolution.
        raster_points = points.to_crs(src.crs)
        coordinates = [
            (geom.x, geom.y)
            for geom in raster_points.geometry
        ]

        samples = src.sample(
            coordinates,
            indexes=1,
            masked=True,
        )

        for i, sample in enumerate(samples):
            value = sample[0]

            if np.ma.is_masked(value):
                invalid_rows.add(i)
                continue

            value = float(value)

            if not np.isfinite(value):
                invalid_rows.add(i)
                continue

            if src.nodata is not None and np.isclose(
                value, src.nodata
            ):
                invalid_rows.add(i)
                continue

            # Clipped reflectance bands use zero outside the AOI.
            # Zero is not a valid sampled reflectance here.
            if feature_name in band_files and value <= 0:
                invalid_rows.add(i)
                continue

            # Spectral indices should be between -1 and +1.
            if feature_name in index_files and not -1.001 <= value <= 1.001:
                invalid_rows.add(i)
                continue

            values[i] = value

    return values


print("\nExtracting Sentinel-2 bands...")

for name, path in band_files.items():
    print(f"  Reading {name}")
    features[name] = extract_raster_values(path, name)

print("\nExtracting spectral indices...")

for name, path in index_files.items():
    print(f"  Reading {name}")
    features[name] = extract_raster_values(path, name)


# --------------------------------------------------
# 5. Build the machine-learning dataset
# --------------------------------------------------

dataset = features.copy()
dataset["class_name"] = points["class_name"]
dataset["class_id"] = points["class_id"]

feature_columns = list(band_files) + list(index_files)

# Report missing features before removing any samples.
missing_by_feature = dataset[feature_columns].isna().sum()

print("\nMissing values per feature:")
print(missing_by_feature[missing_by_feature > 0])

# Identify which samples would be removed.
valid_rows = dataset[feature_columns].notna().all(axis=1)
removed_count = int((~valid_rows).sum())

if removed_count:
    print("\nSamples with missing/invalid predictors:")
    print(
        dataset.loc[
            ~valid_rows, ["class_name", "class_id"]
        ].assign(
            sample_id=dataset.index[~valid_rows] + 1
        ).to_string(index=False)
    )

dataset = dataset.loc[valid_rows].copy()

dataset.insert(
    0,
    "sample_id",
    dataset.index + 1,
)

output_file = OUTPUTS_DIR / "training_dataset.csv"
dataset.to_csv(output_file, index=False)


# --------------------------------------------------
# 6. Report results
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRAINING DATA EXTRACTION COMPLETE")
print("=" * 50)

print(f"Original points: {len(points)}")
print(f"Rows removed: {removed_count}")
print(f"Valid training samples: {len(dataset)}")
print(f"Predictor features: {len(feature_columns)}")
print(f"Output file: {output_file}")

print("\nFinal samples per class:")
print(dataset["class_name"].value_counts())

print("\nDataset preview:")
print(dataset.head())

if dataset.empty:
    raise ValueError(
        "No valid samples remain. Check raster coverage, CRS, "
        "NoData values, and index masks."
    )

if dataset["class_id"].nunique() < 2:
    raise ValueError(
        "At least two classes need valid samples for classification."
    )

print("\nNext step: inspect class balance and prepare model training.")

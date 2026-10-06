from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "dynamic_world_candidates.gpkg"
)

EXTRA_TREES_RASTER = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "land_cover_extra_trees.tif"
)

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

OUTPUT_FILE = (
    REPORTS_DIR / "dynamic_world_candidate_predictions.csv"
)


# --------------------------------------------------
# Existing project class mapping
# --------------------------------------------------

PROJECT_CLASSES = {
    1: "vegetation",
    2: "built_up",
    3: "bare_soil",
    4: "water",
    5: "road",
}


def main():
    print("Loading Dynamic World candidate points...")

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {CANDIDATES_FILE}"
        )

    if not EXTRA_TREES_RASTER.exists():
        raise FileNotFoundError(
            f"Extra Trees raster not found: {EXTRA_TREES_RASTER}"
        )

    points = gpd.read_file(CANDIDATES_FILE)

    required_columns = {
        "class_id",
        "class_name",
        "longitude",
        "latitude",
    }

    missing = required_columns - set(points.columns)

    if missing:
        raise ValueError(
            f"Missing candidate fields: {sorted(missing)}"
        )

    print(f"Candidate points loaded: {len(points)}")
    print("Original Dynamic World labels preserved.")

    with rasterio.open(EXTRA_TREES_RASTER) as src:
        print("\nReading Extra Trees classification raster...")
        print(f"Raster CRS: {src.crs}")
        print(f"Raster size: {src.width} x {src.height}")

        # Reproject point geometries to the raster CRS.
        points_raster_crs = points.to_crs(src.crs)

        predicted_ids = []
        prediction_status = []

        for point in points_raster_crs.geometry:
            if point is None or point.is_empty:
                predicted_ids.append(None)
                prediction_status.append("invalid_geometry")
                continue

            row, col = rowcol(
                src.transform,
                point.x,
                point.y,
            )

            if not (
                0 <= row < src.height
                and 0 <= col < src.width
            ):
                predicted_ids.append(None)
                prediction_status.append("outside_raster")
                continue

            value = src.read(1)[row, col]

            if (
                src.nodata is not None
                and np.isclose(value, src.nodata)
            ):
                predicted_ids.append(None)
                prediction_status.append("nodata")
                continue

            predicted_ids.append(int(value))
            prediction_status.append("predicted")

        points["extra_trees_class_id"] = pd.array(
            predicted_ids,
            dtype="Int64",
        )

        points["extra_trees_class_name"] = (
            points["extra_trees_class_id"]
            .map(PROJECT_CLASSES)
        )

        points["prediction_status"] = prediction_status

    # Keep this as a point-level inspection dataset.
    # Dynamic World and project labels are different schemes.
    output_columns = [
        "class_id",
        "class_name",
        "longitude",
        "latitude",
        "extra_trees_class_id",
        "extra_trees_class_name",
        "prediction_status",
    ]

    result = points[output_columns].copy()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Prediction extraction summary ---")
    print(f"Total candidate points: {len(result)}")
    print("\nPrediction status:")
    print(result["prediction_status"].value_counts().to_string())

    print("\nOriginal Dynamic World class counts:")
    print(result["class_name"].value_counts().to_string())

    print("\nExtracted Extra Trees class counts:")
    print(
        result["extra_trees_class_name"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(f"\nSaved separate report:\n{OUTPUT_FILE}")

    print(
        "\nImportant: this report extracts predictions only. "
        "It does not calculate accuracy because Dynamic World "
        "labels do not directly match the project's five classes."
    )


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "dynamic_world_candidates_cleaned.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "dynamic_world_candidates.gpkg"
)

LAYER_NAME = "dynamic_world_candidates"


def main():
    print("Loading cleaned candidate samples...")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "class_id",
        "class_name",
        "longitude",
        "latitude",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs="EPSG:4326",
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Remove only this separate output if rerunning the script.
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    points.to_file(
        OUTPUT_FILE,
        layer=LAYER_NAME,
        driver="GPKG",
    )

    print(f"Exported points: {len(points)}")
    print(f"CRS: {points.crs}")
    print(f"Saved: {OUTPUT_FILE}")
    print("\nClass counts:")
    print(points["class_name"].value_counts().to_string())


if __name__ == "__main__":
    main()

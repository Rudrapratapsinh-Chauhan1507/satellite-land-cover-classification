
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    Path.home()
    / "Downloads"
    / "DynamicWorld_candidate_samples.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
OUTPUT_FILE = OUTPUT_DIR / "dynamic_world_candidates_cleaned.csv"

# Study-area bounds from the existing project, in longitude/latitude.
MIN_LON = 72.55744464
MIN_LAT = 23.15596316
MAX_LON = 72.60813514
MAX_LAT = 23.18032797

# Preserve original Dynamic World labels; do not convert them
# to the project's five classes at this stage.
DYNAMIC_WORLD_CLASSES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Shrub and scrub",
    6: "Built",
    7: "Bare",
    8: "Snow and ice",
}


def extract_coordinates(geo_value):
    """Extract longitude and latitude from Earth Engine .geo JSON."""
    try:
        geometry = json.loads(geo_value)
        coordinates = geometry["coordinates"]

        if geometry.get("type") != "Point" or len(coordinates) < 2:
            return None, None

        lon = float(coordinates[0])
        lat = float(coordinates[1])

        return lon, lat
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None, None


def main():
    print("Loading Dynamic World candidate samples...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"CSV not found: {INPUT_FILE}\n"
            "Check the download location and update INPUT_FILE if needed."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"class_id", ".geo"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    coordinates = df[".geo"].apply(extract_coordinates)
    df["longitude"] = coordinates.apply(lambda value: value[0])
    df["latitude"] = coordinates.apply(lambda value: value[1])

    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")
    df["class_name"] = df["class_id"].map(DYNAMIC_WORLD_CLASSES)

    df["coordinate_valid"] = (
        df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
    )

    df["inside_study_area"] = (
        df["longitude"].between(MIN_LON, MAX_LON)
        & df["latitude"].between(MIN_LAT, MAX_LAT)
    )

    print(f"Input rows: {len(df)}")
    print(f"Missing/invalid coordinates: {(~df['coordinate_valid']).sum()}")
    print(f"Inside study area: {df['inside_study_area'].sum()}")
    print("\nDynamic World class counts:")
    print(df["class_name"].value_counts(dropna=False).to_string())

    # Keep valid points inside the existing study area.
    cleaned = df.loc[
        df["coordinate_valid"]
        & df["inside_study_area"]
        & df["class_name"].notna(),
        [
            "system:index",
            "class_id",
            "class_name",
            "longitude",
            "latitude",
        ],
    ].copy()

    cleaned["class_id"] = cleaned["class_id"].astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_FILE, index=False)

    print(f"\nClean candidate rows: {len(cleaned)}")
    print(f"Saved separate candidate file:\n{OUTPUT_FILE}")
    print(
        "\nImportant: these labels are preserved as Dynamic World classes. "
        "They have NOT been mapped to the project's five target classes."
    )


if __name__ == "__main__":
    main()


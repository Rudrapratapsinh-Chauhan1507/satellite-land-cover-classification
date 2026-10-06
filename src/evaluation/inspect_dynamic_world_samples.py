
import json
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "DynamicWorld_candidate_samples.csv"
)


# Dynamic World class IDs
DYNAMIC_WORLD_CLASSES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Shrub and scrub",
    6: "Built",
    7: "Bare ground",
    8: "Snow and ice",
}


def extract_coordinates(geo_value):
    """Extract longitude and latitude from the .geo column."""
    try:
        geo = json.loads(geo_value)

        # Handle JSON stored with doubled quotation marks.
        if isinstance(geo, str):
            geo = json.loads(geo)

        coordinates = geo["coordinates"]

        return pd.Series({
            "longitude": coordinates[0],
            "latitude": coordinates[1],
        })

    except (TypeError, ValueError, KeyError, IndexError):
        return pd.Series({
            "longitude": None,
            "latitude": None,
        })


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_FILE}\n"
            "Check that the file is inside data/reference/."
        )

    print("Loading Dynamic World candidate samples...")
    df = pd.read_csv(CSV_FILE)

    print("\n--- Original dataset ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    # Validate expected columns.
    required_columns = {"class_id", ".geo"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: "
            f"{required_columns - set(df.columns)}"
        )

    # Convert class IDs to numeric values.
    df["class_id"] = pd.to_numeric(
        df["class_id"], errors="coerce"
    )

    # Decode geographic coordinates.
    coordinates = df[".geo"].apply(extract_coordinates)
    df = pd.concat([df, coordinates], axis=1)

    # Add readable Dynamic World class names.
    df["class_name"] = df["class_id"].map(
        DYNAMIC_WORLD_CLASSES
    )

    print("\n--- Class distribution ---")
    print(
        df["class_id"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\n--- Class names and counts ---")
    print(
        df.groupby(
            ["class_id", "class_name"],
            dropna=False
        ).size().to_string()
    )

    print("\n--- Coordinate validation ---")
    print(
        "Missing coordinates:",
        df[["longitude", "latitude"]].isna().any(axis=1).sum()
    )

    valid_coordinates = (
        df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
    )

    print(
        "Valid geographic coordinates:",
        int(valid_coordinates.sum()),
        "of",
        len(df),
    )

    print("\n--- Geographic extent ---")
    print(f"Longitude range: {df['longitude'].min()} to "
          f"{df['longitude'].max()}")
    print(f"Latitude range: {df['latitude'].min()} to "
          f"{df['latitude'].max()}")

    # Save an inspection copy; original CSV remains unchanged.
    output_dir = PROJECT_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "dynamic_world_samples_inspected.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nInspection file saved to:")
    print(output_file)


if __name__ == "__main__":
    main()

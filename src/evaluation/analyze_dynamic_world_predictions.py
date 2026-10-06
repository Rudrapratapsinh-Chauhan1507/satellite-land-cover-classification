from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "dynamic_world_candidate_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "dynamic_world_prediction_crosstab.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Exclude points without a valid raster prediction.
    valid = df[
        (df["prediction_status"] == "predicted")
        & df["extra_trees_class_name"].notna()
    ].copy()

    print(f"All candidate points: {len(df)}")
    print(f"Points with valid predictions: {len(valid)}")

    table = pd.crosstab(
        valid["class_name"],
        valid["extra_trees_class_name"],
        margins=True,
    )

    print("\nDynamic World labels vs Extra Trees predictions:")
    print(table.to_string())

    # Row percentages describe prediction distributions,
    # not classification accuracy.
    row_percentages = (
        pd.crosstab(
            valid["class_name"],
            valid["extra_trees_class_name"],
            normalize="index",
        ) * 100
    ).round(2)

    print("\nPrediction percentages within each Dynamic World class:")
    print(row_percentages.to_string())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE)

    print(f"\nSaved crosstab: {OUTPUT_FILE}")
    print(
        "\nNote: these are descriptive comparisons, not accuracy "
        "metrics, because the class definitions differ."
    )


if __name__ == "__main__":
    main()
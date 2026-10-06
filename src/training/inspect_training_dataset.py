import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.paths import TRAINING_DATASET_FILE

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
print("Loading training dataset...")

df = pd.read_csv(TRAINING_DATASET_FILE)

feature_columns = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "NDBI",
]

print("\nDataset shape:", df.shape)
print("Predictor features:", len(feature_columns))
print("Total samples:", len(df))

# --------------------------------------------------
# 2. Class balance
# --------------------------------------------------
print("\nSamples per class:")
print(df["class_name"].value_counts())

print("\nClass percentages:")
print(
    (df["class_name"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str)
    .add("%")
)

# --------------------------------------------------
# 3. Data quality
# --------------------------------------------------
print("\nMissing values per column:")
print(df.isna().sum())

print("\nDuplicate complete rows:", df.duplicated().sum())

print(
    "Duplicate predictor vectors:",
    df.duplicated(subset=feature_columns).sum()
)

print("\nNon-finite predictor values:")
print(
    (~df[feature_columns].apply(
        lambda column: pd.to_numeric(
            column, errors="coerce"
        ).map(lambda value: pd.notna(value) and abs(value) != float("inf"))
    )).sum()
)

# --------------------------------------------------
# 4. Feature ranges
# --------------------------------------------------
print("\nFeature summary:")
print(df[feature_columns].describe().T.round(4))

# --------------------------------------------------
# 5. Cross-validation readiness
# --------------------------------------------------
smallest_class = int(df["class_name"].value_counts().min())

print("\nSmallest class size:", smallest_class)

if smallest_class >= 3:
    print("3-fold stratified cross-validation is possible.")
else:
    print(
        "WARNING: Some classes have fewer than 3 samples. "
        "Collect more samples before using 3-fold CV."
    )

print("\nInspection complete.")
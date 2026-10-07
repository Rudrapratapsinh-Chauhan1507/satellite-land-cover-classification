
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from src.utils.paths import TRAINING_SAMPLES_FILE, TRAINING_DATASET_FILE


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

N_SPATIAL_GROUPS = 2
RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B8A",
    "B11", "B12",
    "NDVI", "NDWI", "NDBI",
]


# --------------------------------------------------
# 2. Load and validate training data
# --------------------------------------------------

data = pd.read_csv(TRAINING_DATASET_FILE)

points = gpd.read_file(
    TRAINING_SAMPLES_FILE,
    layer="training_samples",
)

print("Training dataset shape:", data.shape)
print("Training points shape:", points.shape)
print("Training dataset classes:")
print(data["class_name"].value_counts())
print()

if len(data) != len(points):
    raise ValueError(
        "CSV and GeoPackage row counts differ. "
        "Cannot safely match features to point coordinates."
    )

if not data["class_name"].reset_index(drop=True).equals(
    points["class_name"].reset_index(drop=True)
):
    raise ValueError(
        "CSV and GeoPackage class order differs. "
        "Do not continue until the rows are matched correctly."
    )

missing_columns = [
    column for column in FEATURE_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing feature columns: {missing_columns}")

X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
y = data["class_name"].astype(str)

if X.isna().any().any():
    raise ValueError(
        "Predictor data contains missing or non-numeric values."
    )

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Predictor data contains infinite values.")

if points.geometry.is_empty.any() or points.geometry.isna().any():
    raise ValueError("Some training points have missing geometry.")


# --------------------------------------------------
# 3. Convert coordinates to metres
# --------------------------------------------------

if points.crs is None:
    raise ValueError("Training points have no CRS defined.")

projected_points = points.to_crs("EPSG:32642")

coordinates = np.column_stack([
    projected_points.geometry.x,
    projected_points.geometry.y,
])


# --------------------------------------------------
# 4. Create geographic groups
# --------------------------------------------------

number_of_groups = min(
    N_SPATIAL_GROUPS,
    len(points),
)

clusterer = KMeans(
    n_clusters=number_of_groups,
    random_state=RANDOM_STATE,
    n_init=20,
)

groups = clusterer.fit_predict(coordinates)

print("Points per spatial group:")
print(pd.Series(groups).value_counts().sort_index())
print()

print("Class distribution by spatial group:")
print(pd.crosstab(groups, y))
print()

group_class_counts = pd.crosstab(groups, y)
missing_class_groups = (
    group_class_counts.reindex(
        columns=sorted(y.unique()),
        fill_value=0,
    ) == 0
)

if missing_class_groups.any().any():
    print(
        "WARNING: Some spatial groups do not contain every class. "
        "Some folds may lack training or validation classes."
    )
    print()


# --------------------------------------------------
# 5. Train and evaluate held-out spatial groups
# --------------------------------------------------

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

splitter = GroupKFold(n_splits=number_of_groups)

all_true = []
all_predicted = []
fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y_encoded, groups=groups),
    start=1,
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    train_classes = set(y_train)
    test_classes = set(y_test)

    missing_from_training = (
        set(range(len(encoder.classes_))) - train_classes
    )

    # Record skipped folds instead of silently omitting them.
    if missing_from_training:
        missing_names = encoder.inverse_transform(
            sorted(missing_from_training)
        ).tolist()

        fold_results.append({
            "fold": fold,
            "training_points": len(train_idx),
            "validation_points": len(test_idx),
            "validation_classes": len(test_classes),
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "status": "skipped_missing_training_class",
            "missing_training_classes": ", ".join(missing_names),
        })

        print(
            f"Fold {fold} SKIPPED: training data lacks "
            f"these classes: {missing_names}"
        )
        print()
        continue

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        average="macro",
        zero_division=0,
    )

    all_true.extend(y_test.tolist())
    all_predicted.extend(predictions.tolist())

    fold_results.append({
        "fold": fold,
        "training_points": len(train_idx),
        "validation_points": len(test_idx),
        "validation_classes": len(test_classes),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "status": "evaluated",
        "missing_training_classes": "",
    })

    print(f"Fold {fold} EVALUATED")
    print(f"  Training points:   {len(train_idx)}")
    print(f"  Validation points: {len(test_idx)}")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")

    if len(test_classes) < len(encoder.classes_):
        absent = (
            set(range(len(encoder.classes_))) - test_classes
        )

        absent_names = encoder.inverse_transform(
            sorted(absent)
        ).tolist()

        print(f"  Note: validation lacks classes {absent_names}")

    print()


# --------------------------------------------------
# 6. Save complete fold results and summarize
# --------------------------------------------------

if not fold_results:
    raise RuntimeError(
        "No folds were recorded. Check the spatial grouping."
    )

results = pd.DataFrame(fold_results)

results_path = OUTPUT_DIR / "spatial_cv_2_group_comparison.csv"
results.to_csv(results_path, index=False)

evaluated = results[results["status"] == "evaluated"]
skipped = results[results["status"] != "evaluated"]

print("=" * 60)
print("COMPLETE SPATIAL CROSS-VALIDATION SUMMARY")
print("=" * 60)

print(results.to_string(index=False))
print()

print(f"Folds attempted: {len(results)}")
print(f"Folds evaluated: {len(evaluated)}")
print(f"Folds skipped:   {len(skipped)}")
print()

if not evaluated.empty:
    print(
        f"Mean evaluated-fold accuracy: "
        f"{evaluated['accuracy'].mean():.4f}"
    )

    print(
        f"Mean evaluated-fold macro F1: "
        f"{evaluated['macro_f1'].mean():.4f}"
    )
else:
    print("No valid folds were evaluated.")

# --------------------------------------------------
# 7. Pooled metrics for evaluated folds only
# --------------------------------------------------

if all_true:
    print("\nPooled held-out confusion matrix:")

    print(
        pd.DataFrame(
            confusion_matrix(
                all_true,
                all_predicted,
                labels=np.arange(len(encoder.classes_)),
            ),
            index=encoder.classes_,
            columns=encoder.classes_,
        )
    )

    print("\nPooled held-out classification report:")

    from sklearn.metrics import classification_report

    print(
        classification_report(
            all_true,
            all_predicted,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

print(f"\nSaved complete fold metrics to: {results_path}")

print(
    "\nImportant: these estimates are exploratory. "
    "There are only 39 points, and spatial groups may have "
    "uneven class distributions. Some folds may be skipped "
    "when a training class is absent. Do not interpret these "
    "scores as definitive map accuracy."
)

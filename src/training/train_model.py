
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import (
    TRAINING_DATASET_FILE,
    MODELS_DIR,
    MODEL_FILE,
)

CSV_PATH = TRAINING_DATASET_FILE
MODEL_DIR = MODELS_DIR
MODEL_PATH = MODEL_FILE

MODEL_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "B02", "B03", "B04",
    "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "NDBI",
]

CLASS_NAMES = [
    "vegetation",
    "built_up",
    "bare_soil",
    "water",
    "road",
]


def main():
    print("=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)

    # Validate expected columns and data.
    required = FEATURES + ["class_name", "class_id"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Missing values found. Fix them before training.")

    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Infinite or invalid feature values found.")

    # Verify class-name / class-ID mapping.
    expected_mapping = {
        "vegetation": 1,
        "built_up": 2,
        "bare_soil": 3,
        "water": 4,
        "road": 5,
    }

    for name, expected_id in expected_mapping.items():
        observed = df.loc[
            df["class_name"] == name, "class_id"
        ].unique()

        if len(observed) != 1 or observed[0] != expected_id:
            raise ValueError(
                f"Class mapping mismatch for {name}: {observed}"
            )

    X = df[FEATURES].astype(float)
    y = df["class_name"].astype(str)

    print(f"\nSamples: {len(df)}")
    print(f"Predictor features: {len(FEATURES)}")
    print("\nSamples per class:")
    print(y.value_counts().reindex(CLASS_NAMES, fill_value=0))
    
    class_counts = y.value_counts()

    missing_classes = [
        name for name in CLASS_NAMES
        if class_counts.get(name, 0) == 0
    ]

    if missing_classes:
        raise ValueError(
            f"Expected classes missing from dataset: {missing_classes}"
        )

    if class_counts.min() < 3:
        raise ValueError(
            "At least three samples per class are needed for 3-fold CV."
        )

    # Each fold retains samples from all five classes.
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Out-of-fold predictions: each sample is predicted by a model
    # that was not trained on that sample.
    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
    )

    print("\n========== CROSS-VALIDATION RESULTS ==========")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(y, predictions, average='macro', zero_division=0):.4f}"
    )

    print("\nPer-class report:")
    print(
        classification_report(
            y,
            predictions,
            labels=CLASS_NAMES,
            zero_division=0,
            digits=3,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(CLASS_NAMES)
    print(confusion_matrix(y, predictions, labels=CLASS_NAMES))

    # Fit a final model using all available samples.
    # This model is for the next mapping stage, not independent evaluation.
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "class_names": CLASS_NAMES,
            "class_mapping": expected_mapping,
        },
        MODEL_PATH,
    )

    print("\n========== MODEL SAVED ==========")
    print(MODEL_PATH)
    print("\nBaseline training complete.")
    print(
        "Important: results are preliminary because the dataset "
        "contains only 39 samples."
    )


if __name__ == "__main__":
    main()

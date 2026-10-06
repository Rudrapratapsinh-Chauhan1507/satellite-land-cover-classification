
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    TRAINING_DATASET_FILE,
    MODEL_FILE,
    REPORTS_DIR,
)

FEATURES = [
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "NDBI",
]

TARGET = "class_id"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check required files
    if not TRAINING_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_DATASET_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    # 2. Load and validate the training dataset
    data = pd.read_csv(TRAINING_DATASET_FILE)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[required].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = data[FEATURES]
    y_true = data[TARGET].astype(int)

    # 3. Load the saved model
    saved_object = joblib.load(MODEL_FILE)

    model = saved_object["model"]
    model_features = saved_object["features"]
    class_mapping = saved_object["class_mapping"]

    # 4. Verify feature order
    if list(X.columns) != model_features:
        raise ValueError(
            f"Feature mismatch.\nExpected: {model_features}\n"
            f"Received: {list(X.columns)}"
        )

    # 5. Predict class names and convert them to numeric class IDs
    y_pred_names = model.predict(X)

    y_pred = [class_mapping[name] for name in y_pred_names]

    # 6. Establish a consistent numeric class order
    class_ids = sorted(class_mapping.values())

    id_to_name = {
        class_id: name
        for name, class_id in class_mapping.items()
    }

    class_names = [id_to_name[class_id] for class_id in class_ids]

    # 7. Calculate diagnostic metrics
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    # 8. Save the diagnostic report
    output_file = REPORTS_DIR / "training_set_diagnostic.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("TRAINING-SET DIAGNOSTIC\n")
        file.write("=" * 50 + "\n")
        file.write(
            "Warning: These are training-set results, "
            "not an independent accuracy assessment.\n\n"
        )

        file.write(f"Number of samples: {len(data)}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")

        file.write("Class ID order (ID: name):\n")
        file.write(str(list(zip(class_ids, class_names))))

        file.write("\n\nClassification report:\n")
        file.write(report)

        file.write("\nConfusion matrix:\n")
        file.write(
            "Rows = actual classes; columns = predicted classes.\n"
        )
        file.write(str(matrix))
        file.write("\n")

    # 9. Display results
    print("TRAINING-SET DIAGNOSTIC")
    print("=" * 50)
    print(f"Samples evaluated: {len(data)}")
    print(f"Training-set accuracy: {accuracy:.4f}")
    print(f"Training-set macro F1: {macro_f1:.4f}")

    print("\nClass ID order (ID: name):")
    print(list(zip(class_ids, class_names)))

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()

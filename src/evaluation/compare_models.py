
from pathlib import Path
import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import TRAINING_DATASET_FILE, REPORTS_DIR

FEATURES = [
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12", "NDVI", "NDWI", "NDBI",
]


def main():
    df = pd.read_csv(TRAINING_DATASET_FILE)

    X = df[FEATURES].astype(float)
    y = df["class_name"].astype(str)

    print("Samples:", len(df))
    print("\nClass counts:")
    print(y.value_counts())

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
            },
            n_jobs=1,
        )

        row = {
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "macro_f1_mean": scores["test_macro_f1"].mean(),
            "macro_f1_std": scores["test_macro_f1"].std(),
        }
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        "macro_f1_mean", ascending=False
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "model_comparison.csv"
    results.to_csv(output, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    ))
    print("\nSaved to:", output)
    print(
        "\nCaution: only 39 labeled samples are available. "
        "These results are preliminary and do not establish "
        "accuracy across unseen geographic areas."
    )


if __name__ == "__main__":
    main()

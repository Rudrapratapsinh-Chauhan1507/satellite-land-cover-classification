
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import TRAINING_DATASET_FILE, REPORTS_DIR

FEATURES = [
    "B02", "B03", "B04",
    "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "NDBI",
]


def main():
    df = pd.read_csv(TRAINING_DATASET_FILE)

    X = df[FEATURES].astype(float)
    y = df["class_name"].astype(str)

    if len(df) != 39:
        print(f"Note: found {len(df)} samples instead of the expected 39.")

    print("Samples:", len(df))
    print("\nClass counts:")
    print(y.value_counts())

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

    # Evaluate the existing baseline configuration.
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="f1_macro"
    )
    print("\n3-fold macro F1 scores:", scores.round(3))
    print(f"Mean macro F1: {scores.mean():.3f}")
    print(f"Standard deviation: {scores.std():.3f}")

    # Fit on all samples for exploratory feature analysis.
    # These importances are descriptive, not independent validation.
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = REPORTS_DIR / "feature_importance.csv"
    output_png = REPORTS_DIR / "feature_importance.png"

    importance.to_csv(output_csv, index=False)

    print("\nFeature importance:")
    print(importance.to_string(index=False, float_format="%.4f"))

    plt.figure(figsize=(9, 6))
    ordered = importance.sort_values("importance")
    plt.barh(ordered["feature"], ordered["importance"])
    plt.xlabel("Random Forest impurity-based importance")
    plt.title("Feature Importance — Exploratory Analysis")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    # Permutation importance measured on the same training data is
    # exploratory and can be optimistic; do not treat it as test accuracy.
    print("\nSaved outputs:")
    print(output_csv)
    print(output_png)
    print("\nInterpretation caution:")
    print(
        "Correlated spectral bands and indices can share importance. "
        "Importance does not prove causation or independent generalization."
    )


if __name__ == "__main__":
    main()


from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import CLASSIFICATION_FILE

EXTRA_TREES_RASTER = (
    ROOT / "outputs/classification/land_cover_extra_trees.tif"
)
REPORT_DIR = ROOT / "outputs/reports"
REPORT_CSV = REPORT_DIR / "classification_map_comparison.csv"

CLASS_NAMES = {
    1: "vegetation",
    2: "built_up",
    3: "bare_soil",
    4: "water",
    5: "road",
}


def main():
    if not CLASSIFICATION_FILE.exists():
        raise FileNotFoundError(
            f"Random Forest raster not found: {CLASSIFICATION_FILE}"
        )

    if not EXTRA_TREES_RASTER.exists():
        raise FileNotFoundError(
            f"Extra Trees raster not found: {EXTRA_TREES_RASTER}"
        )

    with rasterio.open(CLASSIFICATION_FILE) as rf_src:
        rf = rf_src.read(1)
        rf_profile = rf_src.profile.copy()
        rf_transform = rf_src.transform
        rf_crs = rf_src.crs
        rf_nodata = rf_src.nodata

    with rasterio.open(EXTRA_TREES_RASTER) as et_src:
        et = et_src.read(1)

        if (
            et_src.width != rf_profile["width"]
            or et_src.height != rf_profile["height"]
            or et_src.transform != rf_transform
            or et_src.crs != rf_crs
        ):
            raise ValueError(
                "The Random Forest and Extra Trees rasters do not "
                "share the same grid."
            )

        et_nodata = et_src.nodata

    # Exclude NoData pixels and unexpected class IDs.
    valid = np.isin(rf, list(CLASS_NAMES)) & np.isin(
        et, list(CLASS_NAMES)
    )

    if rf_nodata is not None:
        valid &= rf != rf_nodata

    if et_nodata is not None:
        valid &= et != et_nodata

    if not valid.any():
        raise ValueError("No overlapping valid pixels were found.")

    rf_valid = rf[valid]
    et_valid = et[valid]

    agreement = float(np.mean(rf_valid == et_valid))

    print("Valid overlapping pixels:", int(valid.sum()))
    print(f"Pixel agreement: {agreement:.4%}")
    print(f"Pixel disagreement: {1 - agreement:.4%}")

    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        rf_count = int(np.count_nonzero(rf_valid == class_id))
        et_count = int(np.count_nonzero(et_valid == class_id))

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "random_forest_pixels": rf_count,
            "extra_trees_pixels": et_count,
            "difference_pixels": et_count - rf_count,
        })

    results = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_CSV, index=False)

    print("\nClass pixel comparison:")
    print(results.to_string(index=False))
    print("\nSaved report:", REPORT_CSV)
    print(
        "\nNote: agreement measures similarity between the two models, "
        "not classification accuracy."
    )


if __name__ == "__main__":
    main()

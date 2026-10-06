from pathlib import Path
import sys

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR

path = PROCESSED_DATA_DIR / (
    "T42QZL_20261003T053651_SCL_20m_clipped.tif"
)

class_names = {
    0: "No data",
    1: "Saturated/defective",
    2: "Dark area pixels",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Bare soil",
    6: "Water",
    7: "Unclassified",
    8: "Clouds: medium probability",
    9: "Clouds: high probability",
    10: "Thin cirrus",
    11: "Snow/ice",
}

with rasterio.open(path) as src:
    scl = src.read(1)
    valid = scl != src.nodata
    values, counts = np.unique(scl[valid], return_counts=True)

    total = counts.sum()
    print("SCL CLASS DISTRIBUTION")
    print("=" * 45)

    for value, count in zip(values, counts):
        name = class_names.get(int(value), "Unknown")
        percentage = count / total * 100
        print(
            f"Class {value}: {name:<27} "
            f"{count:>6} pixels ({percentage:.2f}%)"
        )

    print(f"\nTotal valid SCL pixels: {total}")
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    TRAINING_SAMPLES_FILE,
    PROCESSED_DATA_DIR,
)

PREFIX = "T42QZL_20261003T053651_"


def read_point_values(filename, points):
    path = PROCESSED_DATA_DIR / filename

    with rasterio.open(path) as src:
        projected_points = points.to_crs(src.crs)
        values = []

        for point in projected_points.geometry:
            value = next(
                src.sample([(point.x, point.y)], masked=True)
            )[0]

            if np.ma.is_masked(value):
                values.append(np.nan)
            else:
                values.append(float(value))

        return np.array(values)


def normalized_difference(a, b):
    denominator = a + b
    result = np.full(a.shape, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (denominator != 0)
    )

    result[valid] = (
        (a[valid] - b[valid]) / denominator[valid]
    )

    return result


def main():
    samples = gpd.read_file(TRAINING_SAMPLES_FILE)

    water = samples[
        samples["class_name"].astype(str).str.lower() == "water"
    ].copy()

    if water.empty:
        print("No water training points found.")
        return

    blue = read_point_values(
        f"{PREFIX}B02_10m_clipped.tif", water
    )
    green = read_point_values(
        f"{PREFIX}B03_10m_clipped.tif", water
    )
    red = read_point_values(
        f"{PREFIX}B04_10m_clipped.tif", water
    )
    nir = read_point_values(
        f"{PREFIX}B08_10m_clipped.tif", water
    )
    swir = read_point_values(
        f"{PREFIX}B11_20m_clipped.tif", water
    )

    ndvi = normalized_difference(nir, red)
    ndwi = normalized_difference(green, nir)
    ndbi = normalized_difference(swir, nir)

    print("\nWater-point indices calculated directly from bands")
    print("=" * 65)

    for i, (_, point) in enumerate(water.iterrows()):
        sample_id = point.get("sample_id", point.name)

        print(f"\nSample {sample_id}")
        print(f"  NDVI: {ndvi[i]:.4f}")
        print(f"  NDWI: {ndwi[i]:.4f}")
        print(f"  NDBI: {ndbi[i]:.4f}")

    print("\nDiagnostic complete. No files were modified.")


if __name__ == "__main__":
    main()
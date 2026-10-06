
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]

POINT_FILE = ROOT / "data" / "aoi" / "training_samples_v2.gpkg"
BAND_DIR = ROOT / "data" / "processed" / "sentinel2"

BANDS = [
    "B02",   # Blue
    "B03",   # Green
    "B04",   # Red
    "B08",   # Near-infrared
    "B11",   # Short-wave infrared
    "B12",   # Short-wave infrared
]


def find_band(band_name):
    matches = list(BAND_DIR.glob(f"*_{band_name}_*_clipped.tif"))

    if not matches:
        matches = list(BAND_DIR.glob(f"*_{band_name}_clipped.tif"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find clipped raster for {band_name} "
            f"in {BAND_DIR}"
        )

    return matches[0]


def main():
    print("=" * 65)
    print("WATER SPECTRAL INSPECTION")
    print("=" * 65)

    points = gpd.read_file(
        POINT_FILE,
        layer="training_samples"
    )

    water_points = points[
        points["class_name"].astype(str).str.strip().str.lower()
        == "water"
    ].copy()

    print(f"Total training points: {len(points)}")
    print(f"Water points: {len(water_points)}")

    if water_points.empty:
        print("No water points found. Check class_name values.")
        return

    for band_name in BANDS:
        raster_path = find_band(band_name)

        print(f"\n--- {band_name} ---")
        print(f"File: {raster_path.name}")

        with rasterio.open(raster_path) as src:
            transformed = water_points.to_crs(src.crs)

            for idx, (point_idx, point) in enumerate(
                transformed.iterrows()
            ):
                x, y = point.geometry.x, point.geometry.y
                values = list(src.sample([(x, y)]))
                value = values[0][0]

                if (
                    src.nodata is not None
                    and np.isclose(value, src.nodata)
                ):
                    value = "NODATA"
                elif not np.isfinite(value):
                    value = "INVALID"
                else:
                    value = round(float(value), 3)

                print(
                    f"Feature {point_idx}: "
                    f"{point['class_name']}, "
                    f"{band_name}={value}"
                )

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETE — NO FILES MODIFIED")
    print("=" * 65)


if __name__ == "__main__":
    main()

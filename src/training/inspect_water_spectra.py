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
    INDICES_DIR,
    TRAINING_INDICES_DIR,
)


def main():
    print("=" * 65)
    print("WATER TRAINING POINT SPECTRAL DIAGNOSTICS")
    print("=" * 65)

    samples = gpd.read_file(TRAINING_SAMPLES_FILE)
    water_samples = samples[
        samples["class_name"].astype(str).str.lower() == "water"
    ]

    if water_samples.empty:
        print("ERROR: No water samples found.")
        return

    print(f"\nWater training points: {len(water_samples)}")
    print(f"Training CRS: {samples.crs}")

    # Discover clipped rasters and match band names.
    clipped_files = sorted(PROCESSED_DATA_DIR.glob("*_clipped.tif"))

    if not clipped_files:
        raise FileNotFoundError(
            f"No clipped rasters found in {PROCESSED_DATA_DIR}"
        )

    print("\nClipped raster files:")
    for path in clipped_files:
        print(f"  {path.name}")

    bands_to_check = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

    for band in bands_to_check:
        matches = [
            path for path in clipped_files
            if f"_{band}_" in path.name.upper()
        ]
        print(f"\n--- {band} ---")

        if not matches:
            print("Matching clipped raster not found.")
            continue

        raster_path = matches[0]

        with rasterio.open(raster_path) as src:
            points = water_samples.to_crs(src.crs)

            print(f"File: {raster_path.name}")
            print(f"NoData: {src.nodata}")

            for _, point in points.iterrows():
                sample_id = point.get("sample_id", point.name)
                x, y = point.geometry.x, point.geometry.y

                value = next(src.sample([(x, y)], masked=True))[0]

                if np.ma.is_masked(value):
                    result = "MASKED"
                else:
                    result = str(value.item())

                print(f"Sample {sample_id}: {result}")

    # Inspect the existing indices without modifying them.
    for index_name in ["ndvi", "ndwi", "ndbi"]:
        index_path = TRAINING_INDICES_DIR / f"{index_name}.tif"

        print(f"\n--- {index_name.upper()} ---")

        if not index_path.exists():
            print(f"File not found: {index_path}")
            continue

        with rasterio.open(index_path) as src:
            points = water_samples.to_crs(src.crs)
            print(f"NoData: {src.nodata}")

            for _, point in points.iterrows():
                sample_id = point.get("sample_id", point.name)
                x, y = point.geometry.x, point.geometry.y

                value = next(src.sample([(x, y)], masked=True))[0]

                if np.ma.is_masked(value):
                    result = "MASKED"
                elif not np.isfinite(value):
                    result = "NaN/Infinite"
                else:
                    result = f"{value.item():.4f}"

                print(f"Sample {sample_id}: {result}")

    print("\n" + "=" * 65)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
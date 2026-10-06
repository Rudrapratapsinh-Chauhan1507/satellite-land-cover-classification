
from pathlib import Path
import sys

import geopandas as gpd
import rasterio
from rasterio.mask import mask

# 1. Resolve project root and import shared paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    RAW_DATA_DIR,
    AOI_FILE,
    PROCESSED_DATA_DIR,
)

RAW_DIR = RAW_DATA_DIR
AOI_PATH = AOI_FILE
OUTPUT_DIR = PROCESSED_DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. List the 11 input bands
BANDS = [
    "T42QZL_20261003T053651_B02_10m.jp2",
    "T42QZL_20261003T053651_B03_10m.jp2",
    "T42QZL_20261003T053651_B04_10m.jp2",
    "T42QZL_20261003T053651_B08_10m.jp2",
    "T42QZL_20261003T053651_B05_20m.jp2",
    "T42QZL_20261003T053651_B06_20m.jp2",
    "T42QZL_20261003T053651_B07_20m.jp2",
    "T42QZL_20261003T053651_B8A_20m.jp2",
    "T42QZL_20261003T053651_B11_20m.jp2",
    "T42QZL_20261003T053651_B12_20m.jp2",
    "T42QZL_20261003T053651_SCL_20m.jp2",
]

# 3. Load and validate the AOI
print("Loading study area...")

if not AOI_PATH.exists():
    raise FileNotFoundError(f"AOI file not found: {AOI_PATH}")

aoi = gpd.read_file(AOI_PATH)

if aoi.empty:
    raise ValueError("The AOI file contains no features.")

aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty]

if aoi.empty:
    raise ValueError("The AOI has no usable geometries.")

if aoi.crs is None:
    raise ValueError("The AOI has no CRS. Check study_area.geojson.")

aoi = aoi.dissolve()

print("AOI loaded.")
print("AOI CRS:", aoi.crs)

# 4. Clip every band
for filename in BANDS:
    input_path = RAW_DIR / filename

    if not input_path.exists():
        print(f"\nMISSING FILE: {filename}")
        continue

    output_path = OUTPUT_DIR / f"{Path(filename).stem}_clipped.tif"

    print(f"\nProcessing: {filename}")

    try:
        with rasterio.open(input_path) as src:
            # Reproject AOI coordinates to the raster CRS.
            aoi_in_raster_crs = aoi.to_crs(src.crs)

            shapes = [
                geom.__geo_interface__
                for geom in aoi_in_raster_crs.geometry
            ]

            # Crop to the AOI bounding rectangle and mask
            # pixels outside the AOI polygon.
            clipped, transform = mask(
                src,
                shapes,
                crop=True,
                filled=False,
            )

            # Zero is the output NoData value.
            # This is suitable for the current workflow because
            # zero-valued pixels are excluded downstream.
            output_nodata = 0
            data = clipped.filled(output_nodata)

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                count=src.count,
                dtype=data.dtype,
                nodata=output_nodata,
                compress="deflate",
                tiled=True,
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

        print("Saved:", output_path.name)
        print("Output dimensions:", data.shape[2], "x", data.shape[1])

    except Exception as error:
        print(f"ERROR processing {filename}: {error}")

print("\nClipping finished.")
print("Output folder:", OUTPUT_DIR)

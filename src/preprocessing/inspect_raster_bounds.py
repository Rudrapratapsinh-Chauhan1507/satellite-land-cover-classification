from pathlib import Path
import sys

import rasterio
import geopandas as gpd
from rasterio.warp import transform_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR, AOI_FILE

raster_path = RAW_DATA_DIR / "T42QZL_20261003T053651_B04_10m.jp2"
aoi_path = AOI_FILE

with rasterio.open(raster_path) as src:
    print("RASTER INFORMATION")
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)

    bounds_4326 = transform_bounds(
        src.crs,
        "EPSG:4326",
        *src.bounds,
        densify_pts=21,
    )

    print("\nRaster bounds transformed to EPSG:4326:")
    print(bounds_4326)

aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")

print("\nSTUDY AREA INFORMATION")
print("CRS:", aoi.crs)
print("Bounds:", aoi.total_bounds)

print("\nDoes the raster cover the study area?")

raster_min_x, raster_min_y, raster_max_x, raster_max_y = bounds_4326
aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y = aoi.total_bounds

overlaps = (
    raster_min_x <= aoi_max_x
    and raster_max_x >= aoi_min_x
    and raster_min_y <= aoi_max_y
    and raster_max_y >= aoi_min_y
)

print("Bounding-box overlap:", overlaps)
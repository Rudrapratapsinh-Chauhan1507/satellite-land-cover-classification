
from pathlib import Path
import sys

import rasterio

# Resolve the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR

DATA_DIR = RAW_DATA_DIR

bands = [
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

print("Sentinel-2 data verification")
print("=" * 50)
print(f"Data directory: {DATA_DIR}")

for filename in bands:
    path = DATA_DIR / filename

    if not path.exists():
        print(f"\nMISSING: {filename}")
        continue

    try:
        with rasterio.open(path) as src:
            print(f"\nFile: {filename}")
            print(f"  CRS: {src.crs}")
            print(f"  Resolution: {src.res}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  NoData: {src.nodata}")
            print(f"  Bands: {src.count}")
    except Exception as error:
        print(f"\nERROR reading {filename}: {error}")

print("\nVerification finished.")

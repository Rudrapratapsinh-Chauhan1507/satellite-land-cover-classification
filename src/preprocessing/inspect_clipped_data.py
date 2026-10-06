
from pathlib import Path
import sys

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR

DATA_DIR = PROCESSED_DATA_DIR

files = sorted(DATA_DIR.glob("*_clipped.tif"))

print(f"Data directory: {DATA_DIR}")
print(f"Found {len(files)} clipped rasters")
print("=" * 60)

if not files:
    raise FileNotFoundError(
        f"No clipped GeoTIFF files found in: {DATA_DIR}"
    )

for path in files:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        valid = data.compressed()
        valid = valid[np.isfinite(valid)]

        print(f"\n{path.name}")
        print("  CRS:", src.crs)
        print("  Resolution:", src.res)
        print("  Dimensions:", src.width, "x", src.height)
        print("  Bounds:", src.bounds)
        print("  NoData:", src.nodata)
        print("  Valid pixels:", valid.size)

        if valid.size:
            print("  Min:", float(valid.min()))
            print("  Max:", float(valid.max()))
            print("  Mean:", float(valid.mean()))

print("\nInspection complete.")


from pathlib import Path
import sys
import rasterio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import PROCESSED_DATA_DIR, TRAINING_INDICES_DIR

files = {
    "B02": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B02_10m_clipped.tif",
    "B05": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B05_20m_clipped.tif",
    "B11": PROCESSED_DATA_DIR / "T42QZL_20261003T053651_B11_20m_clipped.tif",
    "NDVI": TRAINING_INDICES_DIR / "ndvi.tif",
    "NDWI": TRAINING_INDICES_DIR / "ndwi.tif",
    "NDBI": TRAINING_INDICES_DIR / "ndbi.tif",
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    print("Path:", path)
    print("Exists:", path.exists())

    if path.exists():
        with rasterio.open(path) as src:
            print("Dimensions:", src.width, "x", src.height)
            print("Resolution:", src.res)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("NoData:", src.nodata)

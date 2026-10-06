
from pathlib import Path
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    PROCESSED_DATA_DIR,
    INDICES_DIR,
)

DATA = PROCESSED_DATA_DIR
OUT = INDICES_DIR
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "T42QZL_20261003T053651_"


def read_band(suffix):
    path = DATA / f"{PREFIX}{suffix}_clipped.tif"

    if not path.exists():
        raise FileNotFoundError(f"Band not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype("float32").filled(np.nan)
        return array, src.profile.copy()


# Read 10 m bands and use B04 as the reference grid.
blue, profile = read_band("B02_10m")
green, _ = read_band("B03_10m")
red, _ = read_band("B04_10m")
nir, _ = read_band("B08_10m")

# Read 20 m SCL and align it to the 10 m grid using nearest neighbour.
scl_path = DATA / f"{PREFIX}SCL_20m_clipped.tif"
with rasterio.open(scl_path) as src:
    scl = np.zeros(red.shape, dtype="uint8")
    reproject(
        source=rasterio.band(src, 1),
        destination=scl,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=Resampling.nearest,
    )

# Mask no-data and quality classes.
# SCL 4, 5 and 6 are vegetation, bare soil and water.
# SCL 2 is retained for now but can be reviewed separately.
valid = (
    np.isin(scl, [4, 5, 6])
    & (red > 0)
    & (green > 0)
    & (blue > 0)
    & (nir > 0)
)

def normalized_difference(a, b):
    denominator = a + b
    result = np.full(a.shape, np.nan, dtype="float32")
    good = valid & (denominator != 0)
    result[good] = (a[good] - b[good]) / denominator[good]
    return result

# NDVI: near-infrared vs red
ndvi = normalized_difference(nir, red)

# NDWI (McFeeters): green vs near-infrared
ndwi = normalized_difference(green, nir)

# NDBI: shortwave infrared vs near-infrared.
# B11 is 20 m, so align it to the 10 m reference grid.
b11_path = DATA / f"{PREFIX}B11_20m_clipped.tif"
with rasterio.open(b11_path) as src:
    swir = np.zeros(red.shape, dtype="float32")
    reproject(
        source=rasterio.band(src, 1),
        destination=swir,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=Resampling.bilinear,
    )

ndbi = normalized_difference(swir, nir)

# Save each index as a GeoTIFF. NaN is used for masked pixels.
profile.update(
    driver="GTiff",
    count=1,
    dtype="float32",
    nodata=np.nan,
    compress="deflate",
)

for name, array in [
    ("NDVI", ndvi),
    ("NDWI", ndwi),
    ("NDBI", ndbi),
]:
    output = OUT / f"{name.lower()}.tif"
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(array, 1)

    finite = array[np.isfinite(array)]
    print(f"\n{name}")
    print("Saved:", output)
    print("Valid pixels:", finite.size)
    if finite.size:
        print("Min:", round(float(finite.min()), 4))
        print("Max:", round(float(finite.max()), 4))
        print("Mean:", round(float(finite.mean()), 4))

print("\nSpectral index calculation finished.")

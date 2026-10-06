
from pathlib import Path
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR, TRAINING_INDICES_DIR

DATA = PROCESSED_DATA_DIR
OUT = TRAINING_INDICES_DIR
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "T42QZL_20261003T053651_"



def read_band(suffix):
    path = DATA / f"{PREFIX}{suffix}_clipped.tif"

    if not path.exists():
        raise FileNotFoundError(f"Band not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype("float32").filled(np.nan)
        return array, src.profile.copy()



# Read 10 m bands on the B04 reference grid.
blue, profile = read_band("B02_10m")
green, _ = read_band("B03_10m")
red, _ = read_band("B04_10m")
nir, _ = read_band("B08_10m")

# Align the 20 m SCL layer to the 10 m grid.
scl_path = DATA / f"{PREFIX}SCL_20m_clipped.tif"

with rasterio.open(scl_path) as src:
    scl = np.zeros(red.shape, dtype="uint8")

    reproject(
        source=rasterio.band(src, 1),
        destination=scl,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

# For training only, retain class 7 as well as classes 4, 5 and 6.
# Class 7 means unclassified; it does not itself confirm water.
valid = (
    np.isin(scl, [4, 5, 6, 7])
    & (blue > 0)
    & (green > 0)
    & (red > 0)
    & (nir > 0)
)


def normalized_difference(a, b, mask):
    denominator = a + b
    result = np.full(a.shape, np.nan, dtype="float32")

    good = mask & np.isfinite(a) & np.isfinite(b) & (denominator != 0)

    result[good] = (
        (a[good] - b[good]) / denominator[good]
    )

    return result


# NDVI: NIR versus red.
ndvi = normalized_difference(nir, red, valid)

# NDWI (McFeeters): green versus NIR.
ndwi = normalized_difference(green, nir, valid)

# Align B11 from 20 m to the 10 m reference grid.
b11_path = DATA / f"{PREFIX}B11_20m_clipped.tif"

with rasterio.open(b11_path) as src:
    swir = np.full(red.shape, np.nan, dtype="float32")

    reproject(
        source=rasterio.band(src, 1),
        destination=swir,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

# Require valid SWIR values specifically for NDBI.
ndbi_valid = valid & (swir > 0)
ndbi = normalized_difference(swir, nir, ndbi_valid)

# Save only to the separate training_indices folder.
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

print("\nTraining spectral indices generated.")
print("Original index files were not modified.")

from pathlib import Path
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shared project paths.
from src.utils.paths import (
    PROCESSED_DATA_DIR,
    INDICES_DIR,
    TRAINING_OUTPUT_DIR,
)

BAND_DIR = PROCESSED_DATA_DIR
INDEX_DIR = INDICES_DIR
OUTPUT_DIR = TRAINING_OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the 10 m B02 raster as the reference grid.
reference_path = BAND_DIR / "T42QZL_20261003T053651_B02_10m_clipped.tif"

features = {}
reference_profile = None
reference_shape = None
reference_transform = None
reference_crs = None

def read_aligned(path, reference):
    """Read a raster and align it to the 10 m reference grid."""
    with rasterio.open(path) as src:
        source = src.read(1).astype("float32")

        destination = np.full(
            (reference.height, reference.width),
            np.nan,
            dtype="float32",
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        return destination


with rasterio.open(reference_path) as reference:
    reference_profile = reference.profile.copy()
    reference_shape = (reference.height, reference.width)
    reference_transform = reference.transform
    reference_crs = reference.crs

    band_files = {
        "B02": "T42QZL_20261003T053651_B02_10m_clipped.tif",
        "B03": "T42QZL_20261003T053651_B03_10m_clipped.tif",
        "B04": "T42QZL_20261003T053651_B04_10m_clipped.tif",
        "B08": "T42QZL_20261003T053651_B08_10m_clipped.tif",
        "B05": "T42QZL_20261003T053651_B05_20m_clipped.tif",
        "B06": "T42QZL_20261003T053651_B06_20m_clipped.tif",
        "B07": "T42QZL_20261003T053651_B07_20m_clipped.tif",
        "B8A": "T42QZL_20261003T053651_B8A_20m_clipped.tif",
        "B11": "T42QZL_20261003T053651_B11_20m_clipped.tif",
        "B12": "T42QZL_20261003T053651_B12_20m_clipped.tif",
    }

    for name, filename in band_files.items():
        path = BAND_DIR / filename
        features[name] = read_aligned(path, reference)
        print(f"Loaded band: {name}")

    for name in ["ndvi", "ndwi", "ndbi"]:
        features[name.upper()] = read_aligned(
            INDEX_DIR / f"{name}.tif",
            reference,
        )
        print(f"Loaded index: {name.upper()}")

    # Reproject the SCL classification layer using nearest-neighbor
    # so class codes remain integers.
    scl_path = BAND_DIR / "T42QZL_20261003T053651_SCL_20m_clipped.tif"
    scl = read_aligned(scl_path, reference)

# Require every feature to be finite and positive for raw reflectance bands.
# The spectral indices may legitimately be negative.
feature_names = list(features.keys())
stack = np.stack([features[name] for name in feature_names], axis=-1)

reflectance_names = [
    "B02", "B03", "B04", "B08", "B05",
    "B06", "B07", "B8A", "B11", "B12",
]
reflectance_indices = [
    feature_names.index(name) for name in reflectance_names
]

valid = np.isfinite(stack).all(axis=-1)
valid &= np.isfinite(scl)
valid &= np.isin(scl, [4, 5, 6])

for index in reflectance_indices:
    valid &= stack[..., index] > 0

# Index files use zero as NoData; exclude those cells too.
for name in ["NDVI", "NDWI", "NDBI"]:
    valid &= np.isfinite(features[name])

# Store features only for pixels passing the validity mask.
X = stack[valid].astype("float32")

np.savez_compressed(
    OUTPUT_DIR / "spectral_features.npz",
    X=X,
    feature_names=np.array(feature_names),
)

# Export the valid-pixel mask for inspection in GIS software.
mask_profile = reference_profile.copy()
mask_profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=0,
    compress="lzw",
)

with rasterio.open(
    OUTPUT_DIR / "valid_pixel_mask.tif",
    "w",
    **mask_profile,
) as dst:
    dst.write(valid.astype("uint8"), 1)

print("\nFeature preparation complete.")
print("Feature names:", feature_names)
print("Raster dimensions:", reference_shape)
print("Valid pixels:", int(valid.sum()))
print("Feature matrix shape:", X.shape)
print("Saved:", OUTPUT_DIR / "spectral_features.npz")
print("Saved:", OUTPUT_DIR / "valid_pixel_mask.tif")
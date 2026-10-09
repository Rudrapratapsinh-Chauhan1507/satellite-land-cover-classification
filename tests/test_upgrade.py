"""
SentinelScope post-upgrade test suite.

Checks:
  1. sklearn version == 1.9.1
  2. Module import
  3. Model loading — no InconsistentVersionWarning
  4. Synthetic end-to-end inference
  5. GeoTIFF round-trip (CRS, dims, class IDs)
"""

import warnings
import sys
import numpy as np

# ── 0. Escalate InconsistentVersionWarning to error ──────────────────────────
# If the warning fires it will crash the script with a non-zero exit code.
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("error", category=InconsistentVersionWarning)

# ── 1. sklearn version ────────────────────────────────────────────────────────
import sklearn
ver = sklearn.__version__
print(f"sklearn version: {ver}")
assert ver == "1.9.1", f"Expected 1.9.1, got {ver}"
print("Test 1 PASSED: sklearn==1.9.1")

# ── 2. Module import ──────────────────────────────────────────────────────────
from src.inference.predict import (
    load_model, run_inference, CLASS_LABELS, FEATURE_ORDER, MODEL_PATH,
)
print(f"\nFEATURE_ORDER : {FEATURE_ORDER}")
print(f"CLASS_LABELS  : {CLASS_LABELS}")
print(f"MODEL_PATH    : {MODEL_PATH}")
print("Test 2 PASSED: Module import OK")

# ── 3. Model loading — must produce NO InconsistentVersionWarning ─────────────
print("\nLoading model (InconsistentVersionWarning is now an error)...")
saved = load_model()
model_type = type(saved["model"]).__name__
print(f"Model type      : {model_type}")
print(f"Stored features : {saved['features']}")
print(f"Class mapping   : {saved['class_mapping']}")
assert model_type == "ExtraTreesClassifier"
assert saved["features"] == FEATURE_ORDER
print("Test 3 PASSED: Model loaded without InconsistentVersionWarning")

# ── 4. Synthetic end-to-end inference ─────────────────────────────────────────
print("\nRunning inference on 50×60 synthetic scene...")
H, W = 50, 60
np.random.seed(42)

def _make_bands():
    bands = {}
    bands["B02"] = np.random.uniform(300, 600,  (H, W)).astype(np.float32)
    bands["B03"] = np.random.uniform(400, 700,  (H, W)).astype(np.float32)
    bands["B04"] = np.random.uniform(300, 600,  (H, W)).astype(np.float32)
    bands["B05"] = np.random.uniform(600, 1000, (H, W)).astype(np.float32)
    bands["B06"] = np.random.uniform(1500,2500, (H, W)).astype(np.float32)
    bands["B07"] = np.random.uniform(1800,3000, (H, W)).astype(np.float32)
    bands["B08"] = np.random.uniform(2000,3500, (H, W)).astype(np.float32)
    bands["B8A"] = np.random.uniform(2000,3500, (H, W)).astype(np.float32)
    bands["B11"] = np.random.uniform(800, 1500, (H, W)).astype(np.float32)
    bands["B12"] = np.random.uniform(400, 900,  (H, W)).astype(np.float32)
    return bands

bands = _make_bands()
classification, pixel_counts, percentages = run_inference(bands, chunk_size=500)

print(f"  Output shape : {classification.shape}")
print(f"  Output dtype : {classification.dtype}")
print(f"  Unique IDs   : {np.unique(classification).tolist()}")
print(f"  Pixel counts : {pixel_counts}")
print(f"  Percentages  : {percentages}")

assert classification.shape == (H, W), "Shape mismatch"
assert classification.dtype == np.uint8, "dtype must be uint8"
assert set(np.unique(classification)).issubset({0, 1, 2, 3, 4, 5}), \
    "Unexpected class IDs in output"
print("Test 4 PASSED: Synthetic inference OK")

# ── 5. GeoTIFF round-trip ─────────────────────────────────────────────────────
print("\nWriting and re-reading classification GeoTIFF in memory...")
import io
import rasterio
from affine import Affine

profile = {
    "driver":    "GTiff",
    "dtype":     "uint8",
    "nodata":    0,
    "width":     W,
    "height":    H,
    "count":     1,
    "crs":       "EPSG:32642",
    "transform": Affine(10, 0, 700000, 0, -10, 2400000),
    "compress":  "deflate",
}

buf = io.BytesIO()
with rasterio.open(buf, "w", **profile) as dst:
    dst.write(classification.astype(np.uint8), 1)
    dst.set_band_description(1, "Extra Trees land-cover class (1–5)")
buf.seek(0)

with rasterio.open(buf) as src:
    rt_arr   = src.read(1)
    rt_crs   = str(src.crs)
    rt_width = src.width
    rt_height = src.height

print(f"  CRS    : {rt_crs}")
print(f"  Dims   : {rt_width} × {rt_height}")
print(f"  IDs    : {np.unique(rt_arr).tolist()}")

assert rt_width  == W,              "GeoTIFF width mismatch"
assert rt_height == H,              "GeoTIFF height mismatch"
assert "32642" in rt_crs,           "GeoTIFF CRS mismatch"
assert set(np.unique(rt_arr)).issubset({0, 1, 2, 3, 4, 5}), \
    "Unexpected class IDs in GeoTIFF"
print("Test 5 PASSED: GeoTIFF round-trip OK")

print("\n" + "="*55)
print("ALL 5 TESTS PASSED — InconsistentVersionWarning is gone.")
print("="*55)

"""
Inference module for Sentinel-2 land-cover classification.

Accepts pre-loaded per-band numpy arrays (float32) and the raster profile
from the reference band (B02), computes NDVI / NDWI / NDBI from the same
formulas used during training, constructs the 13-feature matrix in the
correct order, and runs the Extra Trees candidate model.

The caller is responsible for loading bands from GeoTIFF files and
passing them to run_inference().  No file paths are hard-coded here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "extra_trees_candidate.joblib"

#: Exact feature order that the model was trained on.  Do NOT change.
FEATURE_ORDER: list[str] = [
    "B02", "B03", "B04",
    "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12",
    "NDVI", "NDWI", "NDBI",
]

#: Class ID → human-readable label
CLASS_LABELS: dict[int, str] = {
    1: "Vegetation",
    2: "Built-up",
    3: "Bare Soil",
    4: "Water",
    5: "Road",
}

#: Class name (lowercase, underscore) → class ID  (matches training)
CLASS_MAPPING: dict[str, int] = {
    "vegetation": 1,
    "built_up":   2,
    "bare_soil":  3,
    "water":      4,
    "road":       5,
}

#: Minimum and maximum valid reflectance value (Sentinel-2 raw DN / scaled)
BAND_MIN_VALID: float = 0.0  # exclusive — zero is treated as nodata


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model_cache: dict | None = None


def load_model() -> dict:
    """
    Load the Extra Trees model from disk.  Cached after the first call.

    Returns
    -------
    dict
        Dictionary with keys: ``model``, ``features``, ``class_names``,
        ``class_mapping``.

    Raises
    ------
    FileNotFoundError
        If the model file is missing.
    ValueError
        If the stored feature order does not match ``FEATURE_ORDER``.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Expected at outputs/models/extra_trees_candidate.joblib"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        saved = joblib.load(MODEL_PATH)

    stored_features = list(saved["features"])
    if stored_features != FEATURE_ORDER:
        raise ValueError(
            f"Stored feature order does not match expected order.\n"
            f"Expected : {FEATURE_ORDER}\n"
            f"Stored   : {stored_features}"
        )

    _model_cache = saved
    return _model_cache


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------

def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute (a - b) / (a + b) element-wise with NaN protection.

    Returns NaN where the denominator is zero or where either input is NaN.
    """
    denominator = a + b
    result = np.full(a.shape, np.nan, dtype=np.float32)
    good = np.isfinite(a) & np.isfinite(b) & (denominator != 0)
    result[good] = (a[good] - b[good]) / denominator[good]
    return result


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)

    Uses B08 (NIR) and B04 (Red) — same formula as the project training code.
    """
    return _normalized_difference(nir, red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDWI (McFeeters) = (Green - NIR) / (Green + NIR)

    Uses B03 (Green) and B08 (NIR) — same formula as the project training code.
    """
    return _normalized_difference(green, nir)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDBI = (SWIR - NIR) / (SWIR + NIR)

    Uses B11 (SWIR) and B08 (NIR) — same formula as the project training code.
    """
    return _normalized_difference(swir, nir)


# ---------------------------------------------------------------------------
# Valid-pixel mask
# ---------------------------------------------------------------------------

def build_valid_mask(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Build a boolean mask of pixels that are valid for classification.

    A pixel is valid when:
    - All 10 reflectance bands are finite and > 0
    - All three indices are finite and within [-1.001, 1.001]

    Parameters
    ----------
    bands:
        Dict mapping feature name → 2-D float32 array.  Must include all 13
        features after index computation.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W).
    """
    ref_shape = next(iter(bands.values())).shape
    valid = np.ones(ref_shape, dtype=bool)

    reflectance_bands = [
        "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B11", "B12",
    ]
    for name in reflectance_bands:
        arr = bands[name]
        valid &= np.isfinite(arr)
        valid &= arr > BAND_MIN_VALID

    for name in ("NDVI", "NDWI", "NDBI"):
        arr = bands[name]
        valid &= np.isfinite(arr)
        valid &= arr >= -1.001
        valid &= arr <= 1.001

    return valid


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

def run_inference(
    band_arrays: Dict[str, np.ndarray],
    chunk_size: int = 50_000,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict[int, int], Dict[str, float]]:
    """
    Run land-cover classification on pre-loaded Sentinel-2 bands.

    The caller must supply the 10 raw reflectance bands.  NDVI, NDWI,
    and NDBI are computed inside this function using the project formulas.

    Parameters
    ----------
    band_arrays:
        Dict with keys ``B02`` … ``B12`` mapping to 2-D float32 numpy
        arrays on a common spatial grid.  Arrays must be aligned before
        calling this function.
    chunk_size:
        Number of pixels per prediction chunk.  Reduce if memory is limited.
    progress_callback:
        Optional callable(fraction: float) called at each chunk boundary.

    Returns
    -------
    classification : np.ndarray (uint8, shape H×W)
        Pixel-level class IDs (1–5).  0 = nodata / invalid pixel.
    pixel_counts : dict[int, int]
        Count of predicted pixels per class ID.
    percentages : dict[str, float]
        Percentage of valid pixels per class label.

    Raises
    ------
    ValueError
        On missing bands, empty valid-pixel set, or invalid inputs.
    RuntimeError
        On model prediction failure.
    """
    # --- Check required reflectance bands are present --------------------
    required_bands = [
        "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B11", "B12",
    ]
    missing = [b for b in required_bands if b not in band_arrays]
    if missing:
        raise ValueError(
            f"Missing required Sentinel-2 bands: {missing}\n"
            "Upload all 10 required bands."
        )

    # --- Verify all arrays share the same shape --------------------------
    shapes = {name: arr.shape for name, arr in band_arrays.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        detail = "\n".join(f"  {k}: {v}" for k, v in shapes.items())
        raise ValueError(
            f"Band arrays have inconsistent shapes:\n{detail}\n"
            "All bands must be aligned to the same spatial grid."
        )

    height, width = next(iter(band_arrays.values())).shape

    # --- Compute spectral indices -----------------------------------------
    nir   = band_arrays["B08"].astype(np.float32)
    red   = band_arrays["B04"].astype(np.float32)
    green = band_arrays["B03"].astype(np.float32)
    swir  = band_arrays["B11"].astype(np.float32)

    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    ndbi = compute_ndbi(swir, nir)

    # --- Assemble full feature dict ---------------------------------------
    all_features: Dict[str, np.ndarray] = {}
    for name in required_bands:
        all_features[name] = band_arrays[name].astype(np.float32)
    all_features["NDVI"] = ndvi
    all_features["NDWI"] = ndwi
    all_features["NDBI"] = ndbi

    # --- Build valid-pixel mask -------------------------------------------
    valid = build_valid_mask(all_features)
    valid_count = int(valid.sum())

    if valid_count == 0:
        raise ValueError(
            "No valid pixels found in the uploaded data.\n"
            "Check that the bands contain positive reflectance values and "
            "are correctly masked."
        )

    # --- Load model ----------------------------------------------------------
    saved = load_model()
    model = saved["model"]

    # --- Build feature matrix (only valid pixels) -------------------------
    flat_valid = np.flatnonzero(valid.ravel())

    # Predict in chunks to limit peak memory
    output = np.zeros(height * width, dtype=np.uint8)
    total_chunks = max(1, len(flat_valid) // chunk_size + 1)

    for chunk_idx, start in enumerate(range(0, len(flat_valid), chunk_size)):
        pixel_indices = flat_valid[start:start + chunk_size]

        chunk_df = pd.DataFrame(
            {feat: all_features[feat].ravel()[pixel_indices] for feat in FEATURE_ORDER},
            columns=FEATURE_ORDER,
        )

        try:
            predicted_names = model.predict(chunk_df)
        except Exception as exc:
            raise RuntimeError(
                f"Model prediction failed on chunk {chunk_idx}: {exc}"
            ) from exc

        predicted_ids = np.array(
            [CLASS_MAPPING[name] for name in predicted_names],
            dtype=np.uint8,
        )
        output[pixel_indices] = predicted_ids

        if progress_callback is not None:
            progress_callback((chunk_idx + 1) / total_chunks)

    classification = output.reshape(height, width)

    # --- Compute statistics -----------------------------------------------
    pixel_counts: Dict[int, int] = {}
    for class_id in CLASS_LABELS:
        pixel_counts[class_id] = int(np.sum(classification == class_id))

    total_valid = sum(pixel_counts.values())
    percentages: Dict[str, float] = {}
    for class_id, label in CLASS_LABELS.items():
        pct = (pixel_counts[class_id] / total_valid * 100) if total_valid > 0 else 0.0
        percentages[label] = round(pct, 2)

    return classification, pixel_counts, percentages

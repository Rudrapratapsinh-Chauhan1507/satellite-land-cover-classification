"""
src/inference — inference utilities for the SentinelScope Streamlit application.

Exports:
    run_inference    — main classification entry point
    load_model       — model loader (singleton)
    CLASS_LABELS     — {class_id: label} dict
    FEATURE_ORDER    — ordered list of the 13 training features
    MODEL_PATH       — Path to the Extra Trees model file
"""
from .predict import (
    run_inference,
    load_model,
    CLASS_LABELS,
    CLASS_MAPPING,
    FEATURE_ORDER,
    MODEL_PATH,
    compute_ndvi,
    compute_ndwi,
    compute_ndbi,
)

__all__ = [
    "run_inference",
    "load_model",
    "CLASS_LABELS",
    "CLASS_MAPPING",
    "FEATURE_ORDER",
    "MODEL_PATH",
    "compute_ndvi",
    "compute_ndwi",
    "compute_ndbi",
]

"""
SentinelScope — Sentinel-2 Land Cover Intelligence
===================================================
Streamlit web application for interactive Sentinel-2 land-cover
classification using the project's trained Extra Trees candidate model.

Run with:
    streamlit run app.py

from the project root directory.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root resolution — makes imports work regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy rasterio import (guard for startup without C libs)
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

# ---------------------------------------------------------------------------
# Inference module
# ---------------------------------------------------------------------------
from src.inference.predict import (
    run_inference,
    load_model,
    CLASS_LABELS,
    FEATURE_ORDER,
    MODEL_PATH,
    compute_ndvi,
    compute_ndwi,
    compute_ndbi,
)

# ===========================================================================
# CONSTANTS
# ===========================================================================

APP_NAME = "SentinelScope"
APP_SUBTITLE = "Sentinel-2 Land Cover Intelligence"

REQUIRED_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

#: Visually intuitive class colours (matches existing project visualization)
CLASS_COLORS: Dict[int, str] = {
    1: "#3CB371",   # Vegetation — medium sea green
    2: "#E05C4B",   # Built-up   — warm red
    3: "#C8A96E",   # Bare Soil  — sandy brown
    4: "#3A9BD5",   # Water      — ocean blue
    5: "#6B6B6B",   # Road       — concrete grey
}

CLASS_COLORS_LIST = [CLASS_COLORS[i] for i in range(1, 6)]

CMAP = ListedColormap(CLASS_COLORS_LIST)
NORM = BoundaryNorm(boundaries=np.arange(0.5, 6.5, 1), ncolors=5)

#: Band → human description (for upload UI)
BAND_DESCRIPTIONS: Dict[str, str] = {
    "B02": "Blue (490 nm)",
    "B03": "Green (560 nm)",
    "B04": "Red (665 nm)",
    "B05": "Red Edge 1 (705 nm)",
    "B06": "Red Edge 2 (740 nm)",
    "B07": "Red Edge 3 (783 nm)",
    "B08": "Near Infrared (842 nm)",
    "B8A": "Narrow NIR (865 nm)",
    "B11": "SWIR 1 (1610 nm)",
    "B12": "SWIR 2 (2190 nm)",
}

EXISTING_CLASSIFICATION = PROJECT_ROOT / "outputs" / "classification" / "land_cover_extra_trees.tif"
EXISTING_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "land_cover_classification_final.png"
TRUE_COLOR_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "true_color_preview.png"

# ===========================================================================
# PAGE CONFIG
# ===========================================================================

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# CUSTOM CSS — dark satellite aesthetic
# ===========================================================================

st.markdown("""
<style>
/* ── Root palette ── */
:root {
    --bg-deep:    #0a0f1e;
    --bg-card:    #111827;
    --bg-card2:   #1a2235;
    --accent:     #38bdf8;
    --accent2:    #0ea5e9;
    --text-main:  #e2e8f0;
    --text-muted: #94a3b8;
    --green:      #3CB371;
    --red:        #E05C4B;
    --sand:       #C8A96E;
    --blue:       #3A9BD5;
    --grey:       #6B6B6B;
    --border:     rgba(56, 189, 248, 0.18);
}

/* ── Global background ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    color: var(--text-main) !important;
}
[data-testid="stHeader"] {
    background: var(--bg-deep) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ── Hero section ── */
.hero-container {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a1628 60%, #060d1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    border: 2px solid rgba(56,189,248,0.12);
    pointer-events: none;
}
.hero-container::after {
    content: '';
    position: absolute;
    top: -90px; right: -90px;
    width: 360px; height: 360px;
    border-radius: 50%;
    border: 1px solid rgba(56,189,248,0.06);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.35);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 0.5rem;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin: 0 0 2rem;
    max-width: 560px;
}
.hero-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 2rem;
}
.hero-meta-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    min-width: 110px;
}
.hero-meta-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}
.hero-meta-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
}

/* ── Section headings ── */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 2rem 0 1rem;
}
.section-sub {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: -0.5rem 0 1.2rem;
    padding-left: 1.05rem;
}

/* ── Card wrapper ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-sm {
    background: var(--bg-card2);
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.9rem;
    margin: 1rem 0;
}
.stat-card {
    flex: 1 1 140px;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}
.stat-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    justify-content: center;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
}
.stat-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}

/* ── Band pill grid ── */
.band-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0;
}
.band-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    font-family: monospace;
}

/* ── Pipeline step ── */
.pipeline-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin: 0.6rem 0;
}
.pipeline-num {
    background: var(--accent2);
    color: #fff;
    font-weight: 800;
    font-size: 0.75rem;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.pipeline-text {
    font-size: 0.88rem;
    color: var(--text-main);
}

/* ── Legend row ── */
.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin: 0.75rem 0;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.82rem;
    color: var(--text-main);
}
.legend-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
}

/* ── Info / warning boxes ── */
.info-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: var(--text-main);
    margin: 0.75rem 0;
}
.warn-box {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #fde68a;
    margin: 0.75rem 0;
}
.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #fca5a5;
    margin: 0.75rem 0;
}
.success-box {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #6ee7b7;
    margin: 0.75rem 0;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.77rem;
    color: var(--text-muted);
    margin-top: 1.5rem;
    font-style: italic;
}

/* ── Streamlit widget overrides ── */
.stFileUploader > label {
    color: var(--text-main) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-card2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: var(--accent2) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    transform: translateY(-1px);
}
.stDownloadButton > button {
    background: rgba(56,189,248,0.12) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
[data-testid="stTabsContent"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 12px 12px 12px !important;
    padding: 1.5rem !important;
}
[data-testid="stTab"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
[data-testid="stTab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpanderToggleIcon"] { color: var(--accent) !important; }
.stProgress > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# HELPERS
# ===========================================================================

@st.cache_resource(show_spinner="Loading Extra Trees model…")
def _load_model_cached():
    """Cache the model so it is loaded only once per server process."""
    return load_model()


def render_html(html: str):
    st.markdown(html, unsafe_allow_html=True)


def legend_html() -> str:
    items = "".join(
        f'<div class="legend-item">'
        f'<div class="legend-swatch" style="background:{CLASS_COLORS[cid]}"></div>'
        f'{label}</div>'
        for cid, label in CLASS_LABELS.items()
    )
    return f'<div class="legend-row">{items}</div>'


def band_grid_html(bands: list[str]) -> str:
    pills = "".join(f'<div class="band-pill">{b}</div>' for b in bands)
    return f'<div class="band-grid">{pills}</div>'


def stat_cards_html(percentages: Dict[str, float], pixel_counts: Dict[int, int]) -> str:
    cards = ""
    for cid, label in CLASS_LABELS.items():
        pct = percentages.get(label, 0.0)
        cnt = pixel_counts.get(cid, 0)
        color = CLASS_COLORS[cid]
        cards += (
            f'<div class="stat-card">'
            f'<div class="stat-label"><span class="stat-dot" style="background:{color}"></span>{label}</div>'
            f'<div class="stat-value">{pct:.1f}%</div>'
            f'<div class="stat-sub">{cnt:,} px</div>'
            f'</div>'
        )
    return f'<div class="stat-row">{cards}</div>'


# ---------------------------------------------------------------------------
# Raster I/O helpers
# ---------------------------------------------------------------------------

def load_single_band_geotiff(uploaded_file) -> Tuple[np.ndarray, dict]:
    """Read a single-band GeoTIFF from a Streamlit UploadedFile."""
    if not RASTERIO_OK:
        raise RuntimeError("rasterio is not installed.")

    raw = uploaded_file.read()
    buf = io.BytesIO(raw)
    with rasterio.open(buf) as src:
        if src.count < 1:
            raise ValueError(f"{uploaded_file.name}: file has no bands.")
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr[arr == nodata] = np.nan
        profile = src.profile.copy()
        profile["_source_name"] = uploaded_file.name
    return arr, profile


def load_multiband_geotiff(
    uploaded_file,
    band_order: list[str],
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Read a multiband GeoTIFF where bands are in ``band_order`` order.

    Returns a dict {band_name: array} and the profile of the first band.
    """
    if not RASTERIO_OK:
        raise RuntimeError("rasterio is not installed.")

    raw = uploaded_file.read()
    buf = io.BytesIO(raw)
    with rasterio.open(buf) as src:
        n = src.count
        if n < len(band_order):
            raise ValueError(
                f"Multiband GeoTIFF has {n} band(s) but {len(band_order)} are "
                f"required ({', '.join(band_order)})."
            )
        profile = src.profile.copy()
        arrays = {}
        for i, name in enumerate(band_order):
            arr = src.read(i + 1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None and np.isfinite(nodata):
                arr[arr == nodata] = np.nan
            arrays[name] = arr
    return arrays, profile


def align_to_reference(
    arr: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
    name: str,
) -> np.ndarray:
    """
    Reproject ``arr`` to the reference grid defined by ``ref_profile``.
    Uses bilinear resampling for continuous data.
    """
    if not RASTERIO_OK:
        return arr

    rh = ref_profile["height"]
    rw = ref_profile["width"]

    # Check if already aligned
    if (
        arr.shape == (rh, rw)
        and src_profile.get("crs") == ref_profile.get("crs")
        and src_profile.get("transform") == ref_profile.get("transform")
    ):
        return arr

    dest = np.full((rh, rw), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reproject(
            source=arr,
            destination=dest,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            src_nodata=np.nan,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
    return dest


def validate_raster_metadata(profile: dict, name: str, ref_profile: Optional[dict] = None):
    """
    Raise ValueError with a descriptive message if the raster metadata
    is invalid or incompatible with the reference.
    """
    if profile.get("crs") is None:
        raise ValueError(
            f"Input validation failed: {name} has no CRS. "
            "GeoTIFF must be georeferenced."
        )
    if profile.get("transform") is None:
        raise ValueError(
            f"Input validation failed: {name} has no spatial transform."
        )
    if profile.get("width", 0) == 0 or profile.get("height", 0) == 0:
        raise ValueError(
            f"Input validation failed: {name} has zero dimensions."
        )

    if ref_profile is not None:
        ref_crs = ref_profile.get("crs")
        src_crs = profile.get("crs")
        if ref_crs and src_crs and str(ref_crs) != str(src_crs):
            raise ValueError(
                f"Input validation failed: {name} CRS ({src_crs}) does not "
                f"match reference CRS ({ref_crs}). Reproject all bands to the "
                "same CRS before uploading."
            )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_classification_figure(
    classification: np.ndarray,
    title: str = "Land-Cover Classification",
    dpi: int = 150,
) -> plt.Figure:
    """Create a matplotlib figure of the classification raster with legend."""
    display = classification.astype(float)
    display[display == 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0a0f1e")
    ax.set_facecolor("#0a0f1e")

    im = ax.imshow(display, cmap=CMAP, norm=NORM, interpolation="nearest")

    handles = [
        mpatches.Patch(
            facecolor=CLASS_COLORS[cid],
            edgecolor="#ffffff44",
            label=f"{cid}  {label}",
        )
        for cid, label in CLASS_LABELS.items()
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.85,
        facecolor="#111827",
        edgecolor="#38bdf844",
        labelcolor="#e2e8f0",
        fontsize=9,
        title="Land Cover",
        title_fontsize=9,
    )
    leg.get_title().set_color("#94a3b8")

    ax.set_title(title, color="#e2e8f0", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Column (pixels)", color="#64748b", fontsize=8)
    ax.set_ylabel("Row (pixels)",    color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    fig.tight_layout(pad=1.5)
    return fig


def make_rgb_preview(
    band_arrays: Dict[str, np.ndarray],
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Create a true-colour (B04/B03/B02) preview of the uploaded data.
    Returns None if bands are unavailable.
    """
    for b in ("B04", "B03", "B02"):
        if b not in band_arrays:
            return None

    r = band_arrays["B04"].copy()
    g = band_arrays["B03"].copy()
    b_ = band_arrays["B02"].copy()

    def _norm(arr: np.ndarray) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr)
        p2, p98 = np.percentile(finite, [2, 98])
        arr = np.clip(arr, p2, p98)
        rng = p98 - p2
        if rng == 0:
            return np.zeros_like(arr)
        return ((arr - p2) / rng).astype(np.float32)

    rgb = np.stack([_norm(r), _norm(g), _norm(b_)], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0a0f1e")
    ax.set_facecolor("#0a0f1e")
    ax.imshow(rgb, interpolation="bilinear")
    ax.set_title("True Colour Preview (B04 / B03 / B02)", color="#e2e8f0",
                 fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel("Column (pixels)", color="#64748b", fontsize=8)
    ax.set_ylabel("Row (pixels)",    color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    fig.tight_layout(pad=1.5)
    return fig


def make_bar_chart(percentages: Dict[str, float]) -> plt.Figure:
    labels = list(CLASS_LABELS.values())
    values = [percentages.get(lbl, 0.0) for lbl in labels]
    colors = [CLASS_COLORS[cid] for cid in CLASS_LABELS]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#111827")
    ax.set_facecolor("#111827")

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="#1e293b", height=0.6)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9,
            color="#e2e8f0", fontweight="bold",
        )

    ax.set_xlabel("Coverage (%)", color="#64748b", fontsize=9)
    ax.set_xlim(0, max(values) * 1.15 + 2)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#334155")

    ax.set_title("Class Distribution", color="#e2e8f0", fontsize=11,
                 pad=8, fontweight="bold")
    fig.tight_layout(pad=1.2)
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def classification_to_geotiff_bytes(
    classification: np.ndarray,
    profile: dict,
) -> bytes:
    """Serialise a classification array to GeoTIFF bytes in memory."""
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )
    # Remove non-rasterio keys
    out_profile.pop("_source_name", None)

    buf = io.BytesIO()
    with rasterio.open(buf, "w", **out_profile) as dst:
        dst.write(classification.astype(np.uint8), 1)
        dst.set_band_description(1, "Extra Trees land-cover class (1–5)")
    buf.seek(0)
    return buf.read()


def stats_to_csv_bytes(
    percentages: Dict[str, float],
    pixel_counts: Dict[int, int],
) -> bytes:
    rows = []
    for cid, label in CLASS_LABELS.items():
        rows.append({
            "class_id": cid,
            "class_name": label,
            "pixel_count": pixel_counts.get(cid, 0),
            "percentage": percentages.get(label, 0.0),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# ===========================================================================
# SECTION RENDERERS
# ===========================================================================

def render_hero():
    render_html("""
    <div class="hero-container">
        <div class="hero-badge">🛰️ Remote Sensing · Machine Learning · Geospatial</div>
        <div class="hero-title">Sentinel-2 Land Cover<br>Classification</div>
        <div class="hero-subtitle">
            Multispectral machine learning for 5-class land-cover mapping
            using Sentinel-2 spectral bands and spectral indices.
        </div>
        <div class="hero-meta-row">
            <div class="hero-meta-card">
                <div class="hero-meta-label">Study Area</div>
                <div class="hero-meta-value">Ahmedabad–Gandhinagar</div>
            </div>
            <div class="hero-meta-card">
                <div class="hero-meta-label">Satellite</div>
                <div class="hero-meta-value">Sentinel-2</div>
            </div>
            <div class="hero-meta-card">
                <div class="hero-meta-label">Resolution</div>
                <div class="hero-meta-value">10 m</div>
            </div>
            <div class="hero-meta-card">
                <div class="hero-meta-label">Classes</div>
                <div class="hero-meta-value">5</div>
            </div>
            <div class="hero-meta-card">
                <div class="hero-meta-label">Model</div>
                <div class="hero-meta-value">Extra Trees</div>
            </div>
        </div>
    </div>
    """)


def render_model_status():
    try:
        _load_model_cached()
        render_html("""
        <div class="success-box">
            ✅ <strong>Model ready</strong> — Extra Trees candidate model loaded successfully
            from <code>outputs/models/extra_trees_candidate.joblib</code>
        </div>
        """)
        return True
    except Exception as exc:
        render_html(f"""
        <div class="error-box">
            ❌ <strong>Model loading failed:</strong> {exc}<br><br>
            Make sure <code>outputs/models/extra_trees_candidate.joblib</code> exists
            in the project root and that scikit-learn is installed.
        </div>
        """)
        return False


def render_existing_classification():
    """Show the existing project classification if available."""
    render_html('<div class="section-heading">📂 Project Classification Output</div>')
    render_html("""
    <div class="section-sub">
        The trained Extra Trees model has already been applied to the study area.
        The georeferenced output is stored in <code>outputs/classification/</code>.
    </div>
    """)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        if EXISTING_FIGURE.exists():
            st.image(str(EXISTING_FIGURE), caption="Land-Cover Classification — Ahmedabad–Gandhinagar",
                     use_container_width=True)
        elif EXISTING_CLASSIFICATION.exists():
            try:
                with rasterio.open(str(EXISTING_CLASSIFICATION)) as src:
                    arr = src.read(1)
                fig = make_classification_figure(
                    arr,
                    title="Extra Trees Land-Cover — Ahmedabad–Gandhinagar",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                render_html(f'<div class="warn-box">Could not render existing classification: {exc}</div>')
        else:
            render_html('<div class="info-box">No existing classification figure found.</div>')

    with col2:
        render_html(legend_html())

        if EXISTING_CLASSIFICATION.exists():
            try:
                with rasterio.open(str(EXISTING_CLASSIFICATION)) as src:
                    arr = src.read(1)
                    profile = src.profile.copy()

                total = int(np.sum(arr > 0))
                pct = {}
                cnts = {}
                for cid, lbl in CLASS_LABELS.items():
                    cnt = int(np.sum(arr == cid))
                    cnts[cid] = cnt
                    pct[lbl] = round(cnt / total * 100, 2) if total > 0 else 0.0

                render_html(stat_cards_html(pct, cnts))

                # Download existing GeoTIFF
                with open(str(EXISTING_CLASSIFICATION), "rb") as f:
                    tif_bytes = f.read()
                st.download_button(
                    "⬇️ Download Classification GeoTIFF",
                    data=tif_bytes,
                    file_name="land_cover_extra_trees.tif",
                    mime="image/tiff",
                )
            except Exception as exc:
                render_html(f'<div class="warn-box">Statistics unavailable: {exc}</div>')


def render_upload_section():
    render_html('<div class="section-heading">📡 Upload New Sentinel-2 Data</div>')
    render_html("""
    <div class="section-sub">
        Classify your own Sentinel-2 scene using the same trained model.
        Upload either a single multiband GeoTIFF or individual per-band GeoTIFFs.
    </div>
    """)

    render_html("""
    <div class="warn-box">
        ⚠️ <strong>Sentinel-2 spectral input required</strong><br>
        This model requires Sentinel-2 multispectral GeoTIFF data.
        Ordinary RGB photographs (JPG/PNG) are <strong>not compatible</strong>
        with this model and will be rejected.
    </div>
    """)

    render_html('<div class="card-sm" style="margin-bottom:1rem"><b>Required bands:</b>')
    render_html(band_grid_html(REQUIRED_BANDS))
    render_html("""
    <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.5rem">
        NDVI, NDWI, and NDBI are computed automatically — do not upload them.
    </div></div>
    """)

    mode = st.radio(
        "Input format",
        options=["Multiband GeoTIFF (10 bands in order)", "Individual per-band GeoTIFFs"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.session_state["upload_mode"] = mode
    return mode


def handle_multiband_upload() -> Optional[Tuple[Dict[str, np.ndarray], dict]]:
    """UI for single multiband GeoTIFF upload."""
    uploaded = st.file_uploader(
        "Upload multiband GeoTIFF (10 bands: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12)",
        type=["tif", "tiff"],
        key="multiband_upload",
    )
    if uploaded is None:
        render_html("""
        <div class="info-box">
            📌 The GeoTIFF must contain exactly 10 bands in this order:<br>
            <code>B02 · B03 · B04 · B05 · B06 · B07 · B08 · B8A · B11 · B12</code><br>
            All bands must be pre-aligned to the same spatial grid and CRS.
        </div>
        """)
        return None

    try:
        with st.spinner("Reading multiband GeoTIFF…"):
            arrays, profile = load_multiband_geotiff(uploaded, REQUIRED_BANDS)
            validate_raster_metadata(profile, uploaded.name)
        render_html(f"""
        <div class="success-box">
            ✅ Multiband GeoTIFF loaded — {profile['width']} × {profile['height']} px,
            CRS: <code>{profile.get('crs', 'N/A')}</code>
        </div>
        """)
        return arrays, profile
    except Exception as exc:
        render_html(f'<div class="error-box">❌ {exc}</div>')
        return None


def handle_perband_upload() -> Optional[Tuple[Dict[str, np.ndarray], dict]]:
    """UI for individual per-band GeoTIFF uploads."""
    render_html("""
    <div class="info-box">
        📌 Upload one single-band GeoTIFF per required band.
        All bands must share the same CRS and spatial grid (or will be aligned automatically).
    </div>
    """)

    uploaded_files: Dict[str, object] = {}

    cols = st.columns(5)
    for idx, band in enumerate(REQUIRED_BANDS):
        with cols[idx % 5]:
            desc = BAND_DESCRIPTIONS.get(band, "")
            f = st.file_uploader(
                f"**{band}**\n{desc}",
                type=["tif", "tiff"],
                key=f"band_{band}",
            )
            if f is not None:
                uploaded_files[band] = f

    missing_bands = [b for b in REQUIRED_BANDS if b not in uploaded_files]

    if missing_bands:
        render_html(
            f'<div class="info-box">⏳ Waiting for bands: '
            f'{" · ".join(missing_bands)}</div>'
        )
        return None

    # All bands uploaded — load and validate
    try:
        with st.spinner("Loading and validating all bands…"):
            arrays: Dict[str, np.ndarray] = {}
            profiles: Dict[str, dict] = {}

            for band in REQUIRED_BANDS:
                arr, profile = load_single_band_geotiff(uploaded_files[band])
                validate_raster_metadata(profile, band)
                arrays[band] = arr
                profiles[band] = profile

            # Use B02 as reference grid
            ref_profile = profiles["B02"]
            ref_profile["height"] = arrays["B02"].shape[0]
            ref_profile["width"]  = arrays["B02"].shape[1]

            # Align all bands to B02 reference
            for band in REQUIRED_BANDS:
                if band == "B02":
                    continue
                validate_raster_metadata(profiles[band], band, ref_profile)
                arrays[band] = align_to_reference(
                    arrays[band], profiles[band], ref_profile, band
                )

        render_html(f"""
        <div class="success-box">
            ✅ All 10 bands loaded and aligned — grid: {ref_profile['width']} × {ref_profile['height']} px,
            CRS: <code>{ref_profile.get('crs', 'N/A')}</code>
        </div>
        """)
        return arrays, ref_profile

    except Exception as exc:
        render_html(f'<div class="error-box">❌ Input validation failed: {exc}</div>')
        return None


def render_classification_pipeline(
    band_arrays: Dict[str, np.ndarray],
    raster_profile: dict,
):
    """Run the inference pipeline and render results."""
    render_html('<div class="section-heading">⚙️ Classification Processing</div>')

    # Show preview tabs
    tab_input, tab_result, tab_stats = st.tabs([
        "📷 Input Image", "🗺️ Classification", "📊 Statistics"
    ])

    with tab_input:
        rgb_fig = make_rgb_preview(band_arrays)
        if rgb_fig:
            st.pyplot(rgb_fig, use_container_width=True)
            plt.close(rgb_fig)
        else:
            render_html('<div class="info-box">True-colour preview unavailable.</div>')

    # Run classification
    classification_result = st.session_state.get("classification_result")
    classification_profile = st.session_state.get("classification_profile")

    if classification_result is None:
        if st.button("🚀 Run Classification", type="primary"):
            progress_bar = st.progress(0.0, text="Initialising…")
            status_text = st.empty()

            def _progress(frac: float):
                progress_bar.progress(min(frac, 1.0), text=f"Classifying pixels… {frac*100:.0f}%")

            try:
                status_text.markdown("*Loading model…*")
                _load_model_cached()
                status_text.markdown("*Running Extra Trees classification…*")

                classification, pixel_counts, percentages = run_inference(
                    band_arrays,
                    chunk_size=50_000,
                    progress_callback=_progress,
                )

                progress_bar.progress(1.0, text="Complete ✓")
                status_text.empty()

                st.session_state["classification_result"] = classification
                st.session_state["pixel_counts"] = pixel_counts
                st.session_state["percentages"] = percentages
                st.session_state["classification_profile"] = raster_profile
                st.rerun()

            except Exception as exc:
                progress_bar.empty()
                status_text.empty()
                render_html(f'<div class="error-box">❌ Classification failed: {exc}</div>')
        return

    # ── Results are ready ──────────────────────────────────────────────────
    pixel_counts  = st.session_state["pixel_counts"]
    percentages   = st.session_state["percentages"]

    render_html('<div class="success-box">✅ Classification completed successfully.</div>')

    with tab_result:
        cls_fig = make_classification_figure(
            classification_result,
            title="Sentinel-2 Land-Cover Classification (Extra Trees)",
        )
        st.pyplot(cls_fig, use_container_width=True)
        plt.close(cls_fig)
        render_html(legend_html())

    with tab_stats:
        render_html('<div class="section-heading" style="margin-top:0">Class Distribution</div>')
        render_html("""
        <div class="info-box" style="margin-bottom:1rem">
            These statistics describe the <strong>class distribution</strong> of the
            current classification output — not model accuracy.
            Do not interpret these percentages as validation metrics.
        </div>
        """)
        render_html(stat_cards_html(percentages, pixel_counts))

        bar_fig = make_bar_chart(percentages)
        st.pyplot(bar_fig, use_container_width=True)
        plt.close(bar_fig)

        # Statistics table
        rows = [
            {"Class ID": cid, "Class": lbl, "Pixels": pixel_counts.get(cid, 0),
             "Coverage (%)": percentages.get(lbl, 0.0)}
            for cid, lbl in CLASS_LABELS.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────
    render_html('<div class="section-heading">⬇️ Download Results</div>')
    dcol1, dcol2, dcol3, _ = st.columns([1, 1, 1, 2])

    with dcol1:
        try:
            tif_bytes = classification_to_geotiff_bytes(
                classification_result, raster_profile
            )
            st.download_button(
                "🗺️ GeoTIFF",
                data=tif_bytes,
                file_name="sentinelscope_classification.tif",
                mime="image/tiff",
            )
        except Exception as exc:
            render_html(f'<div class="error-box">GeoTIFF export failed: {exc}</div>')

    with dcol2:
        png_bytes = fig_to_png_bytes(
            make_classification_figure(classification_result)
        )
        st.download_button(
            "🖼️ PNG Map",
            data=png_bytes,
            file_name="sentinelscope_classification.png",
            mime="image/png",
        )

    with dcol3:
        csv_bytes = stats_to_csv_bytes(percentages, pixel_counts)
        st.download_button(
            "📄 Statistics CSV",
            data=csv_bytes,
            file_name="sentinelscope_statistics.csv",
            mime="text/csv",
        )

    # ── Reset button ──────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Classify Another Image"):
        for key in [
            "classification_result", "pixel_counts",
            "percentages", "classification_profile",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


def render_methodology():
    with st.expander("📖 How It Works — Methodology", expanded=False):
        col1, col2 = st.columns([1.2, 1])

        with col1:
            render_html("""
            <div class="card">
            <div style="font-size:0.9rem; color:#94a3b8; line-height:2.2">
                <div class="pipeline-row">
                    <div class="pipeline-num">1</div>
                    <div class="pipeline-text">
                        <strong style="color:#e2e8f0">Sentinel-2 Input</strong><br>
                        10 spectral bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
                    </div>
                </div>
                <div class="pipeline-row">
                    <div class="pipeline-num">2</div>
                    <div class="pipeline-text">
                        <strong style="color:#e2e8f0">Spectral Indices</strong><br>
                        NDVI = (B08 − B04) / (B08 + B04)<br>
                        NDWI = (B03 − B08) / (B03 + B08)<br>
                        NDBI = (B11 − B08) / (B11 + B08)
                    </div>
                </div>
                <div class="pipeline-row">
                    <div class="pipeline-num">3</div>
                    <div class="pipeline-text">
                        <strong style="color:#e2e8f0">13-Feature Matrix</strong><br>
                        10 bands + 3 indices → pixel feature vector
                    </div>
                </div>
                <div class="pipeline-row">
                    <div class="pipeline-num">4</div>
                    <div class="pipeline-text">
                        <strong style="color:#e2e8f0">Extra Trees Classifier</strong><br>
                        Pre-trained on 33 Ahmedabad–Gandhinagar samples.
                        No retraining occurs at inference time.
                    </div>
                </div>
                <div class="pipeline-row">
                    <div class="pipeline-num">5</div>
                    <div class="pipeline-text">
                        <strong style="color:#e2e8f0">5-Class Land-Cover Map</strong><br>
                        Vegetation · Built-up · Bare Soil · Water · Road
                    </div>
                </div>
            </div>
            </div>
            """)

        with col2:
            render_html("""
            <div class="card">
            <div style="font-size:0.85rem; color:#94a3b8">
                <p><strong style="color:#e2e8f0">Model</strong><br>
                Extra Trees Classifier (scikit-learn)<br>
                300 trees · balanced class weights · sqrt max_features</p>

                <p><strong style="color:#e2e8f0">Training data</strong><br>
                33 valid labelled point samples across 5 land-cover classes.<br>
                Collected in the Ahmedabad–Gandhinagar study area.</p>

                <p><strong style="color:#e2e8f0">Spatial validation</strong><br>
                Mean accuracy across 3 evaluated spatial folds: <strong style="color:#38bdf8">78.7%</strong><br>
                Macro F1: <strong style="color:#38bdf8">0.44</strong><br>
                (1 fold skipped — water class absent from training partition)</p>

                <p><strong style="color:#e2e8f0">CRS</strong><br>
                Reference: EPSG:32642 (UTM Zone 42N)</p>

                <p><strong style="color:#e2e8f0">Inference</strong><br>
                The model file is loaded from<br>
                <code>outputs/models/extra_trees_candidate.joblib</code>.
                It is never retrained during inference.</p>
            </div>
            </div>
            """)


def render_disclaimer():
    render_html("""
    <div class="disclaimer">
        ⚠️ <strong>Scientific disclaimer:</strong> This is an experimental land-cover
        classification workflow. Results depend on input quality, spatial distribution,
        acquisition conditions, and the limitations of the training dataset (33 samples,
        small geographic area). Predictions should not be treated as authoritative
        land-use/land-cover maps. Model accuracy metrics shown reflect cross-validation
        on the original training data — not accuracy on user-uploaded images.
    </div>
    """)


# ===========================================================================
# MAIN APP
# ===========================================================================

def main():
    # ── Hero ─────────────────────────────────────────────────────────────
    render_hero()

    # ── Navigation anchors ────────────────────────────────────────────────
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])
    with nav_col1:
        st.markdown("[📊 View Results](#project-classification-output)", unsafe_allow_html=True)
    with nav_col2:
        st.markdown("[📖 Methodology](#how-it-works)", unsafe_allow_html=True)

    st.markdown("---")

    # ── Model status ──────────────────────────────────────────────────────
    model_ok = render_model_status()

    if not RASTERIO_OK:
        render_html("""
        <div class="error-box">
            ❌ <strong>rasterio is not installed.</strong><br>
            Install it with: <code>pip install rasterio</code>
        </div>
        """)

    st.markdown("---")

    # ── Existing project output ────────────────────────────────────────────
    render_existing_classification()

    st.markdown("---")

    # ── Upload / classify new data ─────────────────────────────────────────
    if model_ok and RASTERIO_OK:
        mode = render_upload_section()

        band_arrays_and_profile = None

        if mode == "Multiband GeoTIFF (10 bands in order)":
            result = handle_multiband_upload()
            if result:
                band_arrays_and_profile = result
        else:
            result = handle_perband_upload()
            if result:
                band_arrays_and_profile = result

        if band_arrays_and_profile is not None:
            band_arrays, raster_profile = band_arrays_and_profile
            st.markdown("---")
            render_classification_pipeline(band_arrays, raster_profile)
    else:
        render_html("""
        <div class="info-box">
            New-image classification requires both the model and rasterio to be available.
        </div>
        """)

    st.markdown("---")

    # ── Methodology ───────────────────────────────────────────────────────
    render_methodology()

    # ── Disclaimer ────────────────────────────────────────────────────────
    render_disclaimer()

    # ── Footer ────────────────────────────────────────────────────────────
    render_html("""
    <div style="text-align:center; margin-top:2.5rem; padding-top:1rem;
                border-top:1px solid #1e293b; font-size:0.75rem; color:#475569">
        SentinelScope · Sentinel-2 Land Cover Intelligence ·
        Built on the Ahmedabad–Gandhinagar research pipeline ·
        Extra Trees · 5-class land cover
    </div>
    """)


if __name__ == "__main__":
    main()

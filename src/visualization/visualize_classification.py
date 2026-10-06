
from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import CLASSIFICATION_FILE, FIGURES_DIR

CLASSIFICATION_PATH = CLASSIFICATION_FILE
OUTPUT_DIR = FIGURES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "land_cover_classification.png"

# --------------------------------------------------
# 2. Land-cover class definitions
# --------------------------------------------------

CLASS_NAMES = {
    1: "Vegetation",
    2: "Built-up",
    3: "Bare soil",
    4: "Water",
    5: "Road",
}

# Colors follow the class IDs above.
CLASS_COLORS = [
    "#228B22",  # 1 - Vegetation
    "#E53935",  # 2 - Built-up
    "#D8B365",  # 3 - Bare soil
    "#2196F3",  # 4 - Water
    "#424242",  # 5 - Road
]

cmap = ListedColormap(CLASS_COLORS)
norm = BoundaryNorm(
    boundaries=np.arange(0.5, 6.5, 1),
    ncolors=len(CLASS_COLORS),
)

# --------------------------------------------------
# 3. Read classification raster
# --------------------------------------------------

if not CLASSIFICATION_PATH.exists():
    raise FileNotFoundError(
        f"Classification raster not found: {CLASSIFICATION_PATH}\n"
        "Run predict_land_cover.py first."
    )

with rasterio.open(CLASSIFICATION_PATH) as src:
    classification = src.read(1, masked=True)
    classification = np.ma.masked_invalid(
        classification.astype("float32").filled(np.nan)
    )

    raster_crs = src.crs
    raster_shape = classification.shape
    raster_bounds = src.bounds

print("Classification raster loaded.")
print("Shape:", raster_shape)
print("CRS:", raster_crs)
print("Bounds:", raster_bounds)

# --------------------------------------------------
# 4. Calculate class pixel counts
# --------------------------------------------------

valid_data = classification.compressed()
valid_data = valid_data[np.isfinite(valid_data)]

print("\nPredicted class distribution:")

for class_id, class_name in CLASS_NAMES.items():
    count = int(np.count_nonzero(valid_data == class_id))
    percentage = (
        count / len(valid_data) * 100
        if len(valid_data) > 0
        else 0
    )
    print(
        f"{class_name}: {count:,} pixels ({percentage:.2f}%)"
    )

# --------------------------------------------------
# 5. Visualize classification
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 9))

image = ax.imshow(
    classification,
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
)

legend_handles = [
    Patch(
        facecolor=CLASS_COLORS[class_id - 1],
        edgecolor="black",
        label=f"{class_id} — {class_name}",
    )
    for class_id, class_name in CLASS_NAMES.items()
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    title="Land-cover classes",
)

ax.set_title(
    "Baseline Land-Cover Classification\n"
    "Sentinel-2 — Ahmedabad/Gandhinagar"
)
ax.set_xlabel("Raster column")
ax.set_ylabel("Raster row")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\nSaved classification visualization:", OUTPUT_PATH)

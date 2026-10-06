
from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import INDICES_DIR, FIGURES_DIR

INDEX_DIR = INDICES_DIR
OUTPUT_DIR = FIGURES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

indices = [
    ("ndvi.tif", "NDVI — Vegetation"),
    ("ndwi.tif", "NDWI — Potential Water"),
    ("ndbi.tif", "NDBI — Built-up Indicator"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, (filename, title) in zip(axes, indices):
    path = INDEX_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Index raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float32")
        data = np.ma.masked_invalid(data.filled(np.nan))

    cmap = "RdYlGn" if filename == "ndvi.tif" else "RdYlBu_r"

    image = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Sentinel-2 Spectral Indices — Ahmedabad/Gandhinagar")
fig.tight_layout()

output_path = OUTPUT_DIR / "spectral_indices_preview.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
plt.show()

print("Saved visualization:", output_path)

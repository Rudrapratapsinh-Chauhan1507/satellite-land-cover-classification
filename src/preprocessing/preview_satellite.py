from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR, FIGURES_DIR

DATA_DIR = RAW_DATA_DIR
OUTPUT_DIR = FIGURES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel-2 bands for true-colour RGB
band_files = {
    "red": "T42QZL_20261003T053651_B04_10m.jp2",
    "green": "T42QZL_20261003T053651_B03_10m.jp2",
    "blue": "T42QZL_20261003T053651_B02_10m.jp2",
}


def read_preview(path, width=1500):
    """Read a smaller version of a raster for quick visualization."""
    with rasterio.open(path) as src:
        height = round(src.height * width / src.width)

        image = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
        ).astype(np.float32)

        print(f"{path.name}: {src.width} x {src.height}")
        return image


def stretch_band(image):
    """Apply percentile contrast stretching for display."""
    valid = np.isfinite(image) & (image > 0)
    values = image[valid]

    if values.size == 0:
        raise ValueError("No valid pixels found in a band.")

    low, high = np.percentile(values, [2, 98])

    if high <= low:
        raise ValueError("Band has insufficient contrast.")

    stretched = np.clip((image - low) / (high - low), 0, 1)
    stretched[~valid] = 0

    return stretched


print("Reading Sentinel-2 bands...")

red = read_preview(DATA_DIR / band_files["red"])
green = read_preview(DATA_DIR / band_files["green"])
blue = read_preview(DATA_DIR / band_files["blue"])

# Stack RGB channels
rgb = np.dstack([
    stretch_band(red),
    stretch_band(green),
    stretch_band(blue),
])

# Display and save
plt.figure(figsize=(12, 10))
plt.imshow(rgb)
plt.title("Sentinel-2 True-Colour Preview — Ahmedabad/Gandhinagar")
plt.axis("off")
plt.tight_layout()

output_path = OUTPUT_DIR / "true_color_preview.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPreview saved to: {output_path}")
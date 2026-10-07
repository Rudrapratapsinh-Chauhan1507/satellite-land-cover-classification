from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import CLASSIFICATION_FILE, FIGURES_DIR

OUTPUT_PATH = FIGURES_DIR / "land_cover_classification_final.png"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    1: "Vegetation",
    2: "Built-up",
    3: "Bare soil",
    4: "Water",
    5: "Road",
}

CLASS_COLORS = [
    "#228B22",
    "#E53935",
    "#D8B365",
    "#2196F3",
    "#424242",
]

cmap = ListedColormap(CLASS_COLORS)
cmap.set_bad("#FFFFFF")

norm = BoundaryNorm(
    np.arange(0.5, 6.5, 1),
    cmap.N,
)

with rasterio.open(CLASSIFICATION_FILE) as src:
    land_cover = src.read(1, masked=True)
    transform = src.transform
    bounds = src.bounds
    crs = src.crs
    resolution = src.res
    nodata = src.nodata

if crs is None:
    raise ValueError("The classification raster has no CRS.")

if not np.isclose(abs(resolution[0]), abs(resolution[1])):
    raise ValueError(
        "This scale-bar implementation requires square raster pixels."
    )

valid = land_cover.compressed()
valid = valid[np.isin(valid, list(CLASS_NAMES))]

if valid.size == 0:
    raise ValueError("No valid land-cover pixels found.")

pixel_area_m2 = abs(transform.a * transform.e)
total_valid_area_km2 = valid.size * pixel_area_m2 / 1_000_000

print("Land-cover class statistics")
print(f"CRS: {crs}")
print(f"Raster resolution: {resolution}")
print(f"Valid pixels: {valid.size:,}")
print(f"Total classified area: {total_valid_area_km2:.3f} km²")
print(f"NoData value: {nodata}")

for class_id, class_name in CLASS_NAMES.items():
    count = int(np.count_nonzero(valid == class_id))
    area_km2 = count * pixel_area_m2 / 1_000_000
    percentage = count / valid.size * 100

    print(
        f"{class_name}: {count:,} pixels | "
        f"{area_km2:.3f} km² | {percentage:.2f}%"
    )

fig, ax = plt.subplots(figsize=(13, 9))

image = ax.imshow(
    land_cover,
    cmap=cmap,
    norm=norm,
    extent=(
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ),
    interpolation="nearest",
    origin="upper",
)

ax.set_title(
    "Land-Cover Classification",
    fontsize=17,
    fontweight="bold",
    pad=18,
)

ax.text(
    0.5,
    1.005,
    "Sentinel-2 | Ahmedabad–Gandhinagar, Gujarat",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
)

ax.set_xlabel(f"Easting ({crs.to_string()})", fontsize=10)
ax.set_ylabel("Northing (m)", fontsize=10)
ax.ticklabel_format(
    style="plain",
    axis="both",
    useOffset=False,
)
ax.tick_params(axis="both", labelsize=8)
ax.grid(
    visible=True,
    linestyle=":",
    linewidth=0.5,
    alpha=0.45,
)

legend_handles = []

for class_id, class_name in CLASS_NAMES.items():
    count = int(np.count_nonzero(valid == class_id))
    percentage = count / valid.size * 100

    legend_handles.append(
        Patch(
            facecolor=CLASS_COLORS[class_id - 1],
            edgecolor="black",
            label=f"{class_name} ({percentage:.1f}%)",
        )
    )

ax.legend(
    handles=legend_handles,
    title="Land-cover classes",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    fontsize=9,
    title_fontsize=10,
)

# North arrow
ax.annotate(
    "N",
    xy=(0.93, 0.89),
    xytext=(0.93, 0.77),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    arrowprops={
        "arrowstyle": "-|>",
        "linewidth": 1.8,
        "color": "black",
    },
)

# Scale bar: 1 km, drawn in projected map coordinates.
scale_length_m = 1000
scale_x = bounds.left + (bounds.right - bounds.left) * 0.06
scale_y = bounds.bottom + (bounds.top - bounds.bottom) * 0.07
bar_height_m = abs(resolution[1]) * 2

ax.plot(
    [scale_x, scale_x + scale_length_m],
    [scale_y, scale_y],
    color="black",
    linewidth=3,
    solid_capstyle="butt",
)

ax.plot(
    [scale_x, scale_x],
    [scale_y - bar_height_m, scale_y + bar_height_m],
    color="black",
    linewidth=1.5,
)

ax.plot(
    [scale_x + scale_length_m, scale_x + scale_length_m],
    [scale_y - bar_height_m, scale_y + bar_height_m],
    color="black",
    linewidth=1.5,
)

ax.text(
    scale_x + scale_length_m / 2,
    scale_y + bar_height_m * 2.5,
    "1 km",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
)

ax.set_aspect("equal", adjustable="box")

fig.text(
    0.01,
    0.015,
    "Classification is model-generated. NoData pixels are excluded "
    "from class statistics; map accuracy has not been independently verified.",
    fontsize=8,
    color="#444444",
)

fig.tight_layout(rect=(0, 0.045, 0.80, 0.97))
fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print(f"\nFinal map saved to: {OUTPUT_PATH}")
print("The baseline raster and existing maps were not modified.")
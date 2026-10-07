
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

from src.utils.paths import TRAINING_SAMPLES_FILE


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

N_SPATIAL_GROUPS = 4
RANDOM_STATE = 42

FIGURE_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_FILE = FIGURE_DIR / "training_point_spatial_distribution.png"
CSV_FILE = REPORT_DIR / "training_point_group_assignments.csv"

CLASS_ORDER = [
    "bare_soil",
    "built_up",
    "road",
    "vegetation",
    "water",
]

CLASS_COLORS = {
    "bare_soil": "#C49A6C",
    "built_up": "#D62728",
    "road": "#4C78A8",
    "vegetation": "#2CA02C",
    "water": "#17BECF",
}

GROUP_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
}


# --------------------------------------------------
# 2. Load training points
# --------------------------------------------------

points = gpd.read_file(
    TRAINING_SAMPLES_FILE,
    layer="training_samples",
)

if points.empty:
    raise ValueError("The training GeoPackage contains no points.")

if points.crs is None:
    raise ValueError("Training points have no defined CRS.")

required_columns = {"class_name", "geometry"}
missing_columns = required_columns - set(points.columns)

if missing_columns:
    raise ValueError(
        f"Training points are missing columns: {missing_columns}"
    )

if points.geometry.is_empty.any() or points.geometry.isna().any():
    raise ValueError("Some training points have missing geometry.")

if not points.geometry.geom_type.eq("Point").all():
    raise ValueError("All training geometries must be Point features.")

unknown_classes = set(points["class_name"]) - set(CLASS_ORDER)

if unknown_classes:
    raise ValueError(f"Unexpected class labels: {unknown_classes}")

print(f"Loaded training points: {len(points)}")
print("\nClass counts:")
print(points["class_name"].value_counts().reindex(CLASS_ORDER, fill_value=0))


# --------------------------------------------------
# 3. Recreate the four spatial groups
# --------------------------------------------------

projected = points.to_crs("EPSG:32642")

coordinates = pd.DataFrame({
    "easting_m": projected.geometry.x,
    "northing_m": projected.geometry.y,
})

number_of_groups = min(N_SPATIAL_GROUPS, len(points))

clusterer = KMeans(
    n_clusters=number_of_groups,
    random_state=RANDOM_STATE,
    n_init=20,
)

group_ids = clusterer.fit_predict(coordinates)

# Use a copy so the source GeoPackage is never modified.
plot_data = projected.copy()
plot_data["spatial_group"] = group_ids
plot_data["easting_m"] = coordinates["easting_m"].to_numpy()
plot_data["northing_m"] = coordinates["northing_m"].to_numpy()

# Export point-level assignments for future analysis.
assignments = pd.DataFrame({
    "point_id": range(1, len(plot_data) + 1),
    "class_name": plot_data["class_name"].to_numpy(),
    "spatial_group": group_ids,
    "easting_m": coordinates["easting_m"].to_numpy(),
    "northing_m": coordinates["northing_m"].to_numpy(),
})

assignments.to_csv(CSV_FILE, index=False)


# --------------------------------------------------
# 4. Build class-by-group count table
# --------------------------------------------------

class_group_counts = pd.crosstab(
    plot_data["class_name"],
    plot_data["spatial_group"],
).reindex(
    index=CLASS_ORDER,
    columns=range(number_of_groups),
    fill_value=0,
)

class_group_counts.columns = [
    f"Group {group_id}" for group_id in class_group_counts.columns
]

class_group_counts["Total"] = class_group_counts.sum(axis=1)

print("\nClass distribution by spatial group:")
print(class_group_counts.to_string())

print("\nPoints per spatial group:")
print(plot_data["spatial_group"].value_counts().sort_index())


# --------------------------------------------------
# 5. Plot map and class-by-group table
# --------------------------------------------------

fig, (ax_map, ax_table) = plt.subplots(
    1,
    2,
    figsize=(15, 8),
    gridspec_kw={"width_ratios": [1.7, 1]},
)

# Plot points: color represents class; marker represents group.
for class_name in CLASS_ORDER:
    for group_id in range(number_of_groups):
        subset = plot_data[
            (plot_data["class_name"] == class_name)
            & (plot_data["spatial_group"] == group_id)
        ]

        if subset.empty:
            continue

        ax_map.scatter(
            subset.geometry.x / 1000,
            subset.geometry.y / 1000,
            color=CLASS_COLORS[class_name],
            marker=GROUP_MARKERS.get(group_id, "o"),
            s=110,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9,
        )

ax_map.set_title(
    "Training Points and Spatial Groups",
    fontsize=14,
    fontweight="bold",
)

ax_map.set_xlabel("Easting (km, EPSG:32642)")
ax_map.set_ylabel("Northing (km, EPSG:32642)")
ax_map.grid(True, linestyle="--", alpha=0.3)
ax_map.set_aspect("equal", adjustable="datalim")

# Separate legend for class colors.
class_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=CLASS_COLORS[name],
        markeredgecolor="black",
        markersize=9,
        label=name.replace("_", " ").title(),
    )
    for name in CLASS_ORDER
]

# Separate legend for spatial-group marker shapes.
group_handles = [
    Line2D(
        [0],
        [0],
        marker=GROUP_MARKERS[group_id],
        linestyle="None",
        color="black",
        markerfacecolor="white",
        markersize=9,
        label=f"Group {group_id}",
    )
    for group_id in range(number_of_groups)
]

class_legend = ax_map.legend(
    handles=class_handles,
    title="Land-cover class",
    loc="upper left",
    fontsize=8,
)
ax_map.add_artist(class_legend)

ax_map.legend(
    handles=group_handles,
    title="Spatial group",
    loc="lower right",
    fontsize=8,
)

# Draw the count table.
ax_table.axis("off")
ax_table.set_title(
    "Class Distribution by Spatial Group",
    fontsize=12,
    fontweight="bold",
    pad=18,
)

table_values = class_group_counts.astype(int).values.tolist()
row_labels = [
    name.replace("_", " ").title() for name in class_group_counts.index
]
column_labels = list(class_group_counts.columns)

table = ax_table.table(
    cellText=table_values,
    rowLabels=row_labels,
    colLabels=column_labels,
    cellLoc="center",
    rowLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.05, 2.0)

ax_table.text(
    0.5,
    0.13,
    (
        "Marker shape = spatial group\n"
        "Point color = land-cover class\n\n"
        "Groups are created by KMeans on projected coordinates.\n"
        "They are not administrative or land-cover boundaries."
    ),
    transform=ax_table.transAxes,
    ha="center",
    va="center",
    fontsize=9,
)

fig.suptitle(
    f"Spatial Coverage of {len(plot_data)} Training Samples",
    fontsize=16,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURE_FILE, dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved distribution map: {FIGURE_FILE}")
print(f"Saved point assignments: {CSV_FILE}")

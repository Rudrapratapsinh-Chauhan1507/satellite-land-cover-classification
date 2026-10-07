
from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import CLASSIFICATION_FILE, FIGURES_DIR

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAP_PATH = FIGURES_DIR / "land_cover_classification_final.png"
PDF_PATH = REPORT_DIR / "land_cover_final_report.pdf"
CV_PATH = REPORT_DIR / "spatial_cv_fold_metrics.csv"

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

if not CLASSIFICATION_FILE.exists():
    raise FileNotFoundError(
        f"Baseline raster not found: {CLASSIFICATION_FILE}"
    )

if not MAP_PATH.exists():
    raise FileNotFoundError(
        f"Final map not found: {MAP_PATH}. "
        "Run create_final_map.py first."
    )

with rasterio.open(CLASSIFICATION_FILE) as src:
    raster = src.read(1, masked=True)
    crs = src.crs
    resolution = src.res
    width, height = src.width, src.height
    bounds = src.bounds
    nodata = src.nodata
    transform = src.transform

valid = raster.compressed()
valid = valid[np.isin(valid, list(CLASS_NAMES))]

if valid.size == 0:
    raise ValueError("No valid classification pixels found.")

pixel_area_m2 = abs(transform.a * transform.e)
total_area_km2 = valid.size * pixel_area_m2 / 1_000_000

class_stats = []

for class_id, name in CLASS_NAMES.items():
    count = int(np.count_nonzero(valid == class_id))
    area = count * pixel_area_m2 / 1_000_000
    percentage = count / valid.size * 100

    class_stats.append(
        (class_id, name, count, area, percentage)
    )

# Read spatial validation results if available.
cv_rows = []

if CV_PATH.exists():
    import csv

    with CV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        cv_rows = list(csv.DictReader(file))

evaluated = [
    row for row in cv_rows
    if row.get("status") == "evaluated"
]

mean_accuracy = (
    np.mean([float(row["accuracy"]) for row in evaluated])
    if evaluated else None
)

mean_macro_f1 = (
    np.mean([float(row["macro_f1"]) for row in evaluated])
    if evaluated else None
)

with PdfPages(PDF_PATH) as pdf:

    # PAGE 1: Final map
    fig = plt.figure(figsize=(11.7, 8.3))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[3.5, 1.25],
        left=0.06, right=0.96,
        top=0.84, bottom=0.12, wspace=0.12
    )

    ax_map = fig.add_subplot(grid[0, 0])
    ax_legend = fig.add_subplot(grid[0, 1])
    ax_legend.axis("off")

    with Image.open(MAP_PATH) as img:
        ax_map.imshow(img.convert("RGB"))

    ax_map.axis("off")

    fig.suptitle(
        "Satellite Land-Cover Classification",
        fontsize=19, fontweight="bold", y=0.95
    )

    fig.text(
        0.5, 0.90,
        "Sentinel-2 | Ahmedabad–Gandhinagar, Gujarat, India",
        ha="center", fontsize=11
    )

    handles = [
        Patch(
            facecolor=CLASS_COLORS[class_id - 1],
            edgecolor="black",
            label=name
        )
        for class_id, name, *_ in class_stats
    ]

    ax_legend.legend(
        handles=handles,
        title="Land-cover classes",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=11
    )

    stats_text = (
        f"\nRaster details\n\n"
        f"CRS: {crs}\n"
        f"Resolution: {resolution[0]:g} m\n"
        f"Dimensions: {width} × {height}\n"
        f"Valid pixels: {valid.size:,}\n"
        f"Classified area: {total_area_km2:.3f} km²"
    )

    ax_legend.text(
        0, 0.63, stats_text,
        transform=ax_legend.transAxes,
        fontsize=9, va="top",
        linespacing=1.5
    )

    fig.text(
        0.06, 0.055,
        "Model-generated classification; independent map accuracy "
        "has not been established.",
        fontsize=8, color="#444444"
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 2: Class-area statistics
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    fig.suptitle(
        "Predicted Land-Cover Distribution",
        fontsize=17, fontweight="bold", y=0.95
    )

    names = [row[1] for row in class_stats]
    percentages = [row[4] for row in class_stats]
    colors = CLASS_COLORS

    bars = ax.barh(
        names[::-1],
        percentages[::-1],
        color=colors[::-1],
        edgecolor="black"
    )

    ax.set_xlabel("Share of valid classified pixels (%)")
    ax.set_xlim(0, max(percentages) * 1.25)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, row in zip(bars, class_stats[::-1]):
        _, name, count, area, percentage = row
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.2f}% | {area:.3f} km² | {count:,} px",
            va="center", fontsize=9
        )

    fig.text(
        0.08, 0.06,
        f"Total classified area: {total_area_km2:.3f} km². "
        "Areas are derived from valid classified pixels, not "
        "independent ground-truth measurements.",
        fontsize=9
    )

    fig.tight_layout(rect=(0.04, 0.12, 0.96, 0.90))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 3: Validation and limitations
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")

    fig.suptitle(
        "Model Evaluation and Limitations",
        fontsize=17, fontweight="bold", y=0.95
    )

    lines = [
        "Spatial cross-validation",
        "",
        "Four spatial folds were attempted. Results below reflect "
        "the latest saved spatial-validation CSV.",
        "",
    ]

    if cv_rows:
        for row in cv_rows:
            fold = row.get("fold", "?")
            status = row.get("status", "unknown")
            n_train = row.get("training_points", "?")
            n_val = row.get("validation_points", "?")

            if status == "evaluated":
                lines.append(
                    f"Fold {fold}: evaluated; training={n_train}, "
                    f"validation={n_val}, "
                    f"accuracy={float(row['accuracy']):.4f}, "
                    f"macro F1={float(row['macro_f1']):.4f}"
                )
            else:
                missing = row.get("missing_training_classes", "")
                lines.append(
                    f"Fold {fold}: skipped; training={n_train}, "
                    f"validation={n_val}; missing training classes: "
                    f"{missing or 'not specified'}"
                )

        lines.extend([
            "",
            f"Evaluated folds: {len(evaluated)} of {len(cv_rows)}",
        ])

        if mean_accuracy is not None:
            lines.append(
                f"Mean accuracy over evaluated folds: "
                f"{mean_accuracy:.4f} ({mean_accuracy * 100:.2f}%)"
            )

        if mean_macro_f1 is not None:
            lines.append(
                f"Mean macro F1 over evaluated folds: "
                f"{mean_macro_f1:.4f}"
            )
    else:
        lines.append(
            "Spatial validation CSV was not found. "
            "No spatial-validation metrics are reported."
        )

    lines.extend([
        "",
        "Interpretation",
        "",
        "• The training dataset contains 39 labelled samples and "
        "13 predictor features across five classes.",
        "• Spatial groups have an uneven distribution of classes.",
        "• All six water samples fall within one spatial group, "
        "which can prevent a split from training on the water class.",
        "• Some validation folds lack classes, so fold metrics do "
        "not represent a complete four-fold evaluation.",
        "• These spatial results are exploratory and are not "
        "independent estimates of whole-map accuracy.",
        "• The class-area statistics describe model predictions, "
        "not independently verified land cover.",
        "",
        "Data and map metadata",
        "",
        f"CRS: {crs}",
        f"Raster dimensions: {width} × {height}",
        f"Pixel resolution: {resolution[0]:g} × "
        f"{resolution[1]:g} m",
        f"NoData value: {nodata}",
        f"Valid classified pixels: {valid.size:,}",
        "",
        "Recommended next steps",
        "",
        "Collect independently verified reference samples, improve "
        "spatial coverage across classes, and compare models using "
        "consistent validation splits before making accuracy claims.",
    ])

    ax.text(
        0.07, 0.88,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        linespacing=1.35,
        wrap=True
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print("Final report created:", PDF_PATH)
print("Pages: map, class-area statistics, evaluation and limitations")
print("Existing PDF and baseline raster were preserved.")

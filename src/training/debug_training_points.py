from pathlib import Path
import sys
from collections import Counter

import geopandas as gpd
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    AOI_DIR,
    TRAINING_SAMPLES_FILE,
    PROCESSED_DATA_DIR,
    INDICES_DIR,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

preferred_file = TRAINING_SAMPLES_FILE

if preferred_file.exists():
    POINTS_PATH = preferred_file
else:
    candidates = sorted(
        AOI_DIR.glob("*training*.gpkg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No training GeoPackage found in {AOI_DIR}"
        )
    POINTS_PATH = candidates[0]

RASTER_DIR = PROCESSED_DATA_DIR
INDEX_DIR = INDICES_DIR

print("=" * 65)
print("TRAINING POINT DIAGNOSTICS")
print("=" * 65)
print("Training file:", POINTS_PATH)

# --------------------------------------------------
# 2. Load and validate training points
# --------------------------------------------------

points = gpd.read_file(POINTS_PATH, layer="training_samples")

if points.crs is None:
    raise ValueError("Training layer has no CRS.")

required = {"class_name", "class_id", "geometry"}
missing = required - set(points.columns)

if missing:
    raise ValueError(f"Missing required fields: {missing}")

points = points.reset_index(drop=True)
points["class_name"] = (
    points["class_name"].astype(str).str.strip().str.lower()
)

print("\nPoint CRS:", points.crs)
print("Total points:", len(points))

print("\nClass counts:")
print(points["class_name"].value_counts(dropna=False))

# --------------------------------------------------
# 3. Find input rasters
# --------------------------------------------------

def find_raster(pattern):
    matches = list(RASTER_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No raster matching {pattern!r} in {RASTER_DIR}"
        )
    return matches[0]


b02_path = find_raster("*B02*10m_clipped.tif")
scl_path = find_raster("*SCL*clipped.tif")

print("\nB02 raster:", b02_path.name)
print("SCL raster:", scl_path.name)

# --------------------------------------------------
# 4. Inspect points against B02
# --------------------------------------------------

print("\n" + "=" * 65)
print("B02 RASTER CHECK")
print("=" * 65)

with rasterio.open(b02_path) as src:
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)
    print("Raster dimensions:", src.width, "x", src.height)
    print("Raster NoData:", src.nodata)

    points_b02 = points.to_crs(src.crs)

    print("\nB02 value at each point:")

    for idx, row in points_b02.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            print(f"Feature {idx}: EMPTY GEOMETRY")
            continue

        x, y = geom.x, geom.y

        inside = (
            src.bounds.left <= x <= src.bounds.right
            and src.bounds.bottom <= y <= src.bounds.top
        )

        try:
            sample = next(src.sample([(x, y)], masked=True))
            value = sample[0]

            if np.ma.is_masked(value):
                value_text = "MASKED"
            else:
                value_text = str(float(value))

            print(
                f"Feature {idx}: "
                f"class={row['class_name']}, "
                f"class_id={row['class_id']}, "
                f"x={x:.2f}, y={y:.2f}, "
                f"B02={value_text}, "
                f"inside={inside}"
            )

        except Exception as exc:
            print(f"Feature {idx}: sampling error: {exc}")

# --------------------------------------------------
# 5. Inspect SCL classification at each point
# --------------------------------------------------

print("\n" + "=" * 65)
print("SCL CLASS CHECK")
print("=" * 65)

with rasterio.open(scl_path) as src:
    print("SCL CRS:", src.crs)
    print("SCL bounds:", src.bounds)
    print("SCL NoData:", src.nodata)

    points_scl = points.to_crs(src.crs)

    for idx, row in points_scl.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        x, y = geom.x, geom.y

        inside = (
            src.bounds.left <= x <= src.bounds.right
            and src.bounds.bottom <= y <= src.bounds.top
        )

        try:
            sample = next(src.sample([(x, y)], masked=True))
            value = sample[0]

            if np.ma.is_masked(value):
                scl_text = "MASKED"
            else:
                scl_text = str(int(value))

            print(
                f"Feature {idx}: "
                f"class={row['class_name']}, "
                f"class_id={row['class_id']}, "
                f"SCL={scl_text}, "
                f"inside={inside}"
            )

        except Exception as exc:
            print(f"Feature {idx}: sampling error: {exc}")

# --------------------------------------------------
# 6. Check spectral-index rasters
# --------------------------------------------------

print("\n" + "=" * 65)
print("SPECTRAL INDEX CHECK")
print("=" * 65)

for index_name in ["ndvi", "ndwi", "ndbi"]:
    index_path = INDEX_DIR / f"{index_name}.tif"

    if not index_path.exists():
        print(f"{index_name.upper()}: FILE NOT FOUND")
        continue

    with rasterio.open(index_path) as src:
        print(
            f"{index_name.upper()}: "
            f"CRS={src.crs}, "
            f"size={src.width}x{src.height}, "
            f"NoData={src.nodata}"
        )

        points_index = points.to_crs(src.crs)

        water_points = points_index[
            points_index["class_name"] == "water"
        ]

        print(f"  Water points checked: {len(water_points)}")

        for idx, row in water_points.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                print(f"  Feature {idx}: EMPTY GEOMETRY")
                continue

            x, y = geom.x, geom.y

            try:
                sample = next(src.sample([(x, y)], masked=True))
                value = sample[0]

                if np.ma.is_masked(value):
                    value_text = "MASKED"
                else:
                    value_text = str(float(value))

                print(
                    f"  Feature {idx}: "
                    f"{index_name.upper()}={value_text}"
                )

            except Exception as exc:
                print(f"  Feature {idx}: error: {exc}")

print("\n" + "=" * 65)
print("DIAGNOSTICS FINISHED")
print("=" * 65)
print("No files were modified.")
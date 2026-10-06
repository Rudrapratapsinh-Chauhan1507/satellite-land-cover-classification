
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import PROCESSED_DATA_DIR


REFERENCE = (
    PROCESSED_DATA_DIR
    / "T42QZL_20261003T053651_B02_10m_clipped.tif"
)

BANDS_TO_ALIGN = ["B05", "B06", "B07", "B8A", "B11", "B12"]

OUTPUT_DIR = PROCESSED_DATA_DIR / "aligned_10m"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE.exists():
        raise FileNotFoundError(
            f"Reference raster not found: {REFERENCE}"
        )

    # Use B02 as the reference grid.
    with rasterio.open(REFERENCE) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    print("Reference grid:", ref_width, "x", ref_height)
    print("Reference CRS:", ref_crs)
    print("Reference resolution:", ref_transform.a, abs(ref_transform.e))

    for band in BANDS_TO_ALIGN:
        source_path = (
            PROCESSED_DATA_DIR
            / f"T42QZL_20261003T053651_{band}_20m_clipped.tif"
        )

        output_path = (
            OUTPUT_DIR
            / f"T42QZL_20261003T053651_{band}_10m_aligned.tif"
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Required source raster not found: {source_path}"
            )

        with rasterio.open(source_path) as src:
            output = np.full(
                (ref_height, ref_width),
                0.0,
                dtype=np.float32,
            )

            reproject(
                source=rasterio.band(src, 1),
                destination=output,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=0.0,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )

        profile = ref_profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=0.0,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output, 1)

        # Verify the saved raster matches the reference grid.
        with rasterio.open(output_path) as check:
            assert check.width == ref_width
            assert check.height == ref_height
            assert check.crs == ref_crs
            assert check.transform == ref_transform

        print(f"Created: {output_path.name}")
        print(f"  Dimensions: {ref_width} x {ref_height}")

    print("\nAll requested bands aligned successfully.")
    print("Original rasters were not modified.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

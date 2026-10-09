
import unittest
import warnings
from pathlib import Path

import numpy as np
import rasterio

from src.inference.predict import run_inference

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed/sentinel2")
ALIGNED_DIR = DATA_DIR / "aligned_10m"
PREFIX = "T42QZL_20261003T053651_"

BAND_PATHS = {
    "B02": DATA_DIR / f"{PREFIX}B02_10m_clipped.tif",
    "B03": DATA_DIR / f"{PREFIX}B03_10m_clipped.tif",
    "B04": DATA_DIR / f"{PREFIX}B04_10m_clipped.tif",
    "B08": DATA_DIR / f"{PREFIX}B08_10m_clipped.tif",
    "B05": ALIGNED_DIR / f"{PREFIX}B05_10m_aligned.tif",
    "B06": ALIGNED_DIR / f"{PREFIX}B06_10m_aligned.tif",
    "B07": ALIGNED_DIR / f"{PREFIX}B07_10m_aligned.tif",
    "B8A": ALIGNED_DIR / f"{PREFIX}B8A_10m_aligned.tif",
    "B11": ALIGNED_DIR / f"{PREFIX}B11_10m_aligned.tif",
    "B12": ALIGNED_DIR / f"{PREFIX}B12_10m_aligned.tif",
}


class TestRealDataInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [str(path) for path in BAND_PATHS.values()
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required Sentinel-2 bands are missing:\n"
                + "\n".join(missing)
            )

        cls.bands = {}
        for name, path in BAND_PATHS.items():
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None and np.isfinite(nodata):
                    arr[arr == nodata] = np.nan
                cls.bands[name] = arr

        cls.classification, cls.pixel_counts, cls.percentages = (
            run_inference(cls.bands)
        )

    def test_output_has_expected_shape(self):
        self.assertEqual(self.classification.shape, (280, 528))

    def test_predictions_use_valid_class_ids(self):
        valid_ids = set(np.unique(self.classification).tolist())
        self.assertTrue(
            valid_ids.issubset({0, 1, 2, 3, 4, 5}),
            f"Unexpected class IDs: {valid_ids}"
        )
        self.assertTrue(
            any(class_id in valid_ids for class_id in {1, 2, 3, 4, 5}),
            "No valid land-cover predictions were produced."
        )

    def test_pixel_counts_match_classification(self):
        for class_id, count in self.pixel_counts.items():
            actual_count = int(np.count_nonzero(
                self.classification == class_id
            ))
            self.assertEqual(actual_count, count)

    def test_percentages_are_valid(self):
        for percentage in self.percentages.values():
            self.assertTrue(np.isfinite(percentage))
            self.assertGreaterEqual(percentage, 0)
            self.assertLessEqual(percentage, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)

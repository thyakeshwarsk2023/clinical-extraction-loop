"""
tests/test_image_preflight.py
=============================
Deterministic unit tests for BioTrace standalone image-quality preflight inspection.

Validates:
  1. Valid, readable synthetic clinical document image -> PASS, ADEQUATE, SHARP, NORMAL, no glare.
  2. Empty image array -> REJECT_DEGENERATE, DEGENERATE resolution.
  3. Invalid image dimensions or channel layout -> REJECT_INVALID_INPUT.
  4. Very low-resolution image -> REJECT_LOW_RESOLUTION.
  5. Severely blurred image -> REJECT_SEVERELY_BLURRED.
  6. Near-black underexposed image -> REJECT_EXTREME_EXPOSURE (UNDEREXPOSED).
  7. Overexposed washed-out image -> REJECT_EXTREME_EXPOSURE (OVEREXPOSED).
  8. Clean image that should not be flagged for glare (standard white/light background).
  9. Document image with localized specular flash flare -> detects possible_glare.
  10. Repeated assessment idempotence & non-destructive input immutability.
  11. Custom PreflightConfig overrides defaults.
"""

import unittest
import numpy as np
import cv2

from src.image_preflight import inspect_image_quality, PreflightConfig


class TestImageQualityPreflight(unittest.TestCase):
    """Deterministic test suite for standalone image quality preflight."""

    @classmethod
    def setUpClass(cls):
        # Create standard synthetic clinical document canvas (400x600 px)
        cls.valid_doc = np.full((400, 600, 3), 245, dtype=np.uint8)
        cv2.putText(
            cls.valid_doc,
            "CLINICAL LABORATORY REPORT - LIPID PANEL",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (15, 15, 15),
            2,
        )
        cv2.putText(
            cls.valid_doc,
            "Total Cholesterol: 210.0 mg/dL",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
        )
        cv2.putText(
            cls.valid_doc,
            "HDL Cholesterol: 52.0 mg/dL",
            (40, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
        )
        cv2.putText(
            cls.valid_doc,
            "LDL Cholesterol: 126.0 mg/dL",
            (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
        )
        cv2.putText(
            cls.valid_doc,
            "Triglycerides: 160.0 mg/dL",
            (40, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
        )

    def test_valid_readable_document_passes_all_checks(self):
        """Standard clear synthetic document must produce PASS and clean statuses."""
        res = inspect_image_quality(self.valid_doc)

        self.assertTrue(res["input_valid"])
        self.assertEqual(res["width"], 600)
        self.assertEqual(res["height"], 400)
        self.assertEqual(res["resolution_status"], "ADEQUATE")
        self.assertGreater(res["blur_score"], 100.0)
        self.assertEqual(res["blur_status"], "SHARP")
        self.assertEqual(res["exposure_status"], "NORMAL")
        self.assertFalse(res["possible_glare"])
        self.assertEqual(res["quality_status"], "PASS")
        self.assertEqual(len(res["warnings"]), 0)

    def test_empty_image_array(self):
        """Empty numpy array (size 0) must trigger REJECT_DEGENERATE."""
        empty_img = np.empty((0, 0, 3), dtype=np.uint8)
        res = inspect_image_quality(empty_img)

        self.assertFalse(res["input_valid"])
        self.assertEqual(res["width"], 0)
        self.assertEqual(res["height"], 0)
        self.assertEqual(res["resolution_status"], "DEGENERATE")
        self.assertIsNone(res["blur_score"])
        self.assertEqual(res["blur_status"], "NOT_ASSESSED")
        self.assertEqual(res["exposure_status"], "NOT_ASSESSED")
        self.assertFalse(res["possible_glare"])
        self.assertEqual(res["quality_status"], "REJECT_DEGENERATE")
        self.assertIn("Empty image array", res["warnings"][0])

    def test_invalid_image_dimensions_and_channels(self):
        """Tests handling of None, non-array inputs, 1D arrays, and unsupported channel shapes."""
        # 1. None input
        res_none = inspect_image_quality(None)
        self.assertFalse(res_none["input_valid"])
        self.assertEqual(res_none["quality_status"], "REJECT_INVALID_INPUT")
        self.assertIn("None", res_none["warnings"][0])

        # 2. Non-numpy input
        res_str = inspect_image_quality("not_an_image")
        self.assertFalse(res_str["input_valid"])
        self.assertEqual(res_str["quality_status"], "REJECT_INVALID_INPUT")

        # 3. 1D array
        res_1d = inspect_image_quality(np.zeros((100,), dtype=np.uint8))
        self.assertFalse(res_1d["input_valid"])
        self.assertEqual(res_1d["quality_status"], "REJECT_INVALID_INPUT")
        self.assertIn("ndim=1", res_1d["warnings"][0])

        # 4. 4D array (e.g. batch tensor)
        res_4d = inspect_image_quality(np.zeros((1, 100, 100, 3), dtype=np.uint8))
        self.assertFalse(res_4d["input_valid"])
        self.assertEqual(res_4d["quality_status"], "REJECT_INVALID_INPUT")

        # 5. 3D array with invalid channel count (e.g. 5 channels)
        res_5ch = inspect_image_quality(np.zeros((100, 100, 5), dtype=np.uint8))
        self.assertFalse(res_5ch["input_valid"])
        self.assertEqual(res_5ch["quality_status"], "REJECT_INVALID_INPUT")
        self.assertIn("Unsupported channel count: 5", res_5ch["warnings"][0])

    def test_very_low_resolution_image(self):
        """Sub-minimum resolution images must trigger REJECT_LOW_RESOLUTION."""
        low_res = np.full((80, 120, 3), 240, dtype=np.uint8)
        res = inspect_image_quality(low_res)

        self.assertTrue(res["input_valid"])
        self.assertEqual(res["width"], 120)
        self.assertEqual(res["height"], 80)
        self.assertEqual(res["resolution_status"], "LOW_RESOLUTION")
        self.assertEqual(res["quality_status"], "REJECT_LOW_RESOLUTION")
        self.assertTrue(any("resolution" in w.lower() for w in res["warnings"]))

    def test_severely_blurred_image(self):
        """Severely defocused or motion-blurred image must trigger REJECT_SEVERELY_BLURRED."""
        blurred_img = cv2.GaussianBlur(self.valid_doc, (35, 35), sigmaX=15.0)
        res = inspect_image_quality(blurred_img)

        self.assertTrue(res["input_valid"])
        self.assertEqual(res["resolution_status"], "ADEQUATE")
        self.assertLess(res["blur_score"], 20.0)
        self.assertEqual(res["blur_status"], "SEVERELY_BLURRED")
        self.assertEqual(res["quality_status"], "REJECT_SEVERELY_BLURRED")
        self.assertTrue(any("blur" in w.lower() for w in res["warnings"]))

    def test_near_black_underexposed_image(self):
        """Near-black underexposed image must trigger REJECT_EXTREME_EXPOSURE with UNDEREXPOSED."""
        black_img = np.full((400, 600, 3), 10, dtype=np.uint8)
        res = inspect_image_quality(black_img)

        self.assertTrue(res["input_valid"])
        self.assertEqual(res["exposure_status"], "UNDEREXPOSED")
        self.assertEqual(res["quality_status"], "REJECT_EXTREME_EXPOSURE")
        self.assertTrue(any("underexposure" in w.lower() for w in res["warnings"]))

    def test_overexposed_washed_out_image(self):
        """Washed-out overexposed image with low text contrast must trigger OVEREXPOSED."""
        overexposed_img = np.full((400, 600, 3), 254, dtype=np.uint8)
        # Add very faint, barely legible text that retains almost zero contrast
        cv2.putText(overexposed_img, "faint text", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1)
        res = inspect_image_quality(overexposed_img)

        self.assertTrue(res["input_valid"])
        self.assertEqual(res["exposure_status"], "OVEREXPOSED")
        self.assertEqual(res["quality_status"], "REJECT_EXTREME_EXPOSURE")
        self.assertTrue(any("overexposure" in w.lower() for w in res["warnings"]))

    def test_clean_image_not_flagged_for_glare(self):
        """Clean scanned documents (both 245 off-white and 255 pure white) must NOT be flagged as glare."""
        # Case A: Standard off-white scan
        res_a = inspect_image_quality(self.valid_doc)
        self.assertFalse(res_a["possible_glare"], "Clean off-white document falsely flagged for glare")

        # Case B: Pure white background (255) with sharp text
        pure_white_doc = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.putText(pure_white_doc, "LABORATORY REPORT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        res_b = inspect_image_quality(pure_white_doc)
        self.assertFalse(res_b["possible_glare"], "Pure white document falsely flagged for glare")

    def test_localized_specular_glare_detection(self):
        """Camera photo with a localized bright specular flash hotspot must be flagged for glare."""
        # Simulated camera photo with grey/beige paper (luminance ~175)
        photo_doc = np.full((400, 600, 3), 175, dtype=np.uint8)
        cv2.putText(photo_doc, "LABORATORY REPORT", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(photo_doc, "Cholesterol: 190.0 mg/dL", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

        # Inject localized glare hotspot (radius 45 circular specular reflection)
        glare_doc = photo_doc.copy()
        cv2.circle(glare_doc, (320, 180), 45, (255, 255, 255), -1)

        res = inspect_image_quality(glare_doc)
        self.assertTrue(res["possible_glare"])
        self.assertEqual(res["quality_status"], "WARN_BORDERLINE")
        self.assertTrue(any("glare" in w.lower() for w in res["warnings"]))

    def test_repeated_assessment_determinism_and_immutability(self):
        """Repeated assessment of the same image must be idempotent and must NOT mutate input."""
        original_copy = self.valid_doc.copy()

        # Run 1
        res1 = inspect_image_quality(self.valid_doc)
        # Run 2
        res2 = inspect_image_quality(self.valid_doc)

        # Assert input array was never modified
        self.assertTrue(np.array_equal(self.valid_doc, original_copy), "inspect_image_quality mutated input image!")

        # Assert identical results across repeated evaluations
        self.assertEqual(res1, res2)

    def test_custom_preflight_config_thresholds(self):
        """Custom configuration parameters must override default thresholds."""
        strict_config = PreflightConfig(
            min_width=800,
            min_height=800,
            blur_threshold_sharp=1500.0,
            blur_threshold_severe=500.0,
        )
        res = inspect_image_quality(self.valid_doc, config=strict_config)

        # 600x400 fails strict 800x800 resolution
        self.assertEqual(res["resolution_status"], "LOW_RESOLUTION")
        self.assertEqual(res["quality_status"], "REJECT_LOW_RESOLUTION")


if __name__ == "__main__":
    unittest.main()

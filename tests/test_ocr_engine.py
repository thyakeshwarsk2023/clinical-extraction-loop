"""
tests/test_ocr_engine.py
=============================================================================
Phase 6C: Unit tests for OCREngine protocol, SidecarEngine, RapidOCREngine,
and clinical layout parser.

Tests:
  1. SidecarEngine contract (oracle_sidecar branding, metadata fidelity)
  2. Clean document image extraction
  3. mmol/L units detection and extraction
  4. Decimal comma handling (4,2 -> 4.2)
  5. Reference-range column discrimination (avoids <150, 0-200, 35-65)
  6. Explicit missing analyte tracking (never guessed)
"""

import os
import sys
import unittest
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ocr_engine import (
    OCREngine, SidecarEngine, get_ocr_engine,
    parse_numeric_token, is_reference_range, extract_unit_from_text
)

try:
    import rapidocr_onnxruntime
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False


class TestSidecarEngine(unittest.TestCase):
    def setUp(self):
        self.engine = get_ocr_engine("oracle_sidecar")

    def test_sidecar_engine_protocol_conformance(self):
        self.assertIsInstance(self.engine, OCREngine)
        self.assertEqual(self.engine.engine_name, "oracle_sidecar")

    def test_sidecar_engine_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.engine.extract("non_existent_file.png", "non_existent_file.json")


class TestLayoutParserHelpers(unittest.TestCase):
    def test_decimal_comma_parsing(self):
        self.assertEqual(parse_numeric_token("4,2"), 4.2)
        self.assertEqual(parse_numeric_token("12,85 mmol/L"), 12.85)
        self.assertEqual(parse_numeric_token("211.0 mg/dL"), 211.0)
        self.assertEqual(parse_numeric_token("  5,99  "), 5.99)
        self.assertIsNone(parse_numeric_token("NEGATIVE"))

    def test_reference_range_detection(self):
        # Range formats should be recognized as reference ranges
        self.assertTrue(is_reference_range("< 200"))
        self.assertTrue(is_reference_range("0-200"))
        self.assertTrue(is_reference_range("100–199"))
        self.assertTrue(is_reference_range("[35-65]"))
        self.assertTrue(is_reference_range("Ref: 40-60"))
        self.assertTrue(is_reference_range(">50"))
        # Clean patient values should NOT be recognized as reference ranges
        self.assertFalse(is_reference_range("211.0"))
        self.assertFalse(is_reference_range("4.2"))
        self.assertFalse(is_reference_range("55"))

    def test_unit_extraction(self):
        self.assertEqual(extract_unit_from_text("Result: 211.0 mg/dL"), "mg/dL")
        self.assertEqual(extract_unit_from_text("5.99 mmol/L"), "mmol/L")
        self.assertEqual(extract_unit_from_text("mmol/litre"), "mmol/L")
        self.assertIsNone(extract_unit_from_text("No unit here"))


@unittest.skipUnless(HAS_RAPIDOCR, "rapidocr-onnxruntime is not installed; skipping live OCR tests")
class TestRapidOCREngineLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_ocr_engine("rapidocr")

    def test_rapidocr_engine_protocol_conformance(self):
        self.assertIsInstance(self.engine, OCREngine)
        self.assertEqual(self.engine.engine_name, "rapidocr")

    def test_clean_render_extraction(self):
        img_path = os.path.join(ROOT, "dataset_B_realistic", "friedewald", "doc_0000.png")
        if not os.path.exists(img_path):
            self.skipTest("Sample document doc_0000.png missing")

        res = self.engine.extract(img_path)
        self.assertEqual(res["extractor_engine"], "rapidocr")
        vals = res["extracted_values"]
        self.assertIn("total_cholesterol", vals)
        self.assertIn("hdl", vals)
        self.assertIn("ldl", vals)
        self.assertIn("triglycerides", vals)
        self.assertAlmostEqual(vals["total_cholesterol"], 211.0, delta=1.0)
        self.assertAlmostEqual(vals["ldl"], 162.2, delta=1.0)
        self.assertEqual(res["unit"], "mg/dL")
        self.assertEqual(res["missing_fields"], [])

    def test_mmol_units_extraction(self):
        img_path = os.path.join(ROOT, "dataset_B_realistic", "friedewald", "doc_0051.png")
        if not os.path.exists(img_path):
            self.skipTest("Sample mmol document doc_0051.png missing")

        res = self.engine.extract(img_path)
        self.assertEqual(res["unit"], "mmol/L")
        vals = res["extracted_values"]
        self.assertAlmostEqual(vals["total_cholesterol"], 5.99, delta=0.1)
        self.assertAlmostEqual(vals["hdl"], 1.16, delta=0.1)

    def test_synthetic_decimal_comma_and_reference_range(self):
        # Create a synthetic lab report image with decimal commas and reference range column
        img = np.full((300, 650, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "LIPID PROFILE REPORT", (30, 40), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Test Name", (30, 80), font, 0.5, (50, 50, 50), 1)
        cv2.putText(img, "Result", (280, 80), font, 0.5, (50, 50, 50), 1)
        cv2.putText(img, "Unit", (400, 80), font, 0.5, (50, 50, 50), 1)
        cv2.putText(img, "Ref Range", (500, 80), font, 0.5, (50, 50, 50), 1)

        # Total Cholesterol with decimal comma
        cv2.putText(img, "Total Cholesterol", (30, 130), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "5,45", (280, 130), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "mmol/L", (400, 130), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "< 5,20", (500, 130), font, 0.5, (80, 80, 80), 1)

        # HDL Cholesterol
        cv2.putText(img, "HDL Cholesterol", (30, 180), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "1,35", (280, 180), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "mmol/L", (400, 180), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "> 1,00", (500, 180), font, 0.5, (80, 80, 80), 1)

        # LDL Cholesterol
        cv2.putText(img, "LDL Cholesterol", (30, 230), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "3,20", (280, 230), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "mmol/L", (400, 230), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "< 3,00", (500, 230), font, 0.5, (80, 80, 80), 1)

        tmp_path = os.path.join(ROOT, "reports", "tmp_test_synth_ocr.png")
        cv2.imwrite(tmp_path, img)

        try:
            res = self.engine.extract(tmp_path)
            vals = res["extracted_values"]
            self.assertIn("total_cholesterol", vals)
            self.assertAlmostEqual(vals["total_cholesterol"], 5.45, delta=0.05)
            self.assertAlmostEqual(vals["hdl"], 1.35, delta=0.05)
            self.assertAlmostEqual(vals["ldl"], 3.20, delta=0.05)
            # Triglycerides was intentionally omitted from image
            self.assertIn("triglycerides", res["missing_fields"])
            self.assertNotIn("triglycerides", vals)
            self.assertEqual(res["unit"], "mmol/L")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

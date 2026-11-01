"""
tests/test_lipid_pipeline_repairs.py
=============================================================================
Regression tests verifying:
  1. Pipeline termination repairs in src/pipeline.py:
     - Safe termination on ABSTAIN_MISSING_ANALYTE (missing HDL, missing LDL)
     - Safe termination on unrecognized / unexpected isolation actions
     - Cyclic state oscillation protection
     - Normal panel convergence
     - Non-maleficence: no unsupported mutations on abstention
     - Preservation of valid correction paths
  2. Analyte parser repairs in src/ocr_engine.py:
     - Discrimination of numerical intervals vs hyphenated analyte names
     - Support for HDL-C, LDL-C, and dash/spacing variations
     - Exclusion of non-cholesterol subfractions (LDL Size, LDL-P, HDL-P)
     - Multi-column row association with reference intervals ignored
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.ocr_engine import is_reference_range, ANALYTE_PATTERNS, RapidOCREngine
from src.pipeline import run_pipeline


class TestPipelineTerminationRepairs(unittest.TestCase):
    """Verifies that inference termination is guaranteed and safe under all conditions."""

    def test_missing_hdl_produces_terminating_abstention(self):
        """Test A: Missing HDL produces a terminating ABSTAIN_MISSING_ANALYTE."""
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.zeros((100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 198.0, "ldl": 112.0, "triglycerides": 148.0},
            "bounding_boxes": {"total_cholesterol": [10, 10, 50, 20]},
            "confidences": {"total_cholesterol": 0.95, "ldl": 0.95, "triglycerides": 0.95},
            "missing_fields": ["hdl"],
            "raw_ocr_tokens": [],
        }
        with patch("src.pipeline.extract_document_state", return_value=mock_doc):
            res = run_pipeline("dummy.png", max_attempts=2)
            self.assertEqual(res["status"], "ABSTAIN_MISSING_ANALYTE")
            self.assertIn("hdl", res.get("missing_fields", []))
            self.assertEqual(res["data"]["total_cholesterol"], 198.0)
            self.assertNotIn("hdl", res["data"])

    def test_missing_ldl_produces_terminating_abstention(self):
        """Test B: Missing LDL produces a safe, terminating abstention."""
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.zeros((100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 248.0, "hdl": 38.0, "triglycerides": 210.0},
            "bounding_boxes": {},
            "confidences": {"total_cholesterol": 0.95, "hdl": 0.95, "triglycerides": 0.95},
            "missing_fields": ["ldl"],
            "raw_ocr_tokens": [],
        }
        with patch("src.pipeline.extract_document_state", return_value=mock_doc):
            res = run_pipeline("dummy.png", max_attempts=2)
            self.assertEqual(res["status"], "ABSTAIN_MISSING_ANALYTE")
            self.assertIn("ldl", res.get("missing_fields", []))

    def test_unrecognized_search_action_terminates_safely(self):
        """Test C: An unrecognized search action cannot cause an infinite loop."""
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.zeros((100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 300.0, "hdl": 50.0, "ldl": 100.0, "triglycerides": 150.0},
            "bounding_boxes": {},
            "confidences": {"total_cholesterol": 0.95, "hdl": 0.95, "ldl": 0.95, "triglycerides": 0.95},
            "missing_fields": [],
            "raw_ocr_tokens": [],
        }
        fake_action = {
            "action": "UNEXPECTED_EXPERIMENTAL_ACTION",
            "reason": "Test unexpected action handler"
        }
        with patch("src.pipeline.extract_document_state", return_value=mock_doc), \
             patch("src.pipeline.hierarchical_isolation_search", return_value=fake_action):
            res = run_pipeline("dummy.png", max_attempts=2)
            self.assertTrue(res["status"].startswith("ABSTAIN_UNRECOGNIZED_ACTION"))
            self.assertEqual(res["action"], "UNEXPECTED_EXPERIMENTAL_ACTION")

    def test_normal_valid_lipid_panel_converges(self):
        """Test D: A normal, valid lipid panel reaches the appropriate downstream processing stage."""
        # 198 = 52 + 112 + 148/5 = 193.6 (residual 4.4 mg/dL; rel_err 2.2% <= 10% tol)
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.zeros((100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 198.0, "hdl": 52.0, "ldl": 112.0, "triglycerides": 148.0},
            "bounding_boxes": {},
            "confidences": {"total_cholesterol": 0.95, "hdl": 0.95, "ldl": 0.95, "triglycerides": 0.95},
            "missing_fields": [],
            "raw_ocr_tokens": [],
        }
        with patch("src.pipeline.extract_document_state", return_value=mock_doc):
            res = run_pipeline("dummy.png", max_attempts=2)
            self.assertEqual(res["status"], "CONVERGED")
            self.assertEqual(res["data"]["total_cholesterol"], 198.0)

    def test_no_unsupported_correction_committed_on_abstention(self):
        """Test E: No unsupported correction is committed when the pipeline abstains."""
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.zeros((100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 250.0, "hdl": 50.0, "ldl": 100.0, "triglycerides": 150.0},
            "bounding_boxes": {},
            "confidences": {"total_cholesterol": 0.95, "hdl": 0.95, "ldl": 0.95, "triglycerides": 0.95},
            "missing_fields": [],
            "raw_ocr_tokens": [],
        }
        # Invariant: 250 != 50 + 100 + 30 = 180 (violates). Search returns abstain due to visual evidence missing
        with patch("src.pipeline.extract_document_state", return_value=mock_doc):
            res = run_pipeline("dummy.png", max_attempts=2)
            self.assertTrue(res["status"].startswith("ABSTAIN"))
            self.assertEqual(res["data"]["total_cholesterol"], 250.0)
            self.assertEqual(res["data"]["ldl"], 100.0)

    def test_cyclic_state_oscillation_protection(self):
        """Test F: A retry or correction transition cannot revisit the same state indefinitely."""
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.zeros((100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 300.0, "hdl": 50.0, "ldl": 100.0, "triglycerides": 150.0},
            "bounding_boxes": {"total_cholesterol": [10, 10, 50, 50]},
            "confidences": {"total_cholesterol": 0.8, "hdl": 0.8, "ldl": 0.8, "triglycerides": 0.8},
            "missing_fields": [],
            "raw_ocr_tokens": [],
        }
        # Neither 280 nor 300 converges (expected is 50 + 100 + 30 = 180)
        call_count = [0]
        def fake_search(*args, **kwargs):
            call_count[0] += 1
            return {
                "action": "CORRECT_K1",
                "field": "total_cholesterol",
                "target_val": 280.0 if call_count[0] == 1 else 300.0,
                "is_fast_path": True,
                "score": 0.5,
                "ranked_candidates": []
            }

        test_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        mock_doc["image_bgr"] = test_img

        with patch("src.pipeline.extract_document_state", return_value=mock_doc), \
             patch("src.pipeline.hierarchical_isolation_search", side_effect=fake_search), \
             patch("src.pipeline.is_valid_bbox", return_value=True), \
             patch("src.pipeline.process_vision_gate", return_value=(test_img, None, None)), \
             patch("src.pipeline.enhance_and_reextract", side_effect=[
                 (280.0, 0.95, 0.15, None, {}),
                 (300.0, 0.95, 0.15, None, {})
             ]):
            res = run_pipeline("dummy.png", max_attempts=5)
            self.assertEqual(res["status"], "ABSTAIN_CYCLIC_STATE_DETECTED")

    def test_valid_correction_path_retains_behavior(self):
        """Test G: Existing valid correction paths retain their intended behavior."""
        # TC has OCR error (280.0), true is 180.0 = 50 + 100 + 150/5
        mock_doc = {
            "image_path": "dummy.png",
            "image_bgr": np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8),
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {"total_cholesterol": 280.0, "hdl": 50.0, "ldl": 100.0, "triglycerides": 150.0},
            "bounding_boxes": {"total_cholesterol": [10, 10, 50, 50]},
            "confidences": {"total_cholesterol": 0.60, "hdl": 0.95, "ldl": 0.95, "triglycerides": 0.95},
            "missing_fields": [],
            "raw_ocr_tokens": [],
        }
        search_ret = {
            "action": "CORRECT_K1",
            "field": "total_cholesterol",
            "target_val": 180.0,
            "is_fast_path": True,
            "score": 0.2,
            "ranked_candidates": []
        }
        test_img = mock_doc["image_bgr"]
        with patch("src.pipeline.extract_document_state", return_value=mock_doc), \
             patch("src.pipeline.hierarchical_isolation_search", return_value=search_ret), \
             patch("src.pipeline.is_valid_bbox", return_value=True), \
             patch("src.pipeline.process_vision_gate", return_value=(test_img, None, None)), \
             patch("src.pipeline.enhance_and_reextract", return_value=(180.0, 0.96, 0.36, None, {"psnr": 28.5})):
            res = run_pipeline("dummy.png", max_attempts=2)
            self.assertEqual(res["status"], "CONVERGED")
            self.assertEqual(res["data"]["total_cholesterol"], 180.0)
            self.assertGreater(len(res["audit_trail"]), 0)


class TestAnalyteParserRepairs(unittest.TestCase):
    """Verifies reference range classification and analyte extraction repairs."""

    def test_hdl_c_accepted_not_reference_range(self):
        """Test H: HDL-C is accepted as an analyte label and not a reference range."""
        self.assertFalse(is_reference_range("HDL-C"))
        self.assertFalse(is_reference_range("HDL - C"))
        self.assertFalse(is_reference_range("HDL–C"))
        self.assertFalse(is_reference_range("HDL—C"))

        matched = any(p.search("HDL-C") for p in ANALYTE_PATTERNS["hdl"])
        self.assertTrue(matched)

    def test_ldl_c_accepted_not_reference_range(self):
        """Test I: LDL-C is accepted as an analyte label and not a reference range."""
        self.assertFalse(is_reference_range("LDL-C"))
        self.assertFalse(is_reference_range("LDL - C"))
        self.assertFalse(is_reference_range("LDL–C"))
        self.assertFalse(is_reference_range("LDL—C"))
        self.assertFalse(is_reference_range("LDL-C(NIH Clac)"))

        matched = any(p.search("LDL-C") for p in ANALYTE_PATTERNS["ldl"])
        self.assertTrue(matched)
        matched_paren = any(p.search("LDL-C(NIH Clac)") for p in ANALYTE_PATTERNS["ldl"])
        self.assertTrue(matched_paren)

    def test_numeric_reference_intervals_recognized(self):
        """Test J: Numeric reference intervals remain recognized."""
        self.assertTrue(is_reference_range("20-40"))
        self.assertTrue(is_reference_range("3.5–5.5"))
        self.assertTrue(is_reference_range("10 - 20"))
        self.assertTrue(is_reference_range("0.5 — 2.0"))
        self.assertTrue(is_reference_range("< 200"))
        self.assertTrue(is_reference_range("<100"))
        self.assertTrue(is_reference_range(">=40"))
        self.assertTrue(is_reference_range("[10-20]"))
        self.assertTrue(is_reference_range("(100-200)"))
        self.assertTrue(is_reference_range("Ref: 40-60"))
        self.assertTrue(is_reference_range("Ref Interval"))
        self.assertTrue(is_reference_range("Normal"))

    def test_ldl_size_never_extracted_as_ldl_c(self):
        """Test K: LDL Size 20.8 is never extracted as LDL-C."""
        engine = RapidOCREngine.__new__(RapidOCREngine)
        engine.engine_name = "rapidocr"

        # Mock tokens containing ONLY LDL Size and other lipids, but no LDL-C
        tokens = [
            {"text": "Cholesterol,Total", "confidence": 0.98, "bbox": [100, 100, 250, 120], "y_center": 110.0, "x_min": 100.0, "x_max": 250.0},
            {"text": "198", "confidence": 0.99, "bbox": [400, 100, 450, 120], "y_center": 110.0, "x_min": 400.0, "x_max": 450.0},
            {"text": "HDL-C", "confidence": 0.98, "bbox": [100, 140, 200, 160], "y_center": 150.0, "x_min": 100.0, "x_max": 200.0},
            {"text": "52", "confidence": 0.99, "bbox": [400, 140, 450, 160], "y_center": 150.0, "x_min": 400.0, "x_max": 450.0},
            {"text": "LDL Size", "confidence": 0.98, "bbox": [100, 180, 200, 200], "y_center": 190.0, "x_min": 100.0, "x_max": 200.0},
            {"text": "20.8", "confidence": 0.99, "bbox": [400, 180, 450, 200], "y_center": 190.0, "x_min": 400.0, "x_max": 450.0},
            {"text": "Triglycerides", "confidence": 0.98, "bbox": [100, 220, 250, 240], "y_center": 230.0, "x_min": 100.0, "x_max": 250.0},
            {"text": "148", "confidence": 0.99, "bbox": [400, 220, 450, 240], "y_center": 230.0, "x_min": 400.0, "x_max": 450.0},
        ]

        dummy_img = np.zeros((300, 600, 3), dtype=np.uint8)
        raw_items = [([t["bbox"]], t["text"], t["confidence"]) for t in tokens]
        engine._ocr = MagicMock(return_value=(raw_items, None))
        with patch("cv2.imread", return_value=dummy_img):
            res = engine.extract("dummy_scan.jpg")

        vals = res["extracted_values"]
        self.assertNotIn("ldl", vals)
        self.assertIn("ldl", res["missing_fields"])

    def test_realistic_token_layout_row_association(self):
        """Test L: HDL-C and LDL-C values are extracted from appropriate rows ignoring ref ranges."""
        engine = RapidOCREngine.__new__(RapidOCREngine)
        engine.engine_name = "rapidocr"

        tokens = [
            # Row 1: LDL-C
            {"text": "LDL-C (NIH Clac)", "confidence": 0.99, "bbox": [100, 100, 250, 120], "y_center": 110.0, "x_min": 100.0, "x_max": 250.0, "box": []},
            {"text": "112 H", "confidence": 0.98, "bbox": [300, 100, 350, 120], "y_center": 110.0, "x_min": 300.0, "x_max": 350.0, "box": []},
            {"text": "mg/dL", "confidence": 0.97, "bbox": [400, 100, 450, 120], "y_center": 110.0, "x_min": 400.0, "x_max": 450.0, "box": []},
            {"text": "<100", "confidence": 0.99, "bbox": [500, 100, 550, 120], "y_center": 110.0, "x_min": 500.0, "x_max": 550.0, "box": []},
            # Row 2: HDL-C
            {"text": "HDL-C", "confidence": 0.99, "bbox": [100, 140, 200, 160], "y_center": 150.0, "x_min": 100.0, "x_max": 200.0, "box": []},
            {"text": "52", "confidence": 0.99, "bbox": [300, 140, 350, 160], "y_center": 150.0, "x_min": 300.0, "x_max": 350.0, "box": []},
            {"text": "mg/dL", "confidence": 0.97, "bbox": [400, 140, 450, 160], "y_center": 150.0, "x_min": 400.0, "x_max": 450.0, "box": []},
            {"text": ">=40", "confidence": 0.99, "bbox": [500, 140, 550, 160], "y_center": 150.0, "x_min": 500.0, "x_max": 550.0, "box": []},
            # Row 3: LDL Size
            {"text": "LDL Size", "confidence": 0.99, "bbox": [100, 180, 200, 200], "y_center": 190.0, "x_min": 100.0, "x_max": 200.0, "box": []},
            {"text": "20.8", "confidence": 0.99, "bbox": [300, 180, 350, 200], "y_center": 190.0, "x_min": 300.0, "x_max": 350.0, "box": []},
            # Row 4: Total Cholesterol
            {"text": "Cholesterol,Total", "confidence": 0.98, "bbox": [100, 220, 260, 240], "y_center": 230.0, "x_min": 100.0, "x_max": 260.0, "box": []},
            {"text": "198", "confidence": 0.99, "bbox": [300, 220, 350, 240], "y_center": 230.0, "x_min": 300.0, "x_max": 350.0, "box": []},
            # Row 5: Triglycerides
            {"text": "Triglycerides", "confidence": 0.98, "bbox": [100, 260, 250, 280], "y_center": 270.0, "x_min": 100.0, "x_max": 250.0, "box": []},
            {"text": "148", "confidence": 0.99, "bbox": [300, 260, 350, 280], "y_center": 270.0, "x_min": 300.0, "x_max": 350.0, "box": []},
        ]

        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
        raw_items = [([t["bbox"]], t["text"], t["confidence"]) for t in tokens]
        engine._ocr = MagicMock(return_value=(raw_items, None))
        with patch("cv2.imread", return_value=dummy_img):
            res = engine.extract("dummy_scan.jpg")

        vals = res["extracted_values"]
        self.assertEqual(vals.get("total_cholesterol"), 198.0)
        self.assertEqual(vals.get("hdl"), 52.0)
        self.assertEqual(vals.get("ldl"), 112.0)
        self.assertEqual(vals.get("triglycerides"), 148.0)
        self.assertEqual(res["missing_fields"], [])

    def test_dash_and_spacing_variations(self):
        """Test M: Common hyphen, dash, spacing variations do not cause valid analytes to be discarded."""
        variations = [
            "HDL-C", "HDL - C", "HDL – C", "HDL — C",
            "LDL-C", "LDL - C", "LDL – C", "LDL — C"
        ]
        for v in variations:
            self.assertFalse(is_reference_range(v), f"Failed for {v}: should not be reference range")
            if "hdl" in v.lower():
                matched = any(p.search(v) for p in ANALYTE_PATTERNS["hdl"])
                self.assertTrue(matched, f"Failed for {v}: should match hdl pattern")
            else:
                matched = any(p.search(v) for p in ANALYTE_PATTERNS["ldl"])
                self.assertTrue(matched, f"Failed for {v}: should match ldl pattern")


if __name__ == "__main__":
    unittest.main()

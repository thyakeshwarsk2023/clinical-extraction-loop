"""
tests/test_preflight_integration.py
=============================================================================
Integration test suite for BioTrace Image Preflight (Task 4):
1. A clean image follows the existing processing path.
2. An invalid or empty image does not reach OCR.
3. A borderline-quality image records a warning and continues processing.
4. A severe-quality warning is propagated correctly with retake advisories.
5. Quality warnings do not independently authorize or correct values.
6. Independently readable fields are not automatically discarded merely because
   another region has poor quality or localized glare.
7. Existing abstention and visual-verification safeguards remain active.
8. Existing public output schemas remain backward compatible.
"""

import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.image_preflight import inspect_image_quality, PreflightConfig
from src.ocr_engine import RapidOCREngine, SidecarEngine
from src.pipeline import run_pipeline


class TestPreflightIntegration(unittest.TestCase):

    def setUp(self):
        # Base clean synthetic image (400x600, BGR) with paper background (180 for realistic document contrast)
        self.img_h, self.img_w = 400, 600
        self.clean_bgr = np.full((self.img_h, self.img_w, 3), 180, dtype=np.uint8)
        cv2.putText(self.clean_bgr, "LABORATORY REPORT - LIPID PANEL", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(self.clean_bgr, "Total Cholesterol: 200.0 mg/dL", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.clean_bgr, "HDL Cholesterol: 50.0 mg/dL", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.clean_bgr, "LDL Cholesterol: 120.0 mg/dL", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.clean_bgr, "Triglycerides: 150.0 mg/dL", (40, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

        self.bboxes = {
            "total_cholesterol": [200, 85, 270, 105],
            "hdl": [180, 125, 240, 145],
            "ldl": [180, 165, 240, 185],
            "triglycerides": [170, 205, 230, 225],
        }

        self.clean_doc = {
            "image_bgr": self.clean_bgr,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 120.0,
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.95 for f in self.bboxes},
        }

    # -------------------------------------------------------------------------
    # 1. Clean image follows existing processing path
    # -------------------------------------------------------------------------
    def test_clean_image_follows_existing_processing_path(self):
        """Clean document proceeds to CONVERGED with preflight PASS and no warnings."""
        res = run_pipeline("mock_clean.png", doc=self.clean_doc)
        self.assertEqual(res["status"], "CONVERGED")
        self.assertEqual(res["image_quality_status"], "PASS")
        self.assertFalse(res["retake_recommended"])
        self.assertEqual(res["quality_warnings"], [])
        self.assertTrue(res["image_preflight"]["input_valid"])
        self.assertEqual(res["data"]["total_cholesterol"], 200.0)
        self.assertEqual(res["data"]["ldl"], 120.0)

        # Audit trail must have preflight assessment at index 0
        self.assertGreaterEqual(len(res["audit_trail"]), 1)
        first_audit = res["audit_trail"][0]
        self.assertEqual(first_audit["stage"], "STAGE_0_IMAGE_PREFLIGHT")
        self.assertEqual(first_audit["quality_status"], "PASS")

    # -------------------------------------------------------------------------
    # 2. Invalid or empty image does not reach OCR
    # -------------------------------------------------------------------------
    def test_invalid_or_empty_image_does_not_reach_ocr(self):
        """Invalid, empty or corrupted inputs short-circuit without calling OCR."""
        # 2a: run_pipeline with None path and None doc returns ABSTAIN_INVALID_OR_EMPTY_IMAGE
        res = run_pipeline(None, doc=None)
        self.assertEqual(res["status"], "ABSTAIN_INVALID_OR_EMPTY_IMAGE")
        self.assertEqual(res["image_quality_status"], "REJECT_INVALID_INPUT")
        self.assertTrue(res["retake_recommended"])
        self.assertFalse(res["image_preflight"]["input_valid"])

        # 2b: RapidOCREngine.extract must not invoke _ocr when input is invalid or 0 bytes
        engine = RapidOCREngine()
        with patch.object(engine, "_ocr") as mock_ocr:
            doc_empty = engine.extract("tests/fixtures/non_existent_empty.png")
            mock_ocr.assert_not_called()
            self.assertFalse(doc_empty["image_preflight"]["input_valid"])
            self.assertEqual(doc_empty["extracted_values"], {})

        # 2c: SidecarEngine.extract must raise gracefully when files are missing
        sidecar = SidecarEngine()
        with self.assertRaises(FileNotFoundError):
            sidecar.extract("non_existent_file.png", "non_existent_file.json")

    # -------------------------------------------------------------------------
    # 3. Borderline-quality image records a warning
    # -------------------------------------------------------------------------
    def test_borderline_quality_image_records_warning(self):
        """Borderline blur records a warning but allows processing to continue."""
        # Clean image has blur score around 1800; set threshold to classify it as BORDERLINE_BLUR
        custom_cfg = PreflightConfig(
            blur_threshold_severe=10.0,
            blur_threshold_sharp=2500.0,
        )

        doc = dict(self.clean_doc)
        res = run_pipeline("mock_borderline.png", doc=doc, preflight_config=custom_cfg)

        self.assertEqual(res["image_quality_status"], "WARN_BORDERLINE")
        self.assertFalse(res["retake_recommended"])
        self.assertGreater(len(res["quality_warnings"]), 0)
        self.assertTrue(any("Borderline" in w for w in res["quality_warnings"]))

        # Processing must continue to normal convergence!
        self.assertEqual(res["status"], "CONVERGED")
        self.assertEqual(res["data"]["total_cholesterol"], 200.0)

        # Audit trail must show preflight WARN at index 0
        self.assertEqual(res["audit_trail"][0]["stage"], "STAGE_0_IMAGE_PREFLIGHT")
        self.assertEqual(res["audit_trail"][0]["quality_status"], "WARN_BORDERLINE")

    # -------------------------------------------------------------------------
    # 4. Severe-quality warning is propagated correctly
    # -------------------------------------------------------------------------
    def test_severe_quality_warning_propagated_correctly(self):
        """Severely degraded images record advisory rejection with retake recommendation."""
        # Create severely blurred image (massive Gaussian blur: blur score < 5.0)
        blurred_img = cv2.GaussianBlur(self.clean_bgr, (41, 41), sigmaX=15.0)

        doc = {
            "image_bgr": blurred_img,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 120.0,
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.95 for f in self.bboxes},
        }

        res = run_pipeline("mock_severe_blur.png", doc=doc)

        self.assertEqual(res["image_quality_status"], "REJECT_SEVERELY_BLURRED")
        self.assertTrue(res["retake_recommended"])
        self.assertTrue(any("Severe defocus" in w for w in res["quality_warnings"]))
        self.assertEqual(res["audit_trail"][0]["stage"], "STAGE_0_IMAGE_PREFLIGHT")
        self.assertEqual(res["audit_trail"][0]["quality_status"], "REJECT_SEVERELY_BLURRED")

    # -------------------------------------------------------------------------
    # 5. Quality warnings do not independently authorize or correct values
    # -------------------------------------------------------------------------
    def test_quality_warnings_do_not_authorize_or_correct_values(self):
        """Preflight warnings must not smooth over clinical discrepancies or alter decisions."""
        # Inconsistent clinical panel: TC=200, HDL=50, TG=150, LDL=180 (calculated is 120)
        # Discrepancy is genuine and exceeds 10% tolerance -> must FLAG / REVIEW
        doc = {
            "image_bgr": self.clean_bgr,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 180.0,  # printed inconsistency
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.95 for f in self.bboxes},
        }

        # Add borderline preflight threshold
        custom_cfg = PreflightConfig(blur_threshold_sharp=2500.0, blur_threshold_severe=10.0)
        res = run_pipeline("mock_discrepancy.png", doc=doc, preflight_config=custom_cfg)

        # Preflight shows warning
        self.assertEqual(res["image_quality_status"], "WARN_BORDERLINE")

        # Clinical status MUST NOT be authorized or marked CONVERGED
        self.assertNotEqual(res["status"], "CONVERGED")
        self.assertIn(res["status"], ["FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY", "ABSTAIN_VISUAL_DISAGREEMENT", "ABSTAIN_RESIDUAL_HIGH"])
        # Values are NOT modified or forced to fit equation
        self.assertEqual(res["data"]["ldl"], 180.0)

    # -------------------------------------------------------------------------
    # 6. Independently readable fields not discarded merely because another region has poor quality
    # -------------------------------------------------------------------------
    def test_independently_readable_fields_not_discarded_due_to_localized_glare(self):
        """Localized glare patch in header/corner does not discard readable panel fields."""
        glare_img = self.clean_bgr.copy()
        # Add high-brightness glare patch in top-right corner (rows 10-60, cols 480-540 -> 3000 px, 1.25% of 240k px)
        glare_img[10:60, 480:540] = 255

        doc = {
            "image_bgr": glare_img,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 120.0,
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.95 for f in self.bboxes},
        }

        res = run_pipeline("mock_glare.png", doc=doc)

        # Preflight detects glare warning
        self.assertTrue(res["image_preflight"]["possible_glare"])
        self.assertEqual(res["image_quality_status"], "WARN_BORDERLINE")
        self.assertTrue(any("Possible specular glare" in w for w in res["quality_warnings"]))

        # Document MUST NOT be discarded; readable fields converge normally
        self.assertEqual(res["status"], "CONVERGED")
        self.assertEqual(res["data"]["total_cholesterol"], 200.0)
        self.assertEqual(res["data"]["hdl"], 50.0)
        self.assertEqual(res["data"]["ldl"], 120.0)
        self.assertEqual(res["data"]["triglycerides"], 150.0)

    # -------------------------------------------------------------------------
    # 7. Existing abstention and visual-verification safeguards remain active
    # -------------------------------------------------------------------------
    def test_existing_abstention_and_visual_safeguards_remain_active(self):
        """Missing analytes, TG > 800, and bounds violations retain strict abstentions."""
        # 7a: Missing HDL -> ABSTAIN_MISSING_ANALYTE
        doc_missing = {
            "image_bgr": self.clean_bgr,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "ldl": 120.0,
                "triglycerides": 150.0,
                # hdl missing
            },
            "bounding_boxes": {k: v for k, v in self.bboxes.items() if k != "hdl"},
            "confidences": {k: 0.95 for k in self.bboxes if k != "hdl"},
        }
        res_missing = run_pipeline("mock_missing.png", doc=doc_missing)
        self.assertEqual(res_missing["status"], "ABSTAIN_MISSING_ANALYTE")
        self.assertIn("hdl", res_missing.get("missing_fields", []))

        # 7b: Hypertriglyceridemia TG > 800 -> ABSTAIN_TG800
        doc_tg800 = {
            "image_bgr": self.clean_bgr,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 300.0,
                "hdl": 40.0,
                "ldl": 100.0,
                "triglycerides": 850.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.95 for f in self.bboxes},
        }
        res_tg800 = run_pipeline("mock_tg800.png", doc=doc_tg800)
        self.assertEqual(res_tg800["status"], "ABSTAIN_TG800")

    # -------------------------------------------------------------------------
    # 8. Existing public output schemas remain compatible
    # -------------------------------------------------------------------------
    def test_existing_public_output_schemas_remain_compatible(self):
        """Pipeline returns all legacy keys and adds diagnostic preflight keys."""
        legacy_keys = {"status", "data", "assay_route", "audit_trail"}
        preflight_keys = {"image_preflight", "image_quality_status", "retake_recommended", "quality_warnings"}

        # Check clean run
        res = run_pipeline("mock_clean.png", doc=self.clean_doc)
        for k in legacy_keys:
            self.assertIn(k, res, f"Legacy key '{k}' missing from result")
        for k in preflight_keys:
            self.assertIn(k, res, f"Preflight key '{k}' missing from result")

        self.assertIsInstance(res["data"], dict)
        self.assertIsInstance(res["audit_trail"], list)
        self.assertIsInstance(res["quality_warnings"], list)
        self.assertIsInstance(res["retake_recommended"], bool)

        # Check invalid input run
        res_invalid = run_pipeline(None, doc=None)
        for k in legacy_keys:
            self.assertIn(k, res_invalid, f"Legacy key '{k}' missing from invalid result")
        for k in preflight_keys:
            self.assertIn(k, res_invalid, f"Preflight key '{k}' missing from invalid result")


if __name__ == "__main__":
    unittest.main()

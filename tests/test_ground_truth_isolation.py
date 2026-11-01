"""
tests/test_ground_truth_isolation.py
=============================================================================
Targeted regression test suite verifying ground-truth isolation, zero-leakage,
unit consistency, visual verification guardrails, and abstention behavior across
the BioTrace inference pipeline.

Tests:
  1. Reference Label Invariance: Poisoning doc with reference metadata (ground_truth_clean)
     does NOT alter pipeline prediction.
  2. Sidecar Independence: Live inference runs directly from image pixels without JSON sidecar.
  3. Empty/Degraded OCR Abstention: Pitch-black or empty OCR returns explicit abstention,
     never silently substituting expected clean values.
  4. Corruption Annotation Independence: Changing or removing synthetic corruption annotations
     does not alter extraction or confidence.
  5. Contradictory Evidence Rejection: When image evidence disagrees with candidate target,
     Stage 4 rejects commitment and returns committed_val = None with zero confidence boost.
  6. Visual Verification Safety (k=2): Missing, dummy, invalid, or unreadable bboxes cannot
     bypass verification or boost confidence in CORRECT_K2.
  7. Analyte-Aware Unit Consistency: mmol/L reports are correctly converted to canonical mg/dL
     using cholesterol (38.67) and triglyceride (88.57) factors before consensus evaluation.
  8. Evaluation Harness Isolation: Empty live OCR output never falls back to injected_display_data.
"""

import os
import shutil
import unittest
import copy
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from src.stage1_extract import extract_document_state
from src.ocr_engine import RapidOCREngine, SidecarEngine
from src.stage4_enhance_extract import enhance_and_reextract
from src.pipeline import run_pipeline
from src.stage3_vision_gate import is_valid_bbox
from evals.run_final_ablation import run_full_architecture
from evals.eval_realworld_dataset import run_evaluation_on_record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _create_clean_lipid_image(path: str, tc: float = 200.0, hdl: float = 50.0, ldl: float = 120.0, tg: float = 150.0):
    """Renders a standard synthetic high-contrast lipid report."""
    img = np.full((320, 600, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "LIPID PANEL REPORT", (30, 40), font, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Method: Calculated by Friedewald Equation", (30, 70), font, 0.45, (80, 80, 80), 1)

    cv2.putText(img, "Total Cholesterol", (30, 120), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, f"{tc:.1f}", (280, 120), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, "mg/dL", (380, 120), font, 0.45, (0, 0, 0), 1)

    cv2.putText(img, "HDL Cholesterol", (30, 160), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, f"{hdl:.1f}", (280, 160), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, "mg/dL", (380, 160), font, 0.45, (0, 0, 0), 1)

    cv2.putText(img, "LDL Cholesterol", (30, 200), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, f"{ldl:.1f}", (280, 200), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, "mg/dL", (380, 200), font, 0.45, (0, 0, 0), 1)

    cv2.putText(img, "Triglycerides", (30, 240), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, f"{tg:.1f}", (280, 240), font, 0.55, (0, 0, 0), 2)
    cv2.putText(img, "mg/dL", (380, 240), font, 0.45, (0, 0, 0), 1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


class TestMockedEngineIsolation(unittest.TestCase):
    """
    Controlled isolation tests verifying pipeline contracts and mathematical independence.
    """

    def test_reference_label_perturbation_invariance(self):
        """
        Critical Test A: Perturbing or injecting evaluator reference labels
        (e.g. ground_truth_clean) into the document must NOT alter prediction.
        """
        doc_clean = {
            "image_path": "mock.png",
            "image_bgr": np.full((100, 100, 3), 255, dtype=np.uint8),
            "unit": "mg/dL",
            "method_text": "Method: Calculated by Friedewald Equation",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 100.0,  # 20 mg/dL residual error
                "triglycerides": 150.0
            },
            "bounding_boxes": {
                "total_cholesterol": [10, 10, 50, 30],
                "hdl": [10, 40, 50, 60],
                "ldl": [10, 70, 50, 90],
                "triglycerides": [10, 100, 50, 120]
            },
            "confidences": {
                "total_cholesterol": 0.95,
                "hdl": 0.95,
                "ldl": 0.95,
                "triglycerides": 0.95
            }
        }

        # 1. Prediction on clean document
        pred_clean, _, status_clean = run_full_architecture(copy.deepcopy(doc_clean))

        # 2. Prediction on document poisoned with extreme reference metadata
        doc_poisoned = copy.deepcopy(doc_clean)
        doc_poisoned["ground_truth_clean"] = {
            "total_cholesterol": 999.0,
            "hdl": 999.0,
            "ldl": 999.0,
            "triglycerides": 999.0
        }
        doc_poisoned["injected_display_data"] = {"ldl": 999.0}
        doc_poisoned["assay_ground_truth"] = "DIRECT_MEASURED"

        pred_poisoned, _, status_poisoned = run_full_architecture(doc_poisoned)

        # Assert predictions are strictly invariant to reference metadata presence
        self.assertEqual(pred_clean["ldl"], pred_poisoned["ldl"])
        self.assertEqual(pred_clean, pred_poisoned)
        self.assertEqual(status_clean, status_poisoned)
        self.assertNotEqual(pred_poisoned["ldl"], 999.0)
        self.assertNotEqual(pred_poisoned["total_cholesterol"], 999.0)

    def test_empty_or_solid_crop_abstains_without_guessing(self):
        """
        Test C: Blank or solid gray crop passed to Stage 4 returns committed_val = None
        and delta_conf = 0.0, rather than substituting the candidate target.
        """
        solid_crop = np.full((30, 80, 3), 200, dtype=np.uint8)  # uniform gray, zero text
        target_val = 148.8
        initial_conf = 0.65

        committed_val, post_conf, delta_conf, sr_4x, metrics = enhance_and_reextract(
            solid_crop,
            candidate_target_val=target_val,
            current_conf=initial_conf
        )

        self.assertIsNone(committed_val, "Stage 4 must NOT commit a target when OCR sees no text!")
        self.assertFalse(metrics["ocr_agreement"])
        self.assertEqual(delta_conf, 0.0, "Confidence must not increase on unverified crop!")
        self.assertEqual(post_conf, initial_conf)

    def test_contradictory_crop_evidence_rejected(self):
        """
        Test E: If crop shows '75.0' but candidate target is '148.8',
        Stage 4 must reject and return committed_val = None.
        """
        crop_75 = np.full((30, 80, 3), 255, dtype=np.uint8)
        cv2.putText(crop_75, "75.0", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        committed_val, post_conf, delta_conf, sr_4x, metrics = enhance_and_reextract(
            crop_75,
            candidate_target_val=148.8,
            current_conf=0.70
        )

        self.assertIsNone(committed_val, "Stage 4 must NOT commit when OCR disagrees with target!")
        self.assertFalse(metrics["ocr_agreement"])
        self.assertEqual(delta_conf, 0.0)


class TestVisualVerificationSafety(unittest.TestCase):
    """
    Regression suite for Phase 2: Ensuring candidate corrections cannot bypass
    visual verification through missing, dummy, or invalid bounding boxes.
    """

    def test_is_valid_bbox_helper(self):
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        # None and empty
        self.assertFalse(is_valid_bbox(None, canvas))
        self.assertFalse(is_valid_bbox([], canvas))
        # Dummy and uninitialized
        self.assertFalse(is_valid_bbox([0, 0, 10, 10], canvas))
        self.assertFalse(is_valid_bbox([0, 0, 0, 0], canvas))
        self.assertFalse(is_valid_bbox([-1, -1, -1, -1], canvas))
        # Inverted coordinates
        self.assertFalse(is_valid_bbox([50, 50, 40, 40], canvas))
        # Degenerate dimensions (< 6px)
        self.assertFalse(is_valid_bbox([10, 10, 13, 13], canvas))
        # Completely out of bounds
        self.assertFalse(is_valid_bbox([200, 200, 300, 300], canvas))
        # Legitimate bounding box
        self.assertTrue(is_valid_bbox([10, 10, 60, 40], canvas))

    @patch("src.pipeline.extract_document_state")
    def test_k2_missing_or_dummy_bbox_abstains(self, mock_extract):
        """
        When CORRECT_K2 triggers, if any field has a dummy [0, 0, 10, 10] or missing bbox,
        it must ABSTAIN and not commit candidate values or boost confidence to 0.95.
        """
        mock_extract.return_value = {
            "image_path": "dummy.png",
            "image_bgr": np.full((300, 300, 3), 255, dtype=np.uint8),
            "unit": "mg/dL",
            "method_text": "Method: Calculated by Friedewald Equation",
            "extracted_values": {
                "total_cholesterol": 240.0,
                "hdl": 50.0,
                "ldl": 100.0,
                "triglycerides": 200.0
            },
            # Dummy bbox [0, 0, 10, 10]
            "bounding_boxes": {
                "total_cholesterol": [0, 0, 10, 10],
                "hdl": [0, 0, 10, 10],
                "ldl": [0, 0, 10, 10],
                "triglycerides": [0, 0, 10, 10]
            },
            "confidences": {
                "total_cholesterol": 0.80,
                "hdl": 0.80,
                "ldl": 0.80,
                "triglycerides": 0.80
            }
        }

        res = run_pipeline("dummy.png")
        # Must abstain because visual evidence is missing/dummy
        self.assertIn("ABSTAIN", res["status"])
        self.assertEqual(res["status"], "ABSTAIN_VISUAL_EVIDENCE_MISSING")
        # Data must preserve original extraction without blind candidate commits
        self.assertEqual(res["data"]["total_cholesterol"], 240.0)


class TestUnitConsistencyAndAnalyteAwareConversion(unittest.TestCase):
    """
    Regression suite for Phase 5: Ensuring OCR readings in mmol/L are accurately
    converted to canonical mg/dL using analyte-specific factors.
    """

    def test_cholesterol_mmol_to_mgdl_consensus(self):
        """
        Crop reads '3.10' (mmol/L). For LDL (cholesterol), factor is 38.67:
        3.10 * 38.67 = 119.877 -> 119.9 mg/dL.
        Candidate target is 120.0 mg/dL.
        Should agree and commit 119.9 mg/dL.
        """
        crop = np.full((30, 80, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "3.10", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        val, post_conf, delta_conf, _, metrics = enhance_and_reextract(
            crop,
            candidate_target_val=120.0,
            current_conf=0.70,
            field="ldl",
            unit="mmol/L"
        )

        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 119.9, places=1)
        self.assertTrue(metrics["ocr_agreement"])
        self.assertEqual(metrics["blind_reread_val"], 3.1)
        self.assertEqual(metrics["blind_reread_val_mgdl"], 119.9)
        self.assertGreater(delta_conf, 0.0)

    def test_triglycerides_mmol_to_mgdl_consensus(self):
        """
        Crop reads '1.70' (mmol/L). For triglycerides, factor is 88.57:
        1.70 * 88.57 = 150.569 -> 150.6 mg/dL.
        Candidate target is 150.0 mg/dL.
        Should agree and commit 150.6 mg/dL.
        """
        crop = np.full((30, 80, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "1.70", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        val, post_conf, delta_conf, _, metrics = enhance_and_reextract(
            crop,
            candidate_target_val=150.0,
            current_conf=0.70,
            field="triglycerides",
            unit="mmol/L"
        )

        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 150.6, places=1)
        self.assertTrue(metrics["ocr_agreement"])
        self.assertEqual(metrics["blind_reread_val"], 1.7)
        self.assertEqual(metrics["blind_reread_val_mgdl"], 150.6)

    def test_unrecognized_or_missing_unit_abstains(self):
        """
        Unknown unit 'g/L' or None must not guess mg/dL; must abstain.
        """
        crop = np.full((30, 80, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "120.0", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        val, post_conf, delta_conf, _, metrics = enhance_and_reextract(
            crop,
            candidate_target_val=120.0,
            current_conf=0.70,
            field="ldl",
            unit="unsupported_unit"
        )

        self.assertIsNone(val)
        self.assertFalse(metrics["ocr_agreement"])
        self.assertEqual(delta_conf, 0.0)
        self.assertIn("UNSUPPORTED_OR_UNKNOWN_UNIT", metrics.get("reason", ""))

    def test_contradictory_visual_reading_in_mmol_abstains(self):
        """
        Crop reads '1.00' mmol/L (38.7 mg/dL) while target is 120.0 mg/dL.
        Must reject commitment.
        """
        crop = np.full((30, 80, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "1.00", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        val, post_conf, delta_conf, _, metrics = enhance_and_reextract(
            crop,
            candidate_target_val=120.0,
            current_conf=0.70,
            field="ldl",
            unit="mmol/L"
        )

        self.assertIsNone(val)
        self.assertFalse(metrics["ocr_agreement"])
        self.assertEqual(delta_conf, 0.0)


class TestLiveRapidOCREngineIsolation(unittest.TestCase):
    """
    Live tests verifying that RapidOCREngine and run_pipeline
    operate directly on image pixels without sidecar files or ground truth.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(ROOT, "reports", "tmp_isolation_test")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.img_clean = os.path.join(cls.test_dir, "doc_clean.png")
        _create_clean_lipid_image(cls.img_clean, tc=200.0, hdl=50.0, ldl=120.0, tg=150.0)

        # Blank pitch-black image for empty OCR test
        cls.img_black = os.path.join(cls.test_dir, "doc_black.png")
        cv2.imwrite(cls.img_black, np.zeros((300, 500, 3), dtype=np.uint8))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_inference_runs_without_json_sidecar(self):
        """
        Test B: Ensure NO .json sidecar exists on disk for doc_clean.png.
        extract_document_state and run_pipeline must succeed directly from image pixels.
        """
        sidecar_path = os.path.splitext(self.img_clean)[0] + ".json"
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

        self.assertFalse(os.path.exists(sidecar_path))

        # 1. Stage 1 live extraction directly from pixels
        doc = extract_document_state(self.img_clean)
        self.assertIn("extracted_values", doc)
        self.assertNotIn("ground_truth_clean", doc, "Stage 1 must NOT output ground_truth_clean!")
        self.assertNotIn("assay_ground_truth", doc)

        vals = doc["extracted_values"]
        self.assertIn("total_cholesterol", vals)
        self.assertAlmostEqual(vals["total_cholesterol"], 200.0, delta=2.0)
        self.assertAlmostEqual(vals["hdl"], 50.0, delta=2.0)
        self.assertAlmostEqual(vals["ldl"], 120.0, delta=2.0)
        self.assertAlmostEqual(vals["triglycerides"], 150.0, delta=2.0)

        # 2. Complete pipeline inference from pixels
        res = run_pipeline(self.img_clean)
        self.assertIn(res["status"], ["CONVERGED", "CONVERGED_DIRECT_ASSAY", "ABSTAIN_CLINICAL_ANOMALY"])
        self.assertIn("data", res)
        self.assertNotIn("ground_truth_clean", res)

    def test_empty_pitch_black_scan_abstains(self):
        """
        Test C: Pitch-black image returns an explicit abstention
        without crashing and without silently substituting clean answers.
        """
        doc = extract_document_state(self.img_black)
        self.assertEqual(len(doc["extracted_values"]), 0)
        self.assertEqual(len(doc["raw_ocr_tokens"]), 0)

        res = run_pipeline(self.img_black)
        self.assertEqual(res["status"], "ABSTAIN_UNREADABLE_OR_EMPTY_OCR")
        self.assertIn("missing_fields", res)
        self.assertEqual(len(res["data"]), 0)

    def test_synthetic_corruption_annotation_independence(self):
        """
        Test D: Synthetic sidecar annotations (e.g. corrupted_fields)
        have zero effect on live OCR extraction or confidences.
        """
        sidecar_path = os.path.splitext(self.img_clean)[0] + ".json"
        import json
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump({
                "corrupted_fields": ["total_cholesterol", "hdl", "ldl", "triglycerides"],
                "injected_display_data": {"total_cholesterol": 999.0},
                "ground_truth_clean": {"total_cholesterol": 123.4}
            }, f)

        try:
            # extract_document_state must NOT read the sidecar in production mode
            doc = extract_document_state(self.img_clean)
            self.assertNotEqual(doc["extracted_values"].get("total_cholesterol"), 999.0)
            self.assertAlmostEqual(doc["extracted_values"]["total_cholesterol"], 200.0, delta=2.0)
            self.assertNotIn("ground_truth_clean", doc)
        finally:
            if os.path.exists(sidecar_path):
                os.remove(sidecar_path)


class TestEvaluationHarnessIsolation(unittest.TestCase):
    """
    Regression suite for Phase 3: Ensuring evaluation harness never populates
    predictions with injected_display_data when OCR returns empty results.
    """

    def test_evaluation_harness_empty_ocr_does_not_leak_injected_display_data(self):
        fake_rec = {
            "doc_id": "test_empty_ocr_001",
            "category": "test_category",
            "error_scenario": "empty_ocr",
            "img_path": "nonexistent_or_black.png",
            "panel_type": "lipid",
            "ground_truth_clean": {"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0},
            "injected_display_data": {"total_cholesterol": 999.0, "hdl": 999.0, "ldl": 999.0, "triglycerides": 999.0},
            "bounding_boxes": {"total_cholesterol": [10, 10, 50, 30]},
            "unit": "mg/dL"
        }

        mock_rapid_engine = MagicMock()
        mock_rapid_engine.extract.return_value = {
            "image_path": fake_rec["img_path"],
            "image_bgr": None,
            "unit": "mg/dL",
            "method_text": "Method: Calculated by Friedewald Equation",
            "extracted_values": {},  # Complete OCR miss
            "bounding_boxes": {},
            "confidences": {},
            "raw_ocr_tokens": []
        }

        eval_res = run_evaluation_on_record(fake_rec, mode="rapidocr", rapid_engine=mock_rapid_engine)
        # Predictions must NOT be populated with 999.0 from injected_display_data
        self.assertEqual(eval_res["baseline_data"], {})
        self.assertEqual(eval_res["symbolic_data"], {})
        self.assertEqual(eval_res["full_data"], {})
        self.assertNotEqual(eval_res["baseline_data"].get("total_cholesterol"), 999.0)
        self.assertNotEqual(eval_res["full_data"].get("total_cholesterol"), 999.0)


if __name__ == "__main__":
    unittest.main()

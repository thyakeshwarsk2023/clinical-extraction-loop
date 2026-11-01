"""
tests/test_error_detection_and_decisions.py
============================================
Comprehensive Regression & Generalization Test Suite for BioTrace:
1. Clean valid reports -> All fields ACCEPT, CONVERGED.
2. Corrupted digits with image support -> Targeted field CORRECT, anchors ACCEPT.
3. Genuine stoichiometric / constraint discrepancy where physical image confirms
   the printed value -> FLAG / REVIEW (clinical rationale preserved, NOT withheld, NOT corrupted).
4. Blank / unreadable crop -> ABSTAIN on suspect field, clean anchors preserved.
5. Severe hypertriglyceridemia (TG > 800 mg/dL) -> ABSTAIN on calculated LDL, anchors ACCEPT.
6. Incomplete panel (missing analyte) -> ABSTAIN on missing field.
7. Unit conversion / live upload normalization (e.g. mmol/L, decimal formats).
"""

import os
import unittest
import numpy as np
import cv2

from src.stage0_classifier import AssayClassifier
from src.stage2_constraints import (
    normalize_to_mgdl,
    evaluate_friedewald_invariant,
    evaluate_sampson_invariant,
    calculate_sampson_ldl,
)
from src.stage2_isolation import hierarchical_isolation_search
from src.stage3_vision_gate import process_vision_gate, is_valid_bbox
from src.stage4_enhance_extract import enhance_and_reextract
from src.pipeline import run_pipeline
from src.report_generator import compute_field_decisions


class TestBioTraceErrorDetectionAndDecisions(unittest.TestCase):

    def setUp(self):
        # Create a clean synthetic clinical lab image
        self.img_h, self.img_w = 400, 600
        self.img_bgr = np.full((self.img_h, self.img_w, 3), 245, dtype=np.uint8)
        # Render clean text lines
        cv2.putText(self.img_bgr, "LABORATORY REPORT - LIPID PANEL", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        cv2.putText(self.img_bgr, "Total Cholesterol: 200.0 mg/dL", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.img_bgr, "HDL Cholesterol: 50.0 mg/dL", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.img_bgr, "LDL Cholesterol: 120.0 mg/dL", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.img_bgr, "Triglycerides: 150.0 mg/dL", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

        self.bboxes = {
            "total_cholesterol": [200, 85, 270, 105],
            "hdl": [180, 125, 240, 145],
            "ldl": [180, 165, 240, 185],
            "triglycerides": [170, 205, 230, 225],
        }

    # -------------------------------------------------------------------------
    # Test 1: Clean values with valid crop -> All 4 fields ACCEPT
    # -------------------------------------------------------------------------
    def test_clean_panel_all_fields_accepted(self):
        doc = {
            "image_bgr": self.img_bgr,
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

        res = run_pipeline("mock_clean.png", doc=doc)
        self.assertEqual(res["status"], "CONVERGED")
        self.assertAlmostEqual(res["residual"], 0.0, places=1)

        decisions = compute_field_decisions(
            extracted_values=doc["extracted_values"],
            final_values=res["data"],
            proposed_candidates={},
            bounding_boxes=doc["bounding_boxes"],
            img_shape=self.img_bgr.shape,
            final_status=res["status"],
            active_equation="Friedewald Equation (1972)",
            curr_residual=res["residual"],
            tolerance_pct=10.0,
            vision_cache={},
        )

        for f in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
            self.assertEqual(decisions[f]["canonical_decision"], "ACCEPT")
            self.assertEqual(decisions[f]["decision_status"], "RETAINED VALUE")
            self.assertIsNotNone(decisions[f]["accepted_value"])

    # -------------------------------------------------------------------------
    # Test 2: Deliberately corrupted digit with visual support -> CORRECT
    # -------------------------------------------------------------------------
    def test_corrupted_digit_corrected_with_visual_evidence(self):
        # OCR extracted 180.0 for LDL instead of true 120.0, but the image has 120.0
        doc = {
            "image_bgr": self.img_bgr,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 180.0,  # Corrupted extraction
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {"total_cholesterol": 0.95, "hdl": 0.95, "ldl": 0.55, "triglycerides": 0.95},
        }

        # Stage 2 search provides ranked plausible candidates
        search = hierarchical_isolation_search(doc["extracted_values"], doc["confidences"], tol=0.10)
        ranked = search.get("ranked_candidates", [search])
        cand_map = {c["field"]: c["target_val"] for c in ranked}
        self.assertIn("ldl", cand_map)
        self.assertAlmostEqual(cand_map["ldl"], 120.0, places=1)

        # Vision gate evaluation on crop
        bb = self.bboxes["ldl"]
        crop = self.img_bgr[bb[1]:bb[3], bb[0]:bb[2]]
        corr_val, new_conf, delta_c, _, qm = enhance_and_reextract(
            crop, candidate_target_val=120.0, current_conf=0.55, field="ldl", unit="mg/dL"
        )

        # When simulated/verified, decisions should mark LDL as CORRECT and anchors as ACCEPT
        decisions = compute_field_decisions(
            extracted_values=doc["extracted_values"],
            final_values={"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0},
            proposed_candidates={"ldl": 120.0},
            bounding_boxes=doc["bounding_boxes"],
            img_shape=self.img_bgr.shape,
            final_status="FULL_VISION_CORRECTED_K1",
            active_equation="Friedewald Equation (1972)",
            curr_residual=0.0,
            tolerance_pct=10.0,
            vision_cache={"ldl": {"qm": qm, "delta_c": delta_c}},
        )

        self.assertEqual(decisions["ldl"]["canonical_decision"], "CORRECT")
        self.assertEqual(decisions["ldl"]["decision_status"], "ACCEPTED CORRECTION")
        self.assertEqual(decisions["ldl"]["accepted_value"], 120.0)

        # Clean anchor fields must be ACCEPT, not withheld or modified!
        for anchor in ["total_cholesterol", "hdl", "triglycerides"]:
            self.assertEqual(decisions[anchor]["canonical_decision"], "ACCEPT")
            self.assertEqual(decisions[anchor]["decision_status"], "RETAINED VALUE")
            self.assertIsNotNone(decisions[anchor]["accepted_value"])

    # -------------------------------------------------------------------------
    # Test 3: Constraint violation where physical image confirms printed value -> FLAG / REVIEW
    # -------------------------------------------------------------------------
    def test_physical_stoichiometric_discrepancy_flags_for_review(self):
        """
        When the physical document actually has a discrepant number (e.g. transcription error
        on paper, or non-linear patient physiology), the system must NOT overwrite it with a
        fabricated number, NOR withhold all fields. It must emit FLAG / REVIEW.
        """
        # Document printed HDL = 35.0, but TC=200, LDL=120, TG=150 (equation gives 50.0)
        # Discrepancy = 15.0 mg/dL.
        img_discrepant = self.img_bgr.copy()
        cv2.putText(img_discrepant, "HDL Cholesterol: 35.0 mg/dL", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

        doc = {
            "image_bgr": img_discrepant,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 35.0,  # Physical paper really says 35.0
                "ldl": 120.0,
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.90 for f in self.bboxes},
        }

        decisions = compute_field_decisions(
            extracted_values=doc["extracted_values"],
            final_values=doc["extracted_values"],
            proposed_candidates={"hdl": 50.0},
            bounding_boxes=doc["bounding_boxes"],
            img_shape=img_discrepant.shape,
            final_status="FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY",
            active_equation="Friedewald Equation (1972)",
            curr_residual=15.0,
            tolerance_pct=10.0,
            vision_cache={},
        )

        # Suspect field should be FLAGGED FOR REVIEW, retaining original value
        self.assertEqual(decisions["hdl"]["canonical_decision"], "FLAG / REVIEW")
        self.assertEqual(decisions["hdl"]["decision_status"], "FLAGGED FOR REVIEW")
        self.assertEqual(decisions["hdl"]["accepted_value"], 35.0)
        self.assertIn("discrepancy", decisions["hdl"]["rationale"].lower())

        # Clean anchor fields must be ACCEPT
        for anchor in ["total_cholesterol", "ldl", "triglycerides"]:
            self.assertEqual(decisions[anchor]["canonical_decision"], "ACCEPT")
            self.assertEqual(decisions[anchor]["decision_status"], "RETAINED VALUE")
            self.assertIsNotNone(decisions[anchor]["accepted_value"])

    # -------------------------------------------------------------------------
    # Test 4: Blank / unreadable crop -> ABSTAIN on suspect field, anchors ACCEPT
    # -------------------------------------------------------------------------
    def test_unreadable_crop_abstains_without_wiping_clean_anchors(self):
        # Blank / collapsed crop for LDL
        img_blank_crop = self.img_bgr.copy()
        bb = self.bboxes["ldl"]
        img_blank_crop[bb[1]:bb[3], bb[0]:bb[2]] = 245  # wipe to background

        doc = {
            "image_bgr": img_blank_crop,
            "method_text": "Method: Calculated by Friedewald Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 200.0,
                "hdl": 50.0,
                "ldl": 999.0,  # Corrupted or unreadable
                "triglycerides": 150.0,
            },
            "bounding_boxes": self.bboxes,
            "confidences": {"total_cholesterol": 0.95, "hdl": 0.95, "ldl": 0.20, "triglycerides": 0.95},
        }

        decisions = compute_field_decisions(
            extracted_values=doc["extracted_values"],
            final_values={"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 999.0, "triglycerides": 150.0},
            proposed_candidates={"ldl": 120.0},
            bounding_boxes=doc["bounding_boxes"],
            img_shape=img_blank_crop.shape,
            final_status="FULL_ABSTAIN_VISUAL_DISAGREEMENT",
            active_equation="Friedewald Equation (1972)",
            curr_residual=879.0,
            tolerance_pct=10.0,
            vision_cache={},
        )

        # LDL should be ABSTAINED (Withheld)
        self.assertEqual(decisions["ldl"]["canonical_decision"], "ABSTAIN")
        self.assertEqual(decisions["ldl"]["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertIsNone(decisions["ldl"]["accepted_value"])

        # Anchors TC, HDL, TG must NOT be wiped out!
        for anchor in ["total_cholesterol", "hdl", "triglycerides"]:
            self.assertEqual(decisions[anchor]["canonical_decision"], "ACCEPT")
            self.assertEqual(decisions[anchor]["decision_status"], "RETAINED VALUE")
            self.assertIsNotNone(decisions[anchor]["accepted_value"])

    # -------------------------------------------------------------------------
    # Test 5: Severe Hypertriglyceridemia (TG > 800 mg/dL) -> ABSTAIN on LDL
    # -------------------------------------------------------------------------
    def test_severe_hypertriglyceridemia_abstention_on_ldl(self):
        doc = {
            "image_bgr": self.img_bgr,
            "method_text": "Method: Calculated by Sampson Equation",
            "unit": "mg/dL",
            "extracted_values": {
                "total_cholesterol": 350.0,
                "hdl": 30.0,
                "ldl": 120.0,
                "triglycerides": 950.0,  # Exceeds 800 mg/dL validity ceiling
            },
            "bounding_boxes": self.bboxes,
            "confidences": {f: 0.90 for f in self.bboxes},
        }

        res = run_pipeline("mock_high_tg.png", doc=doc)
        self.assertEqual(res["status"], "ABSTAIN_TG800")

        decisions = compute_field_decisions(
            extracted_values=doc["extracted_values"],
            final_values=res["data"],
            proposed_candidates={},
            bounding_boxes=doc["bounding_boxes"],
            img_shape=self.img_bgr.shape,
            final_status=res["status"],
            active_equation="Sampson Equation (NIH 2020 Equation 2)",
            curr_residual=res["residual"],
            tolerance_pct=10.0,
            vision_cache={},
        )

        # Calculated LDL must be ABSTAINED due to TG > 800
        self.assertEqual(decisions["ldl"]["canonical_decision"], "ABSTAIN")
        self.assertEqual(decisions["ldl"]["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertIsNone(decisions["ldl"]["accepted_value"])

        # TG, TC, HDL are directly measurable and should be retained
        self.assertEqual(decisions["triglycerides"]["canonical_decision"], "ACCEPT")
        self.assertEqual(decisions["triglycerides"]["decision_status"], "RETAINED VALUE")
        self.assertEqual(decisions["total_cholesterol"]["canonical_decision"], "ACCEPT")
        self.assertEqual(decisions["hdl"]["canonical_decision"], "ACCEPT")

    # -------------------------------------------------------------------------
    # Test 6: Missing analyte (e.g. HDL missing from scan) -> ABSTAIN on missing
    # -------------------------------------------------------------------------
    def test_missing_analyte_abstention(self):
        extracted = {
            "total_cholesterol": 200.0,
            "ldl": 120.0,
            "triglycerides": 150.0,
            # HDL is missing
        }
        decisions = compute_field_decisions(
            extracted_values=extracted,
            final_values=extracted,
            proposed_candidates={},
            bounding_boxes=self.bboxes,
            img_shape=self.img_bgr.shape,
            final_status="ABSTAIN_MISSING_FIELD",
            active_equation="Friedewald Equation (1972)",
            curr_residual=0.0,
            tolerance_pct=10.0,
            vision_cache={},
        )

        self.assertEqual(decisions["hdl"]["canonical_decision"], "ABSTAIN")
        self.assertEqual(decisions["hdl"]["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertIsNone(decisions["hdl"]["accepted_value"])

    # -------------------------------------------------------------------------
    # Test 7: Live upload unit conversion (mmol/L -> mg/dL)
    # -------------------------------------------------------------------------
    def test_live_upload_unit_conversion_mmol_per_liter(self):
        mmol_panel = {
            "total_cholesterol": 5.17,  # ~200 mg/dL
            "hdl": 1.29,                # ~50 mg/dL
            "ldl": 3.10,                # ~120 mg/dL
            "triglycerides": 1.70,      # ~150 mg/dL (1.70 * 88.57 = 150.57)
        }
        mgdl_panel = normalize_to_mgdl(mmol_panel, "mmol/L")

        self.assertAlmostEqual(mgdl_panel["total_cholesterol"], 200.0, delta=1.0)
        self.assertAlmostEqual(mgdl_panel["hdl"], 50.0, delta=1.0)
        self.assertAlmostEqual(mgdl_panel["ldl"], 120.0, delta=1.0)
        self.assertAlmostEqual(mgdl_panel["triglycerides"], 150.6, delta=1.0)

        # Evaluated invariant in mg/dL satisfies tolerance
        status, res_abs, rel_err = evaluate_friedewald_invariant(mgdl_panel, tolerance=0.10)
        self.assertEqual(status, "CONVERGED")


if __name__ == "__main__":
    unittest.main()

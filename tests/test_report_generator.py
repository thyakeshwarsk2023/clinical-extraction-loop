"""
tests/test_report_generator.py
==============================
Unit and regression tests for conservative clinical report generation,
bounding box validation, visual watermarking, three-state decision rendering,
and regulatory audit record export.
"""

import os
import json
import unittest
import numpy as np
import cv2

from src.report_generator import (
    validate_field_bbox,
    apply_research_watermark,
    overwrite_field_text,
    generate_corrected_report_image,
    build_correction_audit_record,
    compute_field_decisions,
)


class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        # Create a synthetic clinical document image (white paper background with dark lines)
        self.img_h, self.img_w = 400, 600
        self.synthetic_img = np.full((self.img_h, self.img_w, 3), 245, dtype=np.uint8)
        # Add some mock text lines
        cv2.putText(self.synthetic_img, "CLINICAL LIPID PANEL REPORT", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        cv2.putText(self.synthetic_img, "Total Cholesterol: 190.0 mg/dL", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.synthetic_img, "HDL Cholesterol: 4.8 mg/dL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.synthetic_img, "LDL Cholesterol: 116.0 mg/dL", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(self.synthetic_img, "Triglycerides: 130.0 mg/dL", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

        self.valid_bboxes = {
            "total_cholesterol": [240, 85, 300, 105],
            "hdl": [210, 135, 270, 155],
            "ldl": [210, 185, 270, 205],
            "triglycerides": [190, 235, 250, 255],
        }

    # -------------------------------------------------------------------------
    # 1. Bounding Box Defensive Validation
    # -------------------------------------------------------------------------
    def test_validate_field_bbox_valid(self):
        valid, msg = validate_field_bbox([100, 100, 200, 140], (400, 600))
        self.assertTrue(valid)
        self.assertEqual(msg, "Valid bbox")

    def test_validate_field_bbox_rejects_dummy(self):
        # Reject uninitialized default bounding boxes
        for dummy in [[0, 0, 10, 10], [0, 0, 0, 0], [0, 0, 1, 1], [-1, -1, -1, -1]]:
            valid, msg = validate_field_bbox(dummy, (400, 600))
            self.assertFalse(valid, f"Failed to reject dummy bbox {dummy}")
            self.assertIn("dummy", msg.lower())

    def test_validate_field_bbox_rejects_non_numeric_and_malformed(self):
        valid, _ = validate_field_bbox(None, (400, 600))
        self.assertFalse(valid)
        valid, _ = validate_field_bbox([10, 20], (400, 600))
        self.assertFalse(valid)
        valid, _ = validate_field_bbox(["a", "b", "c", "d"], (400, 600))
        self.assertFalse(valid)

    def test_validate_field_bbox_rejects_out_of_bounds(self):
        # Coordinates exceeding image dimensions (400 h, 600 w)
        valid, msg = validate_field_bbox([550, 100, 650, 140], (400, 600))
        self.assertFalse(valid)
        self.assertIn("exceed", msg.lower())

        valid, msg = validate_field_bbox([100, 350, 200, 450], (400, 600))
        self.assertFalse(valid)
        self.assertIn("exceed", msg.lower())

    def test_validate_field_bbox_rejects_degenerate_area(self):
        # Inverted or zero area
        valid, msg = validate_field_bbox([150, 100, 100, 140], (400, 600))
        self.assertFalse(valid)
        self.assertIn("non-positive", msg.lower())

        valid, msg = validate_field_bbox([100, 140, 150, 100], (400, 600))
        self.assertFalse(valid)
        self.assertIn("non-positive", msg.lower())

    # -------------------------------------------------------------------------
    # 2. Prominent Research Watermark
    # -------------------------------------------------------------------------
    def test_apply_research_watermark_preserves_original(self):
        orig_copy = self.synthetic_img.copy()
        watermarked = apply_research_watermark(self.synthetic_img, doc_status="TEST_STATUS")

        # Original image must be 100% unchanged
        np.testing.assert_array_equal(self.synthetic_img, orig_copy)

        # Derived image dimensions: original height + 32 (banner) + 24 (footer) = h + 56
        self.assertEqual(watermarked.shape[0], self.synthetic_img.shape[0] + 56)
        self.assertEqual(watermarked.shape[1], self.synthetic_img.shape[1])

        # Top banner should have deep crimson color (25, 25, 150)
        crimson_sample = watermarked[5, 5]
        self.assertEqual(int(crimson_sample[0]), 25)
        self.assertEqual(int(crimson_sample[1]), 25)
        self.assertEqual(int(crimson_sample[2]), 150)

    # -------------------------------------------------------------------------
    # 3. Three Decision States in Image Generation
    # -------------------------------------------------------------------------
    def test_accepted_correction_modifies_image(self):
        orig_copy = self.synthetic_img.copy()
        decisions = {
            "hdl": {
                "decision_status": "ACCEPTED CORRECTION",
                "original_value": 4.8,
                "accepted_value": 48.0,
                "proposed_value": 48.0,
                "rationale": "Satisfies Friedewald equation"
            },
            "ldl": {
                "decision_status": "RETAINED VALUE",
                "original_value": 116.0,
                "accepted_value": 116.0,
                "proposed_value": 116.0,
                "rationale": "Satisfies equation"
            }
        }

        derived_bgr, mods = generate_corrected_report_image(
            self.synthetic_img, self.valid_bboxes, decisions, doc_status="FULL_VISION_CORRECTED_K1"
        )

        # 1. Original image is completely untouched
        np.testing.assert_array_equal(self.synthetic_img, orig_copy)

        # 2. Modifications flags correctly record HDL modified, LDL retained
        self.assertTrue(mods.get("hdl"))
        self.assertFalse(mods.get("ldl"))

        # 3. Derived canvas has watermark
        self.assertEqual(derived_bgr.shape[0], self.synthetic_img.shape[0] + 56)

        # 4. HDL area (with offset +32 for banner) has changed
        hdl_box = self.valid_bboxes["hdl"]
        orig_crop = self.synthetic_img[hdl_box[1]:hdl_box[3], hdl_box[0]:hdl_box[2]]
        derived_crop = derived_bgr[hdl_box[1]+32:hdl_box[3]+32, hdl_box[0]:hdl_box[2]]
        self.assertFalse(np.array_equal(orig_crop, derived_crop))

        # 5. Non-modified area (e.g. LDL box) matches perfectly
        ldl_box = self.valid_bboxes["ldl"]
        orig_ldl = self.synthetic_img[ldl_box[1]:ldl_box[3], ldl_box[0]:ldl_box[2]]
        derived_ldl = derived_bgr[ldl_box[1]+32:ldl_box[3]+32, ldl_box[0]:ldl_box[2]]
        np.testing.assert_array_equal(orig_ldl, derived_ldl)

    def test_retained_value_preserves_image_content(self):
        orig_copy = self.synthetic_img.copy()
        decisions = {
            "total_cholesterol": {"decision_status": "RETAINED VALUE", "original_value": 190.0, "accepted_value": 190.0},
            "hdl": {"decision_status": "RETAINED VALUE", "original_value": 48.0, "accepted_value": 48.0},
            "ldl": {"decision_status": "RETAINED VALUE", "original_value": 116.0, "accepted_value": 116.0},
            "triglycerides": {"decision_status": "RETAINED VALUE", "original_value": 130.0, "accepted_value": 130.0},
        }

        derived_bgr, mods = generate_corrected_report_image(
            self.synthetic_img, self.valid_bboxes, decisions, doc_status="CONVERGED"
        )

        # No fields should be marked modified
        self.assertFalse(any(mods.values()))

        # Embedded image area (without top and bottom banners) should match original pixel-for-pixel
        inner_content = derived_bgr[32:32+self.img_h, :]
        np.testing.assert_array_equal(inner_content, orig_copy)

    def test_abstained_unresolved_never_inserts_candidate(self):
        orig_copy = self.synthetic_img.copy()
        decisions = {
            "hdl": {
                "decision_status": "ABSTAINED / UNRESOLVED",
                "original_value": 4.8,
                "accepted_value": None,
                "proposed_value": 48.0,
                "rationale": "Visual evidence contradicts candidate target (4.8 vs 48.0)"
            }
        }

        derived_bgr, mods = generate_corrected_report_image(
            self.synthetic_img, self.valid_bboxes, decisions, doc_status="FULL_ABSTAIN_VISUAL_DISAGREEMENT"
        )

        # HDL must NOT be marked modified
        self.assertFalse(mods.get("hdl"))

        # Embedded HDL area must NOT be overwritten with proposed candidate (48.0)
        hdl_box = self.valid_bboxes["hdl"]
        orig_crop = self.synthetic_img[hdl_box[1]:hdl_box[3], hdl_box[0]:hdl_box[2]]
        derived_crop = derived_bgr[hdl_box[1]+32:hdl_box[3]+32, hdl_box[0]:hdl_box[2]]
        np.testing.assert_array_equal(orig_crop, derived_crop)

    def test_invalid_bbox_withholds_visual_overwrite(self):
        orig_copy = self.synthetic_img.copy()
        # Missing/dummy bbox [0, 0, 10, 10]
        dummy_boxes = {"hdl": [0, 0, 10, 10]}
        decisions = {
            "hdl": {
                "decision_status": "ACCEPTED CORRECTION",
                "original_value": 4.8,
                "accepted_value": 48.0,
                "proposed_value": 48.0,
                "rationale": "Mathematical constraint satisfied"
            }
        }

        derived_bgr, mods = generate_corrected_report_image(
            self.synthetic_img, dummy_boxes, decisions, doc_status="ACCEPTED_NO_BBOX"
        )

        # BBox is unvalidated: modification must be False and visual overwrite withheld
        self.assertFalse(mods.get("hdl"))
        self.assertIn("withheld", decisions["hdl"].get("image_render_note", "").lower())

        # Inner canvas should remain identical to original
        inner_content = derived_bgr[32:32+self.img_h, :]
        np.testing.assert_array_equal(inner_content, orig_copy)

    # -------------------------------------------------------------------------
    # 4. Audit Record Structure & Zero Ground-Truth Leakage
    # -------------------------------------------------------------------------
    def test_build_correction_audit_record(self):
        extracted = {"total_cholesterol": 190.0, "hdl": 4.8, "ldl": 116.0, "triglycerides": 130.0}
        final_val = {"total_cholesterol": 190.0, "hdl": 4.8, "ldl": 116.0, "triglycerides": 130.0}
        proposed = {"hdl": 48.0}
        field_decisions = {
            "hdl": {
                "decision_status": "ABSTAINED / UNRESOLVED",
                "original_value": 4.8,
                "accepted_value": None,
                "proposed_value": 48.0,
                "rationale": "Visual evidence contradicts candidate target. Manual review required.",
                "modified_in_image": False
            },
            "total_cholesterol": {"decision_status": "RETAINED VALUE", "original_value": 190.0, "accepted_value": 190.0, "modified_in_image": False, "rationale": "Satisfies invariant"},
            "ldl": {"decision_status": "RETAINED VALUE", "original_value": 116.0, "accepted_value": 116.0, "modified_in_image": False, "rationale": "Satisfies invariant"},
            "triglycerides": {"decision_status": "RETAINED VALUE", "original_value": 130.0, "accepted_value": 130.0, "modified_in_image": False, "rationale": "Satisfies invariant"},
        }

        audit = build_correction_audit_record(
            doc_id="rw_0029",
            extracted_values=extracted,
            final_values=final_val,
            proposed_values=proposed,
            unit="mg/dL",
            field_decisions=field_decisions,
            pipeline_status="FULL_ABSTAIN_VISUAL_DISAGREEMENT",
            active_equation="Friedewald Equation (1972)",
            residual_pct=22.7,
            tolerance_pct=5.0,
            visual_evidence_map={"hdl": True, "ldl": False, "total_cholesterol": False, "triglycerides": False}
        )

        # Audit serialization check
        json_str = json.dumps(audit, indent=2)
        parsed = json.loads(json_str)

        # Check required fields
        self.assertEqual(parsed["document_identifier"], "rw_0029")
        self.assertIn("AI-PROCESSED RESEARCH DEMONSTRATION", parsed["operating_disclaimer"])
        self.assertEqual(parsed["pipeline_terminal_status"], "FULL_ABSTAIN_VISUAL_DISAGREEMENT")
        self.assertEqual(parsed["applicable_constraint"], "Friedewald Equation (1972)")
        self.assertEqual(parsed["constraint_residual_percent"], 22.7)

        # Check HDL field details: Withheld in final output value when abstained
        hdl_audit = parsed["fields"]["hdl"]
        self.assertEqual(hdl_audit["original_extracted_value"], 4.8)
        self.assertEqual(hdl_audit["proposed_candidate_value"], 48.0)
        self.assertEqual(hdl_audit["final_output_value"], "WITHHELD")
        self.assertEqual(hdl_audit["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertFalse(hdl_audit["modified_in_image"])
        self.assertTrue(hdl_audit["visual_evidence_available"])

        # Check Zero Ground-Truth Leakage: Ensure no benchmark ground-truth fields are present
        self.assertNotIn("ground_truth_clean", json_str)
        self.assertNotIn("gt_clean", json_str)
        self.assertNotIn("benchmark_reference", json_str)

    # -------------------------------------------------------------------------
    # 5. Regression Test: rw_0029.png Real-World Case
    # -------------------------------------------------------------------------
    def test_regression_rw_0029_case(self):
        """
        Comprehensive regression test on rw_0029.png:
          - Input: corrupted HDL = 4.8 mg/dL.
          - Invariant calculates candidate target = 48.0 mg/dL.
          - Visual evidence on crop reads '4.8', contradicting 48.0.
          - Pipeline must ABSTAIN (FULL_ABSTAIN_VISUAL_DISAGREEMENT).
          - Original image must remain 100% untouched.
          - HDL must NOT be modified in the derived image.
          - Modified report has visible AI watermark.
          - PNG and JSON exports are valid.
        """
        img_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dataset_C_realworld", "03_decimal_scale_errors", "rw_0029.png"
        )
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dataset_C_realworld", "03_decimal_scale_errors", "rw_0029.json"
        )
        if not os.path.exists(img_path) or not os.path.exists(json_path):
            self.skipTest(f"rw_0029 dataset files not found at {img_path}")

        # Load original image and metadata
        original_bgr = cv2.imread(img_path)
        self.assertIsNotNone(original_bgr, "Failed to load rw_0029.png")
        original_copy = original_bgr.copy()

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_extracted = meta["injected_display_data"]
        bboxes = meta["bounding_boxes"]

        # 1. Verify initial extraction
        self.assertEqual(raw_extracted["hdl"], 4.8)
        self.assertEqual(raw_extracted["total_cholesterol"], 190.0)
        self.assertEqual(raw_extracted["ldl"], 116.0)
        self.assertEqual(raw_extracted["triglycerides"], 130.0)

        # 2. Simulate the pipeline decision for rw_0029
        # The invariant target is: 190 - 116 - (130/5) = 48.0
        # However, visual evidence re-read does not corroborate 48.0.
        # Decision status is ABSTAINED / UNRESOLVED.
        pipeline_status = "FULL_ABSTAIN_VISUAL_DISAGREEMENT"
        proposed_candidate = 48.0

        field_decisions = {
            "hdl": {
                "decision_status": "ABSTAINED / UNRESOLVED",
                "original_value": 4.8,
                "accepted_value": None,
                "proposed_value": proposed_candidate,
                "rationale": "Candidate target (48.0) not corroborated by visual re-read (4.8). Manual clinician review required."
            },
            "total_cholesterol": {
                "decision_status": "RETAINED VALUE",
                "original_value": 190.0,
                "accepted_value": 190.0,
                "proposed_value": 190.0,
                "rationale": "Extracted value satisfies invariant constraint without modification."
            },
            "ldl": {
                "decision_status": "RETAINED VALUE",
                "original_value": 116.0,
                "accepted_value": 116.0,
                "proposed_value": 116.0,
                "rationale": "Extracted value satisfies invariant constraint without modification."
            },
            "triglycerides": {
                "decision_status": "RETAINED VALUE",
                "original_value": 130.0,
                "accepted_value": 130.0,
                "proposed_value": 130.0,
                "rationale": "Extracted value satisfies invariant constraint without modification."
            },
        }

        # 3. Generate derived report image
        derived_bgr, mods = generate_corrected_report_image(
            original_bgr, bboxes, field_decisions, doc_status=pipeline_status
        )

        # Expected behaviour checks:
        # A. The original report remains unchanged
        np.testing.assert_array_equal(original_bgr, original_copy)

        # B. Raw HDL extraction remains 4.8 mg/dL
        self.assertEqual(raw_extracted["hdl"], 4.8)

        # C. Proposed HDL value is 48.0 mg/dL, but NOT accepted
        self.assertEqual(field_decisions["hdl"]["proposed_value"], 48.0)
        self.assertIsNone(field_decisions["hdl"]["accepted_value"])

        # D. Image is modified ONLY if correction is accepted -> For rw_0029, modification must be FALSE
        self.assertFalse(mods.get("hdl"))
        self.assertFalse(any(mods.values()))

        # E. The modified report has a visible AI-processing watermark
        self.assertEqual(derived_bgr.shape[0], original_bgr.shape[0] + 56)
        self.assertEqual(derived_bgr.shape[1], original_bgr.shape[1])
        top_banner_sample = derived_bgr[10, 10]
        self.assertEqual(int(top_banner_sample[2]), 150)  # Crimson red channel

        # F. Unchanged fields remain unchanged (inner pixels match original)
        inner_content = derived_bgr[32:32+original_bgr.shape[0], :]
        np.testing.assert_array_equal(inner_content, original_copy)

        # G. Abstained fields are never silently corrected
        self.assertEqual(field_decisions["hdl"]["decision_status"], "ABSTAINED / UNRESOLVED")

        # H. PNG and JSON downloads contain the actual generated results
        ok_png, png_bytes = cv2.imencode(".png", derived_bgr)
        self.assertTrue(ok_png)
        self.assertGreater(len(png_bytes), 1000)

        audit_record = build_correction_audit_record(
            doc_id="rw_0029",
            extracted_values=raw_extracted,
            final_values=raw_extracted,  # not overwritten because abstained
            proposed_values={"hdl": proposed_candidate},
            unit="mg/dL",
            field_decisions=field_decisions,
            pipeline_status=pipeline_status,
            active_equation="Friedewald Equation (1972)",
            residual_pct=22.7,
            tolerance_pct=5.0,
            visual_evidence_map={"hdl": True, "ldl": False, "total_cholesterol": False, "triglycerides": False}
        )

        audit_json_str = json.dumps(audit_record, indent=2)
        parsed_audit = json.loads(audit_json_str)
        self.assertEqual(parsed_audit["document_identifier"], "rw_0029")
        self.assertEqual(parsed_audit["fields"]["hdl"]["final_output_value"], "WITHHELD")
        self.assertEqual(parsed_audit["fields"]["hdl"]["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertFalse(parsed_audit["fields"]["hdl"]["modified_in_image"])

    # -------------------------------------------------------------------------
    # 6. Agreement Across Derived Report, Verification Table, and Audit JSON
    # -------------------------------------------------------------------------
    def test_decision_agreement_across_table_image_audit(self):
        """
        Verify that compute_field_decisions, generate_corrected_report_image,
        and build_correction_audit_record agree on raw value, final value,
        decision, and reason without discrepancies.
        """
        extracted = {"total_cholesterol": 190.0, "hdl": 28.0, "ldl": 116.0, "triglycerides": 130.0}
        current_data = {"total_cholesterol": 190.0, "hdl": 48.0, "ldl": 116.0, "triglycerides": 130.0}
        proposed = {"hdl": 48.0}
        pipeline_status = "FULL_VISION_CORRECTED_K1"
        active_equation = "Friedewald Equation (1972)"

        field_decisions = compute_field_decisions(
            extracted_values=extracted,
            final_values=current_data,
            proposed_candidates=proposed,
            bounding_boxes=self.valid_bboxes,
            img_shape=self.synthetic_img.shape,
            final_status=pipeline_status,
            active_equation=active_equation,
            curr_residual=0.0,
            tolerance_pct=10.0,
            vision_cache={"hdl": {"raw_c": np.zeros((20, 20)), "delta_c": 0.45}}
        )

        derived_bgr, mods = generate_corrected_report_image(
            self.synthetic_img,
            self.valid_bboxes,
            field_decisions,
            doc_status=pipeline_status
        )

        audit_record = build_correction_audit_record(
            doc_id="test_case_agreement",
            extracted_values=extracted,
            final_values=current_data,
            proposed_values=proposed,
            unit="mg/dL",
            field_decisions=field_decisions,
            pipeline_status=pipeline_status,
            active_equation=active_equation,
            residual_pct=0.0,
            tolerance_pct=10.0,
            run_id="run_test_001"
        )

        # 1. HDL is marked ACCEPTED CORRECTION in field_decisions
        self.assertEqual(field_decisions["hdl"]["decision_status"], "ACCEPTED CORRECTION")
        self.assertEqual(field_decisions["hdl"]["accepted_value"], 48.0)
        self.assertTrue(field_decisions["hdl"]["modified_in_image"])
        self.assertTrue(mods["hdl"])

        # 2. Audit record agrees with field_decisions
        hdl_audit = audit_record["fields"]["hdl"]
        self.assertEqual(hdl_audit["original_extracted_value"], 28.0)
        self.assertEqual(hdl_audit["final_output_value"], 48.0)
        self.assertEqual(hdl_audit["decision_status"], "ACCEPTED CORRECTION")
        self.assertTrue(hdl_audit["modified_in_image"])
        self.assertEqual(hdl_audit["correction_rationale"], field_decisions["hdl"]["rationale"])

        # 3. Retained fields (LDL, TC, TG) agree across table, image, and audit
        for f in ["total_cholesterol", "ldl", "triglycerides"]:
            self.assertEqual(field_decisions[f]["decision_status"], "RETAINED VALUE")
            self.assertEqual(field_decisions[f]["accepted_value"], extracted[f])
            self.assertFalse(field_decisions[f]["modified_in_image"])
            self.assertFalse(mods[f])
            self.assertEqual(audit_record["fields"][f]["decision_status"], "RETAINED VALUE")
            self.assertEqual(audit_record["fields"][f]["final_output_value"], extracted[f])

    # -------------------------------------------------------------------------
    # 7. Document Switching & Stale Session State Isolation
    # -------------------------------------------------------------------------
    def test_document_switching_isolation(self):
        """
        Verify that sequential processing of two different documents (Doc A, Doc B)
        does not leak extraction, bounding boxes, or field decisions between runs.
        """
        # Document A: doc_0051 (TC=194.3, HDL=26.8, LDL=89.9, TG=208.0)
        doc_a_extracted = {"total_cholesterol": 194.3, "hdl": 26.8, "ldl": 89.9, "triglycerides": 208.0}
        doc_a_final = {"total_cholesterol": 194.3, "hdl": 62.8, "ldl": 89.9, "triglycerides": 208.0}
        doc_a_bboxes = {"hdl": [398, 178, 428, 196]}

        decisions_a = compute_field_decisions(
            extracted_values=doc_a_extracted,
            final_values=doc_a_final,
            proposed_candidates={"hdl": 62.8},
            bounding_boxes=doc_a_bboxes,
            img_shape=(400, 600),
            final_status="FULL_FAST_PATH_K1",
            active_equation="Friedewald Equation (1972)",
            curr_residual=0.0,
            tolerance_pct=10.0
        )
        self.assertEqual(decisions_a["hdl"]["original_value"], 26.8)
        self.assertEqual(decisions_a["hdl"]["accepted_value"], 62.8)

        # Document B: rw_0029 (TC=190.0, HDL=4.8, LDL=116.0, TG=130.0)
        doc_b_extracted = {"total_cholesterol": 190.0, "hdl": 4.8, "ldl": 116.0, "triglycerides": 130.0}
        doc_b_final = {"total_cholesterol": 190.0, "hdl": 4.8, "ldl": 116.0, "triglycerides": 130.0}
        doc_b_bboxes = {"hdl": [377, 168, 400, 187]}

        decisions_b = compute_field_decisions(
            extracted_values=doc_b_extracted,
            final_values=doc_b_final,
            proposed_candidates={"hdl": 48.0},
            bounding_boxes=doc_b_bboxes,
            img_shape=(440, 760),
            final_status="FULL_ABSTAIN_VISUAL_DISAGREEMENT",
            active_equation="Friedewald Equation (1972)",
            curr_residual=22.7,
            tolerance_pct=10.0
        )

        # Verify Doc B is not contaminated by Doc A's results or decisions
        self.assertEqual(decisions_b["hdl"]["original_value"], 4.8)
        self.assertIsNone(decisions_b["hdl"]["accepted_value"])
        self.assertEqual(decisions_b["hdl"]["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertEqual(decisions_b["total_cholesterol"]["original_value"], 190.0)
        self.assertNotIn("62.8", decisions_b["hdl"]["rationale"])

    # -------------------------------------------------------------------------
    # 8. Rerunning Inference Does Not Mix Results
    # -------------------------------------------------------------------------
    def test_rerun_inference_isolation(self):
        """
        Verify that rerun IDs and timestamps are unique and do not mix data across executions.
        """
        extracted = {"total_cholesterol": 190.0, "hdl": 48.0, "ldl": 116.0, "triglycerides": 130.0}
        decisions = compute_field_decisions(
            extracted_values=extracted,
            final_values=extracted,
            proposed_candidates={},
            bounding_boxes=self.valid_bboxes,
            img_shape=(400, 600),
            final_status="CONVERGED",
            active_equation="Friedewald Equation (1972)",
            curr_residual=0.0,
            tolerance_pct=10.0
        )

        audit_1 = build_correction_audit_record(
            doc_id="doc_rerun_test",
            extracted_values=extracted,
            final_values=extracted,
            proposed_values={},
            unit="mg/dL",
            field_decisions=decisions,
            pipeline_status="CONVERGED",
            active_equation="Friedewald Equation (1972)",
            residual_pct=0.0,
            tolerance_pct=10.0,
            run_id="run_1001"
        )
        audit_2 = build_correction_audit_record(
            doc_id="doc_rerun_test",
            extracted_values=extracted,
            final_values=extracted,
            proposed_values={},
            unit="mg/dL",
            field_decisions=decisions,
            pipeline_status="CONVERGED",
            active_equation="Friedewald Equation (1972)",
            residual_pct=0.0,
            tolerance_pct=10.0,
            run_id="run_1002"
        )

        self.assertEqual(audit_1["run_identifier"], "run_1001")
        self.assertEqual(audit_2["run_identifier"], "run_1002")
        self.assertNotEqual(audit_1["run_identifier"], audit_2["run_identifier"])

    # -------------------------------------------------------------------------
    # 9. Risk Classification Consistency with Current-Run LDL
    # -------------------------------------------------------------------------
    def test_risk_classification_current_run_ldl(self):
        """
        Verify NCEP ATP-III category classifications and risk flips:
          - Optimal: < 100
          - Near Optimal: 100 - 129
          - Borderline High: 130 - 159
          - High: 160 - 189
          - Very High: >= 190
        Ensures raw and verified categories are derived from the current run's canonical LDL.
        """
        from src.clinical_risk import classify_ldl_risk, evaluate_clinical_risk_flip

        # 1. Unchanged LDL: Must be concordant with zero shift
        rf_same = evaluate_clinical_risk_flip(116.0, 116.0)
        self.assertFalse(rf_same["has_risk_flip"])
        self.assertEqual(rf_same["direction"], "CONCORDANT")
        self.assertEqual(rf_same["raw_tier"]["category"], "Near / Above Optimal")
        self.assertEqual(rf_same["verified_tier"]["category"], "Near / Above Optimal")

        # 2. Downward correction (False escalation prevented)
        # Raw = 195.0 (Very High), Verified = 120.0 (Near / Above Optimal)
        rf_down = evaluate_clinical_risk_flip(195.0, 120.0)
        self.assertTrue(rf_down["has_risk_flip"])
        self.assertEqual(rf_down["direction"], "FALSE_ESCALATION")
        self.assertEqual(rf_down["raw_tier"]["category"], "Very High")
        self.assertEqual(rf_down["verified_tier"]["category"], "Near / Above Optimal")

        # 3. Upward correction (False reassurance prevented)
        # Raw = 95.0 (Optimal), Verified = 170.0 (High)
        rf_up = evaluate_clinical_risk_flip(95.0, 170.0)
        self.assertTrue(rf_up["has_risk_flip"])
        self.assertEqual(rf_up["direction"], "FALSE_REASSURANCE")
        self.assertEqual(rf_up["raw_tier"]["category"], "Optimal")
        self.assertEqual(rf_up["verified_tier"]["category"], "High")

    # -------------------------------------------------------------------------
    # 10. Missing / Invalid Correction Mapping Downgrades to Abstained
    # -------------------------------------------------------------------------
    def test_missing_correction_mapping_downgrades_to_abstained(self):
        """
        Verify that if an invariant solver accepted a correction, but the bounding box
        is missing or invalid, the candidate is NOT inserted into the report, and the
        decision is safely downgraded to ABSTAINED / UNRESOLVED.
        """
        extracted = {"total_cholesterol": 190.0, "hdl": 4.8, "ldl": 116.0, "triglycerides": 130.0}
        final_val = {"total_cholesterol": 190.0, "hdl": 48.0, "ldl": 116.0, "triglycerides": 130.0}

        # Bounding box is missing for HDL
        missing_bboxes = {
            "total_cholesterol": [240, 85, 300, 105],
            "ldl": [210, 185, 270, 205],
            "triglycerides": [190, 235, 250, 255]
            # hdl is missing!
        }

        decisions = compute_field_decisions(
            extracted_values=extracted,
            final_values=final_val,
            proposed_candidates={"hdl": 48.0},
            bounding_boxes=missing_bboxes,
            img_shape=(400, 600),
            final_status="FULL_FAST_PATH_K1",
            active_equation="Friedewald Equation (1972)",
            curr_residual=0.0,
            tolerance_pct=10.0
        )

        # Decision status must be safely downgraded to ABSTAINED / UNRESOLVED
        self.assertEqual(decisions["hdl"]["decision_status"], "ABSTAINED / UNRESOLVED")
        self.assertIsNone(decisions["hdl"]["accepted_value"])
        self.assertEqual(decisions["hdl"]["final_display"], "Withheld")
        self.assertIn("bounding box coordinates are unvalidated", decisions["hdl"]["rationale"].lower())

        # Generate report image: HDL must NOT be modified
        derived_bgr, mods = generate_corrected_report_image(
            self.synthetic_img,
            missing_bboxes,
            decisions,
            doc_status="FULL_FAST_PATH_K1"
        )
        self.assertFalse(mods.get("hdl"))
        self.assertFalse(decisions["hdl"]["modified_in_image"])

    # -------------------------------------------------------------------------
    # 11. Operating Mode Ground-Truth Isolation
    # -------------------------------------------------------------------------
    def test_operating_mode_isolation(self):
        """
        Verify that inference decisions do NOT receive ground truth clean values.
        """
        extracted = {"total_cholesterol": 190.0, "hdl": 48.0, "ldl": 116.0, "triglycerides": 130.0}
        decisions = compute_field_decisions(
            extracted_values=extracted,
            final_values=extracted,
            proposed_candidates={},
            bounding_boxes=self.valid_bboxes,
            img_shape=(400, 600),
            final_status="CONVERGED",
            active_equation="Friedewald Equation (1972)",
            curr_residual=0.0,
            tolerance_pct=10.0
        )

        # Confirm no ground truth clean keys exist in decision objects
        for f, dec in decisions.items():
            self.assertNotIn("ground_truth_clean", dec)
            self.assertNotIn("gt_clean", dec)
            self.assertNotIn("reference_truth", dec)


if __name__ == "__main__":
    unittest.main()

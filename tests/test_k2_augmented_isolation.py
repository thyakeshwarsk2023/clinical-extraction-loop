"""
tests/test_k2_augmented_isolation.py
=============================================================================
Unit tests for K=2 Dual-Field Fault Resolution via Extra-Field Equation Augmentation.

Tests:
  - An underdetermined K=2 case (unsolvable with 1 core identity under require_independent_evidence)
    becomes solvable in closed-form when augmented with a high-confidence extra equation (non_hdl or vldl).
  - A K=2 case with low-confidence extra fields (< 0.80 threshold) discards the untrusted equation
    and correctly abstains with ABSTAIN_ALGEBRAIC_INDETERMINATE.
  - Closed-form recovery of ground truth analyte values across multiple field pairs ({TC, LDL}, {LDL, TG}).
  - Direct unit verification of solve_k2_augmented_closed_form.
"""

import unittest
from src.stage2_isolation import (
    hierarchical_isolation_search,
    solve_k2_augmented_closed_form
)


class TestK2AugmentedIsolation(unittest.TestCase):

    def setUp(self):
        # Ground truth biological profile:
        # TC=200.0, HDL=50.0, LDL=120.0, TG=150.0
        # Expected Non-HDL = 200.0 - 50.0 = 150.0
        # Expected VLDL = 150.0 / 5.0 = 30.0
        self.clean_panel = {
            "total_cholesterol": 200.0,
            "hdl": 50.0,
            "ldl": 120.0,
            "triglycerides": 150.0,
            "non_hdl_cholesterol": 150.0,
            "vldl": 30.0
        }

    def test_k2_unsolvable_with_one_equation_becomes_solvable_with_non_hdl(self):
        """
        Corrupts TC and LDL: TC 200 -> 915, LDL 120 -> 550.
        All single-field targets have edit distance >= 3 or violate bounds.
        With only 1 equation (core Friedewald), system is underdetermined and abstains.
        When augmented with high-confidence non_hdl (150.0, conf 0.95), it solves in closed-form.
        """
        corrupted_panel = {
            "total_cholesterol": 915.0,
            "hdl": 50.0,
            "ldl": 550.0,
            "triglycerides": 150.0,
        }
        confidences = {
            "total_cholesterol": 0.85,
            "hdl": 0.95,
            "ldl": 0.85,
            "triglycerides": 0.95
        }

        # 1. Without extra field, underdetermined K=2 must abstain
        res_no_extra = hierarchical_isolation_search(
            corrupted_panel,
            confidences=confidences,
            require_independent_evidence=True
        )
        self.assertEqual(res_no_extra["action"], "ABSTAIN_ALGEBRAIC_INDETERMINATE")

        # 2. Add high-confidence non_hdl (conf = 0.95 >= 0.80)
        panel_with_extra = corrupted_panel.copy()
        panel_with_extra["non_hdl_cholesterol"] = 150.0
        conf_with_extra = confidences.copy()
        conf_with_extra["non_hdl_cholesterol"] = 0.95

        res_with_extra = hierarchical_isolation_search(
            panel_with_extra,
            confidences=conf_with_extra,
            require_independent_evidence=True
        )
        self.assertEqual(res_with_extra["action"], "CORRECT_K2")
        self.assertEqual(res_with_extra.get("resolution_mechanism"), "extra_field_equation")
        self.assertAlmostEqual(res_with_extra["corrections"]["total_cholesterol"], 200.0, places=1)
        self.assertAlmostEqual(res_with_extra["corrections"]["ldl"], 120.0, places=1)

    def test_k2_with_low_confidence_extra_field_correctly_abstains(self):
        """
        Extra field is present but has low confidence (0.55 < 0.80 threshold).
        Must NOT be blindly trusted; system remains underdetermined and abstains.
        """
        corrupted_panel = {
            "total_cholesterol": 915.0,
            "hdl": 50.0,
            "ldl": 550.0,
            "triglycerides": 150.0,
            "non_hdl_cholesterol": 150.0
        }
        confidences = {
            "total_cholesterol": 0.85,
            "hdl": 0.95,
            "ldl": 0.85,
            "triglycerides": 0.95,
            "non_hdl_cholesterol": 0.55  # Below acceptance threshold (0.80)
        }

        res = hierarchical_isolation_search(
            corrupted_panel,
            confidences=confidences,
            require_independent_evidence=True
        )
        self.assertEqual(res["action"], "ABSTAIN_ALGEBRAIC_INDETERMINATE")

    def test_k2_solvable_with_vldl_equation(self):
        """
        Corrupts LDL and TG: LDL 120 -> 210 (transposition), TG 150 -> 750 (severe smudge).
        With 1 equation, underdetermined K=2 abstains.
        When augmented with high-confidence VLDL (30.0, conf 0.95),
        solves TG = 5 * VLDL = 150 and LDL = TC - HDL - VLDL = 120 in closed form.
        """
        corrupted_panel = {
            "total_cholesterol": 200.0,
            "hdl": 50.0,
            "ldl": 210.0,
            "triglycerides": 750.0,
        }
        confidences = {
            "total_cholesterol": 0.95,
            "hdl": 0.95,
            "ldl": 0.85,
            "triglycerides": 0.85,
        }

        # Without VLDL: must abstain
        res_no_vldl = hierarchical_isolation_search(
            corrupted_panel,
            confidences=confidences,
            require_independent_evidence=True
        )
        self.assertEqual(res_no_vldl["action"], "ABSTAIN_ALGEBRAIC_INDETERMINATE")

        # With high-confidence VLDL: solves in closed form
        panel_with_vldl = corrupted_panel.copy()
        panel_with_vldl["vldl"] = 30.0
        conf_with_vldl = confidences.copy()
        conf_with_vldl["vldl"] = 0.95

        res_vldl = hierarchical_isolation_search(
            panel_with_vldl,
            confidences=conf_with_vldl,
            require_independent_evidence=True
        )
        self.assertEqual(res_vldl["action"], "CORRECT_K2")
        self.assertEqual(res_vldl.get("resolution_mechanism"), "extra_field_equation")
        self.assertAlmostEqual(res_vldl["corrections"]["triglycerides"], 150.0, places=1)
        self.assertAlmostEqual(res_vldl["corrections"]["ldl"], 120.0, places=1)

    def test_k2_vldl_low_confidence_abstains(self):
        """VLDL present but low confidence (0.45 < 0.80) correctly abstains."""
        corrupted_panel = {
            "total_cholesterol": 200.0,
            "hdl": 50.0,
            "ldl": 210.0,
            "triglycerides": 750.0,
            "vldl": 30.0
        }
        confidences = {
            "total_cholesterol": 0.95,
            "hdl": 0.95,
            "ldl": 0.85,
            "triglycerides": 0.85,
            "vldl": 0.45  # Low confidence
        }

        res = hierarchical_isolation_search(
            corrupted_panel,
            confidences=confidences,
            require_independent_evidence=True
        )
        self.assertEqual(res["action"], "ABSTAIN_ALGEBRAIC_INDETERMINATE")

    def test_direct_solve_k2_augmented_closed_form(self):
        """Directly verifies solve_k2_augmented_closed_form interface and metadata."""
        panel = {
            "total_cholesterol": 270.0,
            "hdl": 45.0,
            "ldl": 180.0,
            "triglycerides": 140.0,
            "non_hdl": 145.0  # true non-hdl: 190 - 45 = 145
        }
        confidences = {f: 0.90 for f in panel}

        sol = solve_k2_augmented_closed_form(panel, confidences=confidences)
        self.assertIsNotNone(sol)
        self.assertEqual(sol["action"], "CORRECT_K2")
        self.assertEqual(sol["resolution_mechanism"], "extra_field_equation")
        # TC* = 145 + 45 = 190.0, LDL* = 145 - 28 = 117.0
        self.assertAlmostEqual(sol["corrections"]["total_cholesterol"], 190.0, places=1)
        self.assertAlmostEqual(sol["corrections"]["ldl"], 117.0, places=1)


if __name__ == "__main__":
    unittest.main()

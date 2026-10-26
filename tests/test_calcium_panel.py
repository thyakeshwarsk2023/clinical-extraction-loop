"""
tests/test_calcium_panel.py
=============================================================================
Unit tests for Clinical Calcium Panel (Measured Calcium, Albumin, Corrected Calcium)
and Renal BUN/Creatinine Ratio bounds check.
Tests:
  - Protocol conformity and PanelRegistry integration
  - Closed-form K=1 round-trip algebraic inversions (residual < 1e-6)
  - NHANES-derived empirical tolerance boundaries
  - Physiological bounds checking and clinical abstention rules
  - Advisory BUN/Creatinine bounds checking
"""

import unittest
from src.panel_registry import Panel, PanelRegistry
import src.panels.calcium  # triggers registration
from src.panels.calcium import CalciumPanel


class TestCalciumPanel(unittest.TestCase):

    def setUp(self):
        self.panel = PanelRegistry.get("calcium")

    def test_protocol_conformance_and_registry(self):
        """Verifies that CalciumPanel satisfies the Panel Protocol and registers properly."""
        self.assertIsInstance(self.panel, Panel)
        self.assertEqual(self.panel.name, "calcium")
        self.assertEqual(self.panel.required_fields, ["calcium", "albumin", "corrected_calcium"])
        self.assertGreater(len(self.panel.abstention_rules), 0)
        self.assertTrue(PanelRegistry.has("calcium"))

    def test_identity_fn(self):
        """Validates exact residual computation |Corrected Ca - (Ca + 0.8 * (4.0 - Alb))|."""
        # Ca = 9.2, Alb = 4.0 -> Corrected Ca = 9.2 + 0.8 * 0 = 9.2
        res = self.panel.identity_fn({"calcium": 9.2, "albumin": 4.0, "corrected_calcium": 9.2})
        self.assertAlmostEqual(res, 0.0, places=6)

        # Hypoalbuminemia: Ca = 8.0, Alb = 2.5 -> Corrected Ca = 8.0 + 0.8 * 1.5 = 9.2
        res2 = self.panel.identity_fn({"calcium": 8.0, "albumin": 2.5, "corrected_calcium": 9.2})
        self.assertAlmostEqual(res2, 0.0, places=6)

        # Deviant: Ca = 8.0, Alb = 2.5, Corrected Ca = 8.5 (residual = 0.7)
        res_dev = self.panel.identity_fn({"calcium": 8.0, "albumin": 2.5, "corrected_calcium": 8.5})
        self.assertAlmostEqual(res_dev, 0.7, places=6)

    def test_round_trip_inversions(self):
        """
        Validates closed-form K=1 inversions for all 3 fields.
        Residual between ground truth and inverted value must be < 1e-6.
        """
        ca_gt = 8.4
        alb_gt = 3.0
        # Expected Corrected Ca = 8.4 + 0.8 * (4.0 - 3.0) = 8.4 + 0.8 = 9.2
        corr_gt = 9.2

        # 1. Invert corrected_calcium
        corr_inv = self.panel.solve_inversion(
            {"calcium": ca_gt, "albumin": alb_gt},
            target_field="corrected_calcium"
        )
        self.assertIsNotNone(corr_inv)
        self.assertAlmostEqual(corr_inv, corr_gt, places=6)

        # 2. Invert calcium
        ca_inv = self.panel.solve_inversion(
            {"corrected_calcium": corr_gt, "albumin": alb_gt},
            target_field="calcium"
        )
        self.assertIsNotNone(ca_inv)
        self.assertAlmostEqual(ca_inv, ca_gt, places=6)

        # 3. Invert albumin
        alb_inv = self.panel.solve_inversion(
            {"corrected_calcium": corr_gt, "calcium": ca_gt},
            target_field="albumin"
        )
        self.assertIsNotNone(alb_inv)
        self.assertAlmostEqual(alb_inv, alb_gt, places=6)

    def test_tolerance_boundaries(self):
        """
        Verifies behavior around the NHANES-derived 2.0% / 0.10 mg/dL tolerance threshold.
        """
        # Exact match
        status, abs_res, rel_err = self.panel.evaluate_invariant({
            "calcium": 9.0,
            "albumin": 4.0,
            "corrected_calcium": 9.0
        })
        self.assertEqual(status, "CONVERGED")
        self.assertEqual(abs_res, 0.0)
        self.assertEqual(rel_err, 0.0)

        # Minor instrument roundoff (0.04 mg/dL observed in NHANES)
        status_round, abs_res_r, rel_err_r = self.panel.evaluate_invariant({
            "calcium": 9.0,
            "albumin": 4.1,
            "corrected_calcium": 8.9  # 9.0 + 0.8*(-0.1) = 8.92 -> delta = 0.02
        })
        self.assertEqual(status_round, "CONVERGED")
        self.assertLessEqual(abs_res_r, 0.05)

        # Gross violation (e.g. digit swap: 9.0 read as 6.0)
        status_bad, abs_res_b, rel_err_b = self.panel.evaluate_invariant({
            "calcium": 6.0,
            "albumin": 4.0,
            "corrected_calcium": 9.0
        })
        self.assertEqual(status_bad, "VIOLATION")
        self.assertGreater(abs_res_b, 2.0)

    def test_physiological_bounds_and_abstentions(self):
        """Tests biological plausibility bounds and non-positive abstentions."""
        # Non-positive values
        ok, reason = self.panel.verify_bounds({"calcium": 0.0, "albumin": 4.0, "corrected_calcium": 9.0})
        self.assertFalse(ok)
        self.assertEqual(reason, "BOUND_VIOLATION_NON_POSITIVE_VALUE")

        # Extreme calcium implausibility
        ok_ext, reason_ext = self.panel.verify_bounds({"calcium": 25.0, "albumin": 4.0, "corrected_calcium": 25.0})
        self.assertFalse(ok_ext)
        self.assertEqual(reason_ext, "BOUND_VIOLATION_EXTREME_CALCIUM")

        # Extreme albumin implausibility
        ok_alb, reason_alb = self.panel.verify_bounds({"calcium": 9.0, "albumin": 10.0, "corrected_calcium": 4.2})
        self.assertFalse(ok_alb)
        self.assertEqual(reason_alb, "BOUND_VIOLATION_EXTREME_ALBUMIN")

    def test_bun_creatinine_bounds(self):
        """Tests advisory BUN/Creatinine physiological bounds checking."""
        # Normal 15:1 ratio
        ok_norm, label_norm, ratio_norm = CalciumPanel.evaluate_bun_creatinine_bounds(bun=15.0, creatinine=1.0)
        self.assertTrue(ok_norm)
        self.assertEqual(label_norm, "NORMAL_BUN_CR_RATIO")
        self.assertEqual(ratio_norm, 15.0)

        # Prerenal azotemia (dehydration, 28:1 ratio)
        ok_pre, label_pre, ratio_pre = CalciumPanel.evaluate_bun_creatinine_bounds(bun=28.0, creatinine=1.0)
        self.assertFalse(ok_pre)
        self.assertEqual(label_pre, "ELEVATED_PRERENAL_RATIO")
        self.assertEqual(ratio_pre, 28.0)

        # Intrarenal / low ratio (6:1 ratio)
        ok_intra, label_intra, ratio_intra = CalciumPanel.evaluate_bun_creatinine_bounds(bun=6.0, creatinine=1.0)
        self.assertFalse(ok_intra)
        self.assertEqual(label_intra, "REDUCED_INTRARENAL_RATIO")
        self.assertEqual(ratio_intra, 6.0)

        # Non-positive
        ok_zero, label_zero, _ = CalciumPanel.evaluate_bun_creatinine_bounds(bun=0.0, creatinine=1.0)
        self.assertFalse(ok_zero)
        self.assertEqual(label_zero, "INVALID_NON_POSITIVE")


if __name__ == "__main__":
    unittest.main()

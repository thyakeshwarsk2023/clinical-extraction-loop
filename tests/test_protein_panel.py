"""
tests/test_protein_panel.py
=============================================================================
Unit tests for Clinical Protein Panel (Total Protein, Albumin, Globulin).
Tests:
  - Protocol conformity and PanelRegistry integration
  - Closed-form K=1 round-trip algebraic inversions (residual < 1e-6)
  - NHANES-derived empirical tolerance boundaries
  - Physiological bounds checking and clinical abstention rules
"""

import unittest
from src.panel_registry import Panel, PanelRegistry
import src.panels.protein  # triggers registration
from src.panels.protein import ProteinPanel


class TestProteinPanel(unittest.TestCase):

    def setUp(self):
        self.panel = PanelRegistry.get("protein")

    def test_protocol_conformance_and_registry(self):
        """Verifies that ProteinPanel satisfies the Panel Protocol and registers properly."""
        self.assertIsInstance(self.panel, Panel)
        self.assertEqual(self.panel.name, "protein")
        self.assertEqual(self.panel.required_fields, ["total_protein", "albumin", "globulin"])
        self.assertGreater(len(self.panel.abstention_rules), 0)
        self.assertTrue(PanelRegistry.has("protein"))

    def test_identity_fn(self):
        """Validates exact residual computation |TP - (Alb + Glob)|."""
        res = self.panel.identity_fn({"total_protein": 7.2, "albumin": 4.2, "globulin": 3.0})
        self.assertAlmostEqual(res, 0.0, places=6)

        res_deviant = self.panel.identity_fn({"total_protein": 7.5, "albumin": 4.2, "globulin": 3.0})
        self.assertAlmostEqual(res_deviant, 0.3, places=6)

    def test_round_trip_inversions(self):
        """
        Validates closed-form K=1 inversions for all 3 fields.
        Residual between ground truth and inverted value must be < 1e-6.
        """
        # Ground truth biological sample
        tp_gt = 7.4
        alb_gt = 4.3
        glob_gt = 3.1

        # 1. Invert total_protein
        tp_inv = self.panel.solve_inversion(
            {"albumin": alb_gt, "globulin": glob_gt},
            target_field="total_protein"
        )
        self.assertIsNotNone(tp_inv)
        self.assertAlmostEqual(tp_inv, tp_gt, places=6)

        # 2. Invert albumin
        alb_inv = self.panel.solve_inversion(
            {"total_protein": tp_gt, "globulin": glob_gt},
            target_field="albumin"
        )
        self.assertIsNotNone(alb_inv)
        self.assertAlmostEqual(alb_inv, alb_gt, places=6)

        # 3. Invert globulin
        glob_inv = self.panel.solve_inversion(
            {"total_protein": tp_gt, "albumin": alb_gt},
            target_field="globulin"
        )
        self.assertIsNotNone(glob_inv)
        self.assertAlmostEqual(glob_inv, glob_gt, places=6)

    def test_tolerance_boundaries(self):
        """
        Verifies behavior around the NHANES-derived 2.0% / 0.10 g/dL tolerance threshold.
        """
        # Perfect concordance
        status, abs_res, rel_err = self.panel.evaluate_invariant({
            "total_protein": 7.0,
            "albumin": 4.0,
            "globulin": 3.0
        })
        self.assertEqual(status, "CONVERGED")
        self.assertEqual(abs_res, 0.0)
        self.assertEqual(rel_err, 0.0)

        # 0.1 g/dL standard reporting roundoff discrepancy (within abs tolerance)
        status_roundoff, abs_res_r, rel_err_r = self.panel.evaluate_invariant({
            "total_protein": 7.1,
            "albumin": 4.1,
            "globulin": 2.9
        })
        self.assertEqual(status_roundoff, "CONVERGED")
        self.assertLessEqual(abs_res_r, 0.1001)

        # Gross violation (e.g., OCR transposition: 7.2 read as 1.2 or globulin 3.0 read as 8.0)
        status_bad, abs_res_b, rel_err_b = self.panel.evaluate_invariant({
            "total_protein": 7.2,
            "albumin": 4.2,
            "globulin": 5.0  # expected 3.0 -> delta = 2.0
        })
        self.assertEqual(status_bad, "VIOLATION")
        self.assertGreater(abs_res_b, 0.5)

    def test_physiological_bounds_and_abstentions(self):
        """Tests physiological boundary constraints and abstention cases."""
        # Non-positive values
        ok, reason = self.panel.verify_bounds({"total_protein": 0.0, "albumin": 4.0, "globulin": 3.0})
        self.assertFalse(ok)
        self.assertEqual(reason, "BOUND_VIOLATION_NON_POSITIVE_VALUE")

        # Albumin >= Total Protein
        ok_alb, reason_alb = self.panel.verify_bounds({"total_protein": 7.0, "albumin": 7.2, "globulin": 2.0})
        self.assertFalse(ok_alb)
        self.assertEqual(reason_alb, "BOUND_VIOLATION_ALBUMIN_EXCEEDS_TOTAL")

        # Globulin >= Total Protein
        ok_glob, reason_glob = self.panel.verify_bounds({"total_protein": 6.5, "albumin": 3.0, "globulin": 6.5})
        self.assertFalse(ok_glob)
        self.assertEqual(reason_glob, "BOUND_VIOLATION_GLOBULIN_EXCEEDS_TOTAL")

        # Extreme values outside biological life limits
        ok_ext, reason_ext = self.panel.verify_bounds({"total_protein": 18.0, "albumin": 5.0, "globulin": 3.0})
        self.assertFalse(ok_ext)
        self.assertEqual(reason_ext, "BOUND_VIOLATION_EXTREME_TOTAL_PROTEIN")


if __name__ == "__main__":
    unittest.main()

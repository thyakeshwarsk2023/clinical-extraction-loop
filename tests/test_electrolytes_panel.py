"""
tests/test_electrolytes_panel.py
=============================================================================
Unit tests for Clinical Electrolyte Panel (Anion Gap & Osmolar Gap).
Tests:
  - Protocol conformity and PanelRegistry integration
  - Closed-form K=1 round-trip algebraic inversions (residual < 1e-6)
  - With-K variant inversions and tolerance checks
  - Anion gap bounds and tolerance checking
  - Osmolar gap soft check (ADVISORY only, non-triggering)
  - Physiological bounds checking and clinical abstention rules
"""

import unittest
from src.panel_registry import Panel, PanelRegistry
import src.panels.electrolytes  # triggers registration
from src.panels.electrolytes import ElectrolytePanel


class TestElectrolytePanel(unittest.TestCase):

    def setUp(self):
        self.panel = PanelRegistry.get("electrolytes")

    def test_protocol_conformance_and_registry(self):
        """Verifies that ElectrolytePanel satisfies the Panel Protocol and registers properly."""
        self.assertIsInstance(self.panel, Panel)
        self.assertEqual(self.panel.name, "electrolytes")
        self.assertEqual(self.panel.required_fields, ["sodium", "chloride", "bicarbonate"])
        self.assertGreater(len(self.panel.abstention_rules), 0)
        self.assertTrue(PanelRegistry.has("electrolytes"))

    def test_round_trip_inversions_standard(self):
        """
        Validates closed-form K=1 inversions for standard 3-field anion gap.
        Residual must be < 1e-6.
        """
        na_gt = 140.0
        cl_gt = 104.0
        hco3_gt = 24.0
        ag_gt = 12.0  # 140 - (104 + 24) = 12.0

        # 1. Invert sodium
        na_inv = self.panel.solve_inversion(
            {"anion_gap": ag_gt, "chloride": cl_gt, "bicarbonate": hco3_gt},
            target_field="sodium"
        )
        self.assertIsNotNone(na_inv)
        self.assertAlmostEqual(na_inv, na_gt, places=6)

        # 2. Invert chloride
        cl_inv = self.panel.solve_inversion(
            {"sodium": na_gt, "bicarbonate": hco3_gt, "anion_gap": ag_gt},
            target_field="chloride"
        )
        self.assertIsNotNone(cl_inv)
        self.assertAlmostEqual(cl_inv, cl_gt, places=6)

        # 3. Invert bicarbonate
        hco3_inv = self.panel.solve_inversion(
            {"sodium": na_gt, "chloride": cl_gt, "anion_gap": ag_gt},
            target_field="bicarbonate"
        )
        self.assertIsNotNone(hco3_inv)
        self.assertAlmostEqual(hco3_inv, hco3_gt, places=6)

        # 4. Invert anion_gap
        ag_inv = self.panel.solve_inversion(
            {"sodium": na_gt, "chloride": cl_gt, "bicarbonate": hco3_gt},
            target_field="anion_gap"
        )
        self.assertIsNotNone(ag_inv)
        self.assertAlmostEqual(ag_inv, ag_gt, places=6)

    def test_round_trip_inversions_with_k(self):
        """Validates round-trip inversions when optional potassium is included."""
        na_gt = 140.0
        k_gt = 4.0
        cl_gt = 104.0
        hco3_gt = 24.0
        ag_k_gt = 16.0  # (140 + 4) - (104 + 24) = 16.0

        na_inv = self.panel.solve_inversion(
            {"anion_gap": ag_k_gt, "potassium": k_gt, "chloride": cl_gt, "bicarbonate": hco3_gt},
            target_field="sodium",
            with_k=True
        )
        self.assertAlmostEqual(na_inv, na_gt, places=6)

        cl_inv = self.panel.solve_inversion(
            {"sodium": na_gt, "potassium": k_gt, "bicarbonate": hco3_gt, "anion_gap": ag_k_gt},
            target_field="chloride",
            with_k=True
        )
        self.assertAlmostEqual(cl_inv, cl_gt, places=6)

        hco3_inv = self.panel.solve_inversion(
            {"sodium": na_gt, "potassium": k_gt, "chloride": cl_gt, "anion_gap": ag_k_gt},
            target_field="bicarbonate",
            with_k=True
        )
        self.assertAlmostEqual(hco3_inv, hco3_gt, places=6)

    def test_tolerance_and_bounds_evaluation(self):
        """Verifies evaluation of explicit reported anion gap and implicit normal bounds."""
        # Exact reported AG concordance
        status, abs_res, rel_err = self.panel.evaluate_invariant({
            "sodium": 140.0,
            "chloride": 104.0,
            "bicarbonate": 24.0,
            "anion_gap": 12.0
        })
        self.assertEqual(status, "CONVERGED")
        self.assertEqual(abs_res, 0.0)

        # 1.0 mEq/L integer reporting discrepancy (within absolute tolerance)
        status_round, abs_res_r, _ = self.panel.evaluate_invariant({
            "sodium": 140.0,
            "chloride": 104.0,
            "bicarbonate": 24.0,
            "anion_gap": 13.0
        })
        self.assertEqual(status_round, "CONVERGED")
        self.assertLessEqual(abs_res_r, 1.0)

        # Gross violation (e.g., misread Na as 120 or AG as 25)
        status_viol, abs_res_v, _ = self.panel.evaluate_invariant({
            "sodium": 140.0,
            "chloride": 104.0,
            "bicarbonate": 24.0,
            "anion_gap": 22.0  # discrepancy = 10.0 mEq/L
        })
        self.assertEqual(status_viol, "VIOLATION")
        self.assertEqual(abs_res_v, 10.0)

        # Implicit check when anion_gap is omitted: calculated 12.0 is in [8, 16]
        status_impl, _, _ = self.panel.evaluate_invariant({
            "sodium": 140.0,
            "chloride": 104.0,
            "bicarbonate": 24.0
        })
        self.assertEqual(status_impl, "CONVERGED")

        # Implicit check outside normal bounds: AG = 140 - (98 + 14) = 28 -> elevated
        status_elev, _, _ = self.panel.evaluate_invariant({
            "sodium": 140.0,
            "chloride": 98.0,
            "bicarbonate": 14.0
        })
        self.assertEqual(status_elev, "VIOLATION")

    def test_osmolar_gap_advisory_flag_non_triggering(self):
        """
        Confirms that an abnormal osmolar gap (> 10 mOsm/kg) generates an ADVISORY flag,
        but does NOT trigger a hard invariant violation or repair cycle.
        """
        # Normal osmolar gap:
        # Calculated = 2*140 + 90/18 + 14/2.8 = 280 + 5 + 5 = 290
        # Measured = 292 -> gap = +2.0 (normal)
        res_normal = self.panel.evaluate_full({
            "sodium": 140.0,
            "chloride": 104.0,
            "bicarbonate": 24.0,
            "glucose": 90.0,
            "bun": 14.0,
            "measured_osmolality": 292.0
        })
        self.assertEqual(res_normal["status"], "CONVERGED")
        self.assertTrue(res_normal["is_valid"])
        self.assertEqual(len(res_normal["advisory_flags"]), 0)

        # Elevated osmolar gap (e.g. unmeasured osmole / toxic alcohol ingestion):
        # Calculated = 290.0, Measured = 315.0 -> gap = +25.0 mOsm/kg
        res_elevated = self.panel.evaluate_full({
            "sodium": 140.0,
            "chloride": 104.0,
            "bicarbonate": 24.0,
            "glucose": 90.0,
            "bun": 14.0,
            "measured_osmolality": 315.0
        })
        # Hard invariant MUST remain CONVERGED
        self.assertEqual(res_elevated["status"], "CONVERGED")
        self.assertTrue(res_elevated["is_valid"])
        # Soft advisory flag MUST be present
        self.assertEqual(len(res_elevated["advisory_flags"]), 1)
        flag = res_elevated["advisory_flags"][0]
        self.assertEqual(flag["rule"], "OSMOLAR_GAP_ADVISORY")
        self.assertEqual(flag["severity"], "ADVISORY_SOFT_FLAG")
        self.assertEqual(flag["osmolar_gap"], 25.0)

    def test_physiological_bounds_and_abstentions(self):
        """Tests physiological boundary constraints and impossible biological values."""
        # Non-positive sodium
        ok, reason = self.panel.verify_bounds({"sodium": 0.0, "chloride": 100.0, "bicarbonate": 24.0})
        self.assertFalse(ok)
        self.assertEqual(reason, "BOUND_VIOLATION_NON_POSITIVE_VALUE")

        # Sum of anions exceeds sodium (negative anion gap)
        ok_anions, reason_anions = self.panel.verify_bounds({"sodium": 135.0, "chloride": 115.0, "bicarbonate": 25.0})
        self.assertFalse(ok_anions)
        self.assertEqual(reason_anions, "BOUND_VIOLATION_ANIONS_EXCEED_SODIUM")

        # Extreme non-viable sodium
        ok_na, reason_na = self.panel.verify_bounds({"sodium": 210.0, "chloride": 100.0, "bicarbonate": 24.0})
        self.assertFalse(ok_na)
        self.assertEqual(reason_na, "BOUND_VIOLATION_EXTREME_SODIUM")


if __name__ == "__main__":
    unittest.main()

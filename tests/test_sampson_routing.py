"""
Unit and integration test suite for Phase 2: Dynamic Equation Routing (Sampson Equation).

Validates:
  1. Sampson LDL calculation and invariant evaluation (400 < TG <= 800 mg/dL).
  2. O(1) closed-form algebraic inversions for TC*, HDL*, TG*, and LDL*.
  3. Dynamic routing thresholds: Friedewald (TG <= 400), Sampson (400 < TG <= 800), and ABSTAIN_TG800 (TG > 800).
  4. Integration with solve_field_closed_form and hierarchical_isolation_search.
  5. Defensive error handling and zero-division guardrails.
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stage2_constraints import (
    calculate_sampson_ldl,
    evaluate_sampson_invariant,
    evaluate_friedewald_invariant,
)
from src.stage2_isolation import (
    solve_field_closed_form,
    hierarchical_isolation_search,
)


class TestSampsonInvariantEvaluation(unittest.TestCase):
    """Tests evaluate_sampson_invariant and calculate_sampson_ldl."""

    def setUp(self):
        self.tc = 250.0
        self.hdl = 40.0
        self.tg = 550.0
        self.expected_ldl = calculate_sampson_ldl(self.tc, self.hdl, self.tg)

    def test_sampson_ldl_calculation_and_convergence(self):
        """Validates that a panel with TC=250, HDL=40, TG=550 calculates the correct LDL and converges."""
        panel = {
            "total_cholesterol": self.tc,
            "hdl": self.hdl,
            "triglycerides": self.tg,
            "ldl": self.expected_ldl,
        }
        status, rel_residual = evaluate_sampson_invariant(panel, tol=0.10)
        self.assertEqual(status, "CONVERGED")
        self.assertLessEqual(rel_residual, 0.10)
        self.assertAlmostEqual(rel_residual, 0.0, places=4)

    def test_sampson_violation_on_corrupted_ldl(self):
        """Asserts that a corrupted LDL value exceeding 10% tolerance returns VIOLATION."""
        panel = {
            "total_cholesterol": self.tc,
            "hdl": self.hdl,
            "triglycerides": self.tg,
            "ldl": round(self.expected_ldl * 1.25, 1),
        }
        status, rel_residual = evaluate_sampson_invariant(panel, tol=0.10)
        self.assertEqual(status, "VIOLATION")
        self.assertGreater(rel_residual, 0.10)

    def test_abstain_tg800_boundary(self):
        """Asserts that TG > 800 mg/dL triggers ABSTAIN_TG800."""
        panel_tg850 = {
            "total_cholesterol": 250.0,
            "hdl": 40.0,
            "triglycerides": 850.0,
            "ldl": 100.0,
        }
        status, rel_residual = evaluate_sampson_invariant(panel_tg850)
        self.assertEqual(status, "ABSTAIN_TG800")
        self.assertEqual(rel_residual, 1.0)

    def test_non_positive_inputs_violation(self):
        """Asserts that zero or negative components return VIOLATION without crashing."""
        panel_zero_tc = {"total_cholesterol": 0.0, "hdl": 40.0, "triglycerides": 550.0, "ldl": 50.0}
        status, _ = evaluate_sampson_invariant(panel_zero_tc)
        self.assertEqual(status, "VIOLATION")

        panel_zero_hdl = {"total_cholesterol": 250.0, "hdl": 0.0, "triglycerides": 550.0, "ldl": 50.0}
        status, _ = evaluate_sampson_invariant(panel_zero_hdl)
        self.assertEqual(status, "VIOLATION")


class TestClosedFormAlgebraicInversions(unittest.TestCase):
    """Tests the O(1) closed-form Sampson inversions for TC*, HDL*, TG*, and LDL*."""

    def setUp(self):
        self.clean_tc = 250.0
        self.clean_hdl = 40.0
        self.clean_tg = 550.0
        self.clean_ldl = calculate_sampson_ldl(self.clean_tc, self.clean_hdl, self.clean_tg)

    def test_tc_star_inversion_recovers_original(self):
        """
        Passes a corrupted TC value into the Sampson TC* inversion formula
        and asserts that it recovers the exact original TC value.
        """
        corrupted_tc = 310.0
        panel = {
            "total_cholesterol": corrupted_tc,
            "hdl": self.clean_hdl,
            "triglycerides": self.clean_tg,
            "ldl": self.clean_ldl,
        }

        # Formula 2: TC* inversion
        tc_star = solve_field_closed_form("total_cholesterol", panel)
        self.assertAlmostEqual(tc_star, self.clean_tc, places=1)

    def test_hdl_star_inversion_recovers_original(self):
        """Passes a corrupted HDL value and asserts recovery of clean HDL via Formula 3."""
        corrupted_hdl = 75.0
        panel = {
            "total_cholesterol": self.clean_tc,
            "hdl": corrupted_hdl,
            "triglycerides": self.clean_tg,
            "ldl": self.clean_ldl,
        }

        # Formula 3: HDL* inversion
        hdl_star = solve_field_closed_form("hdl", panel)
        self.assertAlmostEqual(hdl_star, self.clean_hdl, places=1)

    def test_tg_star_inversion_recovers_original(self):
        """Passes a corrupted TG value and asserts recovery of clean TG via Formula 4."""
        corrupted_tg = 680.0
        panel = {
            "total_cholesterol": self.clean_tc,
            "hdl": self.clean_hdl,
            "triglycerides": corrupted_tg,
            "ldl": self.clean_ldl,
        }

        # Formula 4: TG* inversion
        tg_star = solve_field_closed_form("triglycerides", panel)
        self.assertAlmostEqual(tg_star, self.clean_tg, places=1)

    def test_ldl_star_inversion_matches_calculate_sampson(self):
        """Asserts that solve_field_closed_form for LDL matches calculate_sampson_ldl directly."""
        panel = {
            "total_cholesterol": self.clean_tc,
            "hdl": self.clean_hdl,
            "triglycerides": self.clean_tg,
            "ldl": 0.0,
        }
        ldl_star = solve_field_closed_form("ldl", panel)
        self.assertAlmostEqual(ldl_star, self.clean_ldl, places=1)

    def test_inversion_invalidated_when_tg_exceeds_800(self):
        """Asserts that solve_field_closed_form returns 0.0 when TG exceeds 800 mg/dL."""
        panel = {
            "total_cholesterol": 250.0,
            "hdl": 40.0,
            "triglycerides": 850.0,
            "ldl": 100.0,
        }
        for field in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
            val = solve_field_closed_form(field, panel)
            self.assertEqual(val, 0.0)


class TestHierarchicalIsolationSearchSampsonRouting(unittest.TestCase):
    """Tests dynamic equation routing inside hierarchical_isolation_search."""

    def test_sampson_k1_correction_when_tg_in_range(self):
        """Asserts hierarchical_isolation_search isolates and repairs single-field error with TG=550."""
        clean_tc = 250.0
        clean_hdl = 40.0
        clean_tg = 550.0
        clean_ldl = calculate_sampson_ldl(clean_tc, clean_hdl, clean_tg)

        # Corrupt total_cholesterol with a plausible single-digit error (250.0 -> 280.0)
        panel = {
            "total_cholesterol": 280.0,
            "hdl": clean_hdl,
            "triglycerides": clean_tg,
            "ldl": clean_ldl,
        }
        confidences = {
            "total_cholesterol": 0.65,
            "hdl": 0.95,
            "triglycerides": 0.95,
            "ldl": 0.95,
        }

        res = hierarchical_isolation_search(panel, confidences)
        self.assertEqual(res["action"], "CORRECT_K1")
        self.assertEqual(res["field"], "total_cholesterol")
        self.assertAlmostEqual(res["target_val"], clean_tc, places=1)

    def test_sampson_abstain_when_tg_exceeds_800(self):
        """Asserts that hierarchical_isolation_search returns ABSTAIN_TG800 for TG > 800."""
        panel = {
            "total_cholesterol": 250.0,
            "hdl": 40.0,
            "triglycerides": 850.0,
            "ldl": 95.0,
        }
        confidences = {"total_cholesterol": 0.8, "hdl": 0.8, "triglycerides": 0.8, "ldl": 0.8}
        res = hierarchical_isolation_search(panel, confidences)
        self.assertEqual(res["action"], "ABSTAIN_TG800")
        self.assertIn("exceed", res["reason"].lower())

    def test_friedewald_routing_when_tg_under_400(self):
        """Asserts that normal TG <= 400 panels route strictly to Friedewald."""
        # Clean Friedewald panel: TC=200, HDL=50, TG=150, LDL=120 (200 == 50 + 120 + 30)
        panel = {
            "total_cholesterol": 200.0,
            "hdl": 50.0,
            "triglycerides": 150.0,
            "ldl": 120.0,
        }
        confidences = {"total_cholesterol": 0.9, "hdl": 0.9, "triglycerides": 0.9, "ldl": 0.9}
        res = hierarchical_isolation_search(panel, confidences)
        self.assertEqual(res["action"], "CONVERGED")
        self.assertEqual(res["equation"], "FRIEDEWALD")


class TestDynamicEquationRoutingThresholds(unittest.TestCase):
    """Tests the 3 distinct regimes: Friedewald, Sampson, and Abstention."""

    def test_friedewald_regime(self):
        panel = {"total_cholesterol": 200.0, "hdl": 50.0, "triglycerides": 399.0, "ldl": 70.0}
        status, _, _ = evaluate_friedewald_invariant(panel)
        self.assertIn(status, ["CONVERGED", "VIOLATION"])

    def test_sampson_regime(self):
        tc, hdl, tg = 260.0, 45.0, 401.0
        ldl = calculate_sampson_ldl(tc, hdl, tg)
        panel = {"total_cholesterol": tc, "hdl": hdl, "triglycerides": tg, "ldl": ldl}
        status, rel_res = evaluate_sampson_invariant(panel)
        self.assertEqual(status, "CONVERGED")
        self.assertLessEqual(rel_res, 0.10)

    def test_abstain_regime(self):
        panel = {"total_cholesterol": 260.0, "hdl": 45.0, "triglycerides": 801.0, "ldl": 80.0}
        status, _ = evaluate_sampson_invariant(panel)
        self.assertEqual(status, "ABSTAIN_TG800")


if __name__ == "__main__":
    unittest.main()

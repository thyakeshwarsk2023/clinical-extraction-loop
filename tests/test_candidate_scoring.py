"""
Unit and integration tests for Stage 2.5 Candidate Scoring in src/stage2_isolation.py.
Verifies:
  - damerau_levenshtein_distance
  - is_decimal_shift
  - is_adjacent_digit_swap
  - compute_candidate_cost
  - hierarchical_isolation_search integration
"""

import unittest
import os
import sys

# Ensure project root is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stage2_isolation import (
    damerau_levenshtein_distance,
    is_decimal_shift,
    is_adjacent_digit_swap,
    compute_candidate_cost,
    hierarchical_isolation_search,
    is_digit_transposition_match,
    check_decimal_shift,
)


class TestDamerauLevenshteinDistance(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(damerau_levenshtein_distance("180", "180"), 0)
        self.assertEqual(damerau_levenshtein_distance("", ""), 0)

    def test_single_deletion_and_insertion(self):
        self.assertEqual(damerau_levenshtein_distance("180", "18"), 1)
        self.assertEqual(damerau_levenshtein_distance("18", "180"), 1)

    def test_single_substitution(self):
        self.assertEqual(damerau_levenshtein_distance("180", "188"), 1)

    def test_adjacent_transposition(self):
        # In standard Levenshtein, '268' -> '628' is distance 2. In Damerau-Levenshtein, it is 1.
        self.assertEqual(damerau_levenshtein_distance("268", "628"), 1)
        self.assertEqual(damerau_levenshtein_distance("26.8", "62.8"), 1)

    def test_decimal_point_edit(self):
        self.assertEqual(damerau_levenshtein_distance("361", "36.1"), 1)


class TestPhysicalArtifactFlags(unittest.TestCase):
    def test_decimal_shift_detection(self):
        # 10x order-of-magnitude shift
        self.assertTrue(is_decimal_shift(361.0, 36.1, "hdl"))
        self.assertTrue(is_decimal_shift(15.0, 150.0, "ldl"))
        self.assertTrue(check_decimal_shift(361.0, 36.1, "hdl"))  # alias test
        # Non-shift
        self.assertFalse(is_decimal_shift(100.0, 180.0, "ldl"))

    def test_adjacent_digit_swap_detection(self):
        # Exact adjacent swap in mg/dL
        self.assertTrue(is_adjacent_digit_swap(26.8, 62.8, "hdl"))
        self.assertTrue(is_digit_transposition_match(26.8, 62.8, "hdl"))  # alias test
        # Non-swap
        self.assertFalse(is_adjacent_digit_swap(26.8, 45.0, "hdl"))


class TestComputeCandidateCost(unittest.TestCase):
    def test_decimal_shift_scoring(self):
        cost, meta = compute_candidate_cost(
            raw_val=361.0,
            target_val=36.1,
            confidence=0.80,
            field="hdl"
        )
        self.assertTrue(meta["is_decimal_shift"])
        self.assertFalse(meta["is_transposition"])
        self.assertEqual(meta["effective_dist"], 1.0)
        # conf_penalty = 0.50 * (1.0 - 0.80) = 0.10
        self.assertAlmostEqual(meta["conf_penalty"], 0.10, places=3)
        # Final cost = 1.0 + 0.10 - 0.30 = 0.80
        self.assertAlmostEqual(cost, 0.80, places=3)
        self.assertEqual(meta["cost"], 0.80)

    def test_transposition_scoring(self):
        cost, meta = compute_candidate_cost(
            raw_val=26.8,
            target_val=62.8,
            confidence=0.60,
            field="hdl"
        )
        self.assertFalse(meta["is_decimal_shift"])
        self.assertTrue(meta["is_transposition"])
        self.assertEqual(meta["effective_dist"], 1.0)
        # conf_penalty = 0.50 * (1.0 - 0.60) = 0.20
        self.assertAlmostEqual(meta["conf_penalty"], 0.20, places=3)
        # Final cost = 1.0 + 0.20 - 0.25 = 0.95
        self.assertAlmostEqual(cost, 0.95, places=3)
        self.assertEqual(meta["cost"], 0.95)

    def test_standard_edit_scoring(self):
        cost, meta = compute_candidate_cost(
            raw_val=180.0,
            target_val=188.0,
            confidence=0.90,
            field="total_cholesterol"
        )
        self.assertFalse(meta["is_decimal_shift"])
        self.assertFalse(meta["is_transposition"])
        self.assertEqual(meta["raw_dist"], 1)
        self.assertEqual(meta["effective_dist"], 1.0)
        # conf_penalty = 0.50 * (1.0 - 0.90) = 0.05
        self.assertAlmostEqual(meta["conf_penalty"], 0.05, places=3)
        # Final cost = 1.0 + 0.05 = 1.05
        self.assertAlmostEqual(cost, 1.05, places=3)

    def test_confidence_penalty_bounds(self):
        # 100% confidence -> 0 penalty
        _, meta_high = compute_candidate_cost(100.0, 100.0, confidence=1.0, field="ldl")
        self.assertEqual(meta_high["conf_penalty"], 0.0)

        # 0% confidence -> 0.5 penalty
        _, meta_low = compute_candidate_cost(100.0, 100.0, confidence=0.0, field="ldl")
        self.assertEqual(meta_low["conf_penalty"], 0.5)

        # Confidence > 1.0 clamped
        _, meta_over = compute_candidate_cost(100.0, 100.0, confidence=1.2, field="ldl")
        self.assertEqual(meta_over["conf_penalty"], 0.0)

        # Confidence < 0.0 clamped
        _, meta_under = compute_candidate_cost(100.0, 100.0, confidence=-0.5, field="ldl")
        self.assertEqual(meta_under["conf_penalty"], 0.5)

    def test_metadata_structure(self):
        _, meta = compute_candidate_cost(150.0, 15.0, confidence=0.75, field="ldl")
        required_keys = {
            "raw_dist",
            "effective_dist",
            "conf_penalty",
            "is_decimal_shift",
            "is_transposition",
            "cost"
        }
        self.assertTrue(required_keys.issubset(meta.keys()))


class TestHierarchicalIsolationIntegration(unittest.TestCase):
    def test_isolation_prefers_decimal_shift(self):
        # True clean panel: TC=200, HDL=50, LDL=130, TG=100
        # Injected error: HDL decimal shifted to 500.0
        corrupted_panel = {
            "total_cholesterol": 200.0,
            "hdl": 500.0,
            "ldl": 130.0,
            "triglycerides": 100.0
        }
        confidences = {
            "total_cholesterol": 0.95,
            "hdl": 0.60,
            "ldl": 0.95,
            "triglycerides": 0.95
        }
        res = hierarchical_isolation_search(corrupted_panel, confidences)
        self.assertEqual(res["action"], "CORRECT_K1")
        self.assertEqual(res["field"], "hdl")
        self.assertAlmostEqual(res["target_val"], 50.0, places=1)
        self.assertTrue(res["is_fast_path"])

    def test_isolation_prefers_adjacent_transposition(self):
        # True clean panel: TC=210, HDL=62.8, LDL=127.2, TG=100 (62.8 + 127.2 + 20 = 210)
        # Injected error: HDL transposed to 26.8
        corrupted_panel = {
            "total_cholesterol": 210.0,
            "hdl": 26.8,
            "ldl": 127.2,
            "triglycerides": 100.0
        }
        confidences = {
            "total_cholesterol": 0.95,
            "hdl": 0.55,
            "ldl": 0.95,
            "triglycerides": 0.95
        }
        res = hierarchical_isolation_search(corrupted_panel, confidences)
        self.assertEqual(res["action"], "CORRECT_K1")
        self.assertEqual(res["field"], "hdl")
        self.assertAlmostEqual(res["target_val"], 62.8, places=1)


if __name__ == "__main__":
    unittest.main()

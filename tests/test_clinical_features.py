"""
tests/test_clinical_features.py
===========================================================================
Verification suite for Clinical Parity & Integration features:
  1. Martin-Hopkins routing, divisor strata, LDL calculation, and invariant evaluation
  2. Counterfactual XAI diagnostics and natural language explanations
  3. Adversarial corruption injection across all error modalities
  4. Clinical Risk Flip stratification (NCEP ATP III / AHA/ACC guidelines)
  5. HL7 FHIR R4 Lipid Bundle generation and LOINC coding
"""

import unittest
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stage2_constraints import (
    get_martin_hopkins_divisor,
    calculate_martin_hopkins_ldl,
    evaluate_martin_hopkins_invariant,
)
from src.stage2_isolation import (
    invert_martin_hopkins,
    solve_field_closed_form,
    hierarchical_isolation_search,
    explain_counterfactual,
)
from src.clinical_risk import classify_ldl_risk, evaluate_clinical_risk_flip
from src.adversarial import inject_adversarial_corruption, ADVERSARIAL_MODES
from src.fhir_export import generate_fhir_r4_lipid_bundle, LOINC_CODES


class TestMartinHopkinsBackend(unittest.TestCase):
    """Tests Martin-Hopkins dynamic divisor and invariant evaluation."""

    def test_divisor_strata(self):
        # TG < 50
        self.assertEqual(get_martin_hopkins_divisor(45.0, 90.0), 3.5)
        self.assertEqual(get_martin_hopkins_divisor(45.0, 110.0), 4.0)
        # 50 <= TG < 100
        self.assertEqual(get_martin_hopkins_divisor(80.0, 80.0), 4.5)
        self.assertEqual(get_martin_hopkins_divisor(80.0, 120.0), 5.0)
        # 100 <= TG <= 400
        self.assertEqual(get_martin_hopkins_divisor(250.0, 95.0), 5.5)
        self.assertEqual(get_martin_hopkins_divisor(250.0, 140.0), 6.0)
        # TG > 400
        self.assertEqual(get_martin_hopkins_divisor(500.0, 150.0), 6.5)

    def test_martin_hopkins_ldl_calculation(self):
        # TC=220, HDL=50, non_HDL=170, TG=180 -> f=6.0 -> VLDL = 180/6 = 30 -> LDL = 220 - 50 - 30 = 140.0
        ldl = calculate_martin_hopkins_ldl(220.0, 50.0, 180.0)
        self.assertEqual(ldl, 140.0)

    def test_martin_hopkins_invariant_convergence_and_violation(self):
        clean_panel = {
            "total_cholesterol": 220.0,
            "hdl": 50.0,
            "ldl": 140.0,
            "triglycerides": 180.0,
        }
        status, rel_err = evaluate_martin_hopkins_invariant(clean_panel, tol=0.10)
        self.assertEqual(status, "CONVERGED")
        self.assertLessEqual(rel_err, 0.10)

        # Corrupt LDL by 30%
        corrupt_panel = clean_panel.copy()
        corrupt_panel["ldl"] = 185.0
        status_corrupt, rel_err_corrupt = evaluate_martin_hopkins_invariant(corrupt_panel, tol=0.10)
        self.assertEqual(status_corrupt, "VIOLATION")
        self.assertGreater(rel_err_corrupt, 0.10)

    def test_martin_hopkins_closed_form_inversions(self):
        panel = {
            "total_cholesterol": 220.0,
            "hdl": 50.0,
            "ldl": 140.0,
            "triglycerides": 180.0,
        }
        # Invert TC
        tc_star = invert_martin_hopkins(panel, "total_cholesterol")
        self.assertAlmostEqual(tc_star, 220.0, delta=1.0)

        # Invert HDL
        hdl_star = invert_martin_hopkins(panel, "hdl")
        self.assertAlmostEqual(hdl_star, 50.0, delta=1.0)

        # Invert LDL
        ldl_star = invert_martin_hopkins(panel, "ldl")
        self.assertAlmostEqual(ldl_star, 140.0, delta=1.0)

        # Invert TG
        tg_star = invert_martin_hopkins(panel, "triglycerides")
        self.assertAlmostEqual(tg_star, 180.0, delta=5.0)

    def test_martin_hopkins_isolation_search_repair(self):
        # Corrupt TC: 220 -> 2200 (decimal shift)
        corrupt_panel = {
            "total_cholesterol": 2200.0,
            "hdl": 50.0,
            "ldl": 140.0,
            "triglycerides": 180.0,
        }
        res = hierarchical_isolation_search(corrupt_panel, method="MARTIN_HOPKINS")
        self.assertEqual(res["action"], "CORRECT_K1")
        self.assertEqual(res["field"], "total_cholesterol")
        self.assertAlmostEqual(res["target_val"], 220.0, delta=2.0)


class TestCounterfactualXAI(unittest.TestCase):
    """Tests explain_counterfactual diagnostics."""

    def test_counterfactual_on_clean_panel(self):
        panel = {"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0}
        exp = explain_counterfactual(panel)
        self.assertTrue(exp["is_converged"])
        self.assertIsNone(exp["isolated_field"])
        self.assertIn("physiological equilibrium", exp["clinical_explanation"])

    def test_counterfactual_on_decimal_shift(self):
        panel = {"total_cholesterol": 2000.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0}
        exp = explain_counterfactual(panel)
        self.assertFalse(exp["is_converged"])
        self.assertEqual(exp["isolated_field"], "total_cholesterol")
        self.assertTrue(exp["field_counterfactuals"]["total_cholesterol"]["is_decimal_shift"])
        self.assertIn("decimal point shift", exp["clinical_explanation"])


class TestClinicalRiskFlip(unittest.TestCase):
    """Tests ASCVD / NCEP ATP III risk stratification and false escalation/reassurance."""

    def test_ldl_risk_tiers(self):
        self.assertEqual(classify_ldl_risk(85.0)["category"], "Optimal")
        self.assertEqual(classify_ldl_risk(115.0)["category"], "Near / Above Optimal")
        self.assertEqual(classify_ldl_risk(145.0)["category"], "Borderline High")
        self.assertEqual(classify_ldl_risk(175.0)["category"], "High")
        self.assertEqual(classify_ldl_risk(210.0)["category"], "Very High")

    def test_false_escalation_detection(self):
        # Raw extracted: 195 mg/dL (Very High), True: 110 mg/dL (Near Optimal)
        flip = evaluate_clinical_risk_flip(195.0, 110.0)
        self.assertTrue(flip["has_risk_flip"])
        self.assertEqual(flip["direction"], "FALSE_ESCALATION")
        self.assertEqual(flip["severity"], "CRITICAL")
        self.assertIn("inappropriate prescription", flip["clinical_narrative"])

    def test_false_reassurance_detection(self):
        # Raw extracted: 80 mg/dL (Optimal), True: 175 mg/dL (High)
        flip = evaluate_clinical_risk_flip(80.0, 175.0)
        self.assertTrue(flip["has_risk_flip"])
        self.assertEqual(flip["direction"], "FALSE_REASSURANCE")
        self.assertEqual(flip["severity"], "CRITICAL")
        self.assertIn("missed diagnosis", flip["clinical_narrative"])

    def test_concordant_risk(self):
        flip = evaluate_clinical_risk_flip(115.0, 120.0)
        self.assertFalse(flip["has_risk_flip"])
        self.assertEqual(flip["direction"], "CONCORDANT")


class TestAdversarialInjection(unittest.TestCase):
    """Tests adversarial error simulation."""

    def setUp(self):
        self.panel = {"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0}

    def test_decimal_shift_injection(self):
        corrupt, confs, meta = inject_adversarial_corruption(self.panel, "total_cholesterol", "DECIMAL_SHIFT_UP")
        self.assertEqual(corrupt["total_cholesterol"], 2000.0)
        self.assertEqual(confs["total_cholesterol"], 0.65)

    def test_digit_transposition_injection(self):
        corrupt, confs, meta = inject_adversarial_corruption(self.panel, "hdl", "DIGIT_TRANSPOSITION")
        self.assertNotEqual(corrupt["hdl"], 50.0)

    def test_extreme_tg_injection(self):
        corrupt, confs, meta = inject_adversarial_corruption(self.panel, "triglycerides", "EXTREME_TG_ABSTAIN")
        self.assertEqual(corrupt["triglycerides"], 850.0)


from src.fhir_export import (
    generate_fhir_r4_lipid_bundle,
    verify_fhir_bundle_sha256,
    LOINC_CODES,
)


class TestFHIRR4Export(unittest.TestCase):
    """Tests HL7 FHIR Release 4 standard compliance, LOINC coding, and SHA-256 seal."""

    def test_fhir_bundle_structure(self):
        raw = {"total_cholesterol": 2000.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0}
        ver = {"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 120.0, "triglycerides": 150.0}
        bundle = generate_fhir_r4_lipid_bundle(
            "doc_test", raw, ver, "mg/dL", "CONVERGED_K1", "FRIEDEWALD", 18.5
        )

        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "collection")
        self.assertGreaterEqual(len(bundle["entry"]), 5)  # 1 DiagnosticReport + 4 Observations

        # Check DiagnosticReport
        report = bundle["entry"][0]["resource"]
        self.assertEqual(report["resourceType"], "DiagnosticReport")
        self.assertEqual(report["code"]["coding"][0]["code"], "24331-1")

        # Check Observations and LOINC codes
        obs_codes = [e["resource"]["code"]["coding"][0]["code"] for e in bundle["entry"][1:]]
        self.assertIn("2093-3", obs_codes)  # TC
        self.assertIn("2085-9", obs_codes)  # HDL
        self.assertIn("13457-7", obs_codes) # LDL
        self.assertIn("2571-8", obs_codes)  # TG

    def test_sha256_seal_integrity_and_mutation_detection(self):
        """Verifies valid SHA-256 seal passes and any 1-byte mutation fails."""
        raw = {"total_cholesterol": 250.0, "hdl": 50.0, "ldl": 150.0, "triglycerides": 250.0}
        ver = {"total_cholesterol": 250.0, "hdl": 50.0, "ldl": 150.0, "triglycerides": 250.0}
        bundle = generate_fhir_r4_lipid_bundle(
            "doc_crypto_test", raw, ver, "mg/dL", "CONVERGED", "FRIEDEWALD", 12.0
        )

        # 1. Unaltered bundle MUST pass verification
        self.assertTrue(verify_fhir_bundle_sha256(bundle), "Unaltered bundle failed SHA-256 seal verification")

        # 2. Mutate a single byte in an observation value: 150.0 -> 150.1
        bundle["entry"][1]["resource"]["valueQuantity"]["value"] = 150.1
        self.assertFalse(verify_fhir_bundle_sha256(bundle), "One-byte mutation was not caught by SHA-256 seal verification!")


if __name__ == "__main__":
    unittest.main()


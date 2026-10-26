"""
tests/test_dataset_generator.py
=============================================================================
Verification suite for BioTrace Dataset Expansion & External Adapters:
  1. BiochemicalProfileGenerator seedable determinism and mathematical invariants
  2. Multi-panel coverage (Lipid, Protein, Electrolyte, Calcium)
  3. Principled abstention flags (TG > 800 mg/dL)
  4. External Adapters (ClinOCRAdapter, MendeleyAdapter, EkaCareAdapter)
  5. Leakage-safe manifest isolation (zero ground-truth in inference input paths)
"""

import os
import sys
import json
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_pipeline.schema import (
    TrackType, SplitType, ArtifactType, ArtifactSeverity, BoundingBox
)
from src.data_pipeline.profiles import BiochemicalProfileGenerator
from src.data_pipeline.adapters.clinocr_adapter import ClinOCRAdapter
from src.data_pipeline.adapters.mendeley_adapter import MendeleyAdapter
from src.data_pipeline.adapters.ekacare_adapter import EkaCareAdapter


class TestBiochemicalProfiles(unittest.TestCase):
    """Verifies physiological realism and mathematical guarantees of profiles."""

    def test_seedable_determinism(self):
        gen1 = BiochemicalProfileGenerator(seed=12345)
        p1 = gen1.generate_lipid_profile()

        gen2 = BiochemicalProfileGenerator(seed=12345)
        p2 = gen2.generate_lipid_profile()

        self.assertEqual(p1["clean_values"], p2["clean_values"])
        self.assertEqual(p1["active_equation"], p2["active_equation"])

    def test_friedewald_concordance(self):
        gen = BiochemicalProfileGenerator(seed=42)
        for _ in range(25):
            p = gen.generate_lipid_profile(category="friedewald")
            vals = p["clean_values"]
            tc, hdl, ldl, tg = vals["total_cholesterol"], vals["hdl"], vals["ldl"], vals["triglycerides"]

            self.assertLessEqual(tg, 400.0)
            expected_tc = hdl + ldl + (tg / 5.0)
            self.assertAlmostEqual(tc, expected_tc, delta=0.15)
            self.assertFalse(p["requires_abstention"])

    def test_abstention_boundary_chylomicronemia(self):
        gen = BiochemicalProfileGenerator(seed=999)
        p = gen.generate_lipid_profile(category="severe_hypertriglyceridemia")
        vals = p["clean_values"]

        self.assertGreater(vals["triglycerides"], 800.0)
        self.assertTrue(p["requires_abstention"])
        self.assertIn("ABSTAIN_TG800", p["abstention_reason"])

    def test_protein_panel_invariants(self):
        gen = BiochemicalProfileGenerator(seed=101)
        for _ in range(15):
            p = gen.generate_protein_profile()
            vals = p["clean_values"]
            tp, alb, glob = vals["total_protein"], vals["albumin"], vals["globulin"]
            self.assertAlmostEqual(tp, alb + glob, delta=0.05)

    def test_electrolyte_anion_gap_invariants(self):
        gen = BiochemicalProfileGenerator(seed=202)
        for _ in range(15):
            p = gen.generate_electrolyte_profile()
            vals = p["clean_values"]
            na, cl, hco3, ag = vals["sodium"], vals["chloride"], vals["bicarbonate"], vals["anion_gap"]
            self.assertAlmostEqual(ag, na - (cl + hco3), delta=0.05)

    def test_calcium_payne_invariants(self):
        gen = BiochemicalProfileGenerator(seed=303)
        for _ in range(15):
            p = gen.generate_calcium_profile()
            vals = p["clean_values"]
            ca, alb, corr_ca = vals["calcium"], vals["albumin"], vals["corrected_calcium"]
            expected_corr = ca + 0.8 * (4.0 - alb)
            self.assertAlmostEqual(corr_ca, expected_corr, delta=0.15)


class TestExternalAdapters(unittest.TestCase):
    """Verifies ClinOCR-Bench, Mendeley, and EkaCare adapters."""

    def test_clinocr_adapter_structure_and_subsets(self):
        clinocr_root = os.path.join(ROOT, "data", "external", "clinocr_bench")
        adapter = ClinOCRAdapter(clinocr_root)
        inspection = adapter.inspect_dataset()

        self.assertEqual(inspection["status"], "available")
        self.assertEqual(inspection["document_count"], 384)

        manifest_out = os.path.join(ROOT, "reports", "test_clinocr_manifest.json")
        manifest = adapter.generate_evaluation_manifest(manifest_out)

        self.assertEqual(manifest["document_count"], 384)
        expected_subsets = ["handwriting", "mixed", "normal", "poor", "rotated", "tables"]
        self.assertEqual(manifest["subsets"], expected_subsets)

        # Verify strict leakage isolation in manifest
        for entry in manifest["entries"][:10]:
            self.assertIn("image_path", entry)
            self.assertIn("evaluation_reference", entry)
            self.assertIn("text", entry["evaluation_reference"])
            # Ground-truth text must NOT be present at root level of entry
            self.assertNotIn("text", entry)
            self.assertNotIn("ground_truth_clean", entry)

        if os.path.exists(manifest_out):
            os.remove(manifest_out)

    def test_mendeley_adapter_status_handling(self):
        mendeley_root = os.path.join(ROOT, "data", "external", "mendeley")
        adapter = MendeleyAdapter(mendeley_root)
        inspection = adapter.inspect_dataset()
        self.assertIn(inspection["status"], ["ready", "empty", "directory_missing"])

    def test_ekacare_adapter_status_handling(self):
        ekacare_root = os.path.join(ROOT, "data", "external", "ekacare")
        adapter = EkaCareAdapter(ekacare_root)
        inspection = adapter.inspect_dataset()
        self.assertIn(inspection["status"], ["ready", "empty", "directory_missing"])


if __name__ == "__main__":
    unittest.main()

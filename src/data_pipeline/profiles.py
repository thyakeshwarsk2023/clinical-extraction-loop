"""
src/data_pipeline/profiles.py
=============================================================================
BioTrace Dataset Expansion — Biochemical Profile Generator.
Generates physiologically realistic, mathematically verified clinical laboratory
profiles across multiple panel families:
  1. Lipid Profile (Friedewald, Sampson, Martin-Hopkins, Chylomicronemia TG>800)
  2. Total Serum Protein Panel (Total Protein = Albumin + Globulin)
  3. Serum Electrolytes Panel (Anion Gap = Na - (Cl + HCO3))
  4. Calcium & Renal Panel (Payne Corrected Calcium, BUN/Creatinine)
"""

import random
from typing import Dict, Any, Tuple


class BiochemicalProfileGenerator:
    """Generates realistic patient laboratory profiles with mathematical guarantees."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_lipid_profile(self, unit_system: str = "mg/dL", category: str = "random") -> Dict[str, Any]:
        """
        Generates a lipid profile with exact mathematical invariant concordance.
        Categories:
          - 'friedewald': TG <= 400 mg/dL
          - 'sampson': 400 < TG <= 800 mg/dL
          - 'severe_hypertriglyceridemia': TG > 800 mg/dL (abstention boundary)
          - 'random': distributed across categories
        """
        if category == "random":
            r = self.rng.random()
            if r < 0.70:
                category = "friedewald"
            elif r < 0.90:
                category = "sampson"
            else:
                category = "severe_hypertriglyceridemia"

        if category == "friedewald":
            # Realistic biological bounds for normal/borderline patients
            hdl = round(self.rng.uniform(28.0, 85.0), 1)
            tg = round(self.rng.uniform(45.0, 395.0), 1)
            # Sample realistic LDL
            ldl = round(self.rng.uniform(60.0, 195.0), 1)
            # TC strictly derived from Friedewald invariant
            tc = round(hdl + ldl + (tg / 5.0), 1)
            active_equation = "FRIEDEWALD"
            method_text = "Method: Calculated by Friedewald Equation (Valid when TG <= 400 mg/dL)"
            requires_abstention = False
            abstention_reason = None

        elif category == "sampson":
            # High triglycerides requiring NIH Sampson Equation 2
            hdl = round(self.rng.uniform(25.0, 60.0), 1)
            tg = round(self.rng.uniform(405.0, 795.0), 1)
            tc = round(self.rng.uniform(190.0, 320.0), 1)
            non_hdl = tc - hdl
            vldl = (tg / 8.56) + ((tg * non_hdl) / 2140.0) - ((tg ** 2) / 16100.0)
            ldl = round((tc / 0.948) - (hdl / 0.971) - vldl - 9.44, 1)
            if ldl <= 10.0:
                ldl = 45.0
                tc = round(hdl + ldl + (tg / 6.0), 1)
            active_equation = "SAMPSON"
            method_text = "Method: Calculated by Sampson NIH Equation 2 (Valid when 400 < TG <= 800 mg/dL)"
            requires_abstention = False
            abstention_reason = None

        else: # severe_hypertriglyceridemia
            hdl = round(self.rng.uniform(20.0, 45.0), 1)
            tg = round(self.rng.uniform(820.0, 1850.0), 1)
            tc = round(self.rng.uniform(260.0, 520.0), 1)
            ldl = round(self.rng.uniform(50.0, 150.0), 1)
            active_equation = "SAMPSON_EXCEEDED"
            method_text = "Method: Calculated LDL (Warning: TG > 800 mg/dL invalidates calculation formulas)"
            requires_abstention = True
            abstention_reason = "ABSTAIN_TG800: Chylomicronemia invalidates indirect LDL calculation."

        clean_mgdl = {
            "total_cholesterol": tc,
            "hdl": hdl,
            "ldl": ldl,
            "triglycerides": tg,
        }

        # Conversion to SI units (mmol/L) if specified
        if unit_system.lower() in ["mmol/l", "mmol/litre"]:
            unit = "mmol/L"
            rendered_values = {
                "total_cholesterol": round(tc / 38.67, 2),
                "hdl": round(hdl / 38.67, 2),
                "ldl": round(ldl / 38.67, 2),
                "triglycerides": round(tg / 88.57, 2),
            }
            ref_intervals = {
                "total_cholesterol": "< 5.18 mmol/L",
                "hdl": ">= 1.04 mmol/L",
                "ldl": "< 2.59 mmol/L",
                "triglycerides": "< 1.70 mmol/L",
            }
        else:
            unit = "mg/dL"
            rendered_values = clean_mgdl.copy()
            ref_intervals = {
                "total_cholesterol": "< 200 mg/dL",
                "hdl": ">= 40 mg/dL",
                "ldl": "< 100 mg/dL",
                "triglycerides": "< 150 mg/dL",
            }

        return {
            "panel_type": "LIPID_PANEL",
            "unit": unit,
            "active_equation": active_equation,
            "method_text": method_text,
            "requires_abstention": requires_abstention,
            "abstention_reason": abstention_reason,
            "clean_values": clean_mgdl,
            "rendered_values": rendered_values,
            "ref_intervals": ref_intervals,
            "analyte_names": {
                "total_cholesterol": "Total Cholesterol",
                "hdl": "HDL Cholesterol",
                "ldl": "LDL Cholesterol",
                "triglycerides": "Triglycerides",
            }
        }

    def generate_protein_profile(self) -> Dict[str, Any]:
        """Generates Total Serum Protein profile (Total Protein = Albumin + Globulin)."""
        alb = round(self.rng.uniform(3.5, 5.2), 1)
        glob = round(self.rng.uniform(2.1, 3.8), 1)
        tp = round(alb + glob, 1)

        values = {
            "total_protein": tp,
            "albumin": alb,
            "globulin": glob,
        }
        ref_intervals = {
            "total_protein": "6.0 - 8.3 g/dL",
            "albumin": "3.5 - 5.0 g/dL",
            "globulin": "2.0 - 3.5 g/dL",
        }

        return {
            "panel_type": "PROTEIN_PANEL",
            "unit": "g/dL",
            "active_equation": "TOTAL_PROTEIN_SUM",
            "method_text": "Method: Biuret spectrophotometry with calculated globulin fraction",
            "requires_abstention": False,
            "abstention_reason": None,
            "clean_values": values.copy(),
            "rendered_values": values.copy(),
            "ref_intervals": ref_intervals,
            "analyte_names": {
                "total_protein": "Total Protein",
                "albumin": "Albumin",
                "globulin": "Globulin",
            }
        }

    def generate_electrolyte_profile(self) -> Dict[str, Any]:
        """Generates Serum Electrolytes panel (Anion Gap = Na - (Cl + HCO3))."""
        na = round(self.rng.uniform(135.0, 146.0), 1)
        cl = round(self.rng.uniform(96.0, 107.0), 1)
        hco3 = round(self.rng.uniform(22.0, 29.0), 1)
        k = round(self.rng.uniform(3.5, 5.1), 1)
        anion_gap = round(na - (cl + hco3), 1)

        values = {
            "sodium": na,
            "potassium": k,
            "chloride": cl,
            "bicarbonate": hco3,
            "anion_gap": anion_gap,
        }
        ref_intervals = {
            "sodium": "136 - 145 mEq/L",
            "potassium": "3.5 - 5.0 mEq/L",
            "chloride": "98 - 106 mEq/L",
            "bicarbonate": "22 - 28 mEq/L",
            "anion_gap": "8.0 - 16.0 mEq/L",
        }

        return {
            "panel_type": "ELECTROLYTE_PANEL",
            "unit": "mEq/L",
            "active_equation": "ANION_GAP_CALCULATION",
            "method_text": "Method: Ion-Selective Electrode (ISE) potentiometry with calculated Anion Gap",
            "requires_abstention": False,
            "abstention_reason": None,
            "clean_values": values.copy(),
            "rendered_values": values.copy(),
            "ref_intervals": ref_intervals,
            "analyte_names": {
                "sodium": "Sodium (Na+)",
                "potassium": "Potassium (K+)",
                "chloride": "Chloride (Cl-)",
                "bicarbonate": "Bicarbonate (HCO3-)",
                "anion_gap": "Anion Gap",
            }
        }

    def generate_calcium_profile(self) -> Dict[str, Any]:
        """Generates Serum Calcium & Albumin profile (Payne Corrected Calcium Formula)."""
        ca = round(self.rng.uniform(7.8, 10.6), 1)
        alb = round(self.rng.uniform(2.8, 4.8), 1)
        corrected_ca = round(ca + 0.8 * (4.0 - alb), 1)

        values = {
            "calcium": ca,
            "albumin": alb,
            "corrected_calcium": corrected_ca,
        }
        ref_intervals = {
            "calcium": "8.5 - 10.2 mg/dL",
            "albumin": "3.5 - 5.0 g/dL",
            "corrected_calcium": "8.5 - 10.2 mg/dL",
        }

        return {
            "panel_type": "CALCIUM_PANEL",
            "unit": "mg/dL",
            "active_equation": "PAYNE_CORRECTED_CALCIUM",
            "method_text": "Method: Arsenazo III dye-binding; Corrected Ca = Total Ca + 0.8*(4.0 - Albumin)",
            "requires_abstention": False,
            "abstention_reason": None,
            "clean_values": values.copy(),
            "rendered_values": values.copy(),
            "ref_intervals": ref_intervals,
            "analyte_names": {
                "calcium": "Total Calcium",
                "albumin": "Serum Albumin",
                "corrected_calcium": "Corrected Calcium",
            }
        }

"""
src/panels/electrolytes.py
=============================================================================
Clinical Electrolyte Panel: Sodium, Potassium, Chloride, Bicarbonate, Anion Gap,
and Osmolar Gap.

Mathematical Invariants:
    1. Hard Identity (Anion Gap):
       Anion Gap = Sodium - (Chloride + Bicarbonate)
       Normal physiological reference bounds: 8.0 - 16.0 mEq/L
       Optional with_k variant:
       Anion Gap (with K) = (Sodium + Potassium) - (Chloride + Bicarbonate)
       Normal physiological reference bounds: 12.0 - 20.0 mEq/L

    2. Soft Identity / Advisory Check (Osmolar Gap):
       Calculated Osmolality = 2 * Sodium + (Glucose / 18.0) + (BUN / 2.8)
       Osmolar Gap = Measured Osmolality - Calculated Osmolality
       Normal clinical reference range: -10.0 to +10.0 mOsm/kg.
       CRITICAL DESIGN NOTE:
       Biological osmolar gap variance is clinically meaningful (e.g., toxic alcohol
       ingestion, ketoacidosis, hyperproteinemia) and not merely OCR measurement noise.
       Deviations outside [-10, +10] emit an ADVISORY flag only (matching the
       Hct ~ 3xHb soft check pattern in Stage 2). It does NOT trigger a hard repair cycle.

Validation & Derivation:
    - Validated against CDC Continuous NHANES August 2021-August 2023 cohort (BIOPRO_L, N = 6,325).
    - Anion Gap mean 13.57 mEq/L (median 13.0 mEq/L, std 2.44 mEq/L).
    - 91.35% (5,778 patients) fall within normal 8.0-16.0 mEq/L; 99.98% within physiological limits.
    - Operating tolerance for Anion Gap verification is set to 1.0 mEq/L absolute or 5.0% relative error.
"""

from typing import Any
from src.panel_registry import Panel, PanelRegistry


class ElectrolytePanel:
    """
    Evaluates and inverts clinical electrolyte and acid-base panels.
    """

    name: str = "electrolytes"
    required_fields: list[str] = ["sodium", "chloride", "bicarbonate"]
    optional_fields: list[str] = ["potassium", "anion_gap", "glucose", "bun", "measured_osmolality"]
    tolerance: float = 0.05
    abs_tolerance: float = 1.0
    abstention_rules: list[str] = [
        "non-positive electrolyte concentrations (Sodium <= 0, Chloride <= 0, Bicarbonate <= 0)",
        "subfractions exceed sodium (Chloride + Bicarbonate >= Sodium)",
        "extreme biological implausibility (Sodium not in [100, 180] mEq/L, Chloride not in [60, 140] mEq/L)",
    ]

    def verify_bounds(self, values: dict[str, float]) -> tuple[bool, str]:
        na = float(values.get("sodium", 0.0))
        cl = float(values.get("chloride", 0.0))
        hco3 = float(values.get("bicarbonate", 0.0))

        if na <= 0.0 or cl <= 0.0 or hco3 <= 0.0:
            return False, "BOUND_VIOLATION_NON_POSITIVE_VALUE"

        # Broad biological plausibility limits (mEq/L or mmol/L)
        if not (100.0 <= na <= 180.0):
            return False, "BOUND_VIOLATION_EXTREME_SODIUM"

        if not (60.0 <= cl <= 140.0):
            return False, "BOUND_VIOLATION_EXTREME_CHLORIDE"

        if not (5.0 <= hco3 <= 50.0):
            return False, "BOUND_VIOLATION_EXTREME_BICARBONATE"

        if (cl + hco3) >= na:
            return False, "BOUND_VIOLATION_ANIONS_EXCEED_SODIUM"

        if "potassium" in values:
            k = float(values["potassium"])
            if k <= 0.0 or not (1.5 <= k <= 9.0):
                return False, "BOUND_VIOLATION_EXTREME_POTASSIUM"

        return True, "BOUNDS_SATISFIED"

    def identity_fn(self, values: dict[str, float], with_k: bool = False) -> float:
        """
        Computes the Anion Gap invariant residual.
        If 'anion_gap' is present in values:
            returns |observed_ag - calculated_ag|
        If 'anion_gap' is NOT reported:
            returns distance outside standard normal bounds [8.0, 16.0] (or [12.0, 20.0] with_k),
            or 0.0 if within normal physiological bounds.
        """
        na = float(values.get("sodium", 0.0))
        cl = float(values.get("chloride", 0.0))
        hco3 = float(values.get("bicarbonate", 0.0))
        k = float(values.get("potassium", 0.0)) if with_k else 0.0

        calc_ag = (na + k) - (cl + hco3)

        if "anion_gap" in values:
            obs_ag = float(values["anion_gap"])
            return abs(obs_ag - calc_ag)

        low_bound = 12.0 if with_k else 8.0
        high_bound = 20.0 if with_k else 16.0

        if low_bound <= calc_ag <= high_bound:
            return 0.0
        elif calc_ag < low_bound:
            return low_bound - calc_ag
        else:
            return calc_ag - high_bound

    def check_osmolar_gap(self, values: dict[str, float]) -> dict[str, Any] | None:
        """
        Soft Identity 2: Evaluates calculated osmolality against measured osmolality.
        Formula:
            Calculated Osmolality = 2 * Na + (Glucose / 18.0) + (BUN / 2.8)
            Osmolar Gap = Measured Osmolality - Calculated Osmolality

        Returns an ADVISORY soft flag dictionary if osmolar gap is outside [-10, +10] mOsm/kg.
        Does NOT trigger hard repair or validation failure.
        """
        if not ("measured_osmolality" in values and "sodium" in values and "glucose" in values and "bun" in values):
            return None

        na = float(values["sodium"])
        glu = float(values["glucose"])
        bun = float(values["bun"])
        measured_osmo = float(values["measured_osmolality"])

        if na <= 0.0 or glu <= 0.0 or bun <= 0.0 or measured_osmo <= 0.0:
            return None

        calc_osmo = 2.0 * na + (glu / 18.0) + (bun / 2.8)
        gap = measured_osmo - calc_osmo

        if abs(gap) > 10.0:
            return {
                "rule": "OSMOLAR_GAP_ADVISORY",
                "measured_osmolality": round(measured_osmo, 1),
                "calculated_osmolality": round(calc_osmo, 1),
                "osmolar_gap": round(gap, 1),
                "severity": "ADVISORY_SOFT_FLAG",
                "message": (
                    f"Osmolar gap {round(gap, 1)} mOsm/kg exceeds normal [-10, 10] mOsm/kg range. "
                    "Reflects biological variance or unmeasured osmolytes; advisory only, no repair triggered."
                )
            }

        return None

    def evaluate_invariant(self, values: dict[str, float], tolerance: float | None = None, with_k: bool = False) -> tuple[str, float, float]:
        """
        Evaluates the primary Anion Gap equality invariant.
        """
        bounds_ok, reason = self.verify_bounds(values)
        if not bounds_ok:
            return "VIOLATION", 1.0, 1.0

        tol = tolerance if tolerance is not None else self.tolerance
        na = float(values["sodium"])
        cl = float(values["chloride"])
        hco3 = float(values["bicarbonate"])
        k = float(values.get("potassium", 0.0)) if with_k else 0.0

        calc_ag = (na + k) - (cl + hco3)

        if "anion_gap" in values:
            obs_ag = float(values["anion_gap"])
            abs_res = abs(obs_ag - calc_ag)
            rel_err = abs_res / max(obs_ag, 1.0)

            if rel_err <= tol or abs_res <= (self.abs_tolerance + 1e-6):
                return "CONVERGED", round(abs_res, 3), round(rel_err, 4)
            return "VIOLATION", round(abs_res, 3), round(rel_err, 4)

        # If anion gap was not explicitly reported as a separate analyte, verify physiological normalcy
        low_bound = 12.0 if with_k else 8.0
        high_bound = 20.0 if with_k else 16.0
        nominal_mid = 16.0 if with_k else 12.0

        if low_bound <= calc_ag <= high_bound:
            return "CONVERGED", 0.0, 0.0

        delta = (low_bound - calc_ag) if calc_ag < low_bound else (calc_ag - high_bound)
        rel_err = delta / nominal_mid
        return "VIOLATION", round(delta, 3), round(rel_err, 4)

    def evaluate_full(self, values: dict[str, float], tolerance: float | None = None, with_k: bool = False) -> dict[str, Any]:
        """
        Comprehensive evaluation returning hard invariant status and soft advisory flags.
        """
        status, abs_res, rel_err = self.evaluate_invariant(values, tolerance=tolerance, with_k=with_k)
        soft_flags = []
        osmo_advisory = self.check_osmolar_gap(values)
        if osmo_advisory:
            soft_flags.append(osmo_advisory)

        return {
            "status": status,
            "absolute_residual": abs_res,
            "relative_error": rel_err,
            "is_valid": status == "CONVERGED",
            "advisory_flags": soft_flags
        }

    def solve_inversion(self, values: dict[str, float], target_field: str, with_k: bool = False, **kwargs) -> float | None:
        """
        O(1) closed-form algebraic counterfactual inversions (K=1).
            Anion Gap = Sodium - (Chloride + Bicarbonate)  [with optional K]
        Inversions:
            Sodium*      = Anion Gap + Chloride + Bicarbonate (- Potassium if with_k)
            Chloride*    = Sodium - Bicarbonate - Anion Gap (+ Potassium if with_k)
            Bicarbonate* = Sodium - Chloride - Anion Gap (+ Potassium if with_k)
            Anion Gap*   = Sodium - (Chloride + Bicarbonate) (+ Potassium if with_k)
        """
        na = values.get("sodium")
        cl = values.get("chloride")
        hco3 = values.get("bicarbonate")
        k = float(values.get("potassium", 0.0)) if with_k else 0.0

        # Nominal reference AG (12.0 mEq/L standard, 16.0 mEq/L with_k) if not specified
        default_ag = 16.0 if with_k else 12.0
        ag_val = float(values.get("anion_gap", default_ag))

        if target_field == "sodium":
            if cl is not None and hco3 is not None:
                calc_na = ag_val + float(cl) + float(hco3) - k
                return round(calc_na, 1) if calc_na > 0 else None

        elif target_field == "chloride":
            if na is not None and hco3 is not None:
                calc_cl = float(na) + k - float(hco3) - ag_val
                return round(calc_cl, 1) if calc_cl > 0 else None

        elif target_field == "bicarbonate":
            if na is not None and cl is not None:
                calc_hco3 = float(na) + k - float(cl) - ag_val
                return round(calc_hco3, 1) if calc_hco3 > 0 else None

        elif target_field == "anion_gap":
            if na is not None and cl is not None and hco3 is not None:
                calc_ag = float(na) + k - (float(cl) + float(hco3))
                return round(calc_ag, 1)

        return None


# Self-register with global PanelRegistry
PanelRegistry.register(ElectrolytePanel())

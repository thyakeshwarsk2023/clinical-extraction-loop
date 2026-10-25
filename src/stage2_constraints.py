def normalize_to_mgdl(panel: dict, unit: str) -> dict:
    unit_lower = unit.lower().strip()
    if unit_lower in ["mg/dl", "mg/d-l"]:
        return panel.copy()
        
    if unit_lower in ["mmol/l", "mmol/litre"]:
        normalized = {}
        for k, v in panel.items():
            if k == "triglycerides":
                normalized[k] = round(v * 88.57, 1)
            else:
                normalized[k] = round(v * 38.67, 1)
        return normalized
        
    return panel.copy()

def verify_physiological_bounds(panel: dict) -> tuple:
    tc = float(panel.get("total_cholesterol", 0.0))
    hdl = float(panel.get("hdl", 0.0))
    ldl = float(panel.get("ldl", 0.0))
    tg = float(panel.get("triglycerides", 0.0))
    
    if tc <= 0 or hdl <= 0 or ldl <= 0 or tg <= 0:
        return False, "BOUND_VIOLATION_NON_POSITIVE_VALUE"
        
    if ldl >= tc:
        return False, "BOUND_VIOLATION_LDL_EXCEEDS_TOTAL"
        
    if hdl >= tc:
        return False, "BOUND_VIOLATION_HDL_EXCEEDS_TOTAL"
        
    if (hdl + ldl) >= tc:
        return False, "BOUND_VIOLATION_SUBFRACTIONS_EXCEED_TOTAL"
        
    return True, "BOUNDS_SATISFIED"

def evaluate_friedewald_invariant(panel: dict, tolerance: float = 0.10) -> tuple:
    tc = float(panel.get("total_cholesterol", 0.0))
    hdl = float(panel.get("hdl", 0.0))
    ldl = float(panel.get("ldl", 0.0))
    tg = float(panel.get("triglycerides", 0.0))
    
    # 1. Chylomicron biological validity boundary
    if tg > 400.0:
        return "ABSTAIN_CLINICAL_ANOMALY", 0.0, 0.0
        
    # 2. Friedewald Residual Calculation
    expected_tc = hdl + ldl + (tg / 5.0)
    residual = abs(tc - expected_tc)
    relative_residual = residual / tc if tc > 0 else 1.0
    
    if relative_residual <= tolerance:
        return "CONVERGED", residual, relative_residual
        
    return "VIOLATION", residual, relative_residual

def calculate_sampson_ldl(tc: float, hdl: float, tg: float) -> float:
    """Calculates LDL using the Sampson (2020) equation."""
    # LDL = TC/1.19 + HDL/1.1 - TG/3.98 - (TG * (TC - HDL)) / 8.2 - 0.17
    non_hdl = tc - hdl
    ldl = (tc / 1.19) + (hdl / 1.1) - (tg / 3.98) - ((tg * non_hdl) / 8.2) - 0.17
    return round(ldl, 1)

def evaluate_sampson_invariant(panel: dict, tol: float = 0.10) -> tuple[str, float]:
    """
    Evaluates Sampson equation for patients with high triglycerides (400 < TG <= 800 mg/dL).
    """
    tc = panel.get("total_cholesterol", 0.0)
    hdl = panel.get("hdl", 0.0)
    ldl_raw = panel.get("ldl", 0.0)
    tg = panel.get("triglycerides", 0.0)

    if tg > 800.0:
        return "ABSTAIN_TG800", 1.0

    if tc <= 0 or hdl <= 0:
        return "VIOLATION", 1.0

    ldl_calc = calculate_sampson_ldl(tc, hdl, tg)
    denom = abs(ldl_raw) if ldl_raw != 0 else 1.0
    rel_residual = abs(ldl_raw - ldl_calc) / denom

    if rel_residual <= tol:
        return "CONVERGED", rel_residual

    if ldl_raw <= 0:
        return "VIOLATION", 1.0

    return "VIOLATION", rel_residual


def get_martin_hopkins_divisor(tg: float, non_hdl: float) -> float:
    """
    Returns the dynamic Martin-Hopkins VLDL divisor based on TG and non-HDL strata.
    Bounds: Divisor ranges from 3.1 to 11.9.
    """
    if tg < 50:
        return 3.5 if non_hdl < 100 else 4.0
    elif tg < 100:
        return 4.5 if non_hdl < 100 else 5.0
    elif tg <= 400:
        return 5.5 if non_hdl < 100 else 6.0
    else:
        return 6.5


def calculate_martin_hopkins_ldl(tc: float, hdl: float, tg: float) -> float:
    non_hdl = tc - hdl
    f = get_martin_hopkins_divisor(tg, non_hdl)
    return round(tc - hdl - (tg / f), 1)


def evaluate_martin_hopkins_invariant(panel: dict, tol: float = 0.10) -> tuple[str, float]:
    tc, hdl, ldl_raw, tg = (
        panel.get("total_cholesterol", 0.0),
        panel.get("hdl", 0.0),
        panel.get("ldl", 0.0),
        panel.get("triglycerides", 0.0),
    )
    if tc <= 0 or hdl <= 0 or tg <= 0:
        return "VIOLATION", 1.0
    ldl_calc = calculate_martin_hopkins_ldl(tc, hdl, tg)
    rel_residual = abs(ldl_raw - ldl_calc) / max(ldl_raw, 1.0)
    return "CONVERGED" if rel_residual <= tol else "VIOLATION", rel_residual


def check_tg_decimal_shift_hypothesis(
    panel: dict,
    tolerance: float = 0.10,
    declared_route: str = None
) -> dict | None:
    """
    Tests decimal-shift hypotheses (TG/10, TG/100, TG*10) against clinical identities.
    Returns hypothesis evidence dictionary if a candidate restores identity; None otherwise.
    """
    raw_tg = panel.get("triglycerides", 0.0)
    if raw_tg <= 0:
        return None

    for scale in [0.1, 0.01, 10.0]:
        hyp_tg = round(raw_tg * scale, 1)
        if hyp_tg <= 0 or hyp_tg > 800.0:
            continue
        test_panel = panel.copy()
        test_panel["triglycerides"] = hyp_tg

        if declared_route == "MARTIN_HOPKINS":
            st, res = evaluate_martin_hopkins_invariant(test_panel, tolerance)
            eq = "MARTIN_HOPKINS"
        elif declared_route == "SAMPSON" or (declared_route is None and hyp_tg > 400.0):
            st, res = evaluate_sampson_invariant(test_panel, tolerance)
            eq = "SAMPSON"
        else:
            st, abs_res, res = evaluate_friedewald_invariant(test_panel, tolerance)
            eq = "FRIEDEWALD"

        if st == "CONVERGED":
            return {
                "hypothesis_tg": hyp_tg,
                "original_tg": raw_tg,
                "scale": scale,
                "equation": eq,
                "residual": round(res, 4)
            }
    return None


def evaluate_cbc_identities(cbc_panel: dict, tol: float = 0.05) -> dict:
    """
    Evaluates Complete Blood Count (CBC) mathematical identities:
      Hard identities:
        MCV = (Hct * 10) / RBC
        MCH = (Hb * 10) / RBC
        MCHC = (Hb * 100) / Hct
      Soft identity (advisory only):
        Hct ~ 3 * Hb
    """
    hb = float(cbc_panel.get("hemoglobin", 0.0))
    hct = float(cbc_panel.get("hematocrit", 0.0))
    rbc = float(cbc_panel.get("rbc", 0.0))
    mcv = float(cbc_panel.get("mcv", 0.0))
    mch = float(cbc_panel.get("mch", 0.0))
    mchc = float(cbc_panel.get("mchc", 0.0))

    violations = []
    soft_flags = []

    # 1. Hard Identity Checks
    if rbc > 0 and mcv > 0:
        expected_mcv = (hct * 10.0) / rbc
        if abs(mcv - expected_mcv) / mcv > tol:
            violations.append("MCV_IDENTITY_VIOLATION")

    if rbc > 0 and mch > 0:
        expected_mch = (hb * 10.0) / rbc
        if abs(mch - expected_mch) / mch > tol:
            violations.append("MCH_IDENTITY_VIOLATION")

    if hct > 0 and mchc > 0:
        expected_mchc = (hb * 100.0) / hct
        if abs(mchc - expected_mchc) / mchc > tol:
            violations.append("MCHC_IDENTITY_VIOLATION")

    # 2. Soft Rule: Hct ~ 3 * Hb (advisory warning only; does NOT trigger hard violation)
    if hb > 0 and hct > 0:
        ratio = hct / hb
        if abs(ratio - 3.0) / 3.0 > 0.15:
            soft_flags.append({
                "rule": "HCT_APPROX_3X_HB",
                "observed_ratio": round(ratio, 2),
                "expected_ratio": 3.0,
                "severity": "ADVISORY_SOFT_FLAG"
            })

    return {
        "is_valid": len(violations) == 0,
        "hard_violations": violations,
        "soft_flags": soft_flags
    }


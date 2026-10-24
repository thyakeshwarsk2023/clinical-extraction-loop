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

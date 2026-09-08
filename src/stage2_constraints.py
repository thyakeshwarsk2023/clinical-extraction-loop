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
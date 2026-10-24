"""
src/adversarial.py
===========================================================================
Adversarial Clinical Injection Module.
Simulates realistic and boundary-case transcription/OCR corruptions for robustness testing.
"""
import copy

ADVERSARIAL_MODES = {
    "DECIMAL_SHIFT_UP": "Order-of-magnitude decimal shift (10x missing decimal point)",
    "DECIMAL_SHIFT_DOWN": "Premature decimal point shift (0.1x scale factor)",
    "DIGIT_TRANSPOSITION": "Adjacent digit swap / transposition (e.g. 26.8 vs 62.8)",
    "VISUAL_SMUDGE": "Optical blur / occlusion smudge (degrades confidence to 0.35)",
    "EXTREME_TG_ABSTAIN": "Severe hypertriglyceridemia (TG = 850.0 mg/dL > 800)",
    "BOUND_VIOLATION_LDL_TC": "Biological bound violation (LDL >= Total Cholesterol)",
}

def inject_adversarial_corruption(
    panel: dict,
    field: str,
    corruption_type: str,
    confidences: dict = None
) -> tuple[dict, dict, dict]:
    """
    Injects a controlled perturbation into the panel and confidences.
    Returns:
        (corrupted_panel, updated_confidences, injection_metadata)
    """
    corrupted_panel = copy.deepcopy(panel)
    updated_confs = copy.deepcopy(confidences) if confidences else {f: 0.95 for f in panel}
    orig_val = float(panel.get(field, 100.0))
    meta = {
        "field": field,
        "orig_val": orig_val,
        "corruption_type": corruption_type,
    }
    
    if corruption_type == "DECIMAL_SHIFT_UP":
        corrupted_val = round(orig_val * 10.0, 1)
        updated_confs[field] = 0.65
        meta["description"] = f"Missing decimal point (10x): {orig_val:.1f} -> {corrupted_val:.1f}"
    elif corruption_type == "DECIMAL_SHIFT_DOWN":
        corrupted_val = round(orig_val * 0.1, 1)
        updated_confs[field] = 0.65
        meta["description"] = f"Premature decimal shift (0.1x): {orig_val:.1f} -> {corrupted_val:.1f}"
    elif corruption_type == "DIGIT_TRANSPOSITION":
        s = f"{orig_val:.1f}".split(".")[0]
        dec = f"{orig_val:.1f}".split(".")[1] if "." in f"{orig_val:.1f}" else "0"
        if len(s) >= 2:
            s_swap = s[1] + s[0] + s[2:]
            corrupted_val = float(f"{s_swap}.{dec}")
        else:
            corrupted_val = float(f"{dec}{s}.0")
        updated_confs[field] = 0.60
        meta["description"] = f"Adjacent digit transposition: {orig_val:.1f} -> {corrupted_val:.1f}"
    elif corruption_type == "VISUAL_SMUDGE":
        corrupted_val = round(orig_val + 35.0, 1)
        updated_confs[field] = 0.35
        meta["description"] = f"Visual smudge (confidence dropped to 0.35): {orig_val:.1f} -> {corrupted_val:.1f}"
    elif corruption_type == "EXTREME_TG_ABSTAIN":
        field = "triglycerides"
        corrupted_val = 850.0
        updated_confs["triglycerides"] = 0.90
        meta["field"] = "triglycerides"
        meta["description"] = "Severe Hypertriglyceridemia (TG = 850.0 mg/dL > 800) triggering ABSTAIN_TG800"
    elif corruption_type == "BOUND_VIOLATION_LDL_TC":
        corrupted_val = round(panel.get("total_cholesterol", 200.0) + 25.0, 1)
        field = "ldl"
        meta["field"] = "ldl"
        meta["description"] = f"Physiological violation (LDL {corrupted_val:.1f} >= TC {panel.get('total_cholesterol', 200.0):.1f})"
    else:
        corrupted_val = orig_val
        meta["description"] = "No modification"

    corrupted_panel[field] = corrupted_val
    meta["corrupted_val"] = corrupted_val
    return corrupted_panel, updated_confs, meta

"""
src/clinical_risk.py
===========================================================================
Clinical Risk Stratification and Misclassification (Risk Flip) Engine.
Evaluates Atherosclerotic Cardiovascular Disease (ASCVD) and NCEP ATP III
risk categorization flips caused by document extraction and transcription errors.
"""

def classify_ldl_risk(ldl_mgdl: float) -> dict:
    """
    Categorizes LDL-C according to NCEP ATP III / 2018 AHA/ACC guidelines.
    Tiers:
      1. Optimal: < 100 mg/dL
      2. Near Optimal / Above Optimal: 100 - 129 mg/dL
      3. Borderline High: 130 - 159 mg/dL
      4. High: 160 - 189 mg/dL
      5. Very High: >= 190 mg/dL
    """
    val = round(float(ldl_mgdl), 1)
    if val < 100.0:
        return {
            "tier_level": 1,
            "category": "Optimal",
            "range": "< 100 mg/dL",
            "color": "#10b981",  # emerald green
            "recommendation": "Lifestyle maintenance; statin generally not indicated.",
            "urgency": "LOW"
        }
    elif val <= 129.0:
        return {
            "tier_level": 2,
            "category": "Near / Above Optimal",
            "range": "100 - 129 mg/dL",
            "color": "#38bdf8",  # sky blue
            "recommendation": "Heart-healthy diet and physical activity; monitor annually.",
            "urgency": "LOW"
        }
    elif val <= 159.0:
        return {
            "tier_level": 3,
            "category": "Borderline High",
            "range": "130 - 159 mg/dL",
            "color": "#f59e0b",  # amber
            "recommendation": "Therapeutic lifestyle changes; consider moderate-intensity statin if risk factors.",
            "urgency": "MODERATE"
        }
    elif val <= 189.0:
        return {
            "tier_level": 4,
            "category": "High",
            "range": "160 - 189 mg/dL",
            "color": "#f97316",  # orange
            "recommendation": "Statin therapy recommended; target >= 50% LDL-C reduction.",
            "urgency": "HIGH"
        }
    else:
        return {
            "tier_level": 5,
            "category": "Very High",
            "range": ">= 190 mg/dL",
            "color": "#ef4444",  # rose red
            "recommendation": "Severe hypercholesterolemia. High-intensity statin + consider Ezetimibe/PCSK9 inhibitor.",
            "urgency": "CRITICAL"
        }


def evaluate_clinical_risk_flip(raw_ldl: float, verified_ldl: float) -> dict:
    """
    Detects clinical risk tier discordance ('Risk Flip') between raw extracted LDL
    and neurosymbolically verified LDL.
    """
    raw_tier = classify_ldl_risk(raw_ldl)
    ver_tier = classify_ldl_risk(verified_ldl)
    
    tier_diff = raw_tier["tier_level"] - ver_tier["tier_level"]
    has_flip = (tier_diff != 0)
    
    if tier_diff > 0:
        direction = "FALSE_ESCALATION"
        direction_label = "False Escalation (Overtreatment Risk)"
        clinical_narrative = (
            f"Extraction corruption artificially elevated LDL from {verified_ldl:.1f} mg/dL ({ver_tier['category']}) "
            f"to {raw_ldl:.1f} mg/dL ({raw_tier['category']}). Without BioTrace verification, patient faced "
            f"inappropriate prescription of statin pharmacotherapy and unnecessary clinical distress."
        )
    elif tier_diff < 0:
        direction = "FALSE_REASSURANCE"
        direction_label = "False Reassurance (Undertreatment Risk)"
        clinical_narrative = (
            f"Extraction corruption artificially depressed LDL from {verified_ldl:.1f} mg/dL ({ver_tier['category']}) "
            f"down to {raw_ldl:.1f} mg/dL ({raw_tier['category']}). BioTrace prevented missed diagnosis of "
            f"actionable hypercholesterolemia."
        )
    else:
        direction = "CONCORDANT"
        direction_label = "Concordant (Zero Risk Shift)"
        clinical_narrative = f"Both raw and verified LDL lie within the same clinical risk band ({ver_tier['category']})."
        
    if abs(tier_diff) >= 3:
        severity = "CRITICAL"
    elif abs(tier_diff) >= 1:
        severity = "MODERATE"
    else:
        severity = "NEGLIGIBLE"

    return {
        "has_risk_flip": has_flip,
        "direction": direction,
        "direction_label": direction_label,
        "raw_tier": raw_tier,
        "verified_tier": ver_tier,
        "raw_ldl": round(float(raw_ldl), 1),
        "verified_ldl": round(float(verified_ldl), 1),
        "tier_step_delta": abs(tier_diff),
        "signed_tier_delta": tier_diff,
        "severity": severity,
        "clinical_narrative": clinical_narrative,
    }

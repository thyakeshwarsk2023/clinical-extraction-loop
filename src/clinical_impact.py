import json
import uuid
import hashlib
from datetime import datetime

def evaluate_atp3_risk_category(ldl: float) -> str:
    """NCEP ATP-III Statin Eligibility Guidelines"""
    val = round(float(ldl), 1)
    if val < 100.0:
        return "Optimal (No Statin)"
    elif val < 130.0:
        return "Near Optimal (Monitor)"
    elif val < 160.0:
        return "Borderline High (Consider Mod-Intensity Statin)"
    elif val < 190.0:
        return "High (High-Intensity Statin Indicated)"
    else:
        return "Very High (Immediate High-Intensity Statin)"


def evaluate_clinical_risk_flip(raw_ldl: float, verified_ldl: float) -> dict:
    """
    Compares the ATP-III risk category of raw extracted LDL against verified LDL
    to detect false escalation (overtreatment) or false reassurance (undertreatment).
    """
    raw_cat = evaluate_atp3_risk_category(raw_ldl)
    ver_cat = evaluate_atp3_risk_category(verified_ldl)
    has_flip = (raw_cat != ver_cat)

    tiers = [
        "Optimal (No Statin)",
        "Near Optimal (Monitor)",
        "Borderline High (Consider Mod-Intensity Statin)",
        "High (High-Intensity Statin Indicated)",
        "Very High (Immediate High-Intensity Statin)",
    ]
    raw_idx = tiers.index(raw_cat) if raw_cat in tiers else 0
    ver_idx = tiers.index(ver_cat) if ver_cat in tiers else 0
    diff = raw_idx - ver_idx

    if diff > 0:
        direction = "FALSE_ESCALATION"
        direction_label = "False Escalation (Overtreatment Risk)"
        narrative = (
            f"Extraction corruption falsely escalated LDL from {verified_ldl:.1f} mg/dL ({ver_cat}) "
            f"to {raw_ldl:.1f} mg/dL ({raw_cat}). Without BioTrace verification, patient faced "
            f"inappropriate statin pharmacotherapy."
        )
    elif diff < 0:
        direction = "FALSE_REASSURANCE"
        direction_label = "False Reassurance (Undertreatment Risk)"
        narrative = (
            f"Extraction corruption falsely depressed LDL from {verified_ldl:.1f} mg/dL ({ver_cat}) "
            f"down to {raw_ldl:.1f} mg/dL ({raw_cat}). BioTrace prevented missed diagnosis of "
            f"actionable hypercholesterolemia."
        )
    else:
        direction = "CONCORDANT"
        direction_label = "Concordant (Zero Risk Shift)"
        narrative = f"Both raw and verified LDL lie within the same clinical risk band ({ver_cat})."

    return {
        "has_risk_flip": has_flip,
        "direction": direction,
        "direction_label": direction_label,
        "raw_category": raw_cat,
        "verified_category": ver_cat,
        "tier_step_delta": abs(diff),
        "signed_tier_delta": diff,
        "clinical_narrative": narrative,
    }


def generate_fhir_r4_bundle(
    panel: dict,
    patient_id: str = "PATIENT-001",
    raw_panel: dict = None,
    status_label: str = "VERIFIED",
    active_equation: str = "FRIEDEWALD",
) -> str:
    """Packages the healed numerical data into a standardized HL7 FHIR R4 DiagnosticReport and Observations."""
    loinc_map = {
        "total_cholesterol": ("2093-3", "Cholesterol in Serum or Plasma"),
        "hdl": ("2085-9", "Cholesterol in HDL in Serum or Plasma"),
        "ldl": ("13457-7", "Cholesterol in LDL in Serum or Plasma by calculation"),
        "triglycerides": ("2571-8", "Triglyceride in Serum or Plasma"),
    }
    
    now_iso = datetime.utcnow().isoformat() + "Z"
    observations = []
    obs_references = []

    for analyte, (loinc, display_name) in loinc_map.items():
        if analyte in panel:
            obs_id = str(uuid.uuid4())
            obs_references.append({
                "reference": f"Observation/{obs_id}",
                "display": display_name,
            })
            
            raw_v = raw_panel.get(analyte, panel[analyte]) if raw_panel else panel[analyte]
            is_repaired = abs(raw_v - panel[analyte]) > 0.01

            obs = {
                "resourceType": "Observation",
                "id": obs_id,
                "status": "amended" if is_repaired else "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc,
                        "display": display_name
                    }],
                    "text": analyte.replace("_", " ").title()
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": now_iso,
                "valueQuantity": {
                    "value": round(float(panel[analyte]), 1),
                    "unit": "mg/dL",
                    "system": "http://unitsofmeasure.org",
                    "code": "mg/dL"
                },
                "extension": [
                    {
                        "url": "http://biotrace.ai/fhir/StructureDefinition/raw-extracted-value",
                        "valueQuantity": {"value": round(float(raw_v), 1), "unit": "mg/dL"}
                    },
                    {
                        "url": "http://biotrace.ai/fhir/StructureDefinition/verification-state",
                        "valueString": "REPAIRED" if is_repaired else "CLEAN"
                    }
                ]
            }
            observations.append(obs)
            
    report_id = str(uuid.uuid4())
    report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "CH",
                "display": "Chemistry"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "24331-1",
                "display": "Lipid 1996 panel - Serum or Plasma"
            }],
            "text": "Lipid Panel with Closed-Loop Neurosymbolic Verification"
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": now_iso,
        "issued": now_iso,
        "result": obs_references,
        "conclusion": f"BioTrace Status: {status_label}. Invariant: {active_equation}.",
    }

    entries = [{"resource": report}] + [{"resource": obs} for obs in observations]

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": now_iso,
        "total": len(entries),
        "entry": entries
    }
    return json.dumps(bundle, indent=2)
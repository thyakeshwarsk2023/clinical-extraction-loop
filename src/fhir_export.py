"""
src/fhir_export.py
===========================================================================
HL7 FHIR R4 Clinical Export Engine.
Produces standard-compliant FHIR R4 JSON Bundles with LOINC-coded
DiagnosticReports and Observations, embedded with BioTrace neurosymbolic
provenance and verification audit signatures.
"""
import json
import time
import hashlib
from datetime import datetime

LOINC_CODES = {
    "total_cholesterol": {
        "code": "2093-3",
        "display": "Cholesterol in Serum or Plasma",
        "ref_high": 200.0,
        "ref_text": "< 200 mg/dL (Desirable)"
    },
    "hdl": {
        "code": "2085-9",
        "display": "Cholesterol in HDL in Serum or Plasma",
        "ref_low": 40.0,
        "ref_text": ">= 40 mg/dL (Desirable)"
    },
    "ldl": {
        "code": "13457-7",
        "display": "Cholesterol in LDL in Serum or Plasma by calculation",
        "ref_high": 100.0,
        "ref_text": "< 100 mg/dL (Optimal)"
    },
    "triglycerides": {
        "code": "2571-8",
        "display": "Triglyceride in Serum or Plasma",
        "ref_high": 150.0,
        "ref_text": "< 150 mg/dL (Normal)"
    }
}

def generate_fhir_r4_lipid_bundle(
    doc_id: str,
    raw_panel: dict,
    verified_panel: dict,
    unit: str,
    pipeline_status: str,
    active_equation: str,
    latency_ms: float,
    corrupted_fields: list = None
) -> dict:
    """
    Constructs a standard HL7 FHIR Release 4 Collection Bundle containing:
      1. DiagnosticReport (LOINC 24331-1: Lipid 1996 panel)
      2. Observation resources for TC, HDL, LDL, and TG with LOINC coding,
         reference intervals, interpretations, and BioTrace verification extensions.
    """
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle_hash = hashlib.sha256(f"{doc_id}-{now_iso}".encode()).hexdigest()[:12]
    bundle_id = f"biotrace-bundle-{bundle_hash}"
    
    entries = []
    obs_references = []
    
    # 1. Observation Resources
    for field, verified_val in verified_panel.items():
        if field not in LOINC_CODES:
            continue
            
        loinc = LOINC_CODES[field]
        obs_id = f"obs-{field}-{bundle_hash[:6]}"
        obs_references.append({
            "reference": f"Observation/{obs_id}",
            "display": loinc["display"]
        })
        
        raw_val = raw_panel.get(field, verified_val)
        is_repaired = abs(raw_val - verified_val) > 0.01
        
        interp_code = "N"
        interp_display = "Normal"
        if field == "hdl" and verified_val < 40.0:
            interp_code = "L"
            interp_display = "Low"
        elif field != "hdl" and verified_val > loinc.get("ref_high", 200.0):
            interp_code = "H"
            interp_display = "High"
            
        obs_resource = {
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
                    "code": loinc["code"],
                    "display": loinc["display"]
                }],
                "text": field.replace("_", " ").title()
            },
            "effectiveDateTime": now_iso,
            "valueQuantity": {
                "value": round(float(verified_val), 1),
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": "mg/dL" if "mg" in unit.lower() else "mmol/L"
            },
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": interp_code,
                    "display": interp_display
                }]
            }],
            "referenceRange": [{
                "text": loinc["ref_text"]
            }],
            "extension": [
                {
                    "url": "http://biotrace.ai/fhir/StructureDefinition/raw-extracted-value",
                    "valueQuantity": {
                        "value": round(float(raw_val), 1),
                        "unit": unit
                    }
                },
                {
                    "url": "http://biotrace.ai/fhir/StructureDefinition/verification-state",
                    "valueString": "REPAIRED_VIA_PHYSICAL_VISION" if is_repaired else "VERIFIED_CLEAN"
                }
            ]
        }
        
        if is_repaired:
            obs_resource["note"] = [{
                "text": f"BioTrace Neurosymbolic Closed Loop: Corrected transcription error from {raw_val:.1f} to {verified_val:.1f} {unit}."
            }]
            
        entries.append({
            "fullUrl": f"urn:uuid:{obs_id}",
            "resource": obs_resource
        })
        
    # 2. DiagnosticReport Resource
    report_id = f"report-lipid-{bundle_hash[:6]}"
    report_resource = {
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
        "effectiveDateTime": now_iso,
        "issued": now_iso,
        "result": obs_references,
        "conclusion": f"BioTrace Status: {pipeline_status}. Invariant: {active_equation}. Execution Latency: {latency_ms:.2f} ms.",
        "extension": [
            {
                "url": "http://biotrace.ai/fhir/StructureDefinition/pipeline-status",
                "valueString": pipeline_status
            },
            {
                "url": "http://biotrace.ai/fhir/StructureDefinition/active-invariant",
                "valueString": active_equation
            },
            {
                "url": "http://biotrace.ai/fhir/StructureDefinition/execution-latency-ms",
                "valueDecimal": round(latency_ms, 2)
            }
        ]
    }
    entries.insert(0, {
        "fullUrl": f"urn:uuid:{report_id}",
        "resource": report_resource
    })
    
    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "meta": {
            "lastUpdated": now_iso,
            "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]
        },
        "type": "collection",
        "timestamp": now_iso,
        "total": len(entries),
        "entry": entries
    }

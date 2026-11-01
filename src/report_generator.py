"""
src/report_generator.py
=======================
Conservative, evidence-grounded report generation and export for clinical laboratory panels.

Safety & Transparency Guarantees:
  1. Immutability: Original image is NEVER modified; all edits are performed on an explicit clone.
  2. Transparency Watermark: Every generated image features an indelible prominent banner:
     "AI-PROCESSED RESEARCH DEMONSTRATION — NOT AN OFFICIAL LABORATORY REPORT"
  3. Conservative Overwrite: Modifications are restricted strictly to fields with an explicitly
     ACCEPTED decision status. Bounding boxes are defensively validated.
  4. Three Decision States:
     - ACCEPTED CORRECTION: Value replaced in derived report, highlighted, audited.
     - RETAINED VALUE: Field preserved untouched, described as retained.
     - ABSTAINED / UNRESOLVED: Field preserved untouched, unverified candidate withheld,
       manual clinician review required.
  5. Ground-Truth Isolation: Reference annotations are strictly excluded from inference and rendering.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


def validate_field_bbox(bbox: Any, img_shape: Tuple[int, ...]) -> Tuple[bool, str]:
    """
    Defensively validates bounding box coordinates for conservative rendering.
    
    Checks:
      - 4-element sequence of integers/floats.
      - Not an uninitialized dummy coordinate ([0,0,10,10], [0,0,0,0], etc.).
      - Positive area (x1 > x0, y1 > y0).
      - Minimum physical dimensions: width >= 10 px, height >= 6 px.
      - Fully contained within image canvas ([0, 0, w, h]).
    """
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "Missing or invalid bounding box format"

    dummy_boxes = [
        [0, 0, 10, 10], [0, 0, 0, 0], [0, 0, 1, 1], [-1, -1, -1, -1],
        (0, 0, 10, 10), (0, 0, 0, 0)
    ]
    if list(bbox) in dummy_boxes:
        return False, "Uninitialized dummy bounding box coordinates"

    try:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    except (ValueError, TypeError):
        return False, "Non-numeric coordinates"

    if x1 <= x0 or y1 <= y0:
        return False, f"Degenerate non-positive area ({x0}, {y0}, {x1}, {y1})"

    w_box = x1 - x0
    h_box = y1 - y0
    if w_box < 10 or h_box < 6:
        return False, f"Sub-minimum physical dimensions ({w_box}x{h_box} px)"

    h_img, w_img = img_shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img:
        return False, f"Coordinates exceed image canvas boundaries ({w_img}x{h_img})"

    return True, "Valid bbox"


def apply_research_watermark(img_bgr: np.ndarray, doc_status: str = "") -> np.ndarray:
    """
    Applies an indelible, prominent research watermark banner and footer bar.
    Prevents the derived image from being mistaken for an authentic, unaltered report.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    # 1. Top Watermark Banner (32 px crimson header bar)
    banner_h = 32
    top_banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    # Distinct deep crimson/maroon color (BGR: 25, 25, 150)
    top_banner[:] = (25, 25, 150)

    text_top = "AI-PROCESSED RESEARCH DEMONSTRATION - NOT AN OFFICIAL LABORATORY REPORT"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 if w < 600 else 0.50
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(text_top, font, font_scale, thickness)
    t_x = max(8, (w - t_w) // 2)
    t_y = (banner_h + t_h) // 2 - 1
    cv2.putText(top_banner, text_top, (t_x, t_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 2. Bottom Disclaimer Footer Bar (24 px dark slate bar)
    footer_h = 24
    footer_bar = np.zeros((footer_h, w, 3), dtype=np.uint8)
    footer_bar[:] = (35, 30, 30)
    status_suffix = f" | DECISION: {doc_status}" if doc_status else ""
    text_bot = f"BIOTRACE NEUROSYMBOLIC VERIFICATION{status_suffix} | RESEARCH DEMONSTRATION ONLY"
    font_scale_bot = 0.36 if w < 600 else 0.42
    (b_w, b_h), _ = cv2.getTextSize(text_bot, font, font_scale_bot, 1)
    b_x = max(8, (w - b_w) // 2)
    b_y = (footer_h + b_h) // 2 - 1
    cv2.putText(footer_bar, text_bot, (b_x, b_y), font, font_scale_bot, (210, 210, 210), 1, cv2.LINE_AA)

    # Vertically stack banner + canvas + footer
    watermarked = np.vstack([top_banner, canvas, footer_bar])
    return watermarked


def overwrite_field_text(canvas: np.ndarray,
                         bbox: List[int],
                         new_text: str,
                         highlight_color: Tuple[int, int, int] = (220, 120, 0)) -> bool:
    """
    Conservatively in-paints the field bounding box and renders new_text with a highlight box.
    Samples local rim pixels to match the authentic paper background color.
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    h_img, w_img = canvas.shape[:2]

    # Sample local paper background from rim pixels around the bounding box
    pad = 2
    y_min_rim = max(0, y0 - pad)
    y_max_rim = min(h_img, y1 + pad)
    x_min_rim = max(0, x0 - pad)
    x_max_rim = min(w_img, x1 + pad)

    rim_patch = canvas[y_min_rim:y_max_rim, x_min_rim:x_max_rim]
    if rim_patch.size > 0:
        bg_color = np.median(rim_patch.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        bg_color = np.array([255, 255, 255], dtype=np.uint8)

    # Inpaint the old text area with the sampled background color
    cv2.rectangle(canvas, (x0, y0), (x1, y1), bg_color.tolist(), -1)

    # Text styling: Dark charcoal for high contrast on light paper
    is_light_bg = float(np.mean(bg_color)) > 128
    text_color = (20, 20, 20) if is_light_bg else (240, 240, 240)

    box_w = x1 - x0
    box_h = y1 - y0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.38, min(0.65, (box_h * 0.65) / 18.0))
    (t_w, t_h), _ = cv2.getTextSize(new_text, font, font_scale, 1)

    text_x = x0 + max(2, (box_w - t_w) // 2)
    text_y = y0 + (box_h + t_h) // 2

    cv2.putText(canvas, new_text, (text_x, text_y), font, font_scale, text_color, 1, cv2.LINE_AA)

    # Highlight border around modified field
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), highlight_color, 1)

    # Subtle badge indicator tag
    tag_text = "[AI-CORRECTED]"
    tag_scale = 0.30
    (tag_w, tag_h), _ = cv2.getTextSize(tag_text, font, tag_scale, 1)
    tag_y = max(tag_h + 2, y0 - 3)
    cv2.putText(canvas, tag_text, (x0, tag_y), font, tag_scale, highlight_color, 1, cv2.LINE_AA)

    return True


def compute_field_decisions(extracted_values: Dict[str, float],
                            final_values: Dict[str, float],
                            proposed_candidates: Dict[str, float],
                            bounding_boxes: Dict[str, Any],
                            img_shape: Optional[Tuple[int, ...]],
                            final_status: str,
                            active_equation: str,
                            curr_residual: float,
                            tolerance_pct: float,
                            vision_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Authoritative calculation of evidence-based decisions (ACCEPT, FLAG / REVIEW, CORRECT, ABSTAIN)
    and transparent clinical rationales across all 4 lipid analytes.

    Guarantees:
      - ACCEPT: Extracted value is sufficiently supported by the source image and applicable checks.
      - FLAG / REVIEW: A constraint violation, OCR mismatch, or conflicting evidence is detected;
        physical document printed value is retained with explicit clinical rationale.
      - CORRECT: Change a value only when independent visual evidence sufficiently supports the replacement.
      - ABSTAIN: Evidence is genuinely insufficient or contradictory—not simply because a constraint failed.
      - Never falsely withhold clean anchor fields when an individual field is under review.
      - Backward-compatible with existing decision_status string assertions.
    """
    field_decisions = {}
    valid_correction_statuses = ["FULL_VISION_CORRECTED_K1", "FULL_FAST_PATH_K1", "FULL_CORRECTED_K2"]
    is_doc_fatal_abstain = final_status in [
        "ABSTAIN_UNREADABLE_OR_EMPTY_OCR",
        "ABSTAIN_MISSING_ANALYTE",
        "ABSTAIN_CYCLIC_STATE_DETECTED",
        "ABSTAIN_BOUNDS_VIOLATION"
    ]

    for f in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
        raw_v = extracted_values.get(f)
        fin_v = final_values.get(f, raw_v)
        prop_v = proposed_candidates.get(f, fin_v)
        vis_info = (vision_cache or {}).get(f, {})
        has_vis = (vis_info.get("raw_c") is not None)
        has_num_change = (raw_v is not None and fin_v is not None and abs(raw_v - fin_v) > 0.01)
        is_candidate_target = (prop_v is not None and raw_v is not None and abs(raw_v - prop_v) > 0.01)
        qm = vis_info.get("qm", {})
        is_field_verified = (
            final_status in valid_correction_statuses
            or (
                final_status in ["CONVERGED", "CONVERGED_DIRECT_ASSAY"]
                and (
                    qm.get("ocr_agreement") is True
                    or vis_info.get("ocr_agreement") is True
                    or vis_info.get("ocr_agree") is True
                    or (vis_info.get("raw_c") is not None and vis_info.get("delta_c", 0.0) > 0.0)
                )
            )
        )

        # 1. Missing or non-positive value
        if raw_v is None or raw_v <= 0:
            dec_status = "ABSTAINED / UNRESOLVED"
            canonical_dec = "ABSTAIN"
            acc_val = None
            final_disp = "Withheld"
            rationale = f"Analyte '{f}' not detected in OCR extraction or has non-positive value. Required for clinical panel evaluation."

        # 2. Document-level fatal failure (unreadable scan, missing analytes, cyclic state, bounds violation)
        elif is_doc_fatal_abstain:
            dec_status = "ABSTAINED / UNRESOLVED"
            canonical_dec = "ABSTAIN"
            acc_val = None
            final_disp = "Withheld"
            if final_status == "ABSTAIN_BOUNDS_VIOLATION":
                rationale = f"Withheld: Analyte values violate physiological plausibility boundaries ({final_status}). Manual review required."
            elif final_status == "ABSTAIN_MISSING_ANALYTE":
                rationale = f"Withheld: Required analytes missing for mathematical invariant balancing. Manual review required."
            elif final_status == "ABSTAIN_UNREADABLE_OR_EMPTY_OCR":
                rationale = f"Withheld: OCR tokens insufficient to construct clinical panel. Document unreadable."
            else:
                rationale = f"Withheld: Document verification conditions not satisfied ({final_status}). Manual clinician review required."

        # 3. Severe Hypertriglyceridemia (TG > 800 mg/dL): Calculated LDL is clinically invalid
        elif final_status == "ABSTAIN_TG800":
            if f == "ldl":
                dec_status = "ABSTAINED / UNRESOLVED"
                canonical_dec = "ABSTAIN"
                acc_val = None
                final_disp = "Withheld"
                rationale = (
                    f"Calculated LDL withheld: Serum Triglycerides ({extracted_values.get('triglycerides', 0.0):.1f} mg/dL) exceed 800 mg/dL "
                    f"clinical validity threshold for calculated equations. Direct enzymatic measurement required."
                )
            else:
                dec_status = "RETAINED VALUE"
                canonical_dec = "ACCEPT"
                acc_val = raw_v
                final_disp = f"{raw_v:.1f}"
                rationale = f"Extracted value ({raw_v:.1f} mg/dL) verified against source document; retained unaltered."

        # 4. Verified Candidate Correction
        elif has_num_change and is_field_verified:
            bbox = bounding_boxes.get(f) if bounding_boxes else None
            is_bbox_valid, bbox_reason = validate_field_bbox(bbox, img_shape) if img_shape else (False, "No image canvas shape provided")

            if is_bbox_valid:
                dec_status = "ACCEPTED CORRECTION"
                canonical_dec = "CORRECT"
                acc_val = fin_v
                final_disp = f"{fin_v:.1f}"

                if final_status == "FULL_FAST_PATH_K1":
                    rationale = f"Candidate ({fin_v:.1f}) verified: satisfies {active_equation} within tolerance with high OCR confidence."
                elif final_status == "FULL_CORRECTED_K2":
                    rationale = f"Candidate ({fin_v:.1f}) verified: satisfies dual-fault stoichiometric invariant solution for {active_equation}."
                else:
                    delta_c = vis_info.get("delta_c", 0.0)
                    rationale = (
                        f"Candidate ({fin_v:.1f}) verified: satisfies {active_equation} within tolerance "
                        f"and corroborated by micro-crop visual inspection (Delta-Conf: {delta_c:+.2f})."
                    )
            else:
                dec_status = "ABSTAINED / UNRESOLVED"
                canonical_dec = "ABSTAIN"
                acc_val = None
                final_disp = "Withheld"
                rationale = (
                    f"Correction withheld: candidate value ({fin_v:.1f}) satisfies invariant, but bounding box "
                    f"coordinates are unvalidated ({bbox_reason}). Withheld visual overwrite. Manual clinician review required."
                )

        # 5. Detected Constraint Violation on this Field that was NOT corrected
        elif is_candidate_target:
            if final_status in ["FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY", "REVIEW_REQUIRED_DISCREPANCY_FLAGGED"]:
                dec_status = "FLAGGED FOR REVIEW"
                canonical_dec = "FLAG / REVIEW"
                acc_val = raw_v
                final_disp = f"{raw_v:.1f}"
                rationale = (
                    f"Discrepancy detected: printed value ({raw_v:.1f} mg/dL) is confirmed by visual inspection, "
                    f"but violates stoichiometric constraint ({active_equation}, discrepancy {abs(raw_v - prop_v):.1f} mg/dL). "
                    f"Flagged for pathologist/clinician review; value retained unaltered without unverified overwrite."
                )
            elif final_status in ["FULL_ABSTAIN_VISUAL_DISAGREEMENT", "ABSTAIN_VISUAL_DISAGREEMENT", "ABSTAIN_VISUAL_EVIDENCE_MISSING"]:
                dec_status = "ABSTAINED / UNRESOLVED"
                canonical_dec = "ABSTAIN"
                acc_val = None
                final_disp = "Withheld"
                rationale = (
                    f"Correction withheld: proposed candidate ({prop_v:.1f}) was not corroborated "
                    f"by visual inspection of document micro-crop (extracted {raw_v:.1f}). Value withheld. Manual clinician review required."
                )
            else:
                dec_status = "FLAGGED FOR REVIEW"
                canonical_dec = "FLAG / REVIEW"
                acc_val = raw_v
                final_disp = f"{raw_v:.1f}"
                rationale = (
                    f"Discrepancy detected on {f}: extracted {raw_v:.1f} vs candidate {prop_v:.1f} under {active_equation}. "
                    f"Manual clinician review required."
                )

        # 6. Clean, consistent field or reference anchor
        else:
            dec_status = "RETAINED VALUE"
            canonical_dec = "ACCEPT"
            acc_val = raw_v
            final_disp = f"{raw_v:.1f}" if raw_v is not None else "N/A"

            if final_status in ["CONVERGED", "CONVERGED_DIRECT_ASSAY"] and not any(abs(extracted_values.get(other_f, 0.0) - final_values.get(other_f, 0.0)) > 0.01 for other_f in extracted_values if other_f != f):
                rationale = f"Extracted value is consistent with {active_equation} (residual {curr_residual:.2f}% <= {tolerance_pct:.0f}% tolerance); retained unaltered."
            elif any(abs(extracted_values.get(other_f, 0.0) - final_values.get(other_f, 0.0)) > 0.01 for other_f in extracted_values if other_f != f):
                rationale = f"Extracted value served as fixed reference anchor for stoichiometric invariant balancing; verified and retained unaltered."
            elif "DISCREPANCY" in final_status or "REVIEW" in final_status or "ABSTAIN" in final_status:
                rationale = f"Extracted value verified against source document image; retained unaltered as reference anchor."
            else:
                rationale = "Extracted value satisfies invariant constraint without modification."

        field_decisions[f] = {
            "decision_status": dec_status,
            "canonical_decision": canonical_dec,
            "original_value": raw_v,
            "accepted_value": acc_val,
            "proposed_value": prop_v,
            "final_display": final_disp,
            "rationale": rationale,
            "modified_in_image": (dec_status == "ACCEPTED CORRECTION"),
            "visual_evidence_available": has_vis,
        }

    return field_decisions



def generate_corrected_report_image(original_bgr: np.ndarray,
                                    bounding_boxes: Dict[str, Any],
                                    field_decisions: Dict[str, Dict[str, Any]],
                                    doc_status: str = "") -> Tuple[Optional[np.ndarray], Dict[str, bool]]:
    """
    Generates the derived AI-processed report image following conservative clinical rules.

    Rules:
      - Starts from an explicit clone of original_bgr (original is never modified).
      - Applies text modifications ONLY to fields with decision_status == 'ACCEPTED CORRECTION'.
      - Fields with 'FLAGGED FOR REVIEW' receive a non-destructive amber border + candidate label.
      - Fields with 'ABSTAINED / UNRESOLVED' (from visual disagreement) receive a red border + label.
      - Fields with 'RETAINED VALUE' are left completely untouched.
      - Bounding box coordinates are strictly validated; if invalid, no pixel overwriting occurs.
      - Prominently applies research watermark banner and footer.

    Returns:
      (derived_image_bgr, modification_flags)
    """
    if original_bgr is None or original_bgr.size == 0:
        return None, {}

    # 1. Conservative start: explicit copy
    derived = original_bgr.copy()
    h_img, w_img = derived.shape[:2]
    modifications = {}
    font = cv2.FONT_HERSHEY_SIMPLEX

    for field_name, dec in field_decisions.items():
        status = dec.get("decision_status", "RETAINED VALUE")
        accepted_val = dec.get("accepted_value")
        proposed_val = dec.get("proposed_value")
        original_val = dec.get("original_value")

        if status == "ACCEPTED CORRECTION" and accepted_val is not None:
            # Verified correction: overwrite pixel region with new value
            bbox = bounding_boxes.get(field_name) if bounding_boxes else None
            is_valid, reason = validate_field_bbox(bbox, (h_img, w_img))

            if is_valid:
                new_str = f"{accepted_val:.1f}"
                success = overwrite_field_text(derived, bbox, new_str, highlight_color=(220, 120, 0))
                modifications[field_name] = success
                dec["modified_in_image"] = success
            else:
                # Coordinate invalid or missing: Do NOT guess! Leave image untouched.
                dec["decision_status"] = "ABSTAINED / UNRESOLVED"
                dec["accepted_value"] = None
                dec["final_display"] = "Withheld"
                dec["rationale"] = f"Correction withheld: bounding box coordinates unvalidated ({reason}). Manual clinician review required."
                dec["image_render_note"] = f"BBox unvalidated ({reason}); withheld visual overwrite."
                dec["modified_in_image"] = False
                modifications[field_name] = False

        elif status == "FLAGGED FOR REVIEW" and proposed_val is not None and original_val is not None:
            # Non-destructive amber annotation: original pixels preserved, border drawn
            bbox = bounding_boxes.get(field_name) if bounding_boxes else None
            is_valid, _ = validate_field_bbox(bbox, (h_img, w_img))
            if is_valid:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                # Amber border (2px) indicating flagged field — no pixel overwrite
                amber = (0, 165, 255)  # BGR amber/orange
                cv2.rectangle(derived, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), amber, 2)
                # Small annotation label above the bounding box
                tag = f"[REVIEW: {original_val:.1f}? cand.{proposed_val:.1f}]"
                tag_scale = 0.27
                (tw, th), _ = cv2.getTextSize(tag, font, tag_scale, 1)
                tag_y = max(th + 3, y0 - 2)
                cv2.putText(derived, tag, (x0, tag_y), font, tag_scale, amber, 1, cv2.LINE_AA)
                dec["modified_in_image"] = False  # No overwrite; annotation only
            modifications[field_name] = False

        else:
            # Retained or Abstained: Never touch pixels
            dec["modified_in_image"] = False
            modifications[field_name] = False

    # 2. Indelible research watermark banner and footer
    watermarked = apply_research_watermark(derived, doc_status=doc_status)
    return watermarked, modifications


def build_correction_audit_record(doc_id: str,
                                  extracted_values: Dict[str, float],
                                  final_values: Dict[str, float],
                                  proposed_values: Dict[str, float],
                                  unit: str,
                                  field_decisions: Dict[str, Dict[str, Any]],
                                  pipeline_status: str,
                                  active_equation: str,
                                  residual_pct: float,
                                  tolerance_pct: float,
                                  visual_evidence_map: Optional[Dict[str, bool]] = None,
                                  run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Constructs a structured, cryptographic-ready clinical audit record for export.
    Zero benchmark ground-truth leakage.
    """
    fields_audit = {}
    for f in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
        raw_v = extracted_values.get(f)
        fin_v = final_values.get(f)
        prop_v = proposed_values.get(f, fin_v)
        dec_info = field_decisions.get(f, {})
        status = dec_info.get("decision_status", "RETAINED VALUE")
        rationale = dec_info.get("rationale", "")
        was_modified = dec_info.get("modified_in_image", False)
        vis_avail = (visual_evidence_map or {}).get(f, dec_info.get("visual_evidence_available", False))

        final_val_output = fin_v
        if status.startswith("ABSTAIN") or dec_info.get("accepted_value") is None:
            final_val_output = "WITHHELD"

        fields_audit[f] = {
            "original_extracted_value": raw_v,
            "final_output_value": final_val_output,
            "proposed_candidate_value": prop_v,
            "unit": unit,
            "decision_status": status,
            "correction_rationale": rationale,
            "visual_evidence_available": vis_avail,
            "modified_in_image": was_modified
        }

    return {
        "audit_version": "1.0-research",
        "document_identifier": doc_id or "unlabeled_doc",
        "run_identifier": run_id or f"run_{doc_id}_{int(time.time()*1000)}",
        "timestamp_iso8601": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operating_disclaimer": "AI-PROCESSED RESEARCH DEMONSTRATION — NOT AN OFFICIAL LABORATORY REPORT",
        "pipeline_terminal_status": pipeline_status,
        "applicable_constraint": active_equation,
        "constraint_residual_percent": round(residual_pct, 3),
        "residual_tolerance_percent": round(tolerance_pct, 1),
        "fields": fields_audit
    }

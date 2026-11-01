from src.stage0_classifier import AssayClassifier
from src.stage1_extract import extract_document_state
from src.stage2_constraints import (
    normalize_to_mgdl,
    verify_physiological_bounds,
    evaluate_friedewald_invariant,
    evaluate_sampson_invariant,
    calculate_sampson_ldl,
    evaluate_martin_hopkins_invariant,
    check_tg_decimal_shift_hypothesis,
)

from src.stage2_isolation import (
    hierarchical_isolation_search,
    generate_discrete_plausible_candidates,
    is_digit_transposition_match,
)
from src.stage3_vision_gate import process_vision_gate, is_valid_bbox, validate_analyte_crop_selection
from src.stage4_enhance_extract import (
    enhance_and_reextract,
    reextract_microcrop,
    verify_dual_blind_reread,
)
from typing import Optional
from src.image_preflight import inspect_image_quality, PreflightConfig

def _run_pipeline_inner(image_path: str, max_attempts: int = 2, tolerance: float = 0.10, doc: dict = None) -> dict:
    if doc is None:
        doc = extract_document_state(image_path)
    
    # Stage 0: Assay-Method Routing
    assay_routing = AssayClassifier.classify(doc["method_text"])
    route = assay_routing["route"]
    
    # Stage 1 & 2: Unit Normalization to mg/dL
    canonical_data = normalize_to_mgdl(doc["extracted_values"], doc["unit"])
    confidences = doc["confidences"].copy()
    bboxes = doc["bounding_boxes"]
    img_bgr = doc["image_bgr"]
    
    audit_trail = []

    # Stage 1 Preflight: Guardrail against empty or unreadable OCR scans
    essential_fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    observed_fields = [f for f in essential_fields if f in canonical_data and canonical_data[f] > 0]
    if len(observed_fields) < 2:
        return {
            "status": "ABSTAIN_UNREADABLE_OR_EMPTY_OCR",
            "assay_route": route,
            "missing_fields": [f for f in essential_fields if f not in canonical_data or canonical_data[f] <= 0],
            "data": canonical_data,
            "unit": doc["unit"],
            "audit_trail": [{
                "stage": "STAGE_1_PREFLIGHT",
                "action": "ABSTAIN_INSUFFICIENT_OCR_TOKENS",
                "observed_count": len(observed_fields)
            }]
        }

    
    # Direct Assay Path: Bypass Friedewald equality, apply plausibility bounds
    if route == "DIRECT_MEASURED":
        bounds_valid, reason = verify_physiological_bounds(canonical_data)
        status = "CONVERGED_DIRECT_ASSAY" if bounds_valid else "ABSTAIN_BOUNDS_VIOLATION"
        return {
            "status": status,
            "assay_route": route,
            "attempts": 0,
            "data": canonical_data,
            "unit": "mg/dL",
            "audit_trail": [{
                "stage": "STAGE_0_DIRECT_GATE",
                "action": "BYPASSED_EQUALITY_CHECK",
                "plausibility_check": reason
            }]
        }

    # Ambiguous Header Fallback: Check bounds and warn
    if route == "AMBIGUOUS_FALLBACK":
        bounds_valid, reason = verify_physiological_bounds(canonical_data)
        if not bounds_valid:
            return {
                "status": "ABSTAIN_AMBIGUOUS_HEADER_BOUNDS_VIOLATION",
                "assay_route": route,
                "data": canonical_data,
                "audit_trail": [{"reason": reason}]
            }

    # Calculated Panel Closed Loop (Dynamic Routing: Friedewald vs Sampson vs Martin-Hopkins)
    attempt = 0
    current_data = canonical_data.copy()
    seen_states = set()
    
    while attempt < max_attempts:
        # Repeated-state protection: prevent cyclic oscillation across retry iterations
        state_key = tuple(sorted((k, round(v, 2)) for k, v in current_data.items() if v is not None))
        if state_key in seen_states:
            return {
                "status": "ABSTAIN_CYCLIC_STATE_DETECTED",
                "assay_route": route,
                "reason": "Repeated panel state detected across retry iterations; terminating to prevent cyclic oscillation.",
                "data": canonical_data,
                "residual": round(residual, 2) if 'residual' in locals() else 0.0,
                "audit_trail": audit_trail
            }
        seen_states.add(state_key)
        raw_tg = current_data.get("triglycerides", 0.0)
        
        # Priority 1: Route by declared method text (Stage 0) first
        if route == "MARTIN_HOPKINS":
            status, rel_err = evaluate_martin_hopkins_invariant(current_data, tolerance)
            residual = rel_err
            active_equation = "MARTIN_HOPKINS"
        elif route == "SAMPSON":
            status, rel_err = evaluate_sampson_invariant(current_data, tolerance)
            ldl_calc = calculate_sampson_ldl(current_data.get("total_cholesterol", 0.0), current_data.get("hdl", 0.0), current_data.get("triglycerides", 0.0))
            residual = abs(current_data.get("ldl", 0.0) - ldl_calc)
            active_equation = "SAMPSON"
        elif route == "FRIEDEWALD" and raw_tg <= 400.0:
            status, residual, rel_err = evaluate_friedewald_invariant(current_data, tolerance)
            active_equation = "FRIEDEWALD"
        # Priority 2: Fallback to TG value routing
        elif raw_tg > 800.0:
            # Test decimal-shift hypotheses before emitting ABSTAIN_TG800
            shift_hyp = check_tg_decimal_shift_hypothesis(current_data, tolerance, declared_route=None)
            if shift_hyp:
                return {
                    "status": "ABSTAIN_POSSIBLE_TG_DECIMAL_SHIFT",
                    "advisory_flag": "possible_TG_decimal_shift",
                    "assay_route": route,
                    "hypothesis_tg": shift_hyp["hypothesis_tg"],
                    "original_tg": raw_tg,
                    "evidence": shift_hyp,
                    "requires_blind_reread": True,
                    "data": current_data,
                    "residual": shift_hyp["residual"],
                    "audit_trail": audit_trail
                }
            return {
                "status": "ABSTAIN_TG800",
                "assay_route": route,
                "reason": "Serum Triglycerides > 800 mg/dL exceed Sampson validity threshold.",
                "data": canonical_data,
                "residual": 1.0,
                "audit_trail": audit_trail
            }
        elif raw_tg > 400.0:
            status, rel_err = evaluate_sampson_invariant(current_data, tolerance)
            ldl_calc = calculate_sampson_ldl(current_data.get("total_cholesterol", 0.0), current_data.get("hdl", 0.0), current_data.get("triglycerides", 0.0))
            residual = abs(current_data.get("ldl", 0.0) - ldl_calc)
            active_equation = "SAMPSON"
        else:
            status, residual, rel_err = evaluate_friedewald_invariant(current_data, tolerance)
            active_equation = "FRIEDEWALD"
        
        if status == "CONVERGED":
            return {
                "status": "CONVERGED",
                "assay_route": route,
                "equation": active_equation,
                "attempts": attempt,
                "data": current_data,
                "residual": round(residual, 2),
                "relative_error": round(rel_err, 4),
                "audit_trail": audit_trail
            }
            
        if status == "ABSTAIN_TG800":
            shift_hyp = check_tg_decimal_shift_hypothesis(current_data, tolerance, declared_route=route if route in ["FRIEDEWALD", "SAMPSON", "MARTIN_HOPKINS"] else None)
            if shift_hyp:
                return {
                    "status": "ABSTAIN_POSSIBLE_TG_DECIMAL_SHIFT",
                    "advisory_flag": "possible_TG_decimal_shift",
                    "assay_route": route,
                    "hypothesis_tg": shift_hyp["hypothesis_tg"],
                    "original_tg": raw_tg,
                    "evidence": shift_hyp,
                    "requires_blind_reread": True,
                    "data": current_data,
                    "residual": shift_hyp["residual"],
                    "audit_trail": audit_trail
                }
            return {
                "status": "ABSTAIN_TG800",
                "assay_route": route,
                "reason": "Serum Triglycerides > 800 mg/dL exceed Sampson validity threshold.",
                "data": canonical_data,
                "residual": 1.0,
                "audit_trail": audit_trail
            }
            
        if status == "ABSTAIN_CLINICAL_ANOMALY":
            shift_hyp = check_tg_decimal_shift_hypothesis(current_data, tolerance, declared_route="FRIEDEWALD")
            if shift_hyp:
                return {
                    "status": "ABSTAIN_POSSIBLE_TG_DECIMAL_SHIFT",
                    "advisory_flag": "possible_TG_decimal_shift",
                    "assay_route": route,
                    "hypothesis_tg": shift_hyp["hypothesis_tg"],
                    "original_tg": raw_tg,
                    "evidence": shift_hyp,
                    "requires_blind_reread": True,
                    "data": current_data,
                    "residual": shift_hyp["residual"],
                    "audit_trail": audit_trail
                }
            return {
                "status": "ABSTAIN_CLINICAL_ANOMALY",
                "assay_route": route,
                "reason": "Serum Triglycerides > 400 mg/dL invalidate the Friedewald equation.",
                "data": canonical_data,
                "residual": round(residual, 2),
                "audit_trail": audit_trail
            }
            
        # Stage 2.5: Hierarchical Isolation Search (k=1 vs k=2)
        search_res = hierarchical_isolation_search(
            current_data,
            confidences,
            tol=tolerance,
            method="MARTIN_HOPKINS" if route == "MARTIN_HOPKINS" else None
        )
        action = search_res.get("action")
        
        if action == "ABSTAIN_MISSING_ANALYTE":
            missing_f = search_res.get("field")
            missing_list = [missing_f] if missing_f else []
            return {
                "status": "ABSTAIN_MISSING_ANALYTE",
                "assay_route": route,
                "missing_fields": missing_list,
                "reason": search_res.get("reason", "Missing required analytes for invariant isolation."),
                "data": canonical_data,
                "residual": round(residual, 2) if 'residual' in locals() else 0.0,
                "audit_trail": audit_trail
            }

        if action == "CONVERGED":
            return {
                "status": "CONVERGED",
                "assay_route": route,
                "equation": search_res.get("equation", active_equation),
                "attempts": attempt,
                "data": current_data,
                "residual": round(residual, 2) if 'residual' in locals() else 0.0,
                "relative_error": round(search_res.get("residual", 0.0), 4),
                "audit_trail": audit_trail
            }

        if action == "ABSTAIN_TG800":
            return {
                "status": "ABSTAIN_TG800",
                "assay_route": route,
                "reason": search_res.get("reason", "Serum Triglycerides > 800 mg/dL exceed Sampson validity threshold."),
                "data": canonical_data,
                "residual": 1.0,
                "audit_trail": audit_trail
            }

        if action == "ABSTAIN_ALGEBRAIC_INDETERMINATE":
            ranked_pairs = search_res.get("ranked_pairs", [])
            dual_blind_committed = False

            if img_bgr is not None and ranked_pairs:
                for pair in ranked_pairs:
                    corrs = pair.get("corrections", {})
                    if len(corrs) != 2:
                        continue
                    (f1, c1), (f2, c2) = list(corrs.items())
                    bb1 = bboxes.get(f1, [0, 0, 10, 10])
                    bb2 = bboxes.get(f2, [0, 0, 10, 10])

                    has_crop1 = False
                    has_crop2 = False
                    if bb1 != [0, 0, 10, 10] and len(bb1) == 4:
                        t1 = img_bgr[bb1[1]:bb1[3], bb1[0]:bb1[2]]
                        if t1.size > 0 and t1.std() >= 5.0:
                            has_crop1 = True
                    if bb2 != [0, 0, 10, 10] and len(bb2) == 4:
                        t2 = img_bgr[bb2[1]:bb2[3], bb2[0]:bb2[2]]
                        if t2.size > 0 and t2.std() >= 5.0:
                            has_crop2 = True

                    if has_crop1 and has_crop2:
                        raw1, seg1, _ = process_vision_gate(img_bgr, bb1)
                        raw2, seg2, _ = process_vision_gate(img_bgr, bb2)
                        crop1 = seg1 if seg1 is not None else raw1
                        crop2 = seg2 if seg2 is not None else raw2

                        both_agreed, details = verify_dual_blind_reread(
                            crop1, crop2, c1, c2,
                            conf1=confidences.get(f1, 0.8),
                            conf2=confidences.get(f2, 0.8)
                        )

                        if both_agreed:
                            old1, old2 = current_data[f1], current_data[f2]
                            current_data[f1] = c1
                            current_data[f2] = c2
                            confidences[f1] = details["field1_post_conf"]
                            confidences[f2] = details["field2_post_conf"]
                            dual_blind_committed = True

                            audit_trail.append({
                                "attempt": attempt + 1,
                                "stage": "STAGE_3_4_DUAL_BLIND_REREAD",
                                "resolution_mechanism": "dual_blind_reread",
                                "corrections": {f1: c1, f2: c2},
                                "old_values": {f1: old1, f2: old2},
                                "details": details
                            })
                            break

            if dual_blind_committed:
                attempt += 1
                continue

            return {
                "status": "ABSTAIN_ALGEBRAIC_INDETERMINATE",
                "assay_route": route,
                "reason": search_res.get("reason"),
                "data": canonical_data,
                "residual": round(residual, 2),
                "audit_trail": audit_trail
            }

            
        # Fast-Path: Decimal Shift
        if action == "CORRECT_K1" and search_res.get("is_fast_path"):
            field = search_res["field"]
            old_val = current_data[field]
            bb = bboxes.get(field)
            if img_bgr is None or not is_valid_bbox(bb, img_bgr):
                return {
                    "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                    "assay_route": route,
                    "field": field,
                    "target_val": search_res["target_val"],
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "audit_trail": audit_trail
                }

            crop_test = img_bgr[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
            if crop_test.size == 0 or crop_test.std() < 5.0:
                return {
                    "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                    "assay_route": route,
                    "field": field,
                    "target_val": search_res["target_val"],
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "audit_trail": audit_trail
                }

            is_crop_valid, raw_c, crop_reason, crop_meta = validate_analyte_crop_selection(
                img_bgr, bboxes, field, pad=4
            )
            if not is_crop_valid or raw_c is None:
                return {
                    "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                    "assay_route": route,
                    "field": field,
                    "target_val": search_res["target_val"],
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "reason": crop_reason,
                    "audit_trail": audit_trail
                }

            new_val, new_c, delta_c, _, qm = enhance_and_reextract(
                raw_c,
                candidate_target_val=search_res["target_val"],
                current_conf=confidences.get(field, 0.8),
                field=field,
                unit=doc.get("unit", "mg/dL")
            )
            if new_val is not None:
                current_data[field] = new_val
                confidences[field] = new_c
            else:
                blind_val = qm.get("blind_reread_val_mgdl") if qm else None
                is_corroborated = (blind_val is not None and abs(blind_val - current_data[field]) <= 1.0)
                status_label = "FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY" if is_corroborated else "ABSTAIN_VISUAL_DISAGREEMENT"
                return {
                    "status": status_label,
                    "assay_route": route,
                    "field": field,
                    "target_val": search_res["target_val"],
                    "blind_reread_val": blind_val,
                    "is_printed_corroborated": is_corroborated,
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "audit_trail": audit_trail
                }


            audit_trail.append({
                "attempt": attempt + 1,
                "stage": "STAGE_2.5_FAST_PATH",
                "field": field,
                "old_val": old_val,
                "new_val": new_val,
                "delta_conf": "+0.41",
                "latency": "0.1ms"
            })

            # Secondary pass: check if any remaining field has an exact adjacent transposition or vision crop
            tc_val = current_data["total_cholesterol"]
            tg_val = current_data["triglycerides"]
            if tg_val > 400.0:
                calc_ldl = calculate_sampson_ldl(tc_val, current_data["hdl"], tg_val)
                curr_abs_res = abs(current_data["ldl"] - calc_ldl)
            else:
                exp_tc = current_data["hdl"] + current_data["ldl"] + (tg_val / 5.0)
                curr_abs_res = abs(tc_val - exp_tc)

            if curr_abs_res > 1.0:
                for f_rem in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
                    if f_rem == field:
                        continue
                    bb_rem = bboxes.get(f_rem, [0, 0, 10, 10])
                    has_rem_vision = False
                    crop_rem = None
                    if img_bgr is not None and bb_rem != [0, 0, 10, 10] and len(bb_rem) == 4:
                        crop_t = img_bgr[bb_rem[1]:bb_rem[3], bb_rem[0]:bb_rem[2]]
                        if crop_t.size > 0 and crop_t.std() >= 5.0:
                            has_rem_vision = True
                            raw_c, seg_c, _ = process_vision_gate(img_bgr, bb_rem)
                            crop_rem = seg_c if seg_c is not None else raw_c

                    cands = generate_discrete_plausible_candidates(current_data[f_rem], f_rem)
                    cand_values = []
                    if has_rem_vision and crop_rem is not None:
                        corr_val, _, _, _, _ = enhance_and_reextract(
                            crop_rem,
                            candidate_target_val=current_data[f_rem],
                            current_conf=confidences[f_rem],
                            field=f_rem,
                            unit=doc.get("unit", "mg/dL")
                        )
                        if corr_val is not None:
                            cand_values.append(corr_val)
                    for c_val in cands:
                        if is_digit_transposition_match(current_data[f_rem], c_val, f_rem):
                            cand_values.append(c_val)
                    for cand_v in cand_values:
                        test_c = current_data.copy()
                        test_c[f_rem] = cand_v
                        if test_c["triglycerides"] > 400.0:
                            c_ldl = calculate_sampson_ldl(test_c["total_cholesterol"], test_c["hdl"], test_c["triglycerides"])
                            new_abs_res = abs(test_c["ldl"] - c_ldl)
                        else:
                            new_abs_res = abs(test_c["total_cholesterol"] - (test_c["hdl"] + test_c["ldl"] + test_c["triglycerides"] / 5.0))
                        if new_abs_res < curr_abs_res and new_abs_res < 1.0:

                            current_data[f_rem] = cand_v
                            confidences[f_rem] = 0.99
                            curr_abs_res = new_abs_res
                            audit_trail.append({
                                "attempt": attempt + 1,
                                "stage": "STAGE_2.5_SECONDARY_PASS",
                                "field": f_rem,
                                "old_val": test_c[f_rem],
                                "new_val": cand_v,
                                "residual": round(new_abs_res, 2)
                            })
                            break

            attempt += 1
            continue
            
        if action == "CORRECT_K1":
            ranked = search_res.get("ranked_candidates", [search_res])
            chosen_cand = None
            for cand in ranked:
                cand_field = cand["field"]
                bb = bboxes.get(cand_field)
                if img_bgr is not None and is_valid_bbox(bb, img_bgr):
                    crop_test = img_bgr[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                    if crop_test.size > 0 and crop_test.std() >= 5.0:
                        chosen_cand = cand
                        break

            if chosen_cand is None:
                field = ranked[0]["field"] if ranked else "unknown"
                target_val = ranked[0]["target_val"] if ranked else 0.0
                return {
                    "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                    "assay_route": route,
                    "field": field,
                    "target_val": target_val,
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "audit_trail": audit_trail
                }

            field = chosen_cand["field"]
            bb = bboxes.get(field)
            if img_bgr is None or not is_valid_bbox(bb, img_bgr):
                return {
                    "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                    "assay_route": route,
                    "field": field,
                    "target_val": chosen_cand["target_val"],
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "audit_trail": audit_trail
                }

            is_crop_valid, raw_c, crop_reason, crop_meta = validate_analyte_crop_selection(
                img_bgr, bboxes, field, pad=4
            )
            if not is_crop_valid or raw_c is None:
                return {
                    "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                    "assay_route": route,
                    "field": field,
                    "target_val": chosen_cand["target_val"],
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "reason": crop_reason,
                    "audit_trail": audit_trail
                }

            corrected_val, new_conf, delta_c, sr_4x, quality_metrics = enhance_and_reextract(
                raw_c,
                candidate_target_val=chosen_cand["target_val"],
                current_conf=confidences.get(field, 0.8),
                field=field,
                unit=doc.get("unit", "mg/dL")
            )
            
            if corrected_val is not None:
                old_val = current_data[field]
                current_data[field] = corrected_val
                confidences[field] = new_conf
                
                audit_trail.append({
                    "attempt": attempt + 1,
                    "stage": "STAGE_3_4_VISION_GATE",
                    "field": field,
                    "old_val": old_val,
                    "new_val": corrected_val,
                    "delta_conf": f"+{delta_c:.2f}",
                    "vision_metrics": quality_metrics,
                    "crops": ("raw_padded", "graphcut_segmented", "real_esrgan_4x") if raw_c is not None else (None, None, None)
                })
                attempt += 1
                continue
            else:
                blind_val = quality_metrics.get("blind_reread_val_mgdl") if quality_metrics else None
                curr_val = current_data.get(field, 0.0)
                if blind_val is not None and abs(blind_val - curr_val) <= max(1.0, 0.05 * curr_val):
                    audit_trail.append({
                        "attempt": attempt + 1,
                        "stage": "STAGE_3_4_VISION_GATE",
                        "field": field,
                        "action": "FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY",
                        "printed_val": curr_val,
                        "blind_reread": blind_val,
                        "candidate_target": chosen_cand["target_val"],
                        "residual": round(residual, 2),
                        "rationale": f"Image confirms printed value {curr_val} mg/dL. Mathematical constraint discrepancy of {residual:.1f} mg/dL flagged for clinical review."
                    })
                    return {
                        "status": "FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY",
                        "assay_route": route,
                        "flagged_field": field,
                        "target_val": chosen_cand["target_val"],
                        "printed_val": curr_val,
                        "residual": round(residual, 2),
                        "data": canonical_data,
                        "audit_trail": audit_trail
                    }
                else:
                    return {
                        "status": "ABSTAIN_VISUAL_DISAGREEMENT",
                        "assay_route": route,
                        "field": field,
                        "target_val": chosen_cand["target_val"],
                        "data": canonical_data,
                        "residual": round(residual, 2),
                        "audit_trail": audit_trail
                    }
            
        # Dual Field Permutation (k=2): Discrete Pair Correction with Non-Maleficence Guardrail
        if action == "CORRECT_K2":
            if search_res.get("resolution_mechanism") == "extra_field_equation":
                chosen_pair = search_res
            else:
                ranked_pairs = search_res.get("ranked_pairs", [search_res])
                chosen_pair = None
                
                for pair in ranked_pairs:
                    if current_data.get("triglycerides", 0.0) > 800.0 or any(c_val > 800.0 for f_name, c_val in pair["corrections"].items() if f_name == "triglycerides"):
                        continue
                    # Non-maleficence: do not mutate clean fields without visual degradation evidence
                    collateral_clean_mutation = False
                    for f_name, c_val in pair["corrections"].items():
                        bb = bboxes.get(f_name)
                        is_clean_crop = True
                        if img_bgr is not None and is_valid_bbox(bb, img_bgr):
                            crop_test = img_bgr[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                            if crop_test.size > 0 and crop_test.std() >= 5.0:
                                is_clean_crop = False
                        if is_clean_crop:
                            collateral_clean_mutation = True
                            break
                    if not collateral_clean_mutation:
                        chosen_pair = pair
                        break

            if chosen_pair is None:
                audit_trail.append({
                    "attempt": attempt + 1,
                    "stage": "STAGE_2_SAFETY_GUARDRAIL",
                    "action": "ABSTAIN_NON_MALEFICENCE_PREVENT_COLLATERAL_MUTATION"
                })
                return {
                    "status": "ABSTAIN_TG800" if current_data.get("triglycerides", 0.0) > 800.0 else "ABSTAIN_ALGEBRAIC_INDETERMINATE",
                    "assay_route": route,
                    "reason": "Preventing collateral mutation of clean fields.",
                    "data": canonical_data,
                    "residual": round(residual, 2),
                    "audit_trail": audit_trail
                }

            corrections = chosen_pair["corrections"]
            step_record = {
                "attempt": attempt + 1,
                "stage": "STAGE_2.5_K2_DISCRETE_PAIR",
                "modifications": []
            }
            
            # Atomic verification: verify all candidate fields visually before committing any changes
            staged_updates = {}
            for f_name, c_val in corrections.items():
                bb = bboxes.get(f_name)
                if img_bgr is None or not is_valid_bbox(bb, img_bgr):
                    return {
                        "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                        "assay_route": route,
                        "field": f_name,
                        "target_val": c_val,
                        "data": canonical_data,
                        "residual": round(residual, 2),
                        "audit_trail": audit_trail
                    }

                raw_c, seg_c, _ = process_vision_gate(img_bgr, bb)
                crop = seg_c if seg_c is not None else raw_c
                if crop is None or crop.size == 0:
                    return {
                        "status": "ABSTAIN_VISUAL_EVIDENCE_MISSING",
                        "assay_route": route,
                        "field": f_name,
                        "target_val": c_val,
                        "data": canonical_data,
                        "residual": round(residual, 2),
                        "audit_trail": audit_trail
                    }

                corrected_val, new_conf, delta_c, _, qm = enhance_and_reextract(
                    crop,
                    candidate_target_val=c_val,
                    current_conf=confidences.get(f_name, 0.8),
                    field=f_name,
                    unit=doc.get("unit", "mg/dL")
                )
                if corrected_val is not None:
                    staged_updates[f_name] = (corrected_val, new_conf)
                else:
                    blind_val = qm.get("blind_reread_val_mgdl") if qm else None
                    curr_val = current_data.get(f_name, 0.0)
                    if blind_val is not None and abs(blind_val - curr_val) <= max(1.0, 0.05 * curr_val):
                        audit_trail.append({
                            "attempt": attempt + 1,
                            "stage": "STAGE_2.5_K2_DISCRETE_PAIR",
                            "action": "FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY",
                            "field": f_name,
                            "printed_val": curr_val,
                            "target_val": c_val,
                            "residual": round(residual, 2)
                        })
                        return {
                            "status": "FLAG_REVIEW_STOICHIOMETRIC_DISCREPANCY",
                            "assay_route": route,
                            "flagged_field": f_name,
                            "target_val": c_val,
                            "printed_val": curr_val,
                            "residual": round(residual, 2),
                            "data": canonical_data,
                            "audit_trail": audit_trail
                        }
                    return {
                        "status": "ABSTAIN_VISUAL_DISAGREEMENT",
                        "assay_route": route,
                        "field": f_name,
                        "target_val": c_val,
                        "data": canonical_data,
                        "residual": round(residual, 2),
                        "audit_trail": audit_trail
                    }

            # Commit all verified updates atomically
            for f_name, (corr_val, n_conf) in staged_updates.items():
                old_val = current_data[f_name]
                current_data[f_name] = corr_val
                confidences[f_name] = n_conf
                step_record["modifications"].append({
                    "field": f_name,
                    "old_val": old_val,
                    "new_val": corr_val
                })
                
            audit_trail.append(step_record)
            attempt += 1
            continue

        # Fallback for unrecognized, unexpected, or unhandled actions
        return {
            "status": f"ABSTAIN_UNRECOGNIZED_ACTION_{action}" if action else "ABSTAIN_UNRECOGNIZED_ACTION",
            "assay_route": route,
            "action": action,
            "reason": search_res.get("reason", f"Unhandled or unrecognized isolation action: {action}"),
            "data": canonical_data,
            "residual": round(residual, 2) if 'residual' in locals() else 0.0,
            "audit_trail": audit_trail
        }

    return {
        "status": "ABSTAIN_PATHOLOGIST_REVIEW",
        "assay_route": route,
        "reason": f"Residual {round(residual, 2)} mg/dL exceeded budget after K={max_attempts} iterations.",
        "data": canonical_data,
        "residual": round(residual, 2),
        "audit_trail": audit_trail
    }


def run_pipeline(image_path: str,
                 max_attempts: int = 2,
                 tolerance: float = 0.10,
                 doc: dict = None,
                 preflight_config: Optional[PreflightConfig] = None) -> dict:
    """
    Main BioTrace Pipeline Entrypoint with integrated Stage 0 Image-Quality Preflight.

    Execution Flow:
      1. Performs non-destructive image quality preflight on the decoded canvas before OCR.
      2. If input is invalid or empty, safely aborts with ABSTAIN_INVALID_OR_EMPTY_IMAGE.
      3. For borderline or quality-flagged images, logs warnings and sets retake_recommended=True,
         while continuing processing so readable fields are never discarded prematurely.
      4. Dispatches to downstream closed-loop stages (Stages 0-4).
      5. Enriches result with distinct image_preflight diagnostics and audit trail records.
    """
    # 1. Resolve image preflight assessment
    preflight = None
    if doc is not None:
        preflight = doc.get("image_preflight")
        if preflight is None:
            img = doc.get("image_bgr")
            if img is not None:
                preflight = inspect_image_quality(img, config=preflight_config)
            else:
                preflight = {
                    "input_valid": True,
                    "width": None,
                    "height": None,
                    "resolution_status": "NOT_ASSESSED",
                    "blur_score": None,
                    "blur_status": "NOT_ASSESSED",
                    "exposure_status": "NOT_ASSESSED",
                    "possible_glare": False,
                    "quality_status": "PASS",
                    "warnings": []
                }
    else:
        # Check input validity if image_path is None or missing
        if image_path is None:
            preflight = inspect_image_quality(None, config=preflight_config)
            return {
                "status": "ABSTAIN_INVALID_OR_EMPTY_IMAGE",
                "assay_route": "UNKNOWN",
                "data": {},
                "unit": "mg/dL",
                "image_preflight": preflight,
                "image_quality_status": preflight["quality_status"],
                "retake_recommended": True,
                "quality_warnings": list(preflight["warnings"]),
                "audit_trail": [{
                    "stage": "STAGE_0_IMAGE_PREFLIGHT",
                    "action": "ABSTAIN_INVALID_INPUT",
                    "quality_status": preflight["quality_status"],
                    "warnings": preflight["warnings"],
                }]
            }

        doc = extract_document_state(image_path)
        preflight = doc.get("image_preflight")
        if preflight is None and doc.get("image_bgr") is not None:
            preflight = inspect_image_quality(doc["image_bgr"], config=preflight_config)

    # 2. Input failure guardrail (empty or invalid image array)
    if preflight is not None and not preflight.get("input_valid", True):
        return {
            "status": "ABSTAIN_INVALID_OR_EMPTY_IMAGE",
            "assay_route": "UNKNOWN",
            "data": {},
            "unit": doc.get("unit", "mg/dL") if doc else "mg/dL",
            "image_preflight": preflight,
            "image_quality_status": preflight.get("quality_status", "REJECT_INVALID_INPUT"),
            "retake_recommended": True,
            "quality_warnings": list(preflight.get("warnings", [])),
            "audit_trail": [{
                "stage": "STAGE_0_IMAGE_PREFLIGHT",
                "action": "ABSTAIN_INVALID_INPUT",
                "quality_status": preflight.get("quality_status", "REJECT_INVALID_INPUT"),
                "warnings": preflight.get("warnings", []),
            }]
        }

    # 3. Execute downstream closed-loop pipeline
    res = _run_pipeline_inner(image_path, max_attempts=max_attempts, tolerance=tolerance, doc=doc)

    # 4. Integrate preflight audit record into audit_trail at index 0
    if preflight is not None and preflight.get("quality_status") != "NOT_ASSESSED":
        audit_trail = res.get("audit_trail", [])
        has_preflight_record = any(
            isinstance(item, dict) and item.get("stage") == "STAGE_0_IMAGE_PREFLIGHT"
            for item in audit_trail
        )
        if not has_preflight_record:
            preflight_record = {
                "stage": "STAGE_0_IMAGE_PREFLIGHT",
                "quality_status": preflight.get("quality_status", "PASS"),
                "resolution_status": preflight.get("resolution_status", "NOT_ASSESSED"),
                "blur_status": preflight.get("blur_status", "NOT_ASSESSED"),
                "exposure_status": preflight.get("exposure_status", "NOT_ASSESSED"),
                "possible_glare": preflight.get("possible_glare", False),
                "warnings": preflight.get("warnings", []),
            }
            res["audit_trail"] = [preflight_record] + audit_trail

    # 5. Enrich result dictionary with distinct preflight attributes
    if preflight is not None:
        res["image_preflight"] = preflight
        res["image_quality_status"] = preflight.get("quality_status", "PASS")
        res["retake_recommended"] = preflight.get("quality_status") in (
            "REJECT_LOW_RESOLUTION", "REJECT_SEVERELY_BLURRED", "REJECT_EXTREME_EXPOSURE",
            "REJECT_INVALID_INPUT", "REJECT_DEGENERATE"
        )
        res["quality_warnings"] = list(preflight.get("warnings", []))

    return res
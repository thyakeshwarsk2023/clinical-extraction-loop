import itertools
import Levenshtein
from src.stage2_constraints import calculate_sampson_ldl, evaluate_sampson_invariant

def evaluate_plausibility_bounds(panel: dict) -> tuple:
    tc, hdl, ldl, tg = panel["total_cholesterol"], panel["hdl"], panel["ldl"], panel["triglycerides"]
    if ldl >= tc or hdl >= tc:
        return False, "BOUND_VIOLATION_SUBFRACTION_EXCEEDS_TOTAL"
    if tc <= 0 or hdl <= 0 or ldl <= 0 or tg <= 0:
        return False, "BOUND_VIOLATION_NEGATIVE_OR_ZERO"
    return True, "BOUNDS_SATISFIED"

def generate_discrete_edit_candidates(val: float, max_edits: int = 1) -> list:
    val_str = f"{val:.1f}"
    candidates = set()
    
    # 1. Decimal Shifts
    candidates.add(round(val * 0.1, 1))
    candidates.add(round(val * 10.0, 1))
    candidates.add(round(val * 0.01, 1))
    candidates.add(round(val * 100.0, 1))
    
    # 2. Local Digit Mutations / Swaps / Drops
    digits = "0123456789"
    for i, ch in enumerate(val_str):
        if ch == '.': 
            continue
        # Replacements (e.g. '1' <-> '7', '8' <-> '0')
        for d in digits:
            mutated = val_str[:i] + d + val_str[i+1:]
            try:
                candidates.add(round(float(mutated), 1))
            except ValueError:
                pass
        # Drops (e.g. '180.0' -> '10.0')
        dropped = val_str[:i] + val_str[i+1:]
        try:
            candidates.add(round(float(dropped), 1))
        except ValueError:
            pass
            
    # Filter non-physiological candidates
    return [c for c in candidates if 0.0 < c < 2000.0 and c != val]

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Damerau-Levenshtein distance (insertions, deletions, substitutions, and adjacent transpositions).
    """
    len1, len2 = len(s1), len(s2)
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost  # substitution
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[len1][len2]

def is_adjacent_digit_swap(orig_val: float, target_val: float, field: str = "hdl") -> bool:
    """
    Detects whether orig_val and target_val represent an exact digit transposition.
    Checks in native mg/dL (1 decimal place) and native mmol/L (2 decimal places) representation spaces.
    """
    if orig_val <= 0 or target_val <= 0:
        return False
    factor = 88.57 if field == "triglycerides" else 38.67
    
    # 1. Native mg/dL (1 decimal place)
    s1 = f"{orig_val:.1f}".replace(".", "")
    s2 = f"{target_val:.1f}".replace(".", "")
    if len(s1) == len(s2) and s1 != s2 and sorted(s1) == sorted(s2):
        diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
        if len(diffs) == 2 and s1[diffs[0]] == s2[diffs[1]] and s1[diffs[1]] == s2[diffs[0]]:
            return True
            
    # 2. Native mmol/L (2 decimal places)
    o_mmol = orig_val / factor
    t_mmol = target_val / factor
    s1_m = f"{o_mmol:.2f}".replace(".", "")
    s2_m = f"{t_mmol:.2f}".replace(".", "")
    if len(s1_m) == len(s2_m) and s1_m != s2_m and sorted(s1_m) == sorted(s2_m):
        diffs = [i for i in range(len(s1_m)) if s1_m[i] != s2_m[i]]
        if len(diffs) == 2 and s1_m[diffs[0]] == s2_m[diffs[1]] and s1_m[diffs[1]] == s2_m[diffs[0]]:
            return True
            
    return False

# Backward compatibility alias
is_digit_transposition_match = is_adjacent_digit_swap

def solve_field_closed_form(field: str, panel: dict) -> float:
    """Solves for the exact Friedewald or Sampson value of field in O(1) closed-form."""
    tc = panel.get("total_cholesterol", 0.0)
    hdl = panel.get("hdl", 0.0)
    ldl = panel.get("ldl", 0.0)
    tg = panel.get("triglycerides", 0.0)
    
    # Invalidate if TG > 800
    if tg > 800.0:
        return 0.0

    is_sampson = (400.0 < tg <= 800.0)
    
    if field == "total_cholesterol":
        if is_sampson:
            denom = (1.0 / 1.19) - (tg / 8.2)
            if abs(denom) < 1e-6:
                return 0.0
            return (ldl - (hdl / 1.1) + (tg / 3.98) - ((tg * hdl) / 8.2) + 0.17) / denom
        else:
            return hdl + ldl + (tg / 5.0)
            
    elif field == "hdl":
        if is_sampson:
            denom = (1.0 / 1.1) + (tg / 8.2)
            if abs(denom) < 1e-6:
                return 0.0
            return (ldl - (tc / 1.19) + (tg / 3.98) + ((tg * tc) / 8.2) + 0.17) / denom
        else:
            return tc - ldl - (tg / 5.0)
            
    elif field == "ldl":
        if is_sampson:
            non_hdl = tc - hdl
            return (tc / 1.19) + (hdl / 1.1) - (tg / 3.98) - ((tg * non_hdl) / 8.2) - 0.17
        else:
            return tc - hdl - (tg / 5.0)
            
    elif field == "triglycerides":
        denom_tg = -(1.0 / 3.98) - ((tc - hdl) / 8.2)
        if is_sampson and abs(denom_tg) > 1e-6:
            return (ldl - (tc / 1.19) - (hdl / 1.1) + 0.17) / denom_tg
        tg_f = 5.0 * (tc - hdl - ldl)
        if tg_f <= 400.0:
            return tg_f
        if abs(denom_tg) > 1e-6:
            tg_s = (ldl - (tc / 1.19) - (hdl / 1.1) + 0.17) / denom_tg
            if 400.0 < tg_s <= 800.0:
                return tg_s
        return tg_f
        
    return 0.0


def generate_discrete_plausible_candidates(val: float, field: str) -> list:
    """
    Generates discrete plausible candidates for a single field:
    scale factors {0.01, 0.1, 10.0, 100.0}, prefix/truncation recovery,
    and adjacent digit transpositions. At most ~15-20 candidates.
    """
    candidates = set()
    if val <= 0:
        return []
    
    # 1. Decimal shifts in mg/dL
    for scale in [0.01, 0.1, 10.0, 100.0]:
        c = round(val * scale, 1)
        if 0.0 < c < 2000.0 and c != val:
            candidates.add(c)
            
    # 2. Representation in mmol/L
    factor = 88.57 if field == "triglycerides" else 38.67
    val_mmol = val / factor
    for scale in [0.1, 10.0, 0.01, 100.0]:
        c = round((val_mmol * scale) * factor, 1)
        if 0.0 < c < 2000.0 and c != val:
            candidates.add(c)
            
    # 3. Adjacent digit transpositions in mg/dL
    for fmt in ["{:.1f}", "{:.0f}"]:
        s = fmt.format(val).replace(".", "")
        for i in range(len(s) - 1):
            if s[i] != s[i+1]:
                swapped = s[:i] + s[i+1] + s[i] + s[i+2:]
                try:
                    v = float(swapped[:-1] + "." + swapped[-1]) if fmt == "{:.1f}" else float(swapped)
                    if 0.0 < v < 2000.0 and v != val:
                        candidates.add(round(v, 1))
                except Exception:
                    pass

    # 4. Adjacent digit transpositions in mmol/L
    for fmt in ["{:.1f}", "{:.2f}"]:
        s = fmt.format(val_mmol).replace(".", "")
        for i in range(len(s) - 1):
            if s[i] != s[i+1]:
                swapped = s[:i] + s[i+1] + s[i] + s[i+2:]
                try:
                    num_dec = 1 if fmt == "{:.1f}" else 2
                    v_mmol = float(swapped[:-num_dec] + "." + swapped[-num_dec:])
                    v_mgdl = round(v_mmol * factor, 1)
                    if 0.0 < v_mgdl < 2000.0 and v_mgdl != val:
                        candidates.add(v_mgdl)
                except Exception:
                    pass

    # 5. Prefix/truncation recovery
    if val_mmol < 1.0:
        for d in range(1, 10):
            candidates.add(round(d * 1.0 * factor, 1))
            
    return [c for c in candidates if 0.0 < c < 2000.0 and c != val]

def is_decimal_shift(orig_val: float, target_val: float, field: str = "ldl") -> bool:
    """Checks if target_val represents an order-of-magnitude decimal shift of orig_val (including rounding/truncation)."""
    factor = 88.57 if field == "triglycerides" else 38.67
    val_pairs = [(orig_val, target_val), (orig_val / factor, target_val / factor)]
    for o_v, t_v in val_pairs:
        if o_v <= 0 or t_v <= 0:
            continue
        ratio = t_v / o_v
        # 1. Direct standard decimal ratios
        for expected_r in [0.1, 10.0, 0.01, 100.0]:
            if abs(ratio - expected_r) / expected_r <= 0.08:
                return True
        # 2. Integer-rounded / truncated decimal shifts (e.g. 3.52 / 10 = 0.352 -> 0.4)
        for scale in [10.0, 100.0]:
            if abs(round(t_v / scale, 1) - round(o_v, 1)) <= 0.05:
                return True
            if abs(round(t_v * scale, 1) - round(o_v, 1)) <= 0.05:
                return True
            if abs(round(o_v / scale, 1) - round(t_v, 1)) <= 0.05:
                return True
            if abs(round(o_v * scale, 1) - round(t_v, 1)) <= 0.05:
                return True
    return False

# Backward compatibility alias
check_decimal_shift = is_decimal_shift

def compute_candidate_cost(
    raw_val: float,
    target_val: float,
    confidence: float,
    field: str
) -> tuple[float, dict]:
    """
    Computes candidate penalty score. Lower score = higher likelihood of being the true error.
    
    Formula:
        base_edit_cost = damerau_levenshtein(raw_str, target_str)
        confidence_penalty = 0.50 * (1.0 - confidence)
        decimal_bonus = 0.30 if is_decimal_shift else 0.0
        transposition_bonus = 0.25 if is_adjacent_digit_swap else 0.0
        
        Final Cost = base_edit_cost + confidence_penalty - decimal_bonus - transposition_bonus
    """
    raw_str = f"{raw_val:.1f}".rstrip('0').rstrip('.')
    target_str = f"{target_val:.1f}".rstrip('0').rstrip('.')
    
    # 1. Base Damerau-Levenshtein Edit Distance (Deletion, Insertion, Substitution, Adjacent Swap)
    raw_dist = damerau_levenshtein_distance(raw_str, target_str)
    
    # 2. Check Physical Artifact Flags
    is_dec = is_decimal_shift(raw_val, target_val, field)
    is_trans = is_adjacent_digit_swap(raw_val, target_val, field)
    
    # 3. If it is an exact transposition or verified decimal shift, clamp effective edit distance to 1.0
    effective_dist = 1.0 if (is_trans or is_dec) else float(raw_dist)
    
    # 4. Confidence Penalty (0.0 penalty if model was 100% confident, up to 0.5 penalty if 0% confident)
    conf_penalty = 0.50 * (1.0 - max(0.0, min(1.0, confidence)))
    
    # 5. Calculate Final Score
    cost = effective_dist + conf_penalty
    if is_dec:
        cost -= 0.30
    if is_trans:
        cost -= 0.25
        
    metadata = {
        "raw_dist": raw_dist,
        "effective_dist": effective_dist,
        "conf_penalty": round(conf_penalty, 3),
        "is_decimal_shift": is_dec,
        "is_transposition": is_trans,
        "cost": round(cost, 3)
    }
    return cost, metadata

def hierarchical_isolation_search(panel: dict, confidences: dict, tol: float = 0.10) -> dict:
    tc, hdl, ldl, tg = panel["total_cholesterol"], panel["hdl"], panel["ldl"], panel["triglycerides"]
    
    # Step 0: Biological validity boundary for Sampson
    if tg > 800.0:
        return {
            "action": "ABSTAIN_TG800",
            "reason": "Serum Triglycerides > 800 mg/dL exceed Sampson validity threshold."
        }
        
    # Step 1: Check Invariant (Dynamic Equation Routing)
    is_sampson = (400.0 < tg <= 800.0)
    if is_sampson:
        ldl_calc = calculate_sampson_ldl(tc, hdl, tg)
        denom = abs(ldl) if ldl != 0 else 1.0
        residual = abs(ldl - ldl_calc) / denom
        if residual <= tol:
            return {"action": "CONVERGED", "fields": [], "residual": residual, "equation": "SAMPSON"}
    else:
        expected_tc = hdl + ldl + (tg / 5.0)
        residual = abs(tc - expected_tc) / tc if tc > 0 else 1.0
        if residual <= tol:
            return {"action": "CONVERGED", "fields": [], "residual": residual, "equation": "FRIEDEWALD"}
        
    # Step 2: k=1 Single-Field Algebraic Counterfactual Search
    single_candidates = {}
    if is_sampson:
        # Sampson closed-form inversions
        single_candidates["ldl"] = calculate_sampson_ldl(tc, hdl, tg)
        
        denom_tc = (1.0 / 1.19) - (tg / 8.2)
        if abs(denom_tc) > 1e-6:
            single_candidates["total_cholesterol"] = (ldl - (hdl / 1.1) + (tg / 3.98) - ((tg * hdl) / 8.2) + 0.17) / denom_tc
            
        denom_hdl = (1.0 / 1.1) + (tg / 8.2)
        if abs(denom_hdl) > 1e-6:
            single_candidates["hdl"] = (ldl - (tc / 1.19) + (tg / 3.98) + ((tg * tc) / 8.2) + 0.17) / denom_hdl
            
        denom_tg = -(1.0 / 3.98) - ((tc - hdl) / 8.2)
        if abs(denom_tg) > 1e-6:
            single_candidates["triglycerides"] = (ldl - (tc / 1.19) - (hdl / 1.1) + 0.17) / denom_tg
    else:
        single_candidates["total_cholesterol"] = hdl + ldl + (tg / 5.0)
        single_candidates["hdl"] = tc - ldl - (tg / 5.0)
        single_candidates["ldl"] = tc - hdl - (tg / 5.0)
        single_candidates["triglycerides"] = 5.0 * (tc - hdl - ldl)
        
    k1_ranked_list = []
    for field, target in single_candidates.items():
        if target is None:
            continue
        # Bounds check: ensure target does not violate physiological bounds
        if field != "ldl" and target <= 0:
            continue
        if not is_sampson and target <= 0:
            continue
        if target > 2000.0 or target < -20000.0:
            continue
        if field == "triglycerides" and target > 800.0:
            continue
        if field == "triglycerides" and not is_sampson and target > 400.0 and panel["triglycerides"] <= 400.0:
            denom_tg = -(1.0 / 3.98) - ((tc - hdl) / 8.2)
            if abs(denom_tg) > 1e-6:
                target_s = (ldl - (tc / 1.19) - (hdl / 1.1) + 0.17) / denom_tg
                if 400.0 < target_s <= 800.0:
                    target = target_s
                else:
                    continue
            else:
                continue

        test_panel = panel.copy()
        test_panel[field] = target
        if test_panel["hdl"] >= test_panel["total_cholesterol"]:
            continue
        if not is_sampson and test_panel["ldl"] >= test_panel["total_cholesterol"]:
            continue

        cost, meta = compute_candidate_cost(panel[field], target, confidences.get(field, 0.8), field)
        
        is_fast_path = meta["is_decimal_shift"]
        is_trans = meta["is_transposition"]
        is_plausible = (meta["effective_dist"] <= 2 or is_fast_path or is_trans)
        
        k1_ranked_list.append({
            "field": field,
            "target_val": round(target, 1),
            "score": cost,
            "is_fast_path": is_fast_path,
            "is_trans": is_trans,
            "is_plausible": is_plausible,
            "raw_dist": meta["raw_dist"],
            "metadata": meta
        })
        
    k1_ranked_list.sort(key=lambda x: x["score"])
    
    # Return best plausible k=1 candidate, along with all ranked candidates for vision fallback
    plausible_k1 = [c for c in k1_ranked_list if c["is_plausible"]]
    if plausible_k1:
        best_k1 = plausible_k1[0]
        return {
            "action": "CORRECT_K1",
            "field": best_k1["field"],
            "target_val": best_k1["target_val"],
            "is_fast_path": best_k1["is_fast_path"],
            "score": best_k1["score"],
            "ranked_candidates": plausible_k1
        }

    # Step 3: k=2 Fast 1D-to-Algebraic Closed-Form Search with Clinical Non-Maleficence Guardrail
    fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    all_k2_pairs = []
    
    for f1, f2 in itertools.permutations(fields, 2):
        c1_list = generate_discrete_plausible_candidates(panel[f1], f1)
        for cand1 in c1_list:
            test_panel = panel.copy()
            test_panel[f1] = cand1
            
            cand2 = solve_field_closed_form(f2, test_panel)
            if cand2 == 0.0 or cand2 >= 2000.0:
                continue
            cand2 = round(cand2, 1)
            test_panel[f2] = cand2
            
            is_sampson_pair = (400.0 < test_panel["triglycerides"] <= 800.0)
            # Check biological plausibility
            if test_panel["triglycerides"] > 800.0 or test_panel["hdl"] >= test_panel["total_cholesterol"]:
                continue
            if not is_sampson_pair and test_panel["ldl"] >= test_panel["total_cholesterol"]:
                continue
            if cand2 <= 0.0 and (not is_sampson_pair or f2 != "ldl"):
                continue
                
            # Both candidates must have physical mutation backing
            b1 = is_decimal_shift(panel[f1], cand1, f1) or is_adjacent_digit_swap(panel[f1], cand1, f1)
            b2 = is_decimal_shift(panel[f2], cand2, f2) or is_adjacent_digit_swap(panel[f2], cand2, f2)
            if not (b1 and b2):
                if not (b1 or b2):
                    continue
                raw1 = f"{panel[f1]:.1f}".rstrip('0').rstrip('.')
                cand_s1 = f"{cand1:.1f}".rstrip('0').rstrip('.')
                raw2 = f"{panel[f2]:.1f}".rstrip('0').rstrip('.')
                cand_s2 = f"{cand2:.1f}".rstrip('0').rstrip('.')
                d_other = damerau_levenshtein_distance(raw2, cand_s2) if b1 else damerau_levenshtein_distance(raw1, cand_s1)
                if d_other > 2:
                    continue

            if is_sampson_pair:
                ldl_calc = calculate_sampson_ldl(test_panel["total_cholesterol"], test_panel["hdl"], test_panel["triglycerides"])
                denom = abs(test_panel["ldl"]) if test_panel["ldl"] != 0 else 1.0
                res = abs(test_panel["ldl"] - ldl_calc) / denom
            else:
                exp_tc = test_panel["hdl"] + test_panel["ldl"] + (test_panel["triglycerides"] / 5.0)
                res = abs(test_panel["total_cholesterol"] - exp_tc) / test_panel["total_cholesterol"] if test_panel["total_cholesterol"] > 0 else 1.0
            
            if res <= tol:
                cost1, _ = compute_candidate_cost(panel[f1], cand1, confidences.get(f1, 0.8), f1)
                cost2, _ = compute_candidate_cost(panel[f2], cand2, confidences.get(f2, 0.8), f2)
                penalty = cost1 + cost2
                if b1 and b2:
                    penalty -= 0.50
                
                all_k2_pairs.append({
                    "action": "CORRECT_K2",
                    "corrections": {f1: cand1, f2: cand2},
                    "penalty": penalty,
                    "residual": res
                })
                    
    if all_k2_pairs:
        all_k2_pairs.sort(key=lambda x: x["penalty"])
        best_k2_pair = all_k2_pairs[0].copy()
        best_k2_pair["ranked_pairs"] = all_k2_pairs
        return best_k2_pair

    # Step 4: Bounded Abstention
    return {
        "action": "ABSTAIN_ALGEBRAIC_INDETERMINATE",
        "reason": "Invariant violated; no 1-field or discrete 2-field edit permutation resolved residual."
    }
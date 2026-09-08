import itertools
import Levenshtein

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

def hierarchical_isolation_search(panel: dict, confidences: dict, tol: float = 0.10) -> dict:
    tc, hdl, ldl, tg = panel["total_cholesterol"], panel["hdl"], panel["ldl"], panel["triglycerides"]
    
    # Step 1: Check Invariant
    expected_tc = hdl + ldl + (tg / 5.0)
    residual = abs(tc - expected_tc) / tc
    if residual <= tol:
        return {"action": "CONVERGED", "fields": [], "residual": residual}
        
    # Step 2: k=1 Single-Field Algebraic Counterfactual Search
    single_candidates = {
        "total_cholesterol": hdl + ldl + (tg / 5.0),
        "hdl": tc - ldl - (tg / 5.0),
        "ldl": tc - hdl - (tg / 5.0),
        "triglycerides": 5.0 * (tc - hdl - ldl)
    }
    
    k1_scores = {}
    for field, target in single_candidates.items():
        if target <= 0:
            continue
        orig_s, target_s = f"{panel[field]:.1f}", f"{target:.1f}"
        d_edit = Levenshtein.distance(orig_s, target_s) / max(len(orig_s), len(target_s))
        c_pen = 1.0 - confidences.get(field, 0.8)
        ratio = target / panel[field] if panel[field] != 0 else 0
        is_mag = round(ratio, 2) in [0.1, 10.0, 0.01, 100.0]
        
        # Calibration weights: α=0.48, β=0.22, γ=0.30
        k1_scores[field] = (0.48 * d_edit) + (0.22 * c_pen) - (0.30 if is_mag else 0.0)
        
    if k1_scores:
        best_k1_field = min(k1_scores, key=k1_scores.get)
        target_val = round(single_candidates[best_k1_field], 1)
        ratio = target_val / panel[best_k1_field] if panel[best_k1_field] != 0 else 0
        is_fast_path = round(ratio, 2) in [0.1, 10.0, 0.01, 100.0]
        
        # If edit distance is plausible, commit k=1
        orig_s, target_s = f"{panel[best_k1_field]:.1f}", f"{target_val:.1f}"
        if Levenshtein.distance(orig_s, target_s) <= 2 or is_fast_path:
            return {
                "action": "CORRECT_K1",
                "field": best_k1_field,
                "target_val": target_val,
                "is_fast_path": is_fast_path,
                "score": k1_scores[best_k1_field]
            }

    # Step 3: k=2 Discrete Combinatorial Search over Field-Pairs
    fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    best_k2_pair = None
    min_combined_penalty = float("inf")
    
    for f1, f2 in itertools.combinations(fields, 2):
        c1_list = generate_discrete_edit_candidates(panel[f1])
        c2_list = generate_discrete_edit_candidates(panel[f2])
        
        for cand1, cand2 in itertools.product(c1_list, c2_list):
            test_panel = panel.copy()
            test_panel[f1] = cand1
            test_panel[f2] = cand2
            
            # Check if this discrete pair resolves the equation
            exp_tc = test_panel["hdl"] + test_panel["ldl"] + (test_panel["triglycerides"] / 5.0)
            res = abs(test_panel["total_cholesterol"] - exp_tc) / test_panel["total_cholesterol"]
            
            if res <= tol:
                d1 = Levenshtein.distance(f"{panel[f1]:.1f}", f"{cand1:.1f}")
                d2 = Levenshtein.distance(f"{panel[f2]:.1f}", f"{cand2:.1f}")
                penalty = d1 + d2 + (2.0 - confidences[f1] - confidences[f2])
                
                if penalty < min_combined_penalty:
                    min_combined_penalty = penalty
                    best_k2_pair = {
                        "action": "CORRECT_K2",
                        "corrections": {f1: cand1, f2: cand2},
                        "penalty": penalty,
                        "residual": res
                    }
                    
    if best_k2_pair:
        return best_k2_pair

    # Step 4: Bounded Abstention
    return {
        "action": "ABSTAIN_ALGEBRAIC_INDETERMINATE",
        "reason": "Invariant violated; no 1-field or discrete 2-field edit permutation resolved residual."
    }
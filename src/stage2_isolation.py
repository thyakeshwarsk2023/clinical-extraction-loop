import itertools
import math
import Levenshtein
from scipy.optimize import brentq
from src.stage2_constraints import (
    calculate_sampson_ldl,
    evaluate_sampson_invariant,
    get_martin_hopkins_divisor,
    calculate_martin_hopkins_ldl,
    evaluate_martin_hopkins_invariant,
)

def invert_martin_hopkins(panel: dict, target_field: str) -> float | None:
    tc = panel.get("total_cholesterol", 0.0)
    hdl = panel.get("hdl", 0.0)
    ldl = panel.get("ldl", 0.0)
    tg = panel.get("triglycerides", 0.0)
    non_hdl = max(tc - hdl, 1.0)
    f = get_martin_hopkins_divisor(tg, non_hdl)
    
    if target_field == "ldl":
        return round(tc - hdl - (tg / f), 1)
    elif target_field == "total_cholesterol":
        return round(ldl + hdl + (tg / f), 1)
    elif target_field == "hdl":
        return round(tc - ldl - (tg / f), 1)
    elif target_field == "triglycerides":
        return round((tc - hdl - ldl) * f, 1)
    return None
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

def solve_field_closed_form(field: str, panel: dict, method: str = None) -> float:
    """Solves for the exact Friedewald, Sampson, or Martin-Hopkins value of field in O(1) closed-form."""
    if method == "MARTIN_HOPKINS":
        inv = invert_martin_hopkins(panel, field)
        return round(inv, 1) if inv is not None else 0.0

    tc = panel.get("total_cholesterol", 0.0)
    hdl = panel.get("hdl", 0.0)
    ldl = panel.get("ldl", 0.0)
    tg = panel.get("triglycerides", 0.0)
    
    # Invalidate if TG > 800
    if tg > 800.0:
        return 0.0

    is_sampson = (method == "SAMPSON") or (400.0 < tg <= 800.0)
    
    if field == "total_cholesterol":
        if is_sampson:
            denom = (1.0 / 0.948) - (tg / 2140.0)
            if abs(denom) < 1e-6:
                return 0.0
            return (ldl + (hdl / 0.971) + (tg / 8.56) - ((tg * hdl) / 2140.0) - (tg**2 / 16100.0) + 9.44) / denom
        else:
            return hdl + ldl + (tg / 5.0)
            
    elif field == "hdl":
        if is_sampson:
            denom = (1.0 / 0.971) - (tg / 2140.0)
            if abs(denom) < 1e-6:
                return 0.0
            return ((tc / 0.948) - ldl - (tg / 8.56) - ((tg * tc) / 2140.0) + (tg**2 / 16100.0) - 9.44) / denom
        else:
            return tc - ldl - (tg / 5.0)
            
    elif field == "ldl":
        if is_sampson:
            return calculate_sampson_ldl(tc, hdl, tg)
        else:
            return tc - hdl - (tg / 5.0)
            
    elif field == "triglycerides":
        def solve_sampson_tg() -> float | None:
            a = 1.0 / 16100.0
            b = -((1.0 / 8.56) + ((tc - hdl) / 2140.0))
            c = (tc / 0.948) - (hdl / 0.971) - 9.44 - ldl
            disc = b**2 - 4.0 * a * c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                r1 = (-b - sqrt_disc) / (2.0 * a)
                r2 = (-b + sqrt_disc) / (2.0 * a)
                valid = [r for r in [r1, r2] if 0.0 <= r <= 800.0]
                if valid:
                    return valid[0]
            try:
                def f_res(x):
                    non_hdl = tc - hdl
                    v = (x / 8.56) + ((x * non_hdl) / 2140.0) - ((x ** 2) / 16100.0)
                    return (tc / 0.948) - (hdl / 0.971) - v - 9.44 - ldl
                return brentq(f_res, 0.0, 800.0)
            except Exception:
                return None

        if is_sampson:
            stg = solve_sampson_tg()
            if stg is not None:
                return stg

        tg_f = 5.0 * (tc - hdl - ldl)
        if tg_f <= 400.0 and method != "SAMPSON":
            return tg_f
        stg = solve_sampson_tg()
        if stg is not None and 400.0 < stg <= 800.0:
            return stg
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
    if abs(raw_val - round(raw_val)) < 1e-3 and abs(target_val - round(target_val)) < 0.20:
        int_dist = damerau_levenshtein_distance(f"{raw_val:.0f}", f"{round(target_val):.0f}")
        raw_dist = min(raw_dist, int_dist)
    
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


def solve_k2_augmented_closed_form(
    panel: dict,
    confidences: dict = None,
    tol: float = 0.10,
    method: str = None,
    extra_field_threshold: float = 0.80
) -> dict | None:
    """
    Augments the K=2 dual-field fault search with extra independent linear equations:
        non_hdl = total_cholesterol - hdl
        vldl    = triglycerides / 5.0 (or tg / f_mh under Martin-Hopkins)
    
    When a K=2 candidate search would otherwise be underdetermined, this checks whether
    high-confidence extra fields make the system exactly or over-determined.
    If so, solves for the candidate pair in closed form using all available equations.
    Extra fields with confidence below extra_field_threshold are discarded as untrusted.
    """
    if confidences is None:
      confidences = {}

    # 1. Extract extra fields
    non_hdl_val = panel.get("non_hdl_cholesterol", panel.get("non_hdl"))
    vldl_val = panel.get("vldl", panel.get("vldl_cholesterol"))

    # 2. Check confidence thresholds
    non_hdl_conf = confidences.get(
        "non_hdl_cholesterol", confidences.get("non_hdl", 0.0)
    )
    vldl_conf = confidences.get(
        "vldl", confidences.get("vldl_cholesterol", 0.0)
    )

    has_non_hdl = (
        non_hdl_val is not None
        and float(non_hdl_val) > 0.0
        and non_hdl_conf >= extra_field_threshold
    )
    has_vldl = (
        vldl_val is not None
        and float(vldl_val) > 0.0
        and vldl_conf >= extra_field_threshold
    )

    if not has_non_hdl and not has_vldl:
      return None

    non_hdl = float(non_hdl_val) if has_non_hdl else None
    vldl = float(vldl_val) if has_vldl else None

    tc_raw = float(panel.get("total_cholesterol", 0.0))
    hdl_raw = float(panel.get("hdl", 0.0))
    ldl_raw = float(panel.get("ldl", 0.0))
    tg_raw = float(panel.get("triglycerides", 0.0))

    is_martin = method == "MARTIN_HOPKINS"
    f_mh = 5.0
    if is_martin:
      non_hdl_ref = (
          non_hdl if non_hdl is not None else max(tc_raw - hdl_raw, 1.0)
      )
      f_mh = get_martin_hopkins_divisor(tg_raw, non_hdl_ref)

    core_fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    candidate_solutions = []

    # 3. Test each pair of suspect fields (f1, f2)
    for f1, f2 in itertools.combinations(core_fields, 2):
      pair_set = {f1, f2}
      c1, c2 = None, None
      equations_used = []
      extra_used = []

      if pair_set == {"total_cholesterol", "ldl"}:
        if has_non_hdl:
          # TC = non_hdl + HDL, LDL = non_hdl - (TG / f)
          c_tc = non_hdl + hdl_raw
          c_ldl = non_hdl - (tg_raw / f_mh)
          c1, c2 = (c_tc, c_ldl) if f1 == "total_cholesterol" else (c_ldl, c_tc)
          equations_used = ["non_hdl = tc - hdl", "primary_identity"]
          extra_used = ["non_hdl_cholesterol"]

      elif pair_set == {"hdl", "ldl"}:
        if has_non_hdl:
          # HDL = TC - non_hdl, LDL = non_hdl - (TG / f)
          c_hdl = tc_raw - non_hdl
          c_ldl = non_hdl - (tg_raw / f_mh)
          c1, c2 = (c_hdl, c_ldl) if f1 == "hdl" else (c_ldl, c_hdl)
          equations_used = ["non_hdl = tc - hdl", "primary_identity"]
          extra_used = ["non_hdl_cholesterol"]

      elif pair_set == {"total_cholesterol", "triglycerides"}:
        if has_vldl:
          c_tg = vldl * f_mh
          c_tc = hdl_raw + ldl_raw + vldl
          c1, c2 = (c_tc, c_tg) if f1 == "total_cholesterol" else (c_tg, c_tc)
          equations_used = ["vldl = tg / 5", "primary_identity"]
          extra_used = ["vldl"]
        elif has_non_hdl:
          c_tc = non_hdl + hdl_raw
          c_tg = (non_hdl - ldl_raw) * f_mh
          c1, c2 = (c_tc, c_tg) if f1 == "total_cholesterol" else (c_tg, c_tc)
          equations_used = ["non_hdl = tc - hdl", "primary_identity"]
          extra_used = ["non_hdl_cholesterol"]

      elif pair_set == {"hdl", "triglycerides"}:
        if has_vldl:
          c_tg = vldl * f_mh
          c_hdl = tc_raw - ldl_raw - vldl
          c1, c2 = (c_hdl, c_tg) if f1 == "hdl" else (c_tg, c_hdl)
          equations_used = ["vldl = tg / 5", "primary_identity"]
          extra_used = ["vldl"]
        elif has_non_hdl:
          c_hdl = tc_raw - non_hdl
          c_tg = (non_hdl - ldl_raw) * f_mh
          c1, c2 = (c_hdl, c_tg) if f1 == "hdl" else (c_tg, c_hdl)
          equations_used = ["non_hdl = tc - hdl", "primary_identity"]
          extra_used = ["non_hdl_cholesterol"]

      elif pair_set == {"ldl", "triglycerides"}:
        if has_vldl:
          c_tg = vldl * f_mh
          c_ldl = tc_raw - hdl_raw - vldl
          c1, c2 = (c_ldl, c_tg) if f1 == "ldl" else (c_tg, c_ldl)
          equations_used = ["vldl = tg / 5", "primary_identity"]
          extra_used = ["vldl"]

      if c1 is None or c2 is None:
        continue

      c1 = round(c1, 1)
      c2 = round(c2, 1)

      # Physiological plausibility checks
      if c1 <= 0.0 or c2 <= 0.0:
        continue
      test_p = panel.copy()
      test_p[f1] = c1
      test_p[f2] = c2

      if test_p["hdl"] >= test_p["total_cholesterol"]:
        continue
      if test_p["ldl"] >= test_p["total_cholesterol"]:
        continue
      if (
          not (50.0 <= test_p["total_cholesterol"] <= 800.0)
          or not (10.0 <= test_p["hdl"] <= 200.0)
          or not (10.0 <= test_p["ldl"] <= 500.0)
          or not (10.0 <= test_p["triglycerides"] <= 800.0)
      ):
        continue
      if test_p["triglycerides"] > 800.0 or (
          not is_martin and test_p["triglycerides"] > 400.0
      ):
        continue

      # Validate residual vs primary equation
      if is_martin:
        st, res = evaluate_martin_hopkins_invariant(test_p, tol)
      else:
        exp_tc = test_p["hdl"] + test_p["ldl"] + (test_p["triglycerides"] / 5.0)
        res = (
            abs(test_p["total_cholesterol"] - exp_tc)
            / test_p["total_cholesterol"]
            if test_p["total_cholesterol"] > 0
            else 1.0
        )
        st = "CONVERGED" if res <= tol else "VIOLATION"

      if st != "CONVERGED":
        continue

      # Score candidate cost using Levenshtein distance and confidence weighting
      cost1, meta1 = compute_candidate_cost(
          panel[f1], c1, confidences.get(f1, 0.8), f1
      )
      cost2, meta2 = compute_candidate_cost(
          panel[f2], c2, confidences.get(f2, 0.8), f2
      )
      total_penalty = round(cost1 + cost2, 3)

      candidate_solutions.append({
          "action": "CORRECT_K2",
          "resolution_mechanism": "extra_field_equation",
          "equations_used": equations_used,
          "extra_fields_used": extra_used,
          "corrections": {f1: c1, f2: c2},
          "penalty": total_penalty,
          "residual": round(res, 4),
          "metadata": {f1: meta1, f2: meta2},
      })

    if not candidate_solutions:
      return None

    candidate_solutions.sort(key=lambda x: x["penalty"])
    best = candidate_solutions[0].copy()
    best["ranked_pairs"] = candidate_solutions
    return best


def hierarchical_isolation_search(
    panel: dict,
    confidences: dict = None,
    tol: float = 0.10,
    method: str = None,
    require_independent_evidence: bool = False,
    independent_evidence: dict = None
) -> dict:
    if confidences is None:
        confidences = {f: 0.85 for f in panel}

    required = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    missing = [f for f in required if f not in panel or panel[f] is None]
    if missing:
        return {
            "action": "ABSTAIN_MISSING_ANALYTE",
            "field": missing[0],
            "target_val": None,
            "residual": 0.0,
            "cost": 999.0,
            "reason": f"Cannot execute invariant isolation: missing analytes {missing}",
            "message": f"Cannot execute invariant isolation: missing analytes {missing}"
        }

    tc = panel.get("total_cholesterol", 0.0)
    hdl = panel.get("hdl", 0.0)
    ldl = panel.get("ldl", 0.0)
    tg = panel.get("triglycerides", 0.0)

    is_martin = (method == "MARTIN_HOPKINS")

    # Step 0: Biological validity boundary for Sampson
    if tg > 800.0 and not is_martin:
        return {
            "action": "ABSTAIN_TG800",
            "reason": "Serum Triglycerides > 800 mg/dL exceed Sampson validity threshold."
        }
        
    # Step 1: Check Invariant (Dynamic Equation Routing)
    is_sampson = (400.0 < tg <= 800.0) and not is_martin
    if is_martin:
        status, rel_err = evaluate_martin_hopkins_invariant(panel, tol)
        if status == "CONVERGED":
            return {"action": "CONVERGED", "fields": [], "residual": rel_err, "equation": "MARTIN_HOPKINS"}
    elif is_sampson:
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
    active_method = "MARTIN_HOPKINS" if is_martin else ("SAMPSON" if is_sampson else "FRIEDEWALD")
    for f in ["total_cholesterol", "hdl", "ldl", "triglycerides"]:
        single_candidates[f] = solve_field_closed_form(f, panel, method=active_method)
        
    k1_ranked_list = []
    for field, target in single_candidates.items():
        if target is None:
            continue
        # Bounds check: ensure target does not violate physiological bounds
        if field != "ldl" and target <= 0:
            continue
        if not is_sampson and not is_martin and target <= 0:
            continue
        if target > 2000.0 or target < -20000.0:
            continue
        if field == "triglycerides" and target > 800.0 and not is_martin:
            continue
        if field == "triglycerides" and not is_sampson and not is_martin and target > 400.0 and panel["triglycerides"] <= 400.0:
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
        if not is_sampson and not is_martin and test_panel["ldl"] >= test_panel["total_cholesterol"]:
            continue

        # Consistency with trusted extra-field equations (if present and high-confidence)
        extra_thresh = 0.80
        nh_val = panel.get("non_hdl_cholesterol", panel.get("non_hdl"))
        nh_conf = confidences.get("non_hdl_cholesterol", confidences.get("non_hdl", 0.0))
        if nh_val is not None and float(nh_val) > 0.0 and nh_conf >= extra_thresh:
            if abs(test_panel["total_cholesterol"] - test_panel["hdl"] - float(nh_val)) / float(nh_val) > tol:
                continue

        vl_val = panel.get("vldl", panel.get("vldl_cholesterol"))
        vl_conf = confidences.get("vldl", confidences.get("vldl_cholesterol", 0.0))
        if vl_val is not None and float(vl_val) > 0.0 and vl_conf >= extra_thresh:
            f_div = 5.0
            if is_martin:
                f_div = get_martin_hopkins_divisor(test_panel["triglycerides"], test_panel["total_cholesterol"] - test_panel["hdl"])
            if abs(test_panel["triglycerides"] / f_div - float(vl_val)) / float(vl_val) > tol:
                continue

        cost, meta = compute_candidate_cost(panel[field], target, confidences.get(field, 0.8), field)
        
        is_fast_path = meta["is_decimal_shift"]
        is_trans = meta["is_transposition"]
        is_plausible = (meta["effective_dist"] <= 2 or is_fast_path or is_trans)
        
        target_rounded = float(round(target)) if abs(target - round(target)) < 0.06 else round(target, 1)
        k1_ranked_list.append({
            "field": field,
            "target_val": target_rounded,
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

    # Step 2.5: k=2 Extra-Field Equation Augmentation (Closed-form multi-equation resolution)
    k2_augmented = solve_k2_augmented_closed_form(
        panel,
        confidences=confidences,
        tol=tol,
        method=active_method,
        extra_field_threshold=0.80
    )
    if k2_augmented is not None:
        return k2_augmented

    # Step 3: k=2 Fast 1D-to-Algebraic Closed-Form Search with Clinical Non-Maleficence Guardrail
    fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    all_k2_pairs = []
    
    for f1, f2 in itertools.permutations(fields, 2):
        c1_list = generate_discrete_plausible_candidates(panel[f1], f1)
        for cand1 in c1_list:
            test_panel = panel.copy()
            test_panel[f1] = cand1
            
            cand2 = solve_field_closed_form(f2, test_panel, method=method)
            if cand2 == 0.0 or cand2 >= 2000.0:
                continue
            cand2 = round(cand2, 1)
            test_panel[f2] = cand2
            
            is_sampson_pair = (400.0 < test_panel.get("triglycerides", 0.0) <= 800.0) and not is_martin
            # Check biological plausibility
            if not is_martin and test_panel.get("triglycerides", 0.0) > 800.0:
                continue
            if test_panel.get("hdl", 0.0) >= test_panel.get("total_cholesterol", 0.0):
                continue
            if not is_sampson_pair and not is_martin and test_panel.get("ldl", 0.0) >= test_panel.get("total_cholesterol", 0.0):
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

            if is_martin:
                st, res = evaluate_martin_hopkins_invariant(test_panel, tol)
            elif is_sampson_pair:
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

        has_independent_evidence = False
        if independent_evidence:
            for field, confirmed_val in independent_evidence.items():
                if field in best_k2_pair["corrections"] and abs(best_k2_pair["corrections"][field] - confirmed_val) <= 0.05:
                    has_independent_evidence = True
                    break

        if require_independent_evidence and not has_independent_evidence:
            return {
                "action": "ABSTAIN_ALGEBRAIC_INDETERMINATE",
                "reason": "K=2 candidates ranked by cost model, but abstained due to lack of independent confirmation.",
                "ranked_pairs": all_k2_pairs
            }
        return best_k2_pair

    # Step 4: Bounded Abstention
    return {
        "action": "ABSTAIN_ALGEBRAIC_INDETERMINATE",
        "reason": "Invariant violated; no 1-field or discrete 2-field edit permutation resolved residual.",
        "ranked_pairs": []
    }


def explain_counterfactual(
    panel: dict,
    confidences: dict = None,
    tol: float = 0.10,
    method: str = None
) -> dict:
    """
    Computes Counterfactual Explainable AI (XAI) diagnostics for a clinical panel.
    Returns:
      - active_equation: FRIEDEWALD, SAMPSON, or MARTIN_HOPKINS
      - is_converged: bool
      - residual: float
      - field_counterfactuals: dict of each field's target, absolute delta, relative delta,
        Damerau-Levenshtein distance, OCR penalty, decimal shift bonus, transposition bonus,
        and net anomaly score.
      - isolated_field: suspect field (if any)
      - clinical_explanation: human-interpretable clinical explanation narrative.
    """
    if confidences is None:
        confidences = {f: 0.85 for f in panel}

    tg = panel.get("triglycerides", 0.0)
    if method == "MARTIN_HOPKINS":
        eq_name = "MARTIN_HOPKINS"
        status, res = evaluate_martin_hopkins_invariant(panel, tol)
    elif tg > 400.0:
        eq_name = "SAMPSON"
        status, res = evaluate_sampson_invariant(panel, tol)
    else:
        eq_name = "FRIEDEWALD"
        expected_tc = panel.get("hdl", 0.0) + panel.get("ldl", 0.0) + (tg / 5.0)
        tc = panel.get("total_cholesterol", 0.0)
        res = abs(tc - expected_tc) / tc if tc > 0 else 1.0
        status = "CONVERGED" if res <= tol else "VIOLATION"

    fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    diagnostics = {}
    for f in fields:
        raw_val = panel.get(f, 0.0)
        target = solve_field_closed_form(f, panel, method=method)
        cost, meta = compute_candidate_cost(raw_val, target, confidences.get(f, 0.85), f)
        delta_abs = abs(raw_val - target)
        delta_pct = (delta_abs / max(raw_val, 1e-3)) * 100.0

        diagnostics[f] = {
            "raw_value": round(raw_val, 1),
            "counterfactual_target": round(target, 1),
            "delta_abs": round(delta_abs, 1),
            "delta_pct": round(delta_pct, 1),
            "edit_distance": meta["raw_dist"],
            "is_decimal_shift": meta["is_decimal_shift"],
            "is_transposition": meta["is_transposition"],
            "confidence_penalty": round(meta.get("conf_penalty", 0.0), 3),
            "composite_anomaly_score": round(cost, 3),
        }

    sorted_fields = sorted(fields, key=lambda f: diagnostics[f]["composite_anomaly_score"])
    best_f = sorted_fields[0]
    best_meta = diagnostics[best_f]

    if status == "CONVERGED":
        explanation = (
            f"Lipid panel is internally consistent under {eq_name} "
            f"(residual: {res*100:.1f}% <= {tol*100:.0f}% tol). All subfractions are in physiological equilibrium."
        )
    else:
        f_title = best_f.replace("_", " ").title()
        if best_meta["is_decimal_shift"]:
            explanation = (
                f"Isolated root cause: {f_title} exhibits an order-of-magnitude decimal point shift "
                f"({best_meta['raw_value']} vs. {best_meta['counterfactual_target']} mg/dL). "
                f"Counterfactual restoration resolves invariant under {eq_name}."
            )
        elif best_meta["is_transposition"]:
            explanation = (
                f"Isolated root cause: {f_title} exhibits an adjacent digit transposition "
                f"({best_meta['raw_value']} vs. {best_meta['counterfactual_target']} mg/dL, Levenshtein distance {best_meta['edit_distance']}). "
                f"Classic transcription typographic error."
            )
        else:
            explanation = (
                f"Isolated suspect analyte: {f_title} with lowest composite anomaly penalty "
                f"({best_meta['composite_anomaly_score']:.2f}). Counterfactual target {best_meta['counterfactual_target']} mg/dL "
                f"achieves physiological consistency under {eq_name}."
            )

    return {
        "active_equation": eq_name,
        "is_converged": (status == "CONVERGED"),
        "residual": round(res, 4),
        "field_counterfactuals": diagnostics,
        "isolated_field": best_f if status != "CONVERGED" else None,
        "clinical_explanation": explanation,
    }

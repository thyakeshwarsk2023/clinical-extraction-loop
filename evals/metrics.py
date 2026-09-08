"""
evals/metrics.py — Metric helper functions for the lipid-panel extraction pipeline.

All functions operate on the long-form field keys used throughout this codebase:
  total_cholesterol, hdl, ldl, triglycerides

Do NOT rename these keys — changing them to TC/HDL/LDL/TRIG would break
stage2_constraints.py, stage2_isolation.py, and synthetic_generator.py.
"""

from collections import Counter

# Canonical field order, matching synthetic_generator.py and stage2_*.py
FIELDS = ["total_cholesterol", "hdl", "ldl", "triglycerides"]

# Modes whose pipeline output should require zero correction attempts
CLEAN_MODES = {"CLEAN", "DIRECT_ASSAY_CLEAN"}

# Modes where ABSTAIN is the correct / expected pipeline outcome
EXPECTED_ABSTAIN_MODES = {"ABSTAIN_HIGH_TG"}

# mg/dL tolerance for field-level match (accounts for floating-point rounding)
FIELD_TOLERANCE_MGDL = 1.0


def field_accuracy(pred_dict: dict, gt_dict: dict,
                   tolerance: float = FIELD_TOLERANCE_MGDL) -> dict:
    """
    Compute per-field binary accuracy (within ±tolerance mg/dL) and an
    overall average.

    Args:
        pred_dict  — pipeline's returned `data` dict (corrected values)
        gt_dict    — ground_truth_clean dict from the sample JSON
        tolerance  — allowable absolute deviation in mg/dL

    Returns:
        dict with keys "total_cholesterol", "hdl", "ldl", "triglycerides",
        "overall". Each field value is True (match), False (mismatch), or
        None (field absent in either dict).
    """
    results = {}
    for f in FIELDS:
        if f in pred_dict and f in gt_dict:
            results[f] = abs(float(pred_dict[f]) - float(gt_dict[f])) <= tolerance
        else:
            results[f] = None

    valid = [v for v in results.values() if v is not None]
    results["overall"] = sum(valid) / len(valid) if valid else 0.0
    return results


def isolation_accuracy(audit_trail: list, corrupted_fields: list) -> bool | None:
    """
    Did Stage 2.5 correctly identify all corrupted field(s)?

    Scans every audit-trail entry for "field" (k=1 path) or "modifications"
    (k=2 path) and collects the set of fields the pipeline chose to correct.
    Compares that set against the ground-truth `corrupted_fields` list.

    Returns:
        True   — all corrupted fields and only corrupted fields were targeted
        False  — wrong field(s) targeted, or no correction was attempted
        None   — sample has no corrupted fields (clean), metric not applicable
    """
    if not corrupted_fields:
        return None  # Not applicable for clean samples

    if not audit_trail:
        return False  # Corruption present but pipeline made no correction

    named_fields: set = set()
    for entry in audit_trail:
        if "field" in entry:
            named_fields.add(entry["field"])
        if "modifications" in entry:
            for mod in entry.get("modifications", []):
                if "field" in mod:
                    named_fields.add(mod["field"])

    return named_fields == set(corrupted_fields)


def status_distribution(results: list) -> dict:
    """
    Return a Counter (as plain dict) over the `status` string of every
    pipeline result in the list.

    Args:
        results — list of row dicts produced by run_ablations.run_eval()

    Returns:
        {status_string: count, ...} sorted by status string
    """
    return dict(sorted(Counter(r["status"] for r in results).items()))


def false_trigger_rate(results: list) -> float:
    """
    Fraction of clean-mode documents that triggered at least one correction
    attempt (i.e., audit_trail is non-empty).

    A non-empty audit_trail on a clean document means the pipeline incorrectly
    detected a Friedewald violation and tried to correct a valid panel.

    Args:
        results — list of row dicts produced by run_ablations.run_eval()

    Returns:
        float in [0.0, 1.0]
    """
    clean_rows = [r for r in results if r["mode"] in CLEAN_MODES]
    if not clean_rows:
        return 0.0
    triggered = [r for r in clean_rows if r.get("audit_trail")]
    return len(triggered) / len(clean_rows)


def correction_precision(results: list) -> float:
    """
    Of all samples where the pipeline attempted a correction (audit_trail
    non-empty), what fraction had actually been corrupted?

    A high value means corrections are targeted; a low value indicates
    spurious corrections on clean or boundary samples.
    """
    attempted = [r for r in results if r.get("audit_trail")]
    if not attempted:
        return 1.0  # no corrections attempted — trivially precise
    actually_corrupted = [r for r in attempted if r.get("corrupted_fields")]
    return len(actually_corrupted) / len(attempted)

#!/usr/bin/env python3
"""
evals/eval_ink_ablation.py
=============================================================================
Phase 5.3 Ink Ablation Damage Curve & Safe Abstention Benchmark.
Progressively erases 10%, 30%, 60%, and 100% of digit ink strokes.
Evaluates the trade-off between Committed-Wrong Rate and Abstention Rate.
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stage2_isolation import hierarchical_isolation_search

ABLATION_LEVELS = [0.10, 0.30, 0.60, 1.00]

def simulate_ink_ablation(clean_panel: dict, target_field: str, ablation_fraction: float) -> tuple[dict, dict]:
    """
    Simulates ink stroke ablation on a field:
      - At 10%: minor stroke drop (e.g. 180 -> 180 with lower confidence)
      - At 30%: partial occlusion / digit swap (e.g. 180 -> 100 or 18)
      - At 60%: major stroke ablation (e.g. 180 -> 10, confidence drops to 0.3)
      - At 100%: total ink obliteration (field value missing/zero, confidence 0.05)
    """
    corrupted = clean_panel.copy()
    val = clean_panel[target_field]
    confidences = {f: 0.95 for f in clean_panel}

    if ablation_fraction <= 0.10:
        # 10% ablation: faint ink, high chance of correct recovery
        confidences[target_field] = 0.85
    elif ablation_fraction <= 0.30:
        # 30% ablation: slight digit confusion
        confidences[target_field] = 0.65
        corrupted[target_field] = round(val * 0.9, 1)  # small smudge shift
    elif ablation_fraction <= 0.60:
        # 60% ablation: major confusion / order of magnitude shift
        confidences[target_field] = 0.40
        corrupted[target_field] = round(val * 0.1, 1)
    else:
        # 100% ablation: complete loss of stroke
        confidences[target_field] = 0.05
        corrupted[target_field] = 0.0

    return corrupted, confidences

def run_ink_ablation_benchmark(n_trials: int = 50, seed: int = 42):
    np.random.seed(seed)
    curve_data = {}

    # Base clean lipid panels (TC = HDL + LDL + TG/5)
    base_panels = [
        {"total_cholesterol": 200.0, "hdl": 50.0, "ldl": 130.0, "triglycerides": 100.0},
        {"total_cholesterol": 220.0, "hdl": 45.0, "ldl": 145.0, "triglycerides": 150.0},
        {"total_cholesterol": 250.0, "hdl": 55.0, "ldl": 155.0, "triglycerides": 200.0},
        {"total_cholesterol": 180.0, "hdl": 40.0, "ldl": 110.0, "triglycerides": 150.0},
        {"total_cholesterol": 210.0, "hdl": 60.0, "ldl": 120.0, "triglycerides": 150.0},
    ]

    for fraction in ABLATION_LEVELS:
        pct_label = f"{int(fraction * 100)}%"
        correct_repaired = 0
        abstained = 0
        committed_wrong = 0
        total = 0

        for trial in range(n_trials):
            clean = base_panels[trial % len(base_panels)]
            field = ["total_cholesterol", "hdl", "ldl", "triglycerides"][trial % 4]
            corrupt_p, confs = simulate_ink_ablation(clean, field, fraction)
            total += 1

            # 1. Bounds Check
            from src.stage2_constraints import verify_physiological_bounds, evaluate_friedewald_invariant
            bounds_ok, _ = verify_physiological_bounds(corrupt_p)
            if not bounds_ok:
                abstained += 1
                continue

            # 2. Invariant Check
            inv_st, _, _ = evaluate_friedewald_invariant(corrupt_p, tolerance=0.10)
            if inv_st == "CONVERGED":
                correct_repaired += 1  # Intact panel passed without false alarms
                continue

            # 3. Isolation & Repair
            res = hierarchical_isolation_search(corrupt_p, confs, tol=0.10)
            action = res.get("action")

            if "ABSTAIN" in action:
                abstained += 1
            elif action == "CORRECT_K1":
                target = res.get("target_val")
                # Check if target restored true clean value within 5%
                if abs(target - clean[field]) / max(clean[field], 1.0) <= 0.05:
                    correct_repaired += 1
                else:
                    committed_wrong += 1
            else:
                committed_wrong += 1

        curve_data[pct_label] = {
            "ablation_fraction": fraction,
            "total_trials": total,
            "correct_repaired_rate": round(correct_repaired / total, 3),
            "abstention_rate": round(abstained / total, 3),
            "committed_wrong_rate": round(committed_wrong / total, 3)
        }

    report = {
        "benchmark": "Ink Ablation Damage vs Safe Abstention Curve",
        "description": "Quantifies system behavior under progressive ink erasure (10% to 100%).",
        "levels": curve_data
    }

    out_json = os.path.join(ROOT, "reports", "ink_ablation_curve.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved ink ablation benchmark to: {out_json}")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_ink_ablation_benchmark()

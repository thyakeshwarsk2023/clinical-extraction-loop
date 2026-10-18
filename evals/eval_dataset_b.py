"""
evals/eval_dataset_b.py -- Evaluation harness for realistic multi-template
documents in dataset_B_realistic.

Evaluates Stage 0 Assay-Method Classification and Stage 2 Invariant Routing
over three realistic clinical cohorts (180 total documents):
  1. dataset_B_realistic/friedewald/         (n=60, calculated Friedewald)
  2. dataset_B_realistic/direct_ldl/         (n=60, direct enzymatic assay, LDL perturbed)
  3. dataset_B_realistic/unlabeled_ambiguous/ (n=60, unknown/unlabeled header)

Key Metrics Measured:
  - Stage 0 Assay Classification Accuracy
  - Direct Assay False-Trigger Rate (should be 0.0% even with perturbed LDL)
  - Post-Routing Convergence Rate
"""

import os
import sys
import csv
import json
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline
from evals.metrics import FIELDS

B_DIR = os.path.join(ROOT, "dataset_B_realistic")

SUBSET_CONFIG = {
    "friedewald": {
        "expected_route": "FRIEDEWALD",
        "method_text": "LDL Cholesterol (Calculated, Friedewald)"
    },
    "direct_ldl": {
        "expected_route": "DIRECT_MEASURED",
        "method_text": "LDL Cholesterol (Direct Measurement)"
    },
    "unlabeled_ambiguous": {
        "expected_route": "AMBIGUOUS_FALLBACK",
        "method_text": ""
    }
}

DUMMY_BBOX = [0, 0, 10, 10]


def run_eval_dataset_b(verbose: bool = False) -> dict:
    results = defaultdict(list)
    total_docs = 0

    print("=" * 74)
    print("  EVALUATION HARNESS: DATASET B (REALISTIC MULTI-TEMPLATE COHORT)")
    print("=" * 74)

    for subset_name, config in SUBSET_CONFIG.items():
        subset_dir = os.path.join(B_DIR, subset_name)
        csv_path = os.path.join(subset_dir, "ground_truth.csv")
        if not os.path.isfile(csv_path):
            print(f"  [skip] Missing CSV: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"  Evaluating {subset_name:<20} : {len(rows)} documents")

        for row in rows:
            total_docs += 1
            doc_id = row["doc_id"]
            img_path = os.path.join(subset_dir, f"{doc_id}.png")
            json_path = os.path.join(subset_dir, f"{doc_id}.json")

            unit = row.get("unit", "mg/dL")
            tc = float(row["TC"])
            hdl = float(row["HDL"])
            ldl = float(row["LDL"])
            tg = float(row["TG"])

            panel_data = {
                "total_cholesterol": tc,
                "hdl": hdl,
                "ldl": ldl,
                "triglycerides": tg
            }

            sidecar = {
                "method_text": config["method_text"],
                "unit": unit,
                "injected_display_data": panel_data,
                "bounding_boxes": {f: DUMMY_BBOX for f in FIELDS},
                "corrupted_fields": [],
                "ground_truth_clean": panel_data,
                "assay_ground_truth": config["expected_route"]
            }

            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(sidecar, jf)

            try:
                pipeline_out = run_pipeline(img_path)
            finally:
                if os.path.exists(json_path):
                    os.remove(json_path)

            status = pipeline_out["status"]
            assay_route = pipeline_out.get("assay_route", "UNKNOWN")
            is_route_correct = (assay_route == config["expected_route"])
            triggered = len(pipeline_out.get("audit_trail", [])) > 0

            record = {
                "doc_id": doc_id,
                "subset": subset_name,
                "unit": unit,
                "status": status,
                "assay_route": assay_route,
                "expected_route": config["expected_route"],
                "is_route_correct": is_route_correct,
                "triggered": triggered,
                "attempts": pipeline_out.get("attempts", 0)
            }
            results[subset_name].append(record)

            if verbose:
                flag = "PASS" if is_route_correct else "FAIL"
                print(f"    [{flag}] {doc_id} -> Route: {assay_route} | Status: {status}")

    return dict(results)


def print_summary_dataset_b(results: dict):
    print("\n" + "=" * 74)
    print("  DATASET B EVALUATION SUMMARY")
    print("=" * 74)

    total_n = sum(len(v) for v in results.values())
    total_route_correct = sum(sum(1 for r in v if r["is_route_correct"]) for v in results.values())
    overall_routing_acc = total_route_correct / total_n if total_n else 0.0

    print(f"  Total Documents Evaluated : {total_n}")
    print(f"  Stage 0 Routing Accuracy  : {total_route_correct}/{total_n} ({overall_routing_acc:.1%})\n")

    print(f"  {'Subset':<22} {'N':>4} {'Expected Route':<18} {'Routing Acc':>12} {'Convergence':>12}")
    print("  " + "-" * 70)

    for subset, rows in results.items():
        n = len(rows)
        correct_route = sum(1 for r in rows if r["is_route_correct"])
        route_acc = correct_route / n if n else 0.0
        converged = sum(1 for r in rows if "CONVERGED" in r["status"])
        conv_rate = converged / n if n else 0.0
        exp_route = SUBSET_CONFIG[subset]["expected_route"]
        print(f"  {subset:<22} {n:>4} {exp_route:<18} {route_acc:>11.1%} {conv_rate:>11.1%}")

    # False-trigger check on Direct LDL
    direct_rows = results.get("direct_ldl", [])
    if direct_rows:
        false_triggers = sum(1 for r in direct_rows if r["status"] == "VIOLATION" or "ABSTAIN_PATHOLOGIST" in r["status"])
        ftr = false_triggers / len(direct_rows)
        print("\n  DIRECT LDL ASSAY CLINICAL SAFETY:")
        print(f"    Bypassed Equality & Satisfied Bounds : {sum(1 for r in direct_rows if 'CONVERGED' in r['status'])}/{len(direct_rows)}")
        print(f"    False-Trigger / Invalidation Rate    : {false_triggers}/{len(direct_rows)} ({ftr:.1%})")

    print("=" * 74 + "\n")


if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    res = run_eval_dataset_b(verbose=verbose)
    print_summary_dataset_b(res)

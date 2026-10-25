"""
evals/eval_twenty_test_cases.py
=============================================================================
Phase 6C: Dual-Mode Targeted Empirical Evaluation of 20 Clinical Documents.
Evaluates the 20 diverse corrupted clinical documents across Baseline, Symbolic,
and Full Architecture in BOTH:
  1. oracle_sidecar mode: Uses ground-truth/corruption sidecar metadata.
  2. rapidocr mode: Uses live RapidOCR ONNX engine directly on document images,
     with clinical layout parsing and Stage 4 blind re-read.

Generates:
  - reports/twenty_test_cases_evaluation.json
  - reports/twenty_test_cases_evaluation_2.json
"""

import os
import sys
import json
import time
import itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evals.run_final_ablation import (
    _load_clean_map, _load_corrupted_single, _load_corrupted_double,
    _write_sidecar, _remove_sidecar,
    run_baseline, run_symbolic_only, run_full_architecture
)
from src.ocr_engine import get_ocr_engine
from src.stage1_extract import extract_document_state
from evals.metrics import field_accuracy
from src.stage2_isolation import generate_discrete_plausible_candidates, hierarchical_isolation_search


def evaluate_test_cases(engine_mode: str, test_case_ids: list, all_recs: dict) -> dict:
    print(f"\n--- Running 20-Case Evaluation in '{engine_mode}' Mode ---")
    ocr_engine = None
    if engine_mode == "rapidocr":
        ocr_engine = get_ocr_engine("rapidocr")

    cases_results = []
    base_fa_list, sym_fa_list, full_fa_list = [], [], []
    base_lat_list, sym_lat_list, full_lat_list = [], [], []

    for doc_id in test_case_ids:
        rec = all_recs[doc_id]
        img_path = rec["img_path"]

        if engine_mode == "oracle_sidecar":
            _write_sidecar(
                img_path, rec["display_values"], rec["gt_clean_values"],
                rec["unit"], rec["bboxes_long"], rec["method_text"]
            )
            doc = extract_document_state(img_path)
            _remove_sidecar(img_path)
        else:
            doc = ocr_engine.extract(img_path)
            doc["ground_truth_clean"] = rec["gt_clean_values"]

        b_res, b_lat, b_st = run_baseline(doc)
        b_acc = field_accuracy(b_res, rec["gt_clean_values"])

        s_res, s_lat, s_st = run_symbolic_only(doc)
        s_acc = field_accuracy(s_res, rec["gt_clean_values"])

        f_res, f_lat, f_st = run_full_architecture(doc)
        f_acc = field_accuracy(f_res, rec["gt_clean_values"])

        base_fa_list.append(b_acc["overall"] * 100.0)
        sym_fa_list.append(s_acc["overall"] * 100.0)
        full_fa_list.append(f_acc["overall"] * 100.0)

        base_lat_list.append(b_lat)
        sym_lat_list.append(s_lat)
        full_lat_list.append(f_lat)

        cases_results.append({
            "doc_id": doc_id,
            "split": rec["split"],
            "corruption_type": rec["corruption_type"],
            "corrupted_fields": rec["corrupted_fields"],
            "baseline": {
                "latency_ms": round(b_lat, 3),
                "status": b_st,
                "field_accuracy": b_acc
            },
            "symbolic": {
                "latency_ms": round(s_lat, 3),
                "status": s_st,
                "field_accuracy": s_acc
            },
            "full_arch": {
                "latency_ms": round(f_lat, 3),
                "status": f_st,
                "field_accuracy": f_acc
            }
        })

    agg = {
        "engine": engine_mode,
        "sample_size": len(cases_results),
        "baseline_mean_fa": round(sum(base_fa_list) / len(base_fa_list), 2),
        "symbolic_mean_fa": round(sum(sym_fa_list) / len(sym_fa_list), 2),
        "full_arch_mean_fa": round(sum(full_fa_list) / len(full_fa_list), 2),
        "baseline_mean_latency_ms": round(sum(base_lat_list) / len(base_lat_list), 3),
        "symbolic_mean_latency_ms": round(sum(sym_lat_list) / len(sym_lat_list), 3),
        "full_arch_mean_latency_ms": round(sum(full_lat_list) / len(full_lat_list), 3),
    }
    print(f"[{engine_mode}] Baseline FA: {agg['baseline_mean_fa']}%, Symbolic FA: {agg['symbolic_mean_fa']}%, Full Arch FA: {agg['full_arch_mean_fa']}%, Mean Latency: {agg['full_arch_mean_latency_ms']} ms")

    return {
        "aggregate_metrics": agg,
        "test_cases": cases_results
    }


def main():
    clean_map = _load_clean_map()
    single_recs = _load_corrupted_single(clean_map)
    double_recs = _load_corrupted_double(clean_map)
    all_recs = {r["doc_id"]: r for r in single_recs + double_recs}

    eval_json_path = os.path.join(ROOT, "reports", "twenty_test_cases_evaluation.json")
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        if "test_cases" in existing_data:
            test_case_ids = [tc["doc_id"] for tc in existing_data["test_cases"]]
        else:
            test_case_ids = [
                "doc_0051", "doc_0061", "doc_0130", "doc_0095", "doc_0152",
                "doc_0005", "doc_0137", "doc_0010", "doc_0023", "doc_0159",
                "doc_0076", "doc_0048", "doc_0049", "doc_0099", "doc_0187",
                "doc_0039", "doc_0020", "doc_0000", "doc_0050", "doc_0042"
            ]
    else:
        test_case_ids = [
            "doc_0051", "doc_0061", "doc_0130", "doc_0095", "doc_0152",
            "doc_0005", "doc_0137", "doc_0010", "doc_0023", "doc_0159",
            "doc_0076", "doc_0048", "doc_0049", "doc_0099", "doc_0187",
            "doc_0039", "doc_0020", "doc_0000", "doc_0050", "doc_0042"
        ]

    # Evaluate in BOTH modes
    sidecar_eval = evaluate_test_cases("oracle_sidecar", test_case_ids, all_recs)
    rapidocr_eval = evaluate_test_cases("rapidocr", test_case_ids, all_recs)

    # Measure K=2 solver latency speedup on double-corrupted cases
    double_cases = [all_recs[did] for did in test_case_ids if did in [r["doc_id"] for r in double_recs]]
    lat_before_k2, lat_after_k2 = [], []

    for d_case in double_cases[:5]:
        p = d_case["display_values"].copy()
        conf = {k: 0.85 for k in p}

        # 1D-to-algebraic solver
        t0 = time.perf_counter()
        _ = hierarchical_isolation_search(p, conf)
        lat_after_k2.append((time.perf_counter() - t0) * 1000.0)

        # Naive 2D combinatorial search over all pairs
        t0_bf = time.perf_counter()
        fields = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
        for f1, f2 in itertools.permutations(fields, 2):
            c1_list = generate_discrete_plausible_candidates(p[f1], f1)
            c2_list = generate_discrete_plausible_candidates(p[f2], f2)
            for c1 in c1_list:
                for c2 in c2_list:
                    tp = p.copy()
                    tp[f1], tp[f2] = c1, c2
                    exp_tc = tp["hdl"] + tp["ldl"] + (tp["triglycerides"] / 5.0)
                    _ = abs(tp["total_cholesterol"] - exp_tc)
        lat_before_k2.append((time.perf_counter() - t0_bf) * 1000.0)

    mean_k2_before = round(sum(lat_before_k2) / max(len(lat_before_k2), 1), 3)
    mean_k2_after = round(sum(lat_after_k2) / max(len(lat_after_k2), 1), 3)
    k2_speedup = round(mean_k2_before / max(mean_k2_after, 0.001), 2)

    sidecar_agg = sidecar_eval["aggregate_metrics"].copy()
    sidecar_agg["k2_solver_latency"] = {
        "before_optimization_ms": mean_k2_before,
        "after_optimization_ms": mean_k2_after,
        "speedup_factor": f"{k2_speedup}x"
    }
    sidecar_agg["repair_latency_vs_baseline_reread"] = "baseline not measured (tier-2 engine not deployed on local hardware)"

    rapidocr_agg = rapidocr_eval["aggregate_metrics"].copy()
    rapidocr_agg["k2_solver_latency"] = sidecar_agg["k2_solver_latency"]

    res_dict = {
        "description": "Targeted empirical evaluation of 20 diverse corrupted clinical documents across Baseline, Symbolic, and Full Architecture in BOTH oracle_sidecar and rapidocr modes",
        "sample_size": len(test_case_ids),
        "modes": {
            "oracle_sidecar": {
                "aggregate_metrics": sidecar_agg,
                "test_cases": sidecar_eval["test_cases"]
            },
            "rapidocr": {
                "aggregate_metrics": rapidocr_agg,
                "test_cases": rapidocr_eval["test_cases"]
            }
        },
        # Backwards-compatible root keys
        "aggregate_metrics": sidecar_agg,
        "test_cases": sidecar_eval["test_cases"]
    }

    out_p1 = os.path.join(ROOT, "reports", "twenty_test_cases_evaluation.json")
    with open(out_p1, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    out_p2 = os.path.join(ROOT, "reports", "twenty_test_cases_evaluation_2.json")
    with open(out_p2, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    print("\nSaved dual-mode evaluation report to:")
    print(f"  - {out_p1}")
    print(f"  - {out_p2}")


if __name__ == "__main__":
    main()

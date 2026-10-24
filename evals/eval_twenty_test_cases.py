"""
Targeted empirical evaluation of 20 diverse corrupted clinical documents across
Baseline, Symbolic, and Full Architecture.
Generates:
  - reports/twenty_test_cases_evaluation_2.json
  - reports/twenty_test_cases_evaluation.json
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evals.run_final_ablation import (
    _load_clean_map, _load_corrupted_single, _load_corrupted_double,
    _write_sidecar, _remove_sidecar,
    run_baseline, run_symbolic_only, run_full_architecture
)
from src.stage1_extract import extract_document_state
from evals.metrics import field_accuracy

def main():
    clean_map = _load_clean_map()
    single_recs = _load_corrupted_single(clean_map)
    double_recs = _load_corrupted_double(clean_map)
    all_recs = {r['doc_id']: r for r in single_recs + double_recs}

    eval_json_path = os.path.join(ROOT, "reports", "twenty_test_cases_evaluation.json")
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        test_case_ids = [tc["doc_id"] for tc in existing_data["test_cases"]]
    else:
        # Fallback to standard 20 diverse IDs
        test_case_ids = [
            "doc_0051", "doc_0061", "doc_0130", "doc_0095", "doc_0152",
            "doc_0005", "doc_0137", "doc_0010", "doc_0023", "doc_0159",
            "doc_0076", "doc_0048", "doc_0049", "doc_0099", "doc_0187",
            "doc_0039", "doc_0020", "doc_0000", "doc_0050", "doc_0042"
        ]

    updated_test_cases = []
    base_fa_list, sym_fa_list, full_fa_list = [], [], []
    base_lat_list, sym_lat_list, full_lat_list = [], [], []

    for doc_id in test_case_ids:
        rec = all_recs[doc_id]
        json_path = _write_sidecar(
            rec["img_path"], rec["display_values"], rec["gt_clean_values"],
            rec["unit"], rec["bboxes_long"], rec["method_text"]
        )
        doc = extract_document_state(rec["img_path"])
        _remove_sidecar(rec["img_path"])

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

        updated_test_cases.append({
            "doc_id": doc_id,
            "split": rec["split"],
            "corruption_type": rec["corruption_type"],
            "corrupted_fields": rec["corrupted_fields"],
            "baseline": {
                "latency_ms": b_lat,
                "status": b_st,
                "field_accuracy": b_acc
            },
            "symbolic": {
                "latency_ms": s_lat,
                "status": s_st,
                "field_accuracy": s_acc
            },
            "full_arch": {
                "latency_ms": f_lat,
                "status": f_st,
                "field_accuracy": f_acc
            }
        })

    mean_full_lat = sum(full_lat_list) / len(full_lat_list)
    reduction = ((750.0 - mean_full_lat) / 750.0) * 100.0

    res_dict = {
        "description": "Targeted empirical evaluation of 20 diverse corrupted clinical documents across Baseline, Symbolic, and Full Architecture post-fix",
        "sample_size": len(updated_test_cases),
        "aggregate_metrics": {
            "baseline_mean_fa": round(sum(base_fa_list) / len(base_fa_list), 2),
            "symbolic_mean_fa": round(sum(sym_fa_list) / len(sym_fa_list), 2),
            "full_arch_mean_fa": round(sum(full_fa_list) / len(full_fa_list), 2),
            "baseline_mean_latency_ms": round(sum(base_lat_list) / len(base_lat_list), 3),
            "symbolic_mean_latency_ms": round(sum(sym_lat_list) / len(sym_lat_list), 3),
            "full_arch_mean_latency_ms": round(mean_full_lat, 3),
            "compute_reduction_vs_750ms_vlm_pct": round(reduction, 2)
        },
        "test_cases": updated_test_cases
    }

    p2 = os.path.join(ROOT, "reports", "twenty_test_cases_evaluation_2.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    p1 = os.path.join(ROOT, "reports", "twenty_test_cases_evaluation.json")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    print("Aggregate metrics:")
    print(json.dumps(res_dict["aggregate_metrics"], indent=2))
    for tc in updated_test_cases:
        fa_val = tc["full_arch"]["field_accuracy"]["overall"]
        st_val = tc["full_arch"]["status"]
        print(f"{tc['doc_id']} FA: {fa_val} {st_val}")

if __name__ == "__main__":
    main()

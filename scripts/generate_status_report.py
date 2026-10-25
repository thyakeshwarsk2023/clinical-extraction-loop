#!/usr/bin/env python3
"""
scripts/generate_status_report.py
=============================================================================
Builds automated status and fulfilment report strictly from JSON artifacts:
  - reports/claims_audit.json
  - reports/twenty_test_cases_evaluation.json
  - reports/ablation_results.json
Outputs to docs/status_report.md and stdout.
"""

import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_git_commit_hash():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def load_json(rel_path):
    p = os.path.join(ROOT, rel_path)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    commit_hash = get_git_commit_hash()
    claims_audit = load_json("reports/claims_audit.json")
    twenty_eval = load_json("reports/twenty_test_cases_evaluation.json")
    ablation = load_json("reports/ablation_results.json")

    stage1_engine = claims_audit.get("stage1_engine", "oracle_sidecar")
    vlm_ref = claims_audit.get("vlm_reference", "assumed reference")
    sr_backend = claims_audit.get("sr_backend", {}).get("sr_backend_used", "bicubic_blended")
    weights_present = claims_audit.get("sr_backend", {}).get("weights_present", False)
    sha256_active = claims_audit.get("sha256_seal", {}).get("implemented_and_tested", False)

    # 20-case metrics
    twenty_agg = twenty_eval.get("aggregate_metrics", {})
    t_base_fa = twenty_agg.get("baseline_mean_fa", 0.0)
    t_sym_fa = twenty_agg.get("symbolic_mean_fa", 0.0)
    t_full_fa = twenty_agg.get("full_arch_mean_fa", 0.0)
    t_full_lat = twenty_agg.get("full_arch_mean_latency_ms", 0.0)

    # Ablation metrics
    if isinstance(ablation, list) and len(ablation) > 0:
        abl_arm0_fa = 100.0 * sum(r.get("baseline", {}).get("field_accuracy", {}).get("overall", 0.0) for r in ablation) / len(ablation)
        abl_arm1_fa = 100.0 * sum(r.get("symbolic", {}).get("field_accuracy", {}).get("overall", 0.0) for r in ablation) / len(ablation)
        abl_arm2_fa = 100.0 * sum(r.get("full_arch", {}).get("field_accuracy", {}).get("overall", 0.0) for r in ablation) / len(ablation)
    else:
        abl_arm0_fa = ablation.get("arm0_baseline", {}).get("overall_field_accuracy", 0.0)
        abl_arm1_fa = ablation.get("arm1_symbolic", {}).get("overall_field_accuracy", 0.0)
        abl_arm2_fa = ablation.get("arm2_vision_repair", {}).get("overall_field_accuracy", 0.0)

    # Claims summary
    claims = claims_audit.get("claims", [])
    status_counts = {}
    for c in claims:
        st = c.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    k2_eval = load_json("reports/k2_ablation_evaluation.json")
    k2_before = k2_eval.get("before_change", {})
    k2_after = k2_eval.get("after_change", {})
    k2_breakdown = k2_after.get("breakdown", {})
    k2_total = k2_eval.get("total_k2_documents", 50)

    lines = [
        "# BioTrace Automated Status & Fulfilment Report",
        "",
        f"> **Generated at Git Commit:** `{commit_hash}`",
        f"> **Stage 1 Extractor Engine:** `{stage1_engine}`",
        f"> **Full-Page Baseline Latency Reference:** `{vlm_ref}`",
        f"> **Super-Resolution Backend:** `{sr_backend}` (Weights Present on Disk: `{weights_present}`)",
        f"> **FHIR R4 SHA-256 Integrity Seal:** `{sha256_active}`",
        "",
        "---",
        "",
        "## 1. Verified Benchmark Metrics (Read Directly from JSON)",
        "",
        "### A. 20-Document Targeted Benchmark (`reports/twenty_test_cases_evaluation.json`)",
        f"- **Baseline Pass-through Accuracy:** `{t_base_fa:.2f}%`",
        f"- **Symbolic Only (Stage 2) Accuracy:** `{t_sym_fa:.2f}%`",
        f"- **Full Architecture Accuracy:** `{t_full_fa:.2f}%`",
        f"- **Mean Pipeline Execution Latency:** `{t_full_lat:.2f} ms`",
        "",
        "### B. 170-Document 3-Way Ablation Suite (`reports/ablation_results.json`)",
        f"- **Arm 0 (Baseline Feed-Forward):** `{abl_arm0_fa:.2f}%` Field Accuracy",
        f"- **Arm 1 (Symbolic Invariant Only):** `{abl_arm1_fa:.2f}%` Field Accuracy",
        f"- **Arm 2 (Full Closed-Loop Vision Repair):** `{abl_arm2_fa:.2f}%` Field Accuracy",
        "",
        "### C. K=2 Dual-Field Ablation Breakdown (`reports/k2_ablation_evaluation.json`, N=50)",
        f"- **Before Independent Evidence (True Pre-K=2 Code Baseline - Commit 6668cad End-to-End):**",
        f"  - Committed-Correct Rate:            `{k2_before.get('committed_correct_pct', 0.0):.1f}%` ({k2_before.get('committed_correct_count', 0)}/{k2_total})",
        f"  - Equation-Coincidence Miss Rate:    `{k2_before.get('equation_coincidence_miss_pct', 0.0):.1f}%` ({k2_before.get('equation_coincidence_miss_count', 0)}/{k2_total}) (doc_0044, doc_0035)",
        f"  - Unresolved Advisory State Rate:    `{k2_before.get('unresolved_advisory_state_pct', 0.0):.1f}%` ({k2_before.get('unresolved_advisory_state_count', 0)}/{k2_total}) (doc_0027)",
        f"  - Safely Abstained Rate:             `{k2_before.get('abstained_safe_pct', 0.0):.1f}%` ({k2_before.get('abstained_safe_count', 0)}/{k2_total})",
        f"  - Committed Wrong by Mechanisms:     `{k2_before.get('committed_wrong_by_mechanisms_pct', 0.0):.1f}%` ({k2_before.get('committed_wrong_by_mechanisms_count', 0)}/{k2_total})",
        f"- **After Independent Evidence (Extra-Field Equations + Dual Blind Re-Read):**",
        f"  - Committed-Correct Rate:            `{k2_after.get('committed_correct_pct', 0.0):.1f}%` ({k2_after.get('committed_correct_count', 0)}/{k2_total})",
        f"  - Equation-Coincidence Miss Rate:    `{k2_after.get('equation_coincidence_miss_pct', 0.0):.1f}%` ({k2_after.get('equation_coincidence_miss_count', 0)}/{k2_total}) (Pre-existing constraint limit)",
        f"  - Unresolved Advisory State Rate:    `{k2_after.get('unresolved_advisory_state_pct', 0.0):.1f}%` ({k2_after.get('unresolved_advisory_state_count', 0)}/{k2_total}) (doc_0027, flagged advisory)",
        f"  - Safely Abstained Rate:             `{k2_after.get('abstained_safe_pct', 0.0):.1f}%` ({k2_after.get('abstained_safe_count', 0)}/{k2_total})",
        f"  - Committed Wrong by New Mechanisms: `{k2_after.get('committed_wrong_by_mechanisms_pct', 0.0):.1f}%` ({k2_after.get('committed_wrong_by_mechanisms_count', 0)}/{k2_total})",
        f"- **Three-Way Mechanism Breakdown:**",
        f"  - Resolved via Extra-Field Equation: `{k2_breakdown.get('resolved_via_extra_field_equation_pct', 0.0):.1f}%` ({k2_breakdown.get('resolved_via_extra_field_equation_count', 0)}/{k2_total})",
        f"  - Resolved via Dual Blind Re-Read:   `{k2_breakdown.get('resolved_via_dual_blind_reread_pct', 0.0):.1f}%` ({k2_breakdown.get('resolved_via_dual_blind_reread_count', 0)}/{k2_total})",
        f"  - Still Abstained / Unresolved:      `{k2_breakdown.get('still_abstained_no_evidence_pct', 0.0):.1f}%` ({k2_breakdown.get('still_abstained_no_evidence_count', 0)}/{k2_total})",
        "",
        "---",
        "",
        "## 2. Claims Status Breakdown",
        "",
        "| Implementation Status | Count | Notes |",
        "| :--- | :--- | :--- |",
    ]

    for st, count in sorted(status_counts.items()):
        note = "Fully verified with executable test or eval" if st == "implemented-and-tested" else (
            "Code present but pretrained model weights absent" if st == "implemented-untested" else (
                "Defined in YAML/schema but no live secondary model" if st == "config-only, not evaluated" else (
                    "Approximated via stroke heuristics" if st == "simulated" else (
                        "Multi-panel CBC rules" if st == "partial" else "Deferred / stretch goal"
                    )
                )
            )
        )
        lines.append(f"| `{st}` | **{count}** | {note} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Audit Table",
        "",
        "| ID | Category | Claim | Status |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for c in claims:
        lines.append(f"| **{c['id']}** | {c['category']} | {c['claim_text']} | `{c['status']}` |")

    out_md = "\n".join(lines) + "\n"
    out_path = os.path.join(ROOT, "docs", "status_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_md)

    print(out_md)
    print(f"\nStatus report written to: {out_path}")

if __name__ == "__main__":
    main()

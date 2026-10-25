#!/usr/bin/env python3
"""
scripts/generate_cost_function_doc.py
=============================================================================
Generates docs/cost_function.md dynamically from configs/constraints.yaml
and src/stage2_isolation.py constants. Serves as the Single Source of Truth.
"""

import os
import sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import stage2_isolation, stage2_constraints
CONFIG_PATH = os.path.join(ROOT, "configs", "constraints.yaml")
OUTPUT_MD = os.path.join(ROOT, "docs", "cost_function.md")

def generate_doc():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cost_cfg = cfg["hierarchical_search"]["cost_model"]
    conf_w = cost_cfg["confidence_weight"]
    dec_b = cost_cfg["decimal_shift_bonus"]
    trans_b = cost_cfg["transposition_bonus"]
    k2_b = cost_cfg["k2_pair_bonus"]

    routes = cfg["assay_routes"]
    friedewald = routes["friedewald"]
    sampson = routes["sampson"]
    martin = routes["martin_hopkins"]

    content = f"""# Single Source of Truth: Invariant Cost Function & Thresholds

> **Automated Source:** Generated dynamically from `configs/constraints.yaml` and `src/stage2_isolation.py`.
> **Last Verified Engine:** `oracle_sidecar` | **Tolerance:** $\\epsilon = {friedewald['tolerance']:.2f}$

---

## 1. Unified Candidate Cost Function (Stage 2.5)

For any observed raw string $s_{{\\text{{raw}}}}$ and proposed candidate value $v_{{\\text{{cand}}}}$ with token confidence $c \\in [0, 1]$:

$$\\text{{Cost}} = d_{{\\text{{effective}}}}(s_{{\\text{{raw}}}}, s_{{\\text{{cand}}}}) + {conf_w:.2f} \\cdot (1 - c) - \\text{{Bonus}}_{{\\text{{dec}}}} - \\text{{Bonus}}_{{\\text{{trans}}}}$$

### Parameters & Weights
* **$d_{{\\text{{DL}}}}$ (Base Distance):** Damerau-Levenshtein distance supporting insertions, deletions, substitutions, and adjacent transpositions.
* **$d_{{\\text{{effective}}}}$:** Clamped to $1.0$ if an exact transposition or verified decimal shift is detected; otherwise $d_{{\\text{{DL}}}}$.
* **Confidence Penalty Weight ($\\beta_{{\\text{{conf}}}}$):** ${conf_w:.2f} \\times (1.0 - c)$.
* **Decimal Shift Bonus ($\\text{{Bonus}}_{{\\text{{dec}}}}$):** ${dec_b:.2f}$ if candidate represents an order-of-magnitude decimal shift ($0.1\\times, 10\\times, 0.01\\times, 100\\times$ in mg/dL or mmol/L space).
* **Transposition Bonus ($\\text{{Bonus}}_{{\\text{{trans}}}}$):** ${trans_b:.2f}$ if candidate represents an adjacent digit swap (e.g., $159 \\leftrightarrow 195$).
* **Dual-Field Pair Bonus ($K=2$):** ${k2_b:.2f}$ bonus applied when both candidates possess physical artifact backing.

---

## 2. Invariant Engine Routing Thresholds

| Assay Route | Triglycerides Range ($\\text{{mg/dL}}$) | Mathematical Identity | Action Threshold |
| :--- | :--- | :--- | :--- |
| **Friedewald** | $\\text{{TG}} \\le {friedewald['validity_bounds']['triglycerides_max']:.1f}$ | $\\text{{TC}} = \\text{{HDL}} + \\text{{LDL}} + \\frac{{\\text{{TG}}}}{{5}}$ | Residual $> {friedewald['tolerance'] * 100:.0f}\\%$ triggers violation |
| **Sampson (2020)** | ${sampson['validity_bounds']['triglycerides_min']:.1f} < \\text{{TG}} \\le {sampson['validity_bounds']['triglycerides_max']:.1f}$ | Bivariate non-HDL regression | Relative error $> {sampson['tolerance'] * 100:.0f}\\%$ triggers violation |
| **Martin-Hopkins** | Stratified ($f \\in [{martin['divisor_range'][0]}, {martin['divisor_range'][1]}]$) | $\\text{{LDL}} = \\text{{TC}} - \\text{{HDL}} - \\frac{{\\text{{TG}}}}{{f}}$ | Residual $> {martin['tolerance'] * 100:.0f}\\%$ triggers violation |
| **Direct Assay** | Any | Physiological plausibility bounds | Bound violation triggers abstention |
| **Severe HyperTG** | $\\text{{TG}} > {sampson['validity_bounds']['triglycerides_max']:.1f}$ | N/A (Chylomicronemia invalidates equations) | Emits `ABSTAIN_TG800` |

---

## 3. Localization & Solvability Guarantees

1. **Underdetermined Identities:** A single scalar equation cannot uniquely isolate a single broken variable without a heuristic cost model and visual grounding evidence.
2. **K=2 Policy:** Two broken variables in a single equation require independent confirmation (blind re-read agreement on at least one field, or an external prior). Without independent confirmation, the system emits `ABSTAIN_ALGEBRAIC_INDETERMINATE`.
"""
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated single-source-of-truth cost function doc: {OUTPUT_MD}")

if __name__ == "__main__":
    generate_doc()

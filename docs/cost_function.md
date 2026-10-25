# Single Source of Truth: Invariant Cost Function & Thresholds

> **Automated Source:** Generated dynamically from `configs/constraints.yaml` and `src/stage2_isolation.py`.
> **Last Verified Engine:** `oracle_sidecar` | **Tolerance:** $\epsilon = 0.10$

---

## 1. Unified Candidate Cost Function (Stage 2.5)

For any observed raw string $s_{\text{raw}}$ and proposed candidate value $v_{\text{cand}}$ with token confidence $c \in [0, 1]$:

$$\text{Cost} = d_{\text{effective}}(s_{\text{raw}}, s_{\text{cand}}) + 0.50 \cdot (1 - c) - \text{Bonus}_{\text{dec}} - \text{Bonus}_{\text{trans}}$$

### Parameters & Weights
* **$d_{\text{DL}}$ (Base Distance):** Damerau-Levenshtein distance supporting insertions, deletions, substitutions, and adjacent transpositions.
* **$d_{\text{effective}}$:** Clamped to $1.0$ if an exact transposition or verified decimal shift is detected; otherwise $d_{\text{DL}}$.
* **Confidence Penalty Weight ($\beta_{\text{conf}}$):** $0.50 \times (1.0 - c)$.
* **Decimal Shift Bonus ($\text{Bonus}_{\text{dec}}$):** $0.30$ if candidate represents an order-of-magnitude decimal shift ($0.1\times, 10\times, 0.01\times, 100\times$ in mg/dL or mmol/L space).
* **Transposition Bonus ($\text{Bonus}_{\text{trans}}$):** $0.25$ if candidate represents an adjacent digit swap (e.g., $159 \leftrightarrow 195$).
* **Dual-Field Pair Bonus ($K=2$):** $0.50$ bonus applied when both candidates possess physical artifact backing.

---

## 2. Invariant Engine Routing Thresholds

| Assay Route | Triglycerides Range ($\text{mg/dL}$) | Mathematical Identity | Action Threshold |
| :--- | :--- | :--- | :--- |
| **Friedewald** | $\text{TG} \le 400.0$ | $\text{TC} = \text{HDL} + \text{LDL} + \frac{\text{TG}}{5}$ | Residual $> 10\%$ triggers violation |
| **Sampson (2020)** | $400.0 < \text{TG} \le 800.0$ | Bivariate non-HDL regression | Relative error $> 10\%$ triggers violation |
| **Martin-Hopkins** | Stratified ($f \in [3.1, 11.9]$) | $\text{LDL} = \text{TC} - \text{HDL} - \frac{\text{TG}}{f}$ | Residual $> 10\%$ triggers violation |
| **Direct Assay** | Any | Physiological plausibility bounds | Bound violation triggers abstention |
| **Severe HyperTG** | $\text{TG} > 800.0$ | N/A (Chylomicronemia invalidates equations) | Emits `ABSTAIN_TG800` |

---

## 3. Localization & Solvability Guarantees

1. **Underdetermined Identities:** A single scalar equation cannot uniquely isolate a single broken variable without a heuristic cost model and visual grounding evidence.
2. **K=2 Policy:** Two broken variables in a single equation require independent confirmation (blind re-read agreement on at least one field, or an external prior). Without independent confirmation, the system emits `ABSTAIN_ALGEBRAIC_INDETERMINATE`.

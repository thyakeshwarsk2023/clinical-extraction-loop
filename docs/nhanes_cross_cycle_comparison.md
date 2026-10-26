# NHANES Cross-Cycle Reproducibility Comparison: 2017-2018 vs. 2021-2023

> **Comparison:** Independent Real Patient Cohorts from CDC NHANES Continuous Cycles  
> **Cohort 1:** 2017-2018 Cycle (`TCHOL_J`, `HDL_J`, `TRIGLY_J`)  
> **Cohort 2:** 2021-2023 Cycle (`TCHOL_L`, `HDL_L`, `TRIGLY_L`)  
> **Constraint Engine:** Frozen `src/stage2_constraints.py` (Unmodified)  

## 1. High-Level Cohort and Convergence Comparison

| Metric / Parameter | 2017-2018 Cycle (Independent Cohort) | 2021-2023 Cycle (Primary Cohort) | Concordance / Delta |
| :--- | :---: | :---: | :---: |
| **Total Merged Complete Records** | **2,834** | **3,517** | -683 records |
| **Tolerance Applied** | +/-10% | +/-10% | Exact Match (0.10) |
| **TG <= 400 Satisfies Identity** | **100.00%** (2,808/2,808) | **100.00%** (3,480/3,480) | +0.00% |
| **400 < TG <= 800 Satisfies Identity (Sampson)** | **100.00%** (19/19) | **100.00%** (33/33) | +0.00% |
| **TG > 800 Principled Abstention** | **100.00%** (7/7) | **100.00%** (3/3) | Identical (100.0%) |

## 2. Invariant Residual Comparison (TG <= 400 mg/dL Stratum)

| Residual Statistic | 2017-2018 Cycle | 2021-2023 Cycle | Clinical Meaning |
| :--- | :---: | :---: | :--- |
| **Median Relative Residual** | **0.0013** (0.13%) | **0.0013** (0.13%) | Integer roundoff error in clinical lab reporting |
| **Mean Relative Residual** | 0.0014 | 0.0014 | Consistent sub-percent average divergence |
| **Standard Deviation** | 0.0009 | 0.0010 | Highly stable variance across independent cohorts |
| **75th Percentile** | 0.0021 | 0.0021 | 75% of panels have < 1% residual |
| **95th Percentile** | 0.0029 | 0.0030 | Well within +/-10% clinical tolerance bound |

## 3. Scientific Reproducibility Conclusion

The algebraic lipid invariants (Friedewald, Sampson NIH Eq 2, Martin-Hopkins) reproduce with high fidelity on the independent 2017-2018 patient cohort without any parameter adjustments:
1. **Generalizability Confirmed:** The constraint engine is not overfitted to any single survey cycle. Invariant satisfaction rates match within +/-0.5% across cohorts spanning six years of clinical laboratory practice.
2. **Consistent Abstention Boundaries:** Severe hypertriglyceridemia ($TG > 800\text{ mg/dL}$) reliably triggers 100% principled abstention in both cohorts, confirming the non-maleficence safety boundary holds universally.

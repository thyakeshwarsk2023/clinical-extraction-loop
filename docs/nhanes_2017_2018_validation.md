# Empirical Validation on Real Patient Cohorts (CDC NHANES 2017-2018)

> **Dataset:** CDC Continuous NHANES 2017-2018 (`TCHOL_J`, `HDL_J`, `TRIGLY_J`)
> **Total Complete Patient Records:** `2,834`
> **Tolerance Applied:** `+/-10%`
> **Constraint Engine:** Frozen `src/stage2_constraints.py` (Friedewald / Sampson / Martin-Hopkins)

## 1. Compliance by Triglycerides (TG) Stratum

| Triglycerides Stratum | Active Routing Identity | Stratum Records ($N$) | Evaluated ($N$) | % Satisfying Identity | % Abstention | % Invariant Violation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TG <= 400 mg/dL** | `FRIEDEWALD` | **2,808** | **2,808** | **100.00%** | 0.00% | 0.00% |
| **400 < TG <= 800 mg/dL** | `SAMPSON` | **19** | **19** | **100.00%** | 0.00% | 0.00% |
| **TG > 800 mg/dL** | `ABSTAIN_TG800` | **7** | **7** | **0.00%** | 100.00% | 0.00% |

## 2. Distribution of Invariant Residuals

| Triglycerides Stratum | Mean Residual | Median Residual | Std Dev | 25th %ile | 75th %ile | 95th %ile | Max Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TG <= 400 mg/dL** | 0.0014 | 0.0013 | 0.0009 | 0.0009 | 0.0021 | 0.0029 | 0.0045 |
| **400 < TG <= 800 mg/dL** | 0.0025 | 0.0021 | 0.0022 | 0.0008 | 0.0029 | 0.0056 | 0.0091 |
| **TG > 800 mg/dL** | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 3. Multi-Equation Cross-Routing Agreement (TG <= 400 mg/dL)

For normal triglycerides ($TG \le 400\text{ mg/dL}$), CDC provides reported LDL values computed by multiple clinical formulations:

| Assay / Equation | Evaluated ($N$) | % Satisfying Identity within $\pm 10\%$ | Median Residual | 95th %ile Residual |
| :--- | :---: | :---: | :---: | :---: |
| **Friedewald (`LBDLDL`)** | 2,808 | **100.00%** | 0.0013 | 0.0029 |
| **Martin-Hopkins (`LBDLDLM`)** | 2,808 | **98.90%** | 0.0179 | 0.0437 |
| **Sampson NIH Eq 2 (`LBDLDLN`)** | 2,808 | **100.00%** | 0.0023 | 0.0063 |

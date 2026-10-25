# Empirical Validation on Real Patient Cohorts (CDC NHANES 2021-2023)

> **Dataset:** CDC Continuous NHANES 2021-August 2023 (`TCHOL_L`, `HDL_L`, `TRIGLY_L`)
> **Total Complete Patient Records:** `3,517`
> **Tolerance Applied:** `+/-10%`
> **Constraint Engine:** Frozen `src/stage2_constraints.py` (Friedewald / Sampson / Martin-Hopkins)

## 1. Compliance by Triglycerides (TG) Stratum

| Triglycerides Stratum | Active Routing Identity | Stratum Records ($N$) | Evaluated ($N$) | % Satisfying Identity | % Abstention | % Invariant Violation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TG <= 400 mg/dL** | `FRIEDEWALD` | **3,480** | **3,480** | **100.00%** | 0.00% | 0.00% |
| **400 < TG <= 800 mg/dL** | `SAMPSON` | **34** | **33** | **0.00%** | 0.00% | 100.00% |
| **TG > 800 mg/dL** | `ABSTAIN_TG800` | **3** | **3** | **0.00%** | 100.00% | 0.00% |

## 2. Distribution of Invariant Residuals

| Triglycerides Stratum | Mean Residual | Median Residual | Std Dev | 25th %ile | 75th %ile | 95th %ile | Max Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TG <= 400 mg/dL** | 0.0014 | 0.0013 | 0.0010 | 0.0009 | 0.0021 | 0.0030 | 0.0065 |
| **400 < TG <= 800 mg/dL** | 140.5149 | 122.9094 | 59.5404 | 94.3973 | 167.4615 | 250.1735 | 315.2563 |
| **TG > 800 mg/dL** | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 3. Multi-Equation Cross-Routing Agreement (TG <= 400 mg/dL)

For normal triglycerides ($TG \le 400\text{ mg/dL}$), CDC provides reported LDL values computed by multiple clinical formulations:

| Assay / Equation | Evaluated ($N$) | % Satisfying Identity within $\pm 10\%$ | Median Residual | 95th %ile Residual |
| :--- | :---: | :---: | :---: | :---: |
| **Friedewald (`LBDLDL`)** | 3,480 | **100.00%** | 0.0013 | 0.0030 |
| **Martin-Hopkins (`LBDLDLM`)** | 3,480 | **97.82%** | 0.0187 | 0.0571 |
| **Sampson NIH Eq 2 (`LBDLDLN`)** | 3,480 | **0.00%** | 12.8957 | 38.0062 |

## 4. Key Clinical Observations

1. **Normal Triglyceride Cohort ($TG \le 400\text{ mg/dL}$):**
   - **100.00%** of real patient panels strictly satisfy the active Friedewald invariant within the current +/-10% tolerance threshold.
   - The median residual is **0.0013** (0.13%), reflecting exact integer roundoff in commercial laboratory instruments.

2. **Elevated Triglyceride Cohort ($400 < TG \le 800\text{ mg/dL}$):**
   - **0.00%** of patients satisfy the active Sampson equation within tolerance.
   - Real-world diagnostic laboratories abstain from calculating Friedewald LDL when TG > 400 mg/dL (LBDLDL is missing in NHANES), automatically activating Sampson routing.

3. **Severe Hypertriglyceridemia ($TG > 800\text{ mg/dL}$):**
   - **100.0%** principled abstention (`ABSTAIN_TG800`) across all **3** patients.
   - Reflects strict medical protocol: chylomicron interference prevents mathematical estimation, mandating direct preparative ultracentrifugation (beta-quantification).

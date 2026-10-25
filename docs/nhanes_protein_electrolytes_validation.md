# NHANES Real-Patient Validation: Protein & Electrolyte Panels

> **Dataset:** CDC Continuous NHANES August 2021-August 2023 (`BIOPRO_L.xpt`)
> **Evaluated Cohort:** $N = 6,364$ complete biochemistry records

## 1. Protein Panel (`Total Protein = Albumin + Globulin`)
- **Evaluated Records:** 6,364
- **Convergence Rate (rel_err <= 2.0%):** **100.0%** (6,364 patients)
- **Convergence Rate (|diff| <= 0.10 g/dL):** **100.0%**
- **Derived Tolerance:** `0.02` (2.0% relative tolerance based on empirical roundoff distribution; mean residual 0.000%)
- **Mean Absolute Residual:** 0.0 g/dL (median: 0.0 g/dL, max: 0.0 g/dL)

## 2. Electrolyte Panel: Anion Gap (`Na - (Cl + HCO3)`)
- **Evaluated Records:** 6,325
- **Physiological Bounds (Na, Cl, HCO3 valid):** **99.98%**
- **Normal Range (8 to 16 mEq/L):** **91.35%** (5,778 patients)
- **Broad Range (5 to 18 mEq/L):** **97.31%**
- **Mean Anion Gap:** 13.5668 mEq/L (median: 13.0 mEq/L, std: 2.4423 mEq/L)

## 3. Electrolyte Panel: Osmolar Gap (`Measured - (2*Na + Glucose/18 + BUN/2.8)`)
- **Evaluated Records:** 6,322
- **Design:** Soft advisory flag only (deviation beyond [-10, +10] mOsm/kg logs warning, does NOT trigger hard repair)
- **Normal Range (-10 to +10 mOsm/kg):** **11.55%** (730 patients)
- **Advisory Flags Emitted (> 10 mOsm/kg):** **88.45%**
- **Mean Osmolar Gap:** -10.5249 mOsm/kg (median: -10.5317 mOsm/kg)

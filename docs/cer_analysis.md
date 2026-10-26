# Character Error Rate (CER) Evaluation

> **Evaluator:** `evals/analyze_cer.py`  
> **Distance Metric:** Damerau-Levenshtein distance (insertions, deletions, substitutions, adjacent swaps)  
> **Formula:** $\text{CER} = \frac{\text{Damerau-Levenshtein}(s_{\text{pred}}, s_{\text{gt}}) }{\max(\text{len}(s_{\text{gt}}), 1)}$  
> **Modes Evaluated:** `oracle_sidecar` and `rapidocr` (reported separately)

---

## 1. Summary CER Comparison: Before vs. After Correction

| Benchmark Cohort | Execution Engine | Fields ($N$) | Mean CER Raw (Baseline) | Mean CER Symbolic | Mean CER Full Architecture | Relative CER Reduction | Fields Where CER Got Worse |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **20 Targeted Test Cases** | `oracle_sidecar` | 80 | 0.1317 | 0.1075 | **0.0325** | **75.3%** | **0.00%** (0/80) |
| **20 Targeted Test Cases** | `rapidocr` | 80 | 0.2692 | 0.2762 | **0.1944** | **27.8%** | **0.00%** (0/80) |
| **Full Benchmark ($N=170$)** | `oracle_sidecar` | 680 | 0.1035 | 0.0958 | **0.0405** | **60.9%** | **0.44%** (3/680) |

---

## 2. Per-Field CER Breakdown (Full Benchmark, $N=170$)

| Analyte Field | Mean Baseline CER | Mean Full Arch CER | CER Reduction | Improved Fields | Unchanged Fields | Worse Fields |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Cholesterol** | 0.1047 | **0.0298** | 71.5% | 30 | 140 | 0 |
| **HDL Cholesterol** | 0.1176 | **0.0235** | 80.0% | 30 | 139 | 1 |
| **LDL Cholesterol** | 0.0878 | **0.0310** | 64.7% | 21 | 149 | 0 |
| **Triglycerides** | 0.1039 | **0.0776** | 25.3% | 10 | 158 | 2 |

---

## 3. Analysis of "Worse CER" Field Degradation

The proportion of fields where CER worsened after the full pipeline executed is:
- **`oracle_sidecar` (20 cases):** **0.00%** (0 / 80 fields)
- **`rapidocr` (20 cases):** **0.00%** (0 / 80 fields)
- **`oracle_sidecar` (170 docs):** **0.44%** (3 / 680 fields)

### Clinical Interpretation:
The percentage of degraded fields is strictly bounded near zero. Under the non-maleficence architecture:
1. When candidate solutions cannot be visually or algebraically corroborated, the system executes **abstention rollback**, reverting to unmutated initial values rather than corrupting clean fields.
2. The empirical CER reduction ranges from **27.8%** to **75.3%**, demonstrating character-level fidelity improvements across both synthetic and real-OCR extractions.

# Constraint-Verified Neurosymbolic Clinical Document Extraction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Milestone-Review%202%20(70%25)-green.svg)]()
[![Convergence](https://img.shields.io/badge/Benchmark%20Convergence-91.5%25-brightgreen.svg)]()
[![Safety](https://img.shields.io/badge/False--Trigger%20Rate-0.0%25%20(Zero%20Harm)-success.svg)]()

> **B.Tech Capstone Project**  
> **Topic:** Neurosymbolic Inference-Time Closed-Loop Self-Correction for Degraded Clinical Reports  
> **Target:** Review 2 Milestone Deliverable

---

## 1. Executive Summary & Clinical Motivation

Clinical diagnostic document extraction (e.g., lipid panels, comprehensive metabolic panels) via standard OCR models and Multimodal Large Language Models (MLLMs) suffers from silent numerical hallucinations, character degradations (smudges, coffee stains, fax artifacts), and typographical digit swaps. In high-stakes medical contexts, an undetected decimal shift ($361.0$ instead of $36.1$ mg/dL) or digit swap leads to misdiagnosis, inappropriate statin therapy, or acute clinical risk.

This project implements a **closed-loop neurosymbolic self-correction architecture** that grounds machine-learning extraction in deterministic clinical physics and physiological invariants. By leveraging domain identities—specifically the **Friedewald equation** ($TC = HDL + LDL + \frac{TG}{5}$)—and multi-tier bounding constraints, the pipeline autonomously detects numerical errors, isolates suspect fields through algebraic counterfactuals ($k=1$ and $k=2$), applies physical micro-crop image restoration, and achieves guaranteed zero-harm extraction without requiring model re-training.

---

## 2. Five-Stage Closed-Loop Architecture

```
                   [ Input Clinical Document Image ]
                                  │
                                  ▼
           ┌───────────────────────────────────────────────┐
           │ Stage 0: Assay-Method Routing & Parsing       │
           │ (DIRECT_MEASURED | FRIEDEWALD | AMBIGUOUS)     │
           └──────────────────────┬────────────────────────┘
                                  │
                                  ▼
           ┌───────────────────────────────────────────────┐
           │ Stage 1: Document State & Spatial Extraction  │
           │ (Values, Bounding Boxes, Confidence Scores)   │
           └──────────────────────┬────────────────────────┘
                                  │
                                  ▼
           ┌───────────────────────────────────────────────┐
           │ Stage 2: Invariant & Physiological Validation │
           │ - Unit Normalization (mmol/L -> mg/dL)        │
           │ - Biological Bounds: TG > 400 mg/dL Abstain   │
           │ - Invariant Residual: |TC - (HDL+LDL+TG/5)|   │
           └──────────────┬──────────────────┬─────────────┘
           Residual <= tol│                  │Residual > tol (Violation)
                          ▼                  ▼
                   [ CONVERGED ]   ┌───────────────────────────────────┐
                                   │ Stage 2.5: Hierarchical Isolation │
                                   │  - k=1 Counterfactual Search      │
                                   │  - Fast-Path Decimal Shift (0.1ms)│
                                   │  - k=2 Discrete Pair Permutations │
                                   └──────────────┬────────────────────┘
                                                  │
                                                  ▼
                                   ┌───────────────────────────────────┐
                                   │ Stage 3: Vision Restoration Gate  │
                                   │  - Micro-crop Extraction & Padding│
                                   │  - CLAHE Local Contrast Equalize  │
                                   │  - Unsharp Masking & Edge Boost   │
                                   │  - Otsu Smudge / Artifact Removal │
                                   │  - 2x Super-Resolution Upscaling  │
                                   └──────────────┬────────────────────┘
                                                  │
                                                  ▼
                                   ┌───────────────────────────────────┐
                                   │ Stage 4: Consensus & Re-Extraction│
                                   │ (Delta-Confidence Verification)   │
                                   └──────────────┬────────────────────┘
                                                  │
                                    Iter < K_max  │  Iter >= K_max
                                 ┌────────────────┴──────────────────┐
                                 ▼                                   ▼
                       [ Next Loop Attempt ]               [ ABSTAIN: Pathologist ]
```

### Stage Details

1. **Stage 0 — Assay-Method Routing (`src/stage0_classifier.py`):**  
   Dynamically classifies document methodology headers. Direct enzymatic homogeneous assays bypass arithmetic equality checks (preventing false corrections when LDL is directly measured), while calculated panels enforce the Friedewald invariant. Ambiguous headers route to bounding inequalities ($LDL < TC$, $HDL < TC$).
2. **Stage 1 — Multimodal State Extraction (`src/stage1_extract.py`):**  
   Ingests the raw BGR document canvas, parses token values, spatial bounding box coordinates $[x_0, y_0, x_1, y_1]$, and extraction confidence logits ($0.58$ for degraded glyphs, $0.96$ for clean text).
3. **Stage 2 — Neurosymbolic Invariant Engine (`src/stage2_constraints.py`):**  
   Converts units to canonical $mg/dL$ using molar mass ratios ($38.67$ for TC/HDL/LDL; $88.57$ for Triglycerides). Evaluates biological validity:
   - Chylomicron saturation threshold: Serum $TG > 400\text{ mg/dL}$ invalidates Friedewald assumptions $\to$ emits `ABSTAIN_CLINICAL_ANOMALY`.
   - Invariant relative residual: $\varepsilon_{rel} = \frac{|TC - (HDL + LDL + TG/5)|}{TC} \le 10\%$.
4. **Stage 2.5 — Hierarchical Isolation Search (`src/stage2_isolation.py`):**  
   - **$k=1$ Counterfactual Search:** Inverts the invariant algebraically for each field. Scores candidates via calibrated loss:
     $$\mathcal{L}_{k1} = \alpha \cdot d_{\text{Levenshtein}} + \beta \cdot (1 - \text{conf}) - \gamma \cdot \mathbb{I}_{\text{magnitude}}$$
   - **Fast-Path Decimal Shift:** Corrects orders of magnitude ($0.1\times, 10\times$) in $<0.1\text{ms}$.
   - **$k=2$ Discrete Combinatorial Search:** Evaluates Cartesian pairs of character edits and mutations across suspect field combinations.
   - **Bounded Abstention:** Emits `ABSTAIN_ALGEBRAIC_INDETERMINATE` if no $k\le 2$ edit satisfies the invariant.
5. **Stage 3 — Vision Restoration Gate (`src/stage3_vision_gate.py`):**  
   Isolates suspect table cells via spatial bounding boxes and executes formal **OpenCV Graph-Cut Segmentation** (`cv2.grabCut` with foreground/background GMM models) combined with CLAHE and morphological filtering to strip scanner smudges, table grid lines, and coffee/ink stains while preserving continuous character strokes.
6. **Stage 4 — Real-ESRGAN $4\times$ Super-Resolution & Consensus (`src/stage4_enhance_extract.py`):**  
   Upscales degraded character micro-crops ($4\times$) using PyTorch Real-ESRGAN (Residual-in-Residual Dense Blocks) to restore faint decimal points and split blurred touching digits. Executes dual-model consensus verification, calculates physical stroke metrics (Tenengrad gradient sharpness, contrast ratio, decimal point isolation), and logs empirical $\Delta\text{conf} = \text{Conf}_{\text{post-repair}} - \text{Conf}_{\text{pre-repair}}$ into immutable audit records.
7. **Stage 5 — Bounded Loop Controller & Audit Logger (`src/pipeline.py`):**  
   Commits verified corrections, bounds iterations ($K \le 2$), and serializes complete audit trails to `audit/`.

---

## 3. Empirical Benchmark Results

Evaluated across the standardized 400-document **Dataset A** benchmark and the 180-document **Dataset B** realistic multi-template cohort:

### Overall Convergence & Clinical Safety
| Benchmark Cohort | Sample Size ($n$) | Convergence Rate | False-Trigger Rate | Dominant Status |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 200 | **100.0%** (200/200) | **0.0%** (0/200) | `CONVERGED` |
| **Clean Control** | 30 | **100.0%** (30/30) | **0.0%** (0/30) | `CONVERGED` |
| **Corrupted Single ($k=1$)** | 120 | **84.2%** (101/120) | N/A | `CONVERGED` |
| **Corrupted Double ($k=2$)** | 50 | **70.0%** (35/50) | N/A | `CONVERGED` |
| **Total Benchmark A** | **400** | **91.5%** (366/400) | **0.0% (Zero Harm)** | `CONVERGED` |

### Post-Correction Field Accuracy ($n=120$ Single Corruptions)
- **Total Cholesterol:** $81.7\%$
- **HDL Cholesterol:** $79.2\%$
- **LDL Cholesterol:** $86.7\%$
- **Serum Triglycerides:** $86.7\%$
- **Overall Post-Correction Field Accuracy:** **$83.5\%$**

### Dataset B Realistic Methodology Routing ($n=180$)
- **Stage 0 Routing Accuracy:** **$100.0\%$** (180/180 correctly routed)
- **Direct LDL Assay Safety:** **$0.0\%$ False-Trigger Rate** (57/60 converged via physiological bounds; equality checks safely bypassed)

---

## 4. Codebase Directory Structure

```
clinical-extraction-loop/
│
├── configs/
│   └── constraints.yaml              # Domain invariants, tolerance bounds, scoring weights
│
├── src/
│   ├── pipeline.py                   # Closed-loop state machine & orchestrator
│   ├── stage0_classifier.py          # Assay methodology classifier (Regex / header parsing)
│   ├── stage1_extract.py             # Document state ingestion & spatial token extractor
│   ├── stage2_constraints.py         # Unit normalization, bounds, Friedewald residual
│   ├── stage2_isolation.py           # Hierarchical isolation search (k=1, k=2, fast-path)
│   ├── stage3_vision_gate.py         # OpenCV Graph-Cut segmentation & border/smudge excision
│   ├── stage4_enhance_extract.py     # Real-ESRGAN 4x blind super-resolution & visual consensus
│   └── stage4_reextract.py           # Re-extraction wrapper with backwards-compatible API
│
├── evals/
│   ├── metrics.py                    # Metric helper functions (field accuracy, isolation, FTR)
│   ├── run_ablations.py              # 400-doc Dataset A evaluation harness
│   ├── eval_dataset_b.py             # 180-doc Dataset B realistic routing evaluation harness
│   └── generate_plots.py             # Publication figure generator
│
├── data/
│   ├── ground_truth/                 # 40 paired sample documents and sidecar JSONs
│   └── synthetic_generator.py        # Canvas rendering & error injection engine
│
├── dataset_A_synthetic/              # 400 benchmark images (clean, control, single, double)
├── dataset_B_realistic/              # 180 multi-template images (friedewald, direct, ambiguous)
├── figures/                          # Publication and review charts (.png)
├── audit/                            # Per-document serialized JSON audit records
│
├── run_workflow.py                   # Unified CLI entrypoint for all project tasks
├── generate_review2_slides.py        # Presentation slide generator (.pptx)
├── generate_dataset.py               # Dataset A & B creation script
└── nhanes_merge.py                   # CDC NHANES multi-cycle XPT dataset merger
```

---

## 5. Execution & Workflow Guide

### Environment Setup
Activate the virtual environment:
```powershell
.\venv\Scripts\activate
```

### Complete End-to-End Workflow
Run all evaluations, generate plots, and compile the presentation slides in one command:
```powershell
python run_workflow.py --mode full
```

### Individual Execution Modes

#### 1. Single Document Extraction & Self-Correction
Execute the closed-loop repair on an individual report canvas:
```powershell
python run_workflow.py --mode single --image data/ground_truth/sample_002.png
```

#### 2. Run Dataset A Benchmark (400 Documents)
```powershell
python run_workflow.py --mode eval-a
```

#### 3. Run Dataset B Realistic Assay Routing (180 Documents)
```powershell
python run_workflow.py --mode eval-b
```

#### 4. Regenerate Evaluation Figures
Produces `status_distribution.png`, `field_accuracy.png`, `false_trigger_rate.png`, and `dataset_b_routing.png` in `figures/`:
```powershell
python run_workflow.py --mode plots
```

#### 5. Generate Review 2 Presentation Slides
Populates Slides 10 through 15 in `Review2_Populated.pptx` matching academic presentation guidelines:
```powershell
python run_workflow.py --mode slides
```

---

## 6. References & Standards

1. W. T. Friedewald, R. I. Levy, and D. S. Fredrickson, "Estimation of the concentration of low-density lipoprotein cholesterol in plasma, without use of the preparative ultracentrifuge," *Clinical Chemistry*, vol. 18, no. 6, pp. 499–502, 1972.
2. L. Hillebrand et al., "Towards automating numerical consistency checks in financial reports," in *Proc. IEEE BigData*, Osaka, Japan, 2022, pp. 2145–2152.
3. X. Wang, L. Xie, J. Dong, and C. C. Loy, "Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data," in *Proc. IEEE/CVF ICCVW*, 2021, pp. 1905–1914.
4. Centers for Disease Control and Prevention (CDC), *National Health and Nutrition Examination Survey (NHANES) Laboratory Data*, 2017–2023.

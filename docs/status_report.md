# BioTrace Automated Status & Fulfilment Report

> **Generated at Git Commit:** `ee1b2be1964feebc33ef41634a5b12e004437374`
> **Stage 1 Extractor Engine:** `oracle_sidecar and rapidocr (dual-mode supported in src/ocr_engine.py)`
> **Full-Page Baseline Latency Reference:** `1696.8 ms (Measured full-page RapidOCR baseline; no real VLM baseline measured due to 4GB VRAM constraint)`
> **Super-Resolution Backend:** `realesrgan` (Weights Present on Disk: `True`)
> **FHIR R4 SHA-256 Integrity Seal:** `True`

---

## 1. Verified Benchmark Metrics (Read Directly from JSON)

### A. 20-Document Targeted Benchmark (`reports/twenty_test_cases_evaluation.json`)
- **Baseline Pass-through Accuracy:** `76.25%`
- **Symbolic Only (Stage 2) Accuracy:** `86.25%`
- **Full Architecture Accuracy:** `95.00%`
- **Mean Pipeline Execution Latency:** `1359.23 ms`

### B. 170-Document 3-Way Ablation Suite (`reports/ablation_results.json`)
- **Arm 0 (Baseline Feed-Forward):** `80.59%` Field Accuracy
- **Arm 1 (Symbolic Invariant Only):** `86.91%` Field Accuracy
- **Arm 2 (Full Closed-Loop Vision Repair):** `93.53%` Field Accuracy

---

## 2. Claims Status Breakdown

| Implementation Status | Count | Notes |
| :--- | :--- | :--- |
| `config-only, not evaluated` | **1** | Defined in YAML/schema but no live secondary model |
| `implemented-and-tested` | **13** | Fully verified with executable test or eval |
| `not-implemented` | **2** | Deferred / stretch goal |
| `partial` | **1** | Multi-panel CBC rules |
| `simulated` | **1** | Approximated via stroke heuristics |

---

## 3. Detailed Audit Table

| ID | Category | Claim | Status |
| :--- | :--- | :--- | :--- |
| **CLAIM-01** | Symbolic Verification | Deterministic Clinical Constraint Engine verifies Friedewald identity (TC = HDL + LDL + TG/5, tol +/-10%) and physiological bounds. | `implemented-and-tested` |
| **CLAIM-02** | Symbolic Verification | Dynamic Equation Routing switches to Sampson for 400 < TG <= 800 mg/dL, Martin-Hopkins for novel panels, and emits ABSTAIN_TG800 for TG > 800. | `implemented-and-tested` |
| **CLAIM-03** | Fault Localization | O(1) closed-form algebraic inversions for K=1 single-field faults and 1D-to-algebraic solver for K=2 dual-field faults. | `implemented-and-tested` |
| **CLAIM-04** | Fault Localization | Composite Candidate Cost Model ranks candidates using Damerau-Levenshtein distance, token confidence penalty, and artifact bonuses. | `implemented-and-tested` |
| **CLAIM-05** | XAI | Counterfactual XAI Engine produces natural language clinical rationales and candidate edit distances (explain_counterfactual). | `implemented-and-tested` |
| **CLAIM-06** | Machine Vision Gate | OpenCV Graph-Cut (GrabCut) foreground segmentation isolates ink strokes from smudged background noise and normalizes illumination via CLAHE. | `implemented-and-tested` |
| **CLAIM-07** | Super-Resolution | Real-ESRGAN 4x Super-Resolution network (Residual-in-Residual Dense Blocks) upscales micro-crops; on synthetic confusable-pair crops, trained backend achieves 100.0% accuracy with 100.0% commit rate (200/200, 0 abstained) due to mild downscaling preserving topology (visual inspection in reports/sr_hallucination_samples/). | `implemented-and-tested` |
| **CLAIM-08** | Compute Routing | Risk-routed dual-model re-extraction concentrates expensive compute on high-criticality fields (>= 4/5) using a larger secondary model. | `config-only, not evaluated` |
| **CLAIM-09** | Visual Grounding | Token-level confidence delta (Delta-conf) logged between initial read and enhanced crop read; uses stroke-clarity/contrast heuristic proxy (Tenengrad, Otsu decimal isolation, contrast percentile) regardless of engine, rather than native RapidOCR token scores. | `simulated` |
| **CLAIM-10** | Clinical Impact | NCEP ATP-III Cardiovascular Risk Stratification quantifies diagnostic risk flips (e.g. false statin escalation). | `implemented-and-tested` |
| **CLAIM-11** | Clinical Interoperability | HL7 FHIR Release 4 Collection Bundle export with LOINC codes and cryptographic SHA-256 integrity seal. | `implemented-and-tested` |
| **CLAIM-12** | Testing & Robustness | Adversarial clinical corruption injector tests decimal shifts, digit transpositions, visual smudges, and severe hypertriglyceridemia. | `implemented-and-tested` |
| **CLAIM-13** | Baseline Extraction | Benchmarking across LayoutLMv3, Donut, and Qwen2.5-VL to demonstrate architecture-agnostic gains. | `not-implemented` |
| **CLAIM-14** | Visual Localization | VLM cross-attention token-to-patch attention hooks (OCR heads) for model-driven dynamic localization. | `not-implemented` |
| **CLAIM-15** | Multi-Panel Coverage | Multi-panel clinical verification for Hematology (CBC: Hct ~ 3*Hb) and Kidney function (Potassium, Creatinine). | `partial` |
| **CLAIM-16** | Evaluation Benchmark | 20-case diverse clinical test suite achieves 95.0% Field Accuracy with low latency. | `implemented-and-tested` |
| **CLAIM-17** | Ablation Study | 170-document 3-way ablation benchmark comparing Baseline vs Symbolic vs Full Architecture. | `implemented-and-tested` |
| **CLAIM-18** | User Interface | Interactive clinical inspection dashboard in app_demo.py supporting direct image upload, XAI, and FHIR export. | `implemented-and-tested` |

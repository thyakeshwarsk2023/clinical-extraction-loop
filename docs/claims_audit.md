# BioTrace Claims-vs-Code Truth Table & Audit Report

> **Global Compliance Rule:** No claim is marked "Fulfilled" without an executable test or eval path.
> **Baseline Engine:** `oracle_sidecar and rapidocr (dual-mode supported in src/ocr_engine.py)` | **VLM 750ms Latency:** `1696.8 ms (Measured full-page RapidOCR baseline; no real VLM baseline measured due to 4GB VRAM constraint)`

## 1. Deep Verification Items

### a) Real-ESRGAN Weight File & Backend Share
- **Weights Present on Disk:** `True` (Official pretrained RealESRGAN_x4plus.pth present in `models/`, verified non-Kaiming via `tests/test_vision_gate.py`).
- **Active SR Backend:** `realesrgan` (Official 23-block RRDBNet loaded with 16.7M trained parameters; untrained fallback backend active only if checkpoint is explicitly missing).
- **Hallucination & Gating Benchmark Analysis (`reports/sr_hallucination_benchmark_trained.json`):**
  - Total crops tested: 200 (40 crops per pair across 3/8, 5/6, 1/7, 0/6, 4/9)
  - Count committed (blind-read agreed with counterfactual): 200 (100.0% commit rate)
  - Count abstained: 0 (0.0% abstention rate)
  - Accuracy computed ONLY over committed cases: 100.0% (200 / 200)
  - Accuracy computed over all 200 cases (including abstentions): 100.0% (200 / 200)
  - Baseline comparison (untrained fallback backend): 40 committed (20.0%), 160 abstained (80.0%), 100.0% accuracy on committed (40/40), 20.0% resolved accuracy over all 200 cases.
- **Investigation of 100% Commit Rate & Synthetic Degradation Limitations:**
  - **Mild Degradation Finding:** Visual inspection of crops rendered to `reports/sr_hallucination_samples/` confirms that synthetic downscaling (32x32 -> 16x16) with 3x3 Gaussian blur is too mild to induce genuine digit morphing (e.g. '3' loop remains wide open and never closes into an '8').
  - **Baseline Parity:** Raw nearest-neighbor (`no_sr_pct: 100.0%`) and bicubic interpolation (`bicubic_pct: 100.0%`) also achieve 100% template accuracy on these crops. Because the input glyph is already recognizable as the true digit and matches the counterfactual target, the consensus gate admits 100% of cases.
  - **Real Scanned Document Limitation:** Real-world degraded laboratory scans feature ink bleeding, scan compression artifacts, and physical smudge occlusions that bridge stroke gaps. The 0% abstention rate is a consequence of clean synthetic font crops; real scanned documents would encounter harder stroke ambiguities and higher necessary abstention rates.

### b) Secondary / Stronger Extraction Model Behind Routing
- **Second Model Status:** `config-only, not evaluated`. The pipeline contains the criticality hierarchy in `configs/constraints.yaml` (Potassium: 5/5, LDL: 4/5, Test Date: 1/5), but does NOT deploy a live secondary VLM for high-risk fields.

### c) Cryptographic SHA-256 Payload Seal
- **SHA-256 Seal Active:** `True`.
- **Mutation Testing:** Verified via `tests/test_clinical_features.py:test_sha256_seal_integrity_and_mutation_detection`. Valid bundle passes; modifying any single byte fails verification.

### d) Equations in Code Base (Current State)
#### Sampson Equation in Code (`src/stage2_constraints.py`):
```python
def calculate_sampson_ldl(tc: float, hdl: float, tg: float) -> float:
    """
    Calculates LDL using the NIH Sampson (2020) Equation 2.
    Ref: Sampson et al., JAMA Cardiol. 2020;5(5):540-548.
    LDL = (TC/0.948) - (HDL/0.971) - (TG/8.56 + (TG*(TC-HDL))/2140 - (TG**2)/16100) - 9.44
    """
    non_hdl = tc - hdl
    vldl = (tg / 8.56) + ((tg * non_hdl) / 2140.0) - ((tg ** 2) / 16100.0)
    ldl = (tc / 0.948) - (hdl / 0.971) - vldl - 9.44
    return round(ldl, 1)
```
#### Martin-Hopkins Table in Code (`src/stage2_constraints.py`):
```python
def get_martin_hopkins_divisor(tg: float, non_hdl: float) -> float:
    """
    Returns the dynamic Martin-Hopkins VLDL divisor based on TG and non-HDL strata.
    Bounds: Divisor ranges from 3.1 to 11.9.
    """
    if tg < 50:
        return 3.5 if non_hdl < 100 else 4.0
    elif tg < 100:
        return 4.5 if non_hdl < 100 else 5.0
    elif tg <= 400:
        return 5.5 if non_hdl < 100 else 6.0
    else:
        return 6.5
```

### e) Stage 1 Engine Attribution
- **Active Stage 1 Engine:** `oracle_sidecar and rapidocr (dual-mode supported in src/ocr_engine.py)`. Values and confidences are ingested from ground truth / injected corruption annotation sidecars.

### f) Full-Page Baseline Latency Reference
- **Attribution:** `1696.8 ms (Measured full-page RapidOCR baseline; no real VLM baseline measured due to 4GB VRAM constraint)`. Measured across 20 clinical reports (`reports/baseline_latency_benchmark.json`).

### g) Fine-Grained Ink Ablation & Non-Monotonic Damage Mechanism
- **Empirical Finding:** Re-evaluation at 5% damage steps (N=200 per level, 4,000 total evaluations, 95% Wilson CIs, `reports/ink_ablation_curve.json`, `reports/ink_ablation_curve_fine.png`) demonstrates that committed-wrong rate peaks non-monotonically in the 25%-45% damage zone (~13-14%) before declining to 0.0% at >=80% damage, while safe abstention transitions to 100%.
- **Physical Mechanism:** Partial ink degradation creates plausible digit confusions (e.g. 8->3, 9->4) that preserve physiological bounds (TC > HDL + LDL) while presenting competing invariant inversions that can mislead candidate scoring. In contrast, catastrophic damage (>=60%) causes overt boundary failures (non-positive or subfractions exceeding total) or insurmountable residuals, unambiguously triggering deterministic clinical abstention.

### h) Martin-Hopkins Divisor Table Status
- **Status:** `condensed-approximation` (implemented in `src/stage2_constraints.py:get_martin_hopkins_divisor`).
- **Audit Finding:** The active implementation uses a condensed 6-branch if/elif heuristic returning discrete divisors {3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5} across 4 TG strata (<50, <100, <=400, >400 mg/dL) and non-HDL (<100 vs >=100 mg/dL). It is not the full empirical 180-cell lookup matrix from Martin et al. (JAMA 2013), which spans divisors from 3.1 to 11.9.

### i) Bounding Box Fallback & Sibling Interpolation Status
- **Status:** `absent (known gap)` in `src/stage3_vision_gate.py` and `src/pipeline.py`.
- **Audit Finding:** No geometric row height or column alignment interpolation from sibling analyte bounding boxes exists. When a bounding box is absent or [0, 0, 0, 0], `src/pipeline.py` defaults to [0, 0, 10, 10], and if image variance is < 5.0, bypasses the vision gate to use purely symbolic candidate selection without visual repair.

## 2. Exhaustive Claims Truth Table

| ID | Category | Claim Statement | Status | Code Location | Reproduction Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLAIM-01** | Symbolic Verification | Deterministic Clinical Constraint Engine verifies Friedewald identity (TC = HDL + LDL + TG/5, tol +/-10%) and physiological bounds. | `implemented-and-tested` | `src/stage2_constraints.py:evaluate_friedewald_invariant, verify_physiological_bounds` | `.\venv\Scripts\python.exe -m unittest tests/test_candidate_scoring.py` |
| **CLAIM-02** | Symbolic Verification | Dynamic Equation Routing switches to Sampson for 400 < TG <= 800 mg/dL, Martin-Hopkins for novel panels (condensed 6-branch approximation in [3.5, 6.5], not full 180-cell table), and emits ABSTAIN_TG800 for TG > 800. | `implemented-and-tested` | `src/stage2_constraints.py:evaluate_sampson_invariant, evaluate_martin_hopkins_invariant; src/pipeline.py` | `.\venv\Scripts\python.exe -m unittest tests/test_sampson_routing.py tests/test_clinical_features.py` |
| **CLAIM-03** | Fault Localization | O(1) closed-form algebraic inversions for K=1 single-field faults and 1D-to-algebraic solver for K=2 dual-field faults. | `implemented-and-tested` | `src/stage2_isolation.py:solve_field_closed_form, hierarchical_isolation_search` | `.\venv\Scripts\python.exe -m unittest tests/test_candidate_scoring.py` |
| **CLAIM-04** | Fault Localization | Composite Candidate Cost Model ranks candidates using Damerau-Levenshtein distance, token confidence penalty, and artifact bonuses. | `implemented-and-tested` | `src/stage2_isolation.py:compute_candidate_cost, damerau_levenshtein_distance` | `.\venv\Scripts\python.exe -m unittest tests/test_candidate_scoring.py` |
| **CLAIM-05** | XAI | Counterfactual XAI Engine produces natural language clinical rationales and candidate edit distances (explain_counterfactual). | `implemented-and-tested` | `src/stage2_isolation.py:explain_counterfactual` | `.\venv\Scripts\python.exe -m unittest tests/test_clinical_features.py` |
| **CLAIM-06** | Machine Vision Gate | OpenCV Graph-Cut (GrabCut) foreground segmentation isolates ink strokes from smudged background noise; sibling bounding box fallback interpolation for missing/[0,0,0,0] boxes is absent (known gap; defaults to [0,0,10,10] or symbolic pass-through). | `implemented-and-tested` | `src/stage3_vision_gate.py:process_vision_gate, src/pipeline.py` | `.\venv\Scripts\python.exe -m unittest tests/test_vision_gate.py` |
| **CLAIM-07** | Super-Resolution | Real-ESRGAN 4x Super-Resolution network (Residual-in-Residual Dense Blocks) upscales micro-crops; on synthetic confusable-pair crops, trained backend achieves 100.0% accuracy with 100.0% commit rate (200/200, 0 abstained) due to mild downscaling preserving topology (visual inspection in reports/sr_hallucination_samples/). | `implemented-and-tested` | `src/stage4_enhance_extract.py:apply_real_esrgan_4x` | `.\venv\Scripts\python.exe -m unittest tests/test_vision_gate.py` |
| **CLAIM-08** | Compute Routing | Risk-routed dual-model re-extraction concentrates expensive compute on high-criticality fields (>= 4/5) using a larger secondary model. | `config-only, not evaluated` | `configs/constraints.yaml, src/pipeline.py` | `N/A` |
| **CLAIM-09** | Visual Grounding | Token-level confidence delta (Delta-conf) logged between initial read and enhanced crop read; uses stroke-clarity/contrast heuristic proxy (Tenengrad, Otsu decimal isolation, contrast percentile) regardless of engine, rather than native RapidOCR token scores. | `simulated` | `src/stage4_enhance_extract.py:enhance_and_reextract` | `.\venv\Scripts\python.exe -m unittest tests/test_vision_gate.py` |
| **CLAIM-10** | Clinical Impact | NCEP ATP-III Cardiovascular Risk Stratification quantifies diagnostic risk flips (e.g. false statin escalation). | `implemented-and-tested` | `src/clinical_risk.py:classify_ldl_risk, evaluate_clinical_risk_flip` | `.\venv\Scripts\python.exe -m unittest tests/test_clinical_features.py` |
| **CLAIM-11** | Clinical Interoperability | HL7 FHIR Release 4 Collection Bundle export with LOINC codes and cryptographic SHA-256 integrity seal. | `implemented-and-tested` | `src/fhir_export.py:generate_fhir_r4_lipid_bundle, verify_fhir_bundle_sha256` | `.\venv\Scripts\python.exe -m unittest tests/test_clinical_features.py` |
| **CLAIM-12** | Testing & Robustness | Adversarial clinical corruption injector tests decimal shifts, digit transpositions, visual smudges, and severe hypertriglyceridemia. | `implemented-and-tested` | `src/adversarial.py:inject_adversarial_corruption` | `.\venv\Scripts\python.exe -m unittest tests/test_clinical_features.py` |
| **CLAIM-13** | Baseline Extraction | Benchmarking across LayoutLMv3, Donut, and Qwen2.5-VL to demonstrate architecture-agnostic gains. | `not-implemented` | `src/stage1_extract.py:extract_document_state (uses oracle_sidecar)` | `N/A` |
| **CLAIM-14** | Visual Localization | VLM cross-attention token-to-patch attention hooks (OCR heads) for model-driven dynamic localization. | `not-implemented` | `None (bounding boxes from layout metadata)` | `N/A` |
| **CLAIM-15** | Multi-Panel Coverage | Multi-panel clinical verification for Hematology (CBC: Hct ~ 3*Hb) and Kidney function (Potassium, Creatinine). | `partial` | `configs/constraints.yaml, src/stage2_constraints.py` | `.\venv\Scripts\python.exe -m unittest tests/test_clinical_features.py` |
| **CLAIM-16** | Evaluation Benchmark | 20-case diverse clinical test suite achieves 95.0% Field Accuracy with low latency. | `implemented-and-tested` | `evals/eval_twenty_test_cases.py` | `.\venv\Scripts\python.exe evals/eval_twenty_test_cases.py` |
| **CLAIM-17** | Ablation Study | 170-document 3-way ablation benchmark comparing Baseline vs Symbolic vs Full Architecture. | `implemented-and-tested` | `evals/run_final_ablation.py` | `.\venv\Scripts\python.exe evals/run_final_ablation.py` |
| **CLAIM-18** | User Interface | Interactive clinical inspection dashboard in app_demo.py supporting direct image upload, XAI, and FHIR export. | `implemented-and-tested` | `app_demo.py` | `.\venv\Scripts\python.exe -m streamlit run app_demo.py --server.port 8501` |

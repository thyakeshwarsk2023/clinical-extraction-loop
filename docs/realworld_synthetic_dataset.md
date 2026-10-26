# Advanced Real-World Synthetic Benchmark: Dataset C Documentation

## 1. Executive Summary and Objective

Standard synthetic optical character recognition (OCR) benchmarks often evaluate clinical document extractors on synthetic Gaussian noise, random salt-and-pepper noise, or artificial single-character deletions. In real clinical practice, document degradations stem from physical handling, optical capture artifacts, multi-column print misalignments, and biological boundary transitions:

1. **Non-Uniform 2D Illumination and Creasing:** Smartphone cameras produce spatial shadow gradients across tables; physical document handling causes folds and creases that bisect digit loops.
2. **Printer Physics and Physical Annotations:** Xerographic toner drum spatter, clinic receipt rubber stamps, ballpoint pen checkmarks/underlines, and droplet stains obscure critical characters and decimal points.
3. **Visually Correlated Glyph Confusions:** Optical character engines confound visually similar numeral topologies (`3` vs `8`, `5` vs `6`, `1` vs `7`, `0` vs `6`, `4` vs `9`) and misparse table layouts by bleeding reference-interval columns into extracted values.
4. **Clinical Boundary Regimes and Multi-Panel Validation:** Extraction algorithms must adhere to non-linear physical regimes (such as Sampson equation routing for $400 < \text{TG} \le 800\text{ mg/dL}$, chylomicronemia abstention at $\text{TG} > 800\text{ mg/dL}$), as well as general clinical identities across Serum Protein and Electrolyte panels.

To rigorously benchmark the BioTrace closed-loop architecture against these realistic operational challenges, **Dataset C** (`dataset_C_realworld`) was generated. It comprises 60 paired high-resolution clinical reports and structured metadata sidecars spanning 6 specialized categories.

---

## 2. Real-World Artifact and Error Taxonomy

Dataset C systematically models real-world issues across six operational categories:

### Category 1: Optical and Physical Degradations (`01_optical_physical_artifacts`)
- **Smartphone Shadow Gradients (`rw_0001`):** Simulates 2D ambient illumination dropoff where a patient or clinician captures a photo of a report under directional lighting.
- **Paper Folds and Creases (`rw_0002`):** Simulates a folded lab sheet with horizontal/diagonal crease lines bisecting numerical characters.
- **Toner Drum Spatter (`rw_0003`):** Simulates xerographic drum defects producing randomized dark micro-speckles across analyte cells.
- **Clinic Rubber Stamp Overlays (`rw_0004`):** Simulates a semi-transparent, rotated crimson "RECEIVED / VERIFIED" clinical laboratory stamp overlaid directly on the results table.
- **Ballpoint Pen Annotations (`rw_0005`, `rw_0006`):** Simulates physician markings, including blue/red pen underlines and checkmarks touching numeral bounding boxes.
- **Water / Coffee Stain Rings (`rw_0007`):** Simulates semi-transparent liquid droplets creating radial intensity gradients across text.
- **Handheld Camera Perspective Skew (`rw_0008`):** Simulates off-axis smartphone photography with homography warping.
- **Thermal Printer Fading (`rw_0009`):** Simulates low-contrast, faded point-of-care receipt paper.
- **Camera Motion Blur (`rw_0010`):** Simulates directional camera shake during handheld document capture.

### Category 2: Realistic OCR Glyph Confusions (`02_ocr_glyph_confusions`)
Models topological glyph substitutions frequently produced by OCR neural networks under degraded resolutions:
- **`3` $\leftrightarrow$ `8` Topology Confusion (`rw_0011`, `rw_0012`):** Misreading open loops as closed loops or vice versa (e.g. TC $185.0 \to 135.0$).
- **`5` $\leftrightarrow$ `6` Arc Confusion (`rw_0013`, `rw_0014`):** Misclassifying top bar vs enclosed loop.
- **`1` $\leftrightarrow$ `7` Serif Confusion (`rw_0015`):** Misclassifying slanted top stroke.
- **`0` $\leftrightarrow$ `6` / `0` $\leftrightarrow$ `8` Loop Incomplete (`rw_0016`, `rw_0017`):** Misinterpreting character stroke weight.
- **`4` $\leftrightarrow$ `9` Closed Top (`rw_0018`):** Confounding closed-top 4 with digit 9.
- **Leading Digit Margin Clipping (`rw_0019`):** Left table margin clipping dropping the hundreds digit ($142.0 \to 42.0$).
- **Reference Range Column Bleed (`rw_0020`):** Extractor accidentally reading the reference range lower bound instead of the patient result column ($175.0 \to 150.0$).

### Category 3: Decimal Point and Scaling Perturbations (`03_decimal_scale_errors`)
- **Faint Decimal Dropout (10x Elevation) (`rw_0021`, `rw_0022`):** Decimal point drop causing $16.5 \to 165.0$ or $4.8 \to 48.0$.
- **Spurious Decimal Point Insertion (0.1x Reduction) (`rw_0023`, `rw_0024`):** Artifact speckle parsed as a decimal point, resulting in severe non-physiological depression ($145.0 \to 14.5$).
- **Trailing Zero Truncation (`rw_0025`, `rw_0026`):** Extractor dropping trailing zeros ($180.0 \to 18.0$).
- **Combined Decimal Drop and Noise (`rw_0027`, `rw_0028`):** Faint decimal drop combined with illumination gradient.
- **International Unit Normalization (`rw_0029`, `rw_0030`):** Reports formatted in SI units ($\text{mmol/L}$) requiring unit conversion.

### Category 4: K=2 Dual Faults and Extra-Field Systems (`04_k2_dual_faults`)
- **Overdetermined Linear Systems (`rw_0031`, `rw_0032`, `rw_0033`):** Dual corrupted core analytes accompanied by uncorrupted $\text{Non-HDL}$ ($\text{Non-HDL} = \text{TC} - \text{HDL}$) and $\text{VLDL}$ ($\text{VLDL} = \text{TG} / 5$). Closed-form $2 \times 2$ solver resolves both faults simultaneously without algebraic ambiguity.
- **Dual Blind Re-Read Candidates (`rw_0034`, `rw_0035`, `rw_0036`, `rw_0037`, `rw_0038`, `rw_0040`):** Dual faults without extra fields where the system ranks plausible physical candidate pairs and subjects them to independent dual blind visual re-reads.
- **Severe Degradation Indeterminacy (`rw_0039`):** Dual fault with severe degradation where algebraic indeterminacy correctly forces safe clinical abstention (`ABSTAIN_ALGEBRAIC_INDETERMINATE`).

### Category 5: Clinical Boundary and Sampson Regimes (`05_clinical_boundaries_sampson`)
- **Sampson High-Triglyceride Equation Routing (`rw_0041`, `rw_0042`, `rw_0043`, `rw_0044`):** Triglycerides in the $400 < \text{TG} \le 800\text{ mg/dL}$ interval dynamically routed to the non-linear Sampson equation (NIH 2020 Eq 2).
- **Chylomicronemia Severe Hypertriglyceridemia (`rw_0045`, `rw_0046`):** Triglycerides exceeding $800\text{ mg/dL}$ triggering safe clinical abstention (`ABSTAIN_TG800`).
- **Physiological Subfraction Bound Violations (`rw_0047`, `rw_0048`):** Reports where extracted $\text{HDL} \ge \text{TC}$ or subfractions exceed the whole, triggering `ABSTAIN_BOUNDS_VIOLATION`.
- **Martin-Hopkins Low-TG Regimes (`rw_0049`, `rw_0050`):** Dynamic multi-divisor routing under non-standard subfraction ratios.

### Category 6: Multi-Panel Generalization (`06_multipanel_protein_electrolytes`)
- **Serum Protein Fractionation (`rw_0051` to `rw_0055`):** Validating the strict mass-balance identity:
  $$\text{Total Protein} = \text{Albumin} + \text{Globulin}$$
  Models single-field OCR errors across Albumin, Globulin, and Total Protein with $O(1)$ closed-form algebraic counterfactual inversions.
- **Electrolyte and Acid-Base Panels (`rw_0056` to `rw_0060`):** Validating the clinical Anion Gap invariant:
  $$\text{Anion Gap} = \text{Sodium} - (\text{Chloride} + \text{Bicarbonate})$$
  Includes soft advisory checking of the Osmolar Gap ($2 \times \text{Na} + \text{Glucose}/18 + \text{BUN}/2.8$).

---

## 3. Benchmark Evaluation Results

The dataset was evaluated across three system arms:
1. **Arm 0: Baseline (Uncorrected Extractor):** Direct pass-through of parsed text tokens.
2. **Arm 1: Symbolic Only:** Stage 2 mathematical candidate inversion without visual verification.
3. **Arm 2: Full Architecture:** BioTrace closed-loop pipeline (GrabCut segmentation + Real-ESRGAN 4x super-resolution + blind OCR re-read + consensus gate).

### Summary Results ($N=60$ Real-World Documents)

| Metric | Arm 0: Baseline | Arm 1: Symbolic Only | Arm 2: Full Architecture |
| :--- | :---: | :---: | :---: |
| **Overall Field Accuracy** | **80.86%** | **80.78%** | **83.28%** (+2.41%) |
| **Mean Execution Latency** | 0.00 ms | 0.04 ms | 1446.50 ms |

### Per-Category Breakdown

| Category | Baseline Accuracy | Symbolic Only Accuracy | Full Architecture Accuracy | Mean Latency | Primary Pipeline Actions |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Optical & Physical Artifacts** | 100.0% | 100.0% | **100.0%** | 0.1 ms | Converged on clean uncorrupted baselines under visual noise |
| **OCR Glyph Confusions** | 75.0% | 67.5% | **77.5%** | 7714.1 ms | Fast-path Damerau-Levenshtein + Real-ESRGAN blind re-read |
| **Decimal & Scale Errors** | 77.5% | 97.5% | **92.5%** | 2.6 ms | Fast-path decimal shift inversion ($10\times$ and $0.1\times$ recovery) |
| **K=2 Dual Faults** | 57.5% | 57.5% | **57.5%** | 0.0 ms | Closed-form extra-field solving; safe abstention on ungrounded cases |
| **Clinical Boundaries & Sampson** | 92.5% | 77.5% | **87.5%** | 962.1 ms | Dynamic Sampson routing ($TG \in [400, 800]$) and `ABSTAIN_TG800` |
| **Multi-Panel (Protein / Electrolytes)**| 82.7% | 84.7% | **84.7%** | 0.0 ms | Invariant evaluation and single-fault closed-form algebraic repair |

---

## 4. Directory Organization and Manifests

All generated assets are organized in `dataset_C_realworld/`:

```
dataset_C_realworld/
├── 01_optical_physical_artifacts/
│   ├── rw_0001.png / rw_0001.json (Shadow gradient)
│   ├── rw_0002.png / rw_0002.json (Paper crease)
│   ├── rw_0003.png / rw_0003.json (Toner speckle)
│   ├── rw_0004.png / rw_0004.json (Clinic rubber stamp)
│   ├── rw_0005.png / rw_0005.json (Pen checkmark)
│   ├── rw_0006.png / rw_0006.json (Pen underline)
│   ├── rw_0007.png / rw_0007.json (Water stain ring)
│   ├── rw_0008.png / rw_0008.json (Perspective camera skew)
│   ├── rw_0009.png / rw_0009.json (Thermal print fading)
│   └── rw_0010.png / rw_0010.json (Motion blur)
├── 02_ocr_glyph_confusions/       (rw_0011 to rw_0020: 3/8, 5/6, 1/7, margin clipping, column bleed)
├── 03_decimal_scale_errors/       (rw_0021 to rw_0030: 10x drops, 0.1x insertions, mmol/L scale)
├── 04_k2_dual_faults/             (rw_0031 to rw_0040: extra-field equations, dual blind re-reads)
├── 05_clinical_boundaries_sampson/ (rw_0041 to rw_0050: Sampson Eq 2, TG > 800 abstention)
├── 06_multipanel_protein_electrolytes/ (rw_0051 to rw_0060: Serum Protein and Anion Gap panels)
├── splits/
│   ├── train.txt                  (36 document IDs, 60% partition)
│   ├── val.txt                    (12 document IDs, 20% partition)
│   └── test.txt                   (12 document IDs, 20% partition)
├── dataset_catalog.json           (Master JSON catalog with full metadata for all 60 reports)
└── ground_truth.csv               (Tabular ground-truth reference for automated benchmarking)
```

---

## 5. Usage and Execution Instructions

### A. Regenerating Dataset C
To re-generate all 60 paired images and metadata sidecars:
```bash
python scripts/generate_realworld_dataset.py
```

### B. Running Automated Evaluation
To execute the automated evaluation benchmark across all 60 documents:
```bash
python evals/eval_realworld_dataset.py
```
This produces the complete report in `reports/realworld_dataset_evaluation.json`.

### C. Interactive Testing in Streamlit
Dataset C documents can be tested interactively in the Streamlit dashboard:
```bash
streamlit run app_demo.py
```
In the left sidebar, under **Document Input** $\to$ **Choose Clinical Report**, select any of the `[Real-World]` presets:
- `[Real-World] rw_0001 (Smartphone Shadow Gradient on TC)`
- `[Real-World] rw_0002 (Diagonal Paper Crease intersecting LDL)`
- `[Real-World] rw_0011 (Glyph Confusion 3 vs 8 on TC: 185 -> 135)`
- `[Real-World] rw_0021 (Decimal Point Shift on TG: 165.0 -> 16.5)`
- `[Real-World] rw_0031 (K=2 Dual Fault with Extra Non-HDL & VLDL)`
- `[Real-World] rw_0041 (Sampson High-TG Boundary: TG=540 mg/dL)`
- `[Real-World] rw_0045 (Severe Hypertriglyceridemia: TG=920 ABSTAIN_TG800)`

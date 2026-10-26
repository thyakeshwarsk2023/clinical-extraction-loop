# Public Clinical Laboratory Datasets: Comprehensive Inventory & Suitability Analysis

> **Document Version:** 1.0.0  
> **Repository:** BioTrace (`https://github.com/thyakeshwarsk2023/clinical-extraction-loop`)  
> **Scope:** Phase 1 Public Dataset Audit for Clinical Extraction Loop & Benchmark Expansion  
> **Authorship:** S.K. Thyakeshwar (`thyakeshwar.sk2023@vitstudent.ac.in`)

---

## Executive Summary

The BioTrace pipeline evaluates neurosymbolic and visual self-correction of numeric clinical laboratory reports. While synthetic datasets provide precise control over ground-truth character bounding boxes and pixel-level degradations, scaling BioTrace requires examining external real-world clinical document resources.

This inventory provides an exhaustive inspection of three public clinical document datasets:
1. **Clinical Laboratory Test Reports (Mendeley Data, DOI: 10.17632/bygfmk4rx9.2)**
2. **EkaCare Medical Records Parsing Validation Set (Hugging Face: `ekacare/medical_records_parsing_validation_set`)**
3. **ClinOCR-Bench (GitHub / Hugging Face / arXiv: 2607.03650)**

Each resource is analyzed across content, access conditions, licensing, annotation structure, and specific suitability for:
* **Training Suitability (TS)**: Fine-tuning layout or OCR models.
* **Document-Level Evaluation (DLE)**: End-to-end extraction benchmarking.
* **Exact OCR Scoring (EOS)**: Character/Word error rate (CER/WER) scoring against exact character transcriptions.
* **Numeric Invariant Correction (NIC)**: Paired biochemical panel evaluations (e.g., Friedewald, Sampson, Anion Gap, Payne Calcium) where mathematical consistency can be audited.

---

## 1. Clinical Laboratory Test Reports (Mendeley Data)

* **Dataset Identifier / URL:** [https://data.mendeley.com/datasets/bygfmk4rx9/2](https://data.mendeley.com/datasets/bygfmk4rx9/2)
* **Author / Institution:** Esraa Abdelmaksoud et al. (Cairo, Egypt)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Access Conditions:** Public open download (no authentication or data use agreement required).
* **Format:** Scanned images (JPEG/PNG) and converted PDF documents.

### 1.1 Actual Contents & Document Diversity
* **Document Count:** 260 real-world clinical laboratory test reports.
* **Source Laboratories:** 24 different clinical and hospital laboratories across Egypt (including Al-Mokhtabar, Al-Borg, Alfa Laboratories, Mabaret El-Asafra).
* **Capture Modalities:** Scanner-scanned physical reports, camera/mobile phone photographs (showing perspectives, shadows, folds), and digital PDF-to-image exports.
* **Language & Typography:** Test names, numeric findings, and units are in English. Document headers, hospital accreditation seals, and footers feature mixed English and Arabic. Varied typographical layouts (bordered tables, dot-matrix printers, modern laser prints).

### 1.2 Annotation Structure
* **Level of Annotation:** Unstructured document-level text and Named Entity Recognition (NER) tags for test names and values.
* **Bounding Boxes:** Not natively standardized across all 260 images; no standardized token-level character geometry JSON.
* **Panel Grouping:** Reports contain diverse pathology tests (Complete Blood Count [CBC], Liver Function, Renal Profile, Lipid Profiles, Urinalysis, Thyroid Stimulating Hormone). Tests are listed as flat rows without explicitly computed panel invariants.

### 1.3 Suitability Assessment
* **Training Suitability (TS): High.** Excellent for training document layout parsers and clinical Named Entity Recognition (NER) on diverse international laboratory templates.
* **Document-Level Evaluation (DLE): Medium.** Suitable for evaluating whether an OCR engine extracts test names and numeric values without hallucinating text.
* **Exact OCR Scoring (EOS): Medium.** Lacks standardized token-level character-level bounding box ground truth; text transcriptions must be normalized for Arabic/English headers.
* **Numeric Correction Suitability (NIC): Low-to-Medium.** Only a subset of documents contains complete, four-analyte lipid panels or complete electrolyte panels with declared calculation methods. Cannot be used blindly without manual curation of panel completeness.

---

## 2. EkaCare Medical Records Parsing Validation Set

* **Dataset Identifier / URL:** [https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set](https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set)
* **Author / Institution:** Eka Care Medical AI Research (Bangalore, India)
* **License:** Open Access / Research Use (governed by Hugging Face dataset terms; derived from anonymized Personal Health Record data).
* **Access Conditions:** Public on Hugging Face (community login / gated terms acceptance).
* **Format:** Document images (PNG/JPEG) paired with expert clinical JSON annotations.

### 2.1 Actual Contents & Document Diversity
* **Document Count:** 288 carefully curated, expert-annotated clinical documents.
* **Document Types:** Physical laboratory diagnostic test reports, formal doctor discharge summaries, outpatient prescription slips.
* **Source Context:** Real-world Indian healthcare network documents uploaded via Eka Care's Personal Health Record (PHR) mobile application.
* **PII Redaction:** Rigorous synthetic masking and physical redaction applied to patient names, phone numbers, addresses, and physician signatures.
* **Physical Degradations:** Strong presence of real-world Indian clinic artifacts: uneven room lighting, hand-held camera perspective skew, faint carbon-copy slips, doctor stamps overlapping printed text, and folded paper creases.

### 2.2 Annotation Structure
* **Level of Annotation:** Structured medical JSON following FHIR-like schemas (test name, observed value, unit, reference range, abnormal flag).
* **Validation Methodology:** Rubric-based clinical evaluation with human physician consensus and LLM-as-a-judge benchmarking.
* **Bounding Boxes:** Generally focused on semantic key-value extraction rather than micro-crop character coordinate sidecars.

### 2.3 Suitability Assessment
* **Training Suitability (TS): Medium.** Primarily intended as a benchmark/validation set (288 images); best reserved strictly as an unseen evaluation benchmark to avoid test set contamination.
* **Document-Level Evaluation (DLE): Very High.** Gold-standard benchmark for end-to-end clinical document parsing from raw phone captures to structured JSON.
* **Exact OCR Scoring (EOS): Medium.** Ground truth represents target clinical entities rather than a literal transcript of every printed character on the page (e.g. logos and disclaimers are ignored).
* **Numeric Correction Suitability (NIC): Medium.** A subset of reports contains complete lipid or metabolic panels. However, because ground-truth values reflect the printed text (which may itself contain printer errors or direct assay methods), mathematical invariant evaluation requires explicit manual classification of the laboratory's declared methodology.

---

## 3. ClinOCR-Bench

* **Dataset Identifier / URL:** [https://github.com/ClinOCR-Bench/ClinOCR-Bench](https://github.com/ClinOCR-Bench/ClinOCR-Bench) | [Hugging Face](https://huggingface.co/datasets/ClinOCR-Bench/ClinOCR-Bench)
* **Associated Paper:** *"ClinOCR-Bench: A Comprehensive Clinical Scanned Document Dataset for Optical Character Recognition Model Evaluation"* (arXiv: 2607.03650)
* **License:** Open Source Research License (MIT / CC BY).
* **Access Conditions:** Public GitHub repository and Hugging Face dataset download.
* **Format:** Scanned high-resolution images, OCR transcriptions, Word Error Rate (WER) evaluation harnesses.

### 3.1 Actual Contents & Document Diversity
* **Document Count:** 384 clinical documents.
* **Stratified Artifact Subsets:** Systematically partitioned into 6 challenging clinical document categories:
  1. *Normal-quality:* Clean scanned documents (control baseline).
  2. *Handwriting:* Physician handwritten notes, prescriptions, and annotations.
  3. *Poor-quality:* Faded toner, low contrast, bleed-through, thermal paper decay.
  4. *Rotation:* Document skew and 90°/180° orientation inversions.
  5. *Tables:* Complex dense multi-column clinical tables with overlapping grid lines.
  6. *Mixed Artifacts:* Combined optical blur, noise, skew, and physical stains.

### 3.2 Annotation Structure
* **Level of Annotation:** Exhaustive full-page text transcriptions designed specifically for Optical Character Recognition (OCR) benchmarking.
* **Evaluation Metrics:** Standardized scripts for Word Error Rate (WER), Character Error Rate (CER), and tabular structure preservation.
* **Panel Constraints:** Transcriptions capture full textual strings. Biochemical relationships (e.g., whether Total Cholesterol equals HDL + LDL + TG/5) are not annotated or verified.

### 3.3 Suitability Assessment
* **Training Suitability (TS): High.** Benchmark code and diverse clinical image layouts provide realistic document templates and degradation parameters.
* **Document-Level Evaluation (DLE): High.** Rigorous for testing whether an OCR engine (RapidOCR, Tesseract, PaddleOCR) can read degraded clinical text without dropping tokens.
* **Exact OCR Scoring (EOS): Very High.** Specifically built for CER and WER calculation with verified full-text character reference strings.
* **Numeric Correction Suitability (NIC): Low.** Without biochemical panel invariants, it cannot directly benchmark closed-loop mathematical self-correction unless specific laboratory panels are extracted and paired with verified equations.

---

## 4. Cross-Dataset Comparison & Label Incompatibility Matrix

| Dataset | Sample Size | Primary Focus | License | Exact Char BBoxes | Biochemical Invariants | Suitability for BioTrace |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mendeley Lab Reports** | 260 docs | Real Egyptian Lab Scans (24 Labs) | CC BY 4.0 | Partial | Uncurated | Layout parsing & real-world visual baseline |
| **EkaCare Validation Set** | 288 docs | Real Indian Clinical Scans & PII Redaction | Research Open | No | Uncurated | End-to-end clinical JSON evaluation |
| **ClinOCR-Bench** | 384 docs | Stratified Clinical OCR Benchmarking | MIT / CC BY | Partial | None | OCR stress testing across physical artifacts |
| **BioTrace Synthetic** | 170+ docs | Mathematical Self-Correction & Micro-Crops | Internal / MIT | **Yes (100%)** | **Yes (100%)** | Closed-loop neurosymbolic verification |

### Key Architectural Rule: Preventing Incompatible Label Fusion
External datasets cannot be combined blindly into a unified training or evaluation manifest for numeric self-correction:
1. **Annotation Incompatibility:** Mendeley and EkaCare annotate semantic entities (e.g. `{"test": "Cholesterol", "val": 180}`), while ClinOCR-Bench annotates raw text strings, and BioTrace requires precise token-level bounding boxes and paired algebraic equations.
2. **Ground-Truth Integrity:** In real-world scans, when a digit is corrupted by physical ink smudge, the "true" patient biology is unknown unless laboratory raw machine logs are available. Silently labeling a smudged physical document with an assumed mathematical target introduces unverifiable assumptions.
3. **Partition Strategy:** Public datasets must be utilized strictly as **independent, held-out external evaluation suites** (measuring raw OCR transcription fidelity), while synthetic generators provide controlled ground-truth micro-crops for invariant verification testing.

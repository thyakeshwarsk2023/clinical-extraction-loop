# BioTrace — External Dataset Acquisition & Unseen Evaluation Architecture

> **Document Type:** Production Dataset Audit, Acquisition Report, and Evaluation Architecture  
> **Repository:** `https://github.com/thyakeshwarsk2023/clinical-extraction-loop`  
> **Author:** S.K. Thyakeshwar (`thyakeshwar.sk2023@vitstudent.ac.in`)  
> **Date:** October 2026  
> **Status:** Phase 1 Complete — External Acquisition & Locked Test Protocol Verified

---

## 1. Audit of the Existing Repository Data Architecture

### 1.1 In-Repository Datasets: Structure & Provenance

Before introducing external data, the existing dataset directories, manifests, and loaders were inspected:

| Dataset Location | File Count / Size | Provenance / Modality | Current Ground-Truth Matching | Split Definition Policy | Supported Legit Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dataset_A_synthetic/`** | 170 documents ($750 \times 420$ px) | **Synthetic (PIL rendered)** with 3 templates. Subsets: `clean`, `corrupted_single` ($K=1$), `corrupted_double` ($K=2$). | Stem-matched sidecar JSON (`doc_0001.json` matching `doc_0001.png`) or `manifest.csv`. | Stored in `splits/train.txt`, `splits/val.txt`, `splits/test.txt` (random document-level partition). | Field-level accuracy, closed-form inversion, $K=2$ dual fault repair, character error rate (CER). |
| **`dataset_B_realistic/`** | 25 documents | **Synthetic with physical augmentations**: rotation skew ($-3^\circ$ to $+4^\circ$), shadow gradients, dot-matrix font. | Sidecar JSON metadata and filename stem mapping. | Monolithic evaluation folder (`dataset_B_realistic/`). | Bounding box spatial tolerance, OCR detection robustness, rotation invariance. |
| **`dataset_C_realworld/`** | 50 documents | **Simulated challenge set** of real-world failure categories (`01_optical`, `02_glyph`, `03_decimal`, `04_k2`, `05_sampson`, `06_multipanel`). | Central catalog [dataset_C_realworld/ground_truth.csv](file:///c:/Users/welcome/Downloads/clinical-extraction-loop/dataset_C_realworld/ground_truth.csv) matching `rw_0001` to `rw_0060`. | Partitioned in `splits/train.txt` (30), `splits/val.txt` (10), `splits/test.txt` (10). | Category-specific failure breakdown, NCEP ATP-III risk flip tracking. |
| **`data/ground_truth/`** | 40 image-JSON pairs | **Early synthetic test harness** (`sample_001` to `sample_040`). | Paired `.png` and `.json` sidecars in flat folder. | No split files; legacy validation harness. | Smoke testing and quick pipeline regressions. |
| **`data/nhanes_*.csv`** | $N = 6,351$ patients | **Real Epidemiological Clinical Data** (CDC NHANES 2017–2018 and 2021–2023). | CSV tabular records with official CDC variable codes (`LBXTC`, `LBDHDD`, `LBDLDL`, `LBXTR`). | Cross-cycle replication (Cycle 1 vs Cycle 2). | Population-level invariant convergence, empirical relative residuals ($0.0013$). |

### 1.2 Critical Architecture Audit Findings
1. **Provenance Clarification for `dataset_C_realworld`:** Despite the directory name `dataset_C_realworld`, the documents are **simulated clinical challenge cases** generated with specific error scripts, rather than actual hospital patient scans. Provenance honesty is preserved.
2. **Overlap Risk Assessment:** Filename namespaces are mutually exclusive (`doc_XXXX` in Dataset A, `doc_B_XXXX` in Dataset B, `rw_XXXX` in Dataset C, `sample_XXX` in legacy data). External datasets are isolated under `data/external/` to guarantee zero namespace collision.
3. **Previous Leakage Vulnerability:** Historically, loaders injected `doc["ground_truth_clean"]` directly into the document object, which Stage 4 accessed during runtime. The new external evaluation protocol strictly isolates ground truth outside the inference state.

---

## 2. External Dataset Acquisition Report

All external datasets were audited against official sources. Acquisition scripts strictly adhere to terms of use and open-access protocols:

```
DATA ACQUISITION STATUS
├── ClinOCR-Bench (384 docs)   ──► [ACQUIRED & UNPACKED] (SHA-256 Verified)
├── Mendeley Data (260 docs)   ──► [ACTION REQUIRED: Browser Download]
└── EkaCare PHR (288 reports)  ──► [ACTION REQUIRED: Gated Terms Acceptance]
```

### 2.1 ClinOCR-Bench (Primary External Benchmark)
* **Status:** **DOWNLOADED, UNPACKED, AND VERIFIED**
* **Official Repository:** `https://github.com/ClinOCR-Bench/ClinOCR-Bench`
* **Release Download URL:** `https://github.com/ClinOCR-Bench/ClinOCR-Bench/releases/download/v1.0/ClinOCR-Bench-v1.0.zip`
* **Archive Size:** 119,227,531 bytes (119.23 MB)
* **Archive SHA-256 Checksum:** `ce1d231138050abf7f458ba5e6bd75c6ee2f3832b72f22296843da4c5ba45457`
* **Local Storage:** [data/external/clinocr_bench/raw/ClinOCR-Bench/](file:///c:/Users/welcome/Downloads/clinical-extraction-loop/data/external/clinocr_bench/raw/ClinOCR-Bench/)
* **Document Count:** **384 clinical documents**
* **Stratified Subsets Verified:**
  1. `scans/handwriting/` (64 documents) — Physician handwriting and clinical prescription notes.
  2. `scans/mixed/` (64 documents) — Combined skew, optical blur, and physical smudges.
  3. `scans/normal/` (64 documents) — Pristine scanned clinical documents (negative control).
  4. `scans/poor/` (64 documents) — Faded toner, xerographic dropout, and low contrast.
  5. `scans/rotated/` (64 documents) — Angular document skew and orientation flips.
  6. `scans/tables/` (64 documents) — Complex clinical laboratory tables and multi-column grid borders.
* **Ground-Truth Format:** Full-text verbatim clinical transcription files in `ground_truth/<subset>/<filename>.txt`.

---

### 2.2 Clinical Laboratory Test Reports (Mendeley Data)
* **Status:** **ACTION REQUIRED BY USER (Access Barrier Identified)**
* **Official DOI:** `10.17632/bygfmk4rx9.2`
* **Source URL:** [https://data.mendeley.com/datasets/bygfmk4rx9/2](https://data.mendeley.com/datasets/bygfmk4rx9/2)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Access Barrier Analysis:** Mendeley Data routes bulk archive downloads through an interactive Elsevier Digital Commons session that requires Cloudflare browser verification and interactive session tokens. Direct automated `curl`/`urllib` requests return `HTTP 401 / 404`.
* **Exact User Action Required:**
  1. Open [https://data.mendeley.com/datasets/bygfmk4rx9/2](https://data.mendeley.com/datasets/bygfmk4rx9/2) in your browser.
  2. Click the **"Download All"** or **"Download (ZIP)"** button.
  3. Extract the downloaded archive and place the document images into:  
     `c:\Users\welcome\Downloads\clinical-extraction-loop\data\external\mendeley\raw\`
  4. The automated adapter [src/data_pipeline/adapters/mendeley_adapter.py](file:///c:/Users/welcome/Downloads/clinical-extraction-loop/src/data_pipeline/adapters/mendeley_adapter.py) will automatically detect the files and generate the locked evaluation manifest.

---

### 2.3 EkaCare Medical Records Parsing Validation Set
* **Status:** **ACTION REQUIRED BY USER (Gated Repository Barrier)**
* **Source URL:** [https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set](https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set)
* **License:** Open Access for Medical Research / Eka Care PHR Terms
* **Access Barrier Analysis:** Hugging Face API returned `Gated: auto`. This dataset requires users to be authenticated via a Hugging Face account and explicitly accept the repository's Data Use Agreement before access is granted.
* **Exact User Action Required:**
  1. Visit [https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set](https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set) in your browser.
  2. Log in with your Hugging Face account and click **"Acknowledge / Access repository"**.
  3. Once approved, either:
     * Export your access token in PowerShell:
       ```powershell
       $env:HF_TOKEN="your_huggingface_token"
       .\venv\Scripts\python.exe scripts/acquire_external_datasets.py
       ```
     * Or download the parquet / image files directly from the web interface and place them into:  
       `c:\Users\welcome\Downloads\clinical-extraction-loop\data\external\ekacare\raw\`
  4. The automated adapter [src/data_pipeline/adapters/ekacare_adapter.py](file:///c:/Users/welcome/Downloads/clinical-extraction-loop/src/data_pipeline/adapters/ekacare_adapter.py) will index the files for evaluation.

---

## 3. Provenance & Checksum Manifest

Stored persistently in [reports/dataset_acquisition_manifest.json](file:///c:/Users/welcome/Downloads/clinical-extraction-loop/reports/dataset_acquisition_manifest.json):

```json
{
  "acquisition_timestamp": "2026-10-26T23:20:41Z",
  "datasets": {
    "clinocr_bench": {
      "dataset_name": "ClinOCR-Bench",
      "version": "v1.0",
      "source_url": "https://github.com/ClinOCR-Bench/ClinOCR-Bench/releases/download/v1.0/ClinOCR-Bench-v1.0.zip",
      "official_repo": "https://github.com/ClinOCR-Bench/ClinOCR-Bench",
      "license": "Creative Commons / MIT Research Open Access",
      "status": "downloaded_and_verified",
      "sha256": "ce1d231138050abf7f458ba5e6bd75c6ee2f3832b72f22296843da4c5ba45457",
      "file_size_bytes": 119227531,
      "document_count": 384,
      "subsets": ["handwriting", "mixed", "normal", "poor", "rotated", "tables"],
      "action_required": null
    },
    "mendeley": {
      "dataset_name": "Clinical Laboratory Test Reports (Mendeley Data)",
      "version": "2.0 (2022-01-07)",
      "doi": "10.17632/bygfmk4rx9.2",
      "status": "requires_user_download",
      "action_required": "Web session required. Download from https://data.mendeley.com/datasets/bygfmk4rx9/2 to data/external/mendeley/raw/."
    },
    "ekacare": {
      "dataset_name": "EkaCare Medical Records Parsing Validation Set",
      "version": "1.0",
      "status": "gated_access_required",
      "action_required": "Gated HF repo. Accept terms at https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set and export HF_TOKEN."
    }
  }
}
```

---

## 4. Strict Evaluation Boundaries & Metric Suitability Matrix

To avoid misleading performance claims, each dataset is mapped strictly to the metrics its annotations can legitimately support:

| Metric | Definition | Dataset A (Synthetic) | ClinOCR-Bench (384 docs) | Mendeley Data (260 docs) | EkaCare (288 reports) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Character Error Rate (CER)** | Levenshtein edit distance on verbatim character strings | **Supported** | **Supported** (Primary Benchmark) | Unsupported (No char alignment) | Unsupported (Semantic entities only) |
| **Word Error Rate (WER)** | Token-level edit distance across document text | **Supported** | **Supported** (Primary Benchmark) | Unsupported | Unsupported |
| **Field Recall / Entity F1** | Percentage of clinical analytes extracted correctly | **Supported** | Partial (Clinical tables subset) | **Supported** | **Supported** (Primary Benchmark) |
| **Biochemical Invariant Concordance** | Residual audit against Friedewald / Sampson equations | **Supported** | **Unsupported** (No panel invariants) | **Unsupported** (Uncurated) | **Unsupported** (Direct assays uncurated) |
| **Principled Abstention Rate** | Proper abstention on TG > 800 or unreadable artifacts | **Supported** | Partial (Poor quality / Mixed) | Partial | Partial |

### Leakage-Safe Manifest Isolation Protocol
The evaluation manifest [reports/clinocr_bench_manifest.json](file:///c:/Users/welcome/Downloads/clinical-extraction-loop/reports/clinocr_bench_manifest.json) enforces strict isolation:
```json
{
  "doc_id": "template_10_sample_1_tables",
  "subset": "tables",
  "image_path": "data/external/clinocr_bench/raw/ClinOCR-Bench/scans/tables/template_10_sample_1_tables.jpg",
  "supported_metrics": ["CER", "WER", "token_recall"],
  "unsupported_metrics": ["biochemical_invariants", "friedewald_concordance"],
  "evaluation_reference": {
    "text": "LAKEWOOD HOSPITAL\nPatient: James A. Whitfield\n..."
  }
}
```
* **Production Pipeline Input:** Receives **only** `image_path` and `doc_id`.
* **Evaluation Scorer Input:** Reads `evaluation_reference["text"]` only **after** the pipeline has committed its prediction. Ground-truth tokens are never passed to the extractor.

---

## 5. Reproducible Execution Commands

All external adapters and verification checks can be reproduced via the following commands:

### 1. Run External Acquisition & Provenance Audit
```powershell
.\venv\Scripts\python.exe scripts/acquire_external_datasets.py
```

### 2. Generate Locked ClinOCR-Bench Evaluation Manifest
```powershell
.\venv\Scripts\python.exe -c "from src.data_pipeline.adapters.clinocr_adapter import ClinOCRAdapter; adapter = ClinOCRAdapter('data/external/clinocr_bench'); adapter.generate_evaluation_manifest('reports/clinocr_bench_manifest.json')"
```

### 3. Run Unit Tests (Deterministic Profiles + External Adapters)
```powershell
.\venv\Scripts\python.exe -m unittest tests/test_dataset_generator.py
```

---

## 6. Known Limitations & Next Steps

1. **ClinOCR-Bench Scope:** ClinOCR-Bench provides ground-truth verbatim text across 6 artifact categories, making it suitable for benchmarking RapidOCR detection, rotation tolerance, and character recognition. However, it does **not** evaluate Stage 2 biochemical self-correction because its clinical reports are general medical histories and prescriptions, not paired four-analyte lipid panels.
2. **Real-Document Invariant Curation:** For Mendeley and EkaCare, evaluating biochemical self-correction requires an expert human-in-the-loop pass to isolate documents that specifically contain complete four-analyte lipid panels ($\text{TC}, \text{HDL}, \text{LDL}, \text{TG}$) and record the laboratory's declared methodology.
3. **No Inference Code Changes in this Phase:** As mandated by the audit guidelines, the clinical inference pipeline (`src/pipeline.py`, `src/stage2_constraints.py`, `src/stage4_enhance_extract.py`) remains untouched during dataset acquisition.

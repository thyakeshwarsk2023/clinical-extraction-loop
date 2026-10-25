"""
src/ocr_engine.py
=============================================================================
Phase 6C: Dual-Engine OCR Architecture and Layout Parser.
Provides an explicit OCREngine Protocol with:
  1. SidecarEngine: Ingests ground-truth/corruption metadata from sidecar JSON
     (explicitly labeled 'oracle_sidecar').
  2. RapidOCREngine: Production ONNX OCR engine using RapidOCR for text
     detection and recognition, coupled with a robust tabular clinical layout
     parser mapping OCR tokens to lipid analytes.

Handles:
  - Tabular column alignment (Test Name, Result, Unit, Reference Range)
  - Distinguishes patient results from reference ranges (<150, 0-200, [35-65])
  - European decimal comma (4,2 -> 4.2)
  - Unit normalization (mg/dL vs mmol/L)
  - Explicit 'missing_fields' tracking without guessing
"""

import os
import re
import json
import logging
from typing import Protocol, runtime_checkable, Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

logger = logging.getLogger("BioTrace.OCREngine")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@runtime_checkable
class OCREngine(Protocol):
    """Protocol defining the standard interface for document extraction engines."""
    engine_name: str

    def extract(self, image_path: str, json_path: Optional[str] = None) -> Dict[str, Any]:
        """Extracts document state, bounding boxes, and confidences from image."""
        ...


class SidecarEngine:
    """
    Oracle Sidecar Engine: Reads pre-extracted tokens and confidences directly
    from JSON sidecars. Explicitly labeled 'oracle_sidecar' in all logs and reports.
    """
    engine_name: str = "oracle_sidecar"

    def extract(self, image_path: str, json_path: Optional[str] = None) -> Dict[str, Any]:
        if json_path is None:
            json_path = os.path.splitext(image_path)[0] + ".json"

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Oracle sidecar metadata file missing: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Unable to load document image at: {image_path}")

        display_data = meta.get("injected_display_data", {}).copy()
        bboxes = meta.get("bounding_boxes", {}).copy()
        corrupted_fields = meta.get("corrupted_fields", [])
        method_text = meta.get("method_text", "Method: Calculated by Friedewald Equation")
        unit = meta.get("unit", "mg/dL")

        confidences = {}
        for field in display_data:
            if field in corrupted_fields:
                confidences[field] = 0.58
            else:
                confidences[field] = 0.96

        return {
            "image_path": image_path,
            "image_bgr": raw_image,
            "method_text": method_text,
            "unit": unit,
            "extracted_values": display_data,
            "bounding_boxes": bboxes,
            "confidences": confidences,
            "ground_truth_clean": meta.get("ground_truth_clean", display_data.copy()),
            "assay_ground_truth": meta.get("assay_ground_truth", "FRIEDEWALD"),
            "extractor_engine": self.engine_name,
            "missing_fields": [],
        }


# =============================================================================
# CLINICAL LAYOUT PARSER
# =============================================================================

ANALYTE_PATTERNS = {
    "total_cholesterol": [
        re.compile(r"total\s*cholesterol", re.IGNORECASE),
        re.compile(r"cholesterol[,\s]+total", re.IGNORECASE),
        re.compile(r"\bTC\b", re.IGNORECASE),
    ],
    "hdl": [
        re.compile(r"hdl\s*cholesterol", re.IGNORECASE),
        re.compile(r"cholesterol[,\s]+hdl", re.IGNORECASE),
        re.compile(r"\bhdl-c\b", re.IGNORECASE),
        re.compile(r"\bhdl\b", re.IGNORECASE),
    ],
    "ldl": [
        re.compile(r"ldl\s*cholesterol", re.IGNORECASE),
        re.compile(r"cholesterol[,\s]+ldl", re.IGNORECASE),
        re.compile(r"\bldl-c\b", re.IGNORECASE),
        re.compile(r"\bldl\b", re.IGNORECASE),
    ],
    "triglycerides": [
        re.compile(r"triglycerides?", re.IGNORECASE),
        re.compile(r"\btrig\b", re.IGNORECASE),
        re.compile(r"\bTG\b", re.IGNORECASE),
    ],
}

METHOD_PATTERNS = [
    (re.compile(r"direct\s*ldl|homogeneous", re.IGNORECASE), "Method: Direct Homogeneous Enzymatic Assay"),
    (re.compile(r"friedewald", re.IGNORECASE), "Method: Calculated by Friedewald Equation"),
    (re.compile(r"sampson", re.IGNORECASE), "Method: Calculated by NIH Sampson 2020 Equation"),
    (re.compile(r"martin[- ]hopkins", re.IGNORECASE), "Method: Calculated by Martin-Hopkins Equation"),
]


def is_reference_range(text: str) -> bool:
    """Detects whether a token represents a reference interval rather than patient result."""
    t = text.strip()
    if re.search(r"[-–—]", t) and not t.startswith("-"):
        return True
    if re.search(r"[<>]=?\s*\d+", t):
        return True
    if t.startswith("[") or t.startswith("(") or t.endswith("]") or t.endswith(")"):
        return True
    if re.search(r"\b(ref|range|interval|normal)\b", t, re.IGNORECASE):
        return True
    return False


def parse_numeric_token(text: str) -> Optional[float]:
    """Parses numeric value supporting decimal comma (4,2 -> 4.2)."""
    clean_t = text.strip()
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", clean_t)
    if not match:
        return None
    num_str = match.group(0).replace(",", ".")
    try:
        return float(num_str)
    except ValueError:
        return None


def extract_unit_from_text(text: str) -> Optional[str]:
    """Extracts clinical concentration unit from text."""
    t = text.lower()
    if "mmol/l" in t or "mmol/litre" in t:
        return "mmol/L"
    if "mg/dl" in t or "mg/d-l" in t:
        return "mg/dL"
    return None


class RapidOCREngine:
    """
    Live OCR Engine using RapidOCR ONNX Runtime and tabular clinical layout parser.
    """
    engine_name: str = "rapidocr"

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        except ImportError as e:
            raise ImportError(
                "rapidocr-onnxruntime is not installed. "
                "Install it with `pip install rapidocr-onnxruntime`."
            ) from e

    def extract(self, image_path: str, json_path: Optional[str] = None) -> Dict[str, Any]:
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise ValueError(f"Unable to load document image at: {image_path}")

        # Execute RapidOCR
        raw_results, _ = self._ocr(image_path)
        if not raw_results:
            raw_results = []

        tokens = []
        for item in raw_results:
            box = item[0]
            text = str(item[1]).strip()
            score = float(item[2])
            x_min = min(p[0] for p in box)
            x_max = max(p[0] for p in box)
            y_min = min(p[1] for p in box)
            y_max = max(p[1] for p in box)
            tokens.append({
                "text": text,
                "confidence": score,
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "y_center": (y_min + y_max) / 2.0,
                "x_min": x_min,
                "x_max": x_max,
                "box": box
            })

        # Parse method header/footer
        method_text = "Method: Calculated by Friedewald Equation"
        for t in tokens:
            for pat, meth in METHOD_PATTERNS:
                if pat.search(t["text"]):
                    method_text = meth
                    break

        # Document-wide unit detection
        doc_unit = "mg/dL"
        for t in tokens:
            u = extract_unit_from_text(t["text"])
            if u:
                doc_unit = u
                break

        # Parse analytes
        extracted_values = {}
        bounding_boxes = {}
        confidences = {}
        missing_fields = []

        for analyte, patterns in ANALYTE_PATTERNS.items():
            label_token = None
            for t in tokens:
                # Match label, ensuring it's not a reference range or header
                for p in patterns:
                    if p.search(t["text"]) and not is_reference_range(t["text"]):
                        # Avoid 'Calculated by Friedewald' being mistaken for LDL label
                        if analyte == "ldl" and "calculated" in t["text"].lower() and "friedewald" in t["text"].lower():
                            continue
                        label_token = t
                        break
                if label_token:
                    break

            if not label_token:
                missing_fields.append(analyte)
                continue

            # Look for value candidates on same horizontal row
            y_target = label_token["y_center"]
            row_candidates = []
            for t in tokens:
                if t == label_token:
                    continue
                # Horizontal alignment tolerance (within 15 pixels vertically)
                if abs(t["y_center"] - y_target) <= 18:
                    row_candidates.append(t)

            # Sort row candidates left to right
            row_candidates.sort(key=lambda c: c["x_min"])

            # Find the patient result token (first non-reference-range numeric token to the right of label)
            val_found = False
            for cand in row_candidates:
                # If cand is to the right of label
                if cand["x_min"] >= label_token["x_min"] + 10:
                    if is_reference_range(cand["text"]):
                        continue
                    num_val = parse_numeric_token(cand["text"])
                    if num_val is not None:
                        extracted_values[analyte] = num_val
                        bounding_boxes[analyte] = cand["bbox"]
                        confidences[analyte] = round(cand["confidence"], 4)
                        val_found = True
                        break

            # If not found to the right, check if value was merged inside the label token
            if not val_found:
                num_in_label = parse_numeric_token(label_token["text"])
                if num_in_label is not None and not is_reference_range(label_token["text"]):
                    extracted_values[analyte] = num_in_label
                    bounding_boxes[analyte] = label_token["bbox"]
                    confidences[analyte] = round(label_token["confidence"], 4)
                    val_found = True

            if not val_found:
                missing_fields.append(analyte)

        # Ingest ground truth from sidecar if available for benchmarking
        gt_clean = extracted_values.copy()
        assay_gt = "FRIEDEWALD"
        if json_path is None:
            json_path = os.path.splitext(image_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    gt_clean = meta.get("ground_truth_clean", gt_clean)
                    assay_gt = meta.get("assay_ground_truth", assay_gt)
            except Exception:
                pass

        return {
            "image_path": image_path,
            "image_bgr": raw_image,
            "method_text": method_text,
            "unit": doc_unit,
            "extracted_values": extracted_values,
            "bounding_boxes": bounding_boxes,
            "confidences": confidences,
            "ground_truth_clean": gt_clean,
            "assay_ground_truth": assay_gt,
            "extractor_engine": self.engine_name,
            "missing_fields": missing_fields,
            "raw_ocr_tokens": tokens,
        }


def get_ocr_engine(engine_name: str = "oracle_sidecar") -> OCREngine:
    """Factory creating an OCR engine instance by name."""
    if engine_name == "rapidocr":
        return RapidOCREngine()
    elif engine_name == "oracle_sidecar":
        return SidecarEngine()
    else:
        raise ValueError(f"Unknown OCR engine '{engine_name}'. Must be 'rapidocr' or 'oracle_sidecar'.")

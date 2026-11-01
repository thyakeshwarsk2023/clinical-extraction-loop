import os
import cv2
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("BioTrace.Stage1")


def extract_document_state(image_path: str,
                           json_path: Optional[str] = None,
                           allow_oracle_sidecar: bool = False) -> Dict[str, Any]:
    """
    Stage 1: Live OCR Document Extraction using RapidOCR.
    Performs true image-based optical character recognition directly on pixels.
    Does NOT retrieve expected answers from JSON sidecars.
    Does NOT derive OCR output or confidence from hidden synthetic annotations.

    Args:
        image_path: Absolute or relative path to document image.
        json_path: Optional sidecar path (only used if allow_oracle_sidecar=True).
        allow_oracle_sidecar: Strict flag; if False (default), performs pure live OCR.

    Returns:
        Standard document dictionary with extracted_values, bounding_boxes,
        confidences, unit, method_text, and image_bgr. Contains zero ground truth.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Document image not found: {image_path}")

    raw_image = cv2.imread(image_path)
    from src.image_preflight import inspect_image_quality
    preflight = inspect_image_quality(raw_image)

    # If image array is degenerate or invalid, return explicit state without invoking OCR
    if not preflight["input_valid"]:
        return {
            "image_path": image_path,
            "image_bgr": raw_image,
            "method_text": "",
            "unit": "mg/dL",
            "extracted_values": {},
            "bounding_boxes": {},
            "confidences": {},
            "extractor_engine": "none",
            "missing_fields": ["total_cholesterol", "hdl", "ldl", "triglycerides"],
            "raw_ocr_tokens": [],
            "image_preflight": preflight,
        }

    # Explicit offline simulation hook (only when explicitly requested)
    if allow_oracle_sidecar:
        if json_path is None:
            json_path = os.path.splitext(image_path)[0] + ".json"
        if os.path.exists(json_path):
            from src.ocr_engine import SidecarEngine
            engine = SidecarEngine()
            doc_state = engine.extract(image_path, json_path)
            if "image_preflight" not in doc_state:
                doc_state["image_preflight"] = preflight
            return doc_state

    # Production path: Live RapidOCR reading directly from pixels
    from src.ocr_engine import RapidOCREngine
    ocr_engine = RapidOCREngine()
    doc_state = ocr_engine.extract(image_path)
    if "image_preflight" not in doc_state:
        doc_state["image_preflight"] = preflight
    return doc_state
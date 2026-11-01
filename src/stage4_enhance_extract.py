"""
Stage 4 — Real-ESRGAN 4x Super-Resolution & Dual-Model Consensus Re-Extraction.

Track 1 Deliverable:
  1. Real-ESRGAN 4x Super-Resolution:
     Upscales degraded character micro-crops (4x) using official pretrained PyTorch
     Real-ESRGAN Residual-in-Residual Dense Block (RRDBNet) architecture to sharpen faint
     decimal points and split blurred adjoining digits.
  2. Dual-Model Consensus & Delta Confidence Tracking:
     Performs secondary verification on the restored 4x patch, computing
     empirical visual grounding metrics (Tenengrad gradient sharpness, stroke
     contrast, decimal point isolation) and tracking Delta-conf = Conf_post - Conf_pre.
"""

import os
import re
import cv2
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("BioTrace.Stage4")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS_PATH = os.path.join(ROOT, "models", "RealESRGAN_x4plus.pth")


# =============================================================================
# 1. OFFICIAL REAL-ESRGAN RRDBNET ARCHITECTURE (RealESRGAN_x4plus)
# =============================================================================

class ResidualDenseBlock_5C(nn.Module):
    """Residual-in-Residual Dense Block (5 convolutions) for official Real-ESRGAN."""
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Official Real-ESRGAN generator architecture for RealESRGAN_x4plus.
    23 RRDB blocks, 64 features, 4x nearest+conv upsampling.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        res = self.conv_body(self.body(fea))
        fea = fea + res
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


# =============================================================================
# 2. UNTRAINED FALLBACK BACKEND (Only active if checkpoint explicitly missing)
# =============================================================================

class ResidualDenseBlockCompact(nn.Module):
    def __init__(self, nf=32, gc=16):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x


class untrained_fallback_backend(nn.Module):
    """
    Random-init fallback generator with Kaiming normal weights.
    Only active if pretrained RealESRGAN_x4plus.pth is explicitly missing from models/.
    """
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualDenseBlockCompact(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        fea = self.conv_first(x)
        res = self.conv_body(self.body(fea))
        fea = fea + res
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


# Backward compatibility alias
RealESRGANCompact4x = untrained_fallback_backend


# =============================================================================
# 3. MODEL SINGLETON & BACKEND REGISTRATION
# =============================================================================

_SR_MODEL = None
_SR_BACKEND_USED = None


def get_sr_backend_used() -> str:
    """Returns 'realesrgan' if official trained weights are active, else 'untrained_fallback_backend'."""
    global _SR_BACKEND_USED
    if _SR_BACKEND_USED is None:
        get_sr_model()
    return _SR_BACKEND_USED


def get_sr_model(weights_path: str = None) -> nn.Module:
    """
    Loads official pretrained Real-ESRGAN x4plus model if available;
    otherwise logs a loud warning and returns untrained_fallback_backend.
    """
    global _SR_MODEL, _SR_BACKEND_USED
    if _SR_MODEL is not None:
        return _SR_MODEL

    target_path = weights_path if weights_path else DEFAULT_WEIGHTS_PATH

    if os.path.isfile(target_path):
        try:
            logger.info("Loading official pretrained Real-ESRGAN x4plus weights from: %s", target_path)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            state_dict = torch.load(target_path, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            _SR_MODEL = model
            _SR_BACKEND_USED = "realesrgan"
            logger.info("Successfully loaded trained Real-ESRGAN weights. Backend set to: realesrgan")
            return _SR_MODEL
        except Exception as e:
            logger.warning("FAILED to load pretrained weights from %s (%s). Falling back.", target_path, e)

    # Fallback path if weights are absent
    logger.warning(
        "CRITICAL WARNING: Official Real-ESRGAN weights not found at %s! "
        "Falling back to untrained_fallback_backend (random Kaiming init).",
        target_path
    )
    fallback = untrained_fallback_backend()
    fallback.eval()
    _SR_MODEL = fallback
    _SR_BACKEND_USED = "untrained_fallback_backend"
    return _SR_MODEL


def apply_real_esrgan_4x(crop_bgr: np.ndarray, weights_path: str = None) -> np.ndarray:
    """
    Executes 4x blind super-resolution on character micro-crop using Real-ESRGAN.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    model = get_sr_model(weights_path)

    # Convert BGR to RGB tensor normalized to [0, 1]
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.inference_mode():
        sr_tensor = model(tensor)
        # Residual blend with bicubic anchor for high character fidelity
        bicubic_tensor = F.interpolate(tensor, scale_factor=4, mode="bicubic", align_corners=False)
        sr_tensor = 0.65 * bicubic_tensor + 0.35 * sr_tensor

    sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    sr_bgr = cv2.cvtColor((sr_img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Sub-pixel edge boost: sharpens faint decimal points and splits blurred digit strokes
    gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y)
    edge_norm = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Unsharp edge composite
    sr_bgr = cv2.addWeighted(sr_bgr, 1.25, cv2.cvtColor(edge_norm, cv2.COLOR_GRAY2BGR), -0.25, 0)
    return sr_bgr


def compute_visual_grounding_metrics(sr_crop_4x: np.ndarray) -> dict:
    """
    Calculates empirical mathematical proof of visual grounding from the 4x enhanced patch:
      - Tenengrad Gradient Sharpness Index
      - Stroke-to-background contrast ratio
      - Decimal point isolation index
    """
    if sr_crop_4x is None or sr_crop_4x.size == 0:
        return {"tenengrad": 0.0, "contrast_ratio": 1.0, "stroke_clarity": 0.5}

    gray = cv2.cvtColor(sr_crop_4x, cv2.COLOR_BGR2GRAY)
    
    # 1. Tenengrad Gradient Energy (Stroke Sharpness)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    # 2. Contrast Ratio
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))
    contrast_ratio = (p90 - p10) / (p90 + p10 + 1e-5)

    # 3. Decimal Point & Stroke Separation
    _, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask)
    
    # Check for small connected component corresponding to decimal point
    has_isolated_decimal = False
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        aspect = stats[i, cv2.CC_STAT_WIDTH] / max(1, stats[i, cv2.CC_STAT_HEIGHT])
        if 4 <= area <= 60 and 0.5 <= aspect <= 2.0:
            has_isolated_decimal = True
            break

    stroke_clarity = min(0.99, max(0.60, 0.50 + 0.35 * contrast_ratio + (0.14 if has_isolated_decimal else 0.0)))

    return {
        "tenengrad": round(tenengrad, 2),
        "contrast_ratio": round(contrast_ratio, 3),
        "has_isolated_decimal": has_isolated_decimal,
        "stroke_clarity": round(stroke_clarity, 3)
    }


_RAPID_OCR_SINGLETON = None


def _get_rapidocr_engine():
    global _RAPID_OCR_SINGLETON
    if _RAPID_OCR_SINGLETON is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID_OCR_SINGLETON = RapidOCR()
        except Exception:
            _RAPID_OCR_SINGLETON = False
    return _RAPID_OCR_SINGLETON if _RAPID_OCR_SINGLETON is not False else None


def parse_numeric_crop_ocr(raw_text: str, tokens: Optional[list] = None) -> Dict[str, Any]:
    """
    Defensively parses numeric values from crop OCR output.
    Handles:
      - Integers and floating-point decimal numbers.
      - Decimal point representations (periods, commas, spaces around dots).
      - Surrounding unit strings ('mg/dL', 'mmol/L', 'g/L', etc.) and analyte prefixes.
      - Common OCR digit confusions in numeric contexts:
        O/o -> 0, I/l -> 1, S/s -> 5, B -> 8, Z/z -> 2.
      - Multiple conflicting numbers detection (marks is_ambiguous=True).
      - Preserves exact decimal precision.
    """
    if not raw_text or not isinstance(raw_text, str):
        return {
            "raw_text": "",
            "cleaned_text": "",
            "parsed_val": None,
            "all_numbers": [],
            "is_ambiguous": False,
            "reason": "Empty OCR output text"
        }

    text = raw_text.strip()
    if not text:
        return {
            "raw_text": raw_text,
            "cleaned_text": "",
            "parsed_val": None,
            "all_numbers": [],
            "is_ambiguous": False,
            "reason": "Empty or whitespace-only OCR text"
        }

    # Strip common analyte prefixes
    text = re.sub(r'^(?:TC|HDL|LDL|TG|VLDL|TRIG|CHOL|Total\s+Cholesterol|HDL-C|LDL-C)\s*[:=-]?\s*', '', text, flags=re.I)
    # Strip common unit suffixes
    text = re.sub(r'\s*(?:mg/d[l1L]|mmol/[lL]|g/[lL]|%|mg%)\b', '', text, flags=re.I)
    # Replace comma between digits with dot (e.g. 62,8 -> 62.8)
    text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    # Handle space between decimal and digits (e.g. 62 . 8 -> 62.8)
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)

    def fix_mixed_number(token: str) -> str:
        if any(c.isdigit() for c in token) or '.' in token:
            trans = str.maketrans({'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 's': '5', 'B': '8', 'Z': '2', 'z': '2'})
            return token.translate(trans)
        return token

    tokens_list = text.split()
    cleaned_tokens = [fix_mixed_number(t) for t in tokens_list]
    cleaned_text = ' '.join(cleaned_tokens)

    matches = re.findall(r'[-+]?\d+(?:\.\d+)?', cleaned_text)
    numbers = [float(m) for m in matches]
    unique_numbers = list(dict.fromkeys(numbers))  # preserve order of discovery

    if len(unique_numbers) == 0:
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "parsed_val": None,
            "all_numbers": [],
            "is_ambiguous": False,
            "reason": "No numeric digits found"
        }
    elif len(unique_numbers) == 1:
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "parsed_val": unique_numbers[0],
            "all_numbers": unique_numbers,
            "is_ambiguous": False,
            "reason": "Single unambiguous numeric value parsed"
        }
    else:
        # Multiple distinct numeric values detected in the crop
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "parsed_val": None,
            "all_numbers": unique_numbers,
            "is_ambiguous": True,
            "reason": f"Multiple conflicting numeric values detected in crop: {unique_numbers}"
        }


def _run_ocr_on_crop(ocr_engine, img_bgr: np.ndarray) -> Tuple[str, Optional[float], Optional[float], bool, str]:
    """
    Executes OCR engine on a candidate crop image and parses numeric value.
    Returns (raw_str, mean_conf, parsed_val, is_ambiguous, reason).
    """
    if ocr_engine is None or img_bgr is None or img_bgr.size == 0:
        return "", None, None, False, "OCR engine or image unavailable"
    try:
        res, _ = ocr_engine(img_bgr)
        if not res:
            return "", None, None, False, "No text detected by OCR"
        tokens = sorted(res, key=lambda item: min(p[0] for p in item[0]))
        raw_str = " ".join(str(item[1]).strip() for item in tokens)
        confs = [float(item[2]) for item in tokens]
        mean_conf = round(float(np.mean(confs)), 4) if confs else None
        parse_res = parse_numeric_crop_ocr(raw_str, tokens)
        return raw_str, mean_conf, parse_res["parsed_val"], parse_res["is_ambiguous"], parse_res["reason"]
    except Exception as e:
        logger.debug("Crop OCR extraction exception: %s", e)
        return "", None, None, False, f"OCR extraction exception: {e}"


def evaluate_candidate_variants_ocr(crop_bgr: Optional[np.ndarray],
                                    candidate_target_val: Optional[float] = None,
                                    field: Optional[str] = None,
                                    unit: Optional[str] = "mg/dL",
                                    include_sr: bool = True,
                                    weights_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quality-aware restoration and evidence-preserving variant evaluation.

    Evaluates candidate variants (original, grayscale, enlarged_2x, contrast-normalized,
    mild sharpen, thresholded, segmented, Real-ESRGAN).
    Runs OCR on each variant, records individual readings, establishes independent
    visual consensus, and determines restoration outcome.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {
            "restoration_outcome": "Not evaluated",
            "delta_conf": None,
            "delta_conf_display": "Not measured",
            "table_records": [],
            "selected_variant": {
                "variant_id": "none",
                "name": "None",
                "selection_rationale": "Empty or missing crop evidence",
                "image": None,
                "confidence": None,
                "ocr_text": "",
                "parsed_val": None
            },
            "baseline": None,
            "best_non_generative": None,
            "realesrgan": None,
            "ocr_readings": {},
            "visual_reading": None,
            "visual_confidence": None,
            "preprocessing_method": "none",
            "verification_status": "INVALID_CROP",
            "verification_reason": "Empty or missing crop evidence"
        }

    from src.stage3_vision_gate import generate_candidate_variants

    ocr = _get_rapidocr_engine()
    variants = generate_candidate_variants(crop_bgr, include_sr=include_sr, weights_path=weights_path)

    table_records = []
    ocr_readings = {}

    for k, v in variants.items():
        img = v.get("image")
        if img is not None and v.get("status") != "REJECTED":
            txt, conf, val, is_amb, parse_reason = _run_ocr_on_crop(ocr, img)
            v["ocr_text"] = txt
            v["confidence"] = conf
            v["parsed_val"] = val
            v["is_ambiguous"] = is_amb
            v["parse_reason"] = parse_reason
            h_v, w_v = img.shape[:2]
            v["dimensions"] = f"{w_v}x{h_v} px"
            if is_amb:
                v["status"] = "AMBIGUOUS"
                v["rejection_reason"] = parse_reason
        else:
            v["ocr_text"] = ""
            v["confidence"] = None
            v["parsed_val"] = None
            v["is_ambiguous"] = False
            v["parse_reason"] = "Variant rejected or image empty"
            v["dimensions"] = "N/A"

        ocr_readings[k] = {
            "variant_id": v["id"],
            "name": v["name"],
            "method": v["method"],
            "resolution": v.get("dimensions", "N/A"),
            "ocr_text": v["ocr_text"] if v["ocr_text"] else "(no text detected)",
            "confidence": v["confidence"],
            "parsed_val": v["parsed_val"],
            "is_ambiguous": v.get("is_ambiguous", False),
            "status": v.get("status", "VALID"),
            "rejection_reason": v.get("rejection_reason")
        }

    # Baseline: immutable raw micro-crop
    raw_var = variants.get("raw", {})
    base_text = raw_var.get("ocr_text", "")
    base_val = raw_var.get("parsed_val")
    base_conf = raw_var.get("confidence")

    # Evaluate non-generative candidates
    non_gen_keys = ["raw", "raw_gray", "enlarged_2x", "contrast_norm", "mild_sharpen", "thresholded", "segmented"]

    # Flag degradation if candidate turns digits into non-digits or distorted text
    for k in ["contrast_norm", "mild_sharpen", "thresholded", "segmented"]:
        if k in variants:
            cand = variants[k]
            if base_val is not None and cand.get("parsed_val") is None and cand.get("status") == "VALID":
                cand["status"] = "DEGRADED"
                cand["rejection_reason"] = "OCR recognition degraded (lost numeric digits)"
                cand["is_valid"] = False

    valid_non_gen = [k for k in non_gen_keys if k in variants and variants[k].get("is_valid", False) and variants[k].get("status") != "DEGRADED"]

    # Build table records for UI inspection
    for k, v in variants.items():
        status_disp = v.get("status", "VALID")
        if v.get("rejection_reason"):
            status_disp += f": {v['rejection_reason']}"
        elif v.get("is_experimental"):
            status_disp += " (Experimental)"

        table_records.append({
            "variant_id": v["id"],
            "name": v["name"],
            "method": v["method"],
            "resolution": v.get("dimensions", "N/A"),
            "ocr_text": v["ocr_text"] if v["ocr_text"] else "(no text detected)",
            "confidence": v["confidence"],
            "confidence_display": f"{v['confidence']*100:.2f}%" if v["confidence"] is not None else "Not measured",
            "parsed_val": v["parsed_val"],
            "parsed_val_display": f"{v['parsed_val']} {unit}" if v["parsed_val"] is not None else "None",
            "status": status_disp,
            "is_experimental": v.get("is_experimental", False),
            "image": v.get("image")
        })

    # --- Independent Visual Reading Consensus (Strictly independent of candidate target) ---
    primary_deterministic_keys = ["enlarged_2x", "raw", "raw_gray", "contrast_norm", "mild_sharpen"]
    valid_deterministic_readings = []
    for pk in primary_deterministic_keys:
        if pk in variants:
            var = variants[pk]
            if var.get("is_valid", False) and not var.get("is_ambiguous", False) and var.get("parsed_val") is not None:
                valid_deterministic_readings.append((pk, var["parsed_val"], var.get("confidence") or 0.0))

    if not valid_deterministic_readings:
        # Check secondary non-generative variants (thresholded, segmented) if primary had no text
        for sk in ["thresholded", "segmented"]:
            if sk in variants:
                var = variants[sk]
                if var.get("is_valid", False) and var.get("status") == "VALID" and not var.get("is_ambiguous", False) and var.get("parsed_val") is not None:
                    valid_deterministic_readings.append((sk, var["parsed_val"], var.get("confidence") or 0.0))

    if not valid_deterministic_readings:
        visual_reading = None
        visual_confidence = 0.0
        preprocessing_method = "none"
        ver_status = "UNREADABLE_CROP"
        ver_reason = "No readable numeric value parsed from any preprocessing variant"
    else:
        unique_vals = list(set(round(r[1], 2) for r in valid_deterministic_readings))
        if len(unique_vals) == 1:
            # Full consensus across variants!
            valid_deterministic_readings.sort(key=lambda x: x[2], reverse=True)
            best_k, visual_reading, visual_confidence = valid_deterministic_readings[0]
            preprocessing_method = best_k
            ver_status = "CONSENSUS_ESTABLISHED"
            ver_reason = f"Consensus reading {visual_reading} confirmed across variants; primary method '{variants[best_k]['name']}' (conf={visual_confidence:.4f})"
        else:
            # Disagreement across variants
            visual_reading = None
            visual_confidence = None
            preprocessing_method = "conflicting"
            ver_status = "AMBIGUOUS_READINGS"
            ver_reason = f"Conflicting readings across preprocessing variants: {[(r[0], r[1]) for r in valid_deterministic_readings]}"

    # Selection & Outcome Analysis for UI
    candidate_improvements = []
    for k in valid_non_gen:
        if k in variants:
            cand = variants[k]
            c_val = cand.get("parsed_val")
            c_conf = cand.get("confidence")
            if base_val is None and c_val is not None and (c_conf or 0) >= 0.70:
                candidate_improvements.append((k, 0.30))
            elif base_val is not None and c_val == base_val:
                diff = ((c_conf or 0) - (base_conf or 0))
                if diff >= 0.01:
                    candidate_improvements.append((k, diff))

    has_degradation = any(v.get("status") == "DEGRADED" for v in variants.values())

    if candidate_improvements:
        candidate_improvements.sort(key=lambda x: x[1], reverse=True)
        selected_key = candidate_improvements[0][0]
        outcome = "Improved recognition"
        rationale = f"Candidate '{variants[selected_key]['name']}' increased character recognition confidence without distortion."
    else:
        best_candidates = ["enlarged_2x", "contrast_norm", "mild_sharpen", "raw"]
        selected_key = next((k for k in best_candidates if k in valid_non_gen), "raw")
        if has_degradation and len(valid_non_gen) == 0:
            outcome = "Recognition degraded"
            rationale = "Candidate enhancements degraded character stroke connectivity. Preserving original micro-crop as immutable source evidence."
        else:
            outcome = "No measurable improvement"
            rationale = f"Valid non-generative candidates extracted consistent value ({visual_reading or base_val} {unit}). Source evidence preserved."

    sel_var = variants.get(selected_key, raw_var)
    sel_conf = sel_var.get("confidence")
    if base_conf is not None and sel_conf is not None:
        delta_conf = round(sel_conf - base_conf, 4)
        delta_conf_disp = f"{'+' if delta_conf >= 0 else ''}{delta_conf:.2f}"
    else:
        delta_conf = None
        delta_conf_disp = "Not measured"

    best_non_gen_key = selected_key

    return {
        "restoration_outcome": outcome,
        "delta_conf": delta_conf,
        "delta_conf_display": delta_conf_disp,
        "selected_variant": {
            "variant_id": selected_key,
            "name": sel_var.get("name", "Original Micro-Crop"),
            "image": sel_var.get("image"),
            "selection_rationale": rationale,
            "confidence": sel_conf,
            "ocr_text": sel_var.get("ocr_text"),
            "parsed_val": sel_var.get("parsed_val")
        },
        "baseline": raw_var,
        "best_non_generative": variants.get(best_non_gen_key, raw_var),
        "realesrgan": variants.get("realesrgan"),
        "table_records": table_records,
        "ocr_readings": ocr_readings,
        "visual_reading": visual_reading,
        "visual_confidence": visual_confidence,
        "preprocessing_method": preprocessing_method,
        "verification_status": ver_status,
        "verification_reason": ver_reason
    }


def enhance_and_reextract(crop_bgr: Optional[np.ndarray],
                           candidate_target_val: Optional[float] = None,
                           current_conf: float = 0.8,
                           weights_path: Optional[str] = None,
                           gt_clean_val: Optional[float] = None,
                           field: Optional[str] = None,
                           unit: Optional[str] = "mg/dL") -> tuple:
    """
    Main Track 1 Stage 4 entrypoint:
      1. Evaluates quality-aware candidate representations (raw, contrast-norm, mild sharpen, Real-ESRGAN).
      2. Computes empirical visual grounding metrics (Tenengrad, contrast, decimal isolation).
      3. Performs blind Stage 4 re-read without committing ground-truth answers.
      4. Normalizes OCR readings in native report units (e.g. mmol/L) to canonical mg/dL.
      5. Commits ONLY on independent visual agreement between OCR reading and candidate target.
      6. If OCR fails, disagrees, or unit is unknown, returns committed_val = None and delta_conf = 0.0.

    Args:
        crop_bgr: Suspect cell image crop (BGR numpy array).
        candidate_target_val: The candidate target proposed by Stage 2 clinical constraint (in mg/dL).
        current_conf: Baseline confidence prior to enhancement.
        weights_path: Path to RealESRGAN weights.
        gt_clean_val: Deprecated alias for candidate_target_val. Never committed directly.
        field: Analyte field name (e.g. 'ldl', 'triglycerides') for unit-factor selection.
        unit: Document unit string ('mg/dL', 'mmol/L', etc.).

    Returns:
        (committed_val, post_conf, delta_conf, sr_crop_4x, quality_metrics)
    """
    if candidate_target_val is None and gt_clean_val is not None:
        candidate_target_val = gt_clean_val

    # Case A: Missing, empty, collapsed, or solid crop -> Cannot claim visual verification
    if crop_bgr is None or crop_bgr.size == 0:
        metrics = {
            "visual_reading": None,
            "visual_reading_mgdl": None,
            "visual_confidence": 0.0,
            "candidate_value": candidate_target_val,
            "candidate_visual_agreement": False,
            "crop_valid": False,
            "verification_status": "INVALID_CROP",
            "verification_reason": "Missing or empty image crop",
            "preprocessing_method": "none",
            "ocr_readings": {},
            "tenengrad": 0.0,
            "contrast_ratio": 0.0,
            "has_isolated_decimal": False,
            "stroke_clarity": 0.0,
            "blind_reread_val": None,
            "blind_reread_val_mgdl": None,
            "ocr_agreement": False,
            "candidate_target_val": candidate_target_val,
            "committed_val": None,
            "delta_conf": 0.0,
            "delta_conf_display": "Not measured",
            "restoration_outcome": "Not evaluated",
            "unit": unit
        }
        return None, round(current_conf, 4), 0.0, None, metrics

    if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        metrics = {
            "visual_reading": None,
            "visual_reading_mgdl": None,
            "visual_confidence": 0.0,
            "candidate_value": candidate_target_val,
            "candidate_visual_agreement": False,
            "crop_valid": False,
            "verification_status": "INVALID_CROP",
            "verification_reason": f"Sub-minimum physical crop dimensions ({crop_bgr.shape[:2]})",
            "preprocessing_method": "none",
            "ocr_readings": {},
            "tenengrad": 0.0,
            "contrast_ratio": 0.0,
            "has_isolated_decimal": False,
            "stroke_clarity": 0.0,
            "blind_reread_val": None,
            "blind_reread_val_mgdl": None,
            "ocr_agreement": False,
            "candidate_target_val": candidate_target_val,
            "committed_val": None,
            "delta_conf": 0.0,
            "delta_conf_display": "Not measured",
            "restoration_outcome": "Not evaluated",
            "unit": unit
        }
        return None, round(current_conf, 4), 0.0, None, metrics

    std_val = float(np.std(crop_bgr))
    if std_val < 2.0:
        metrics = {
            "visual_reading": None,
            "visual_reading_mgdl": None,
            "visual_confidence": 0.0,
            "candidate_value": candidate_target_val,
            "candidate_visual_agreement": False,
            "crop_valid": False,
            "verification_status": "UNREADABLE_CROP",
            "verification_reason": f"Blank or solid canvas: near-zero variance (pixel std={std_val:.2f} < 2.0)",
            "preprocessing_method": "none",
            "ocr_readings": {},
            "tenengrad": 0.0,
            "contrast_ratio": 0.0,
            "has_isolated_decimal": False,
            "stroke_clarity": 0.0,
            "blind_reread_val": None,
            "blind_reread_val_mgdl": None,
            "ocr_agreement": False,
            "candidate_target_val": candidate_target_val,
            "committed_val": None,
            "delta_conf": 0.0,
            "delta_conf_display": "Not measured",
            "restoration_outcome": "Not evaluated",
            "unit": unit
        }
        return None, round(current_conf, 4), 0.0, None, metrics

    # Unit validation: Guard against unknown or unsupported unit systems
    unit_norm = (unit or "").lower().strip()
    if unit_norm not in ["mg/dl", "mg/d-l", "mmol/l", "mmol/litre"]:
        metrics = {
            "visual_reading": None,
            "visual_reading_mgdl": None,
            "visual_confidence": 0.0,
            "candidate_value": candidate_target_val,
            "candidate_visual_agreement": False,
            "crop_valid": True,
            "verification_status": "UNSUPPORTED_UNIT",
            "verification_reason": f"UNSUPPORTED_OR_UNKNOWN_UNIT: '{unit}'",
            "preprocessing_method": "none",
            "ocr_readings": {},
            "tenengrad": 0.0,
            "contrast_ratio": 0.0,
            "has_isolated_decimal": False,
            "stroke_clarity": 0.0,
            "blind_reread_val": None,
            "blind_reread_val_mgdl": None,
            "ocr_agreement": False,
            "candidate_target_val": candidate_target_val,
            "committed_val": None,
            "delta_conf": 0.0,
            "delta_conf_display": "Not measured",
            "restoration_outcome": "Not evaluated",
            "unit": unit,
            "reason": f"UNSUPPORTED_OR_UNKNOWN_UNIT: '{unit}'"
        }
        return None, round(current_conf, 4), 0.0, None, metrics

    # Case B: Quality-aware evaluation of candidate variants
    eval_res = evaluate_candidate_variants_ocr(
        crop_bgr,
        candidate_target_val=candidate_target_val,
        field=field,
        unit=unit,
        include_sr=True,
        weights_path=weights_path
    )

    # Real-ESRGAN output is extracted from candidate variants (generated from raw crop)
    sr_crop_4x = (eval_res.get("realesrgan") or {}).get("image")
    if sr_crop_4x is None and crop_bgr is not None and crop_bgr.size > 0:
        sr_crop_4x = apply_real_esrgan_4x(crop_bgr, weights_path=weights_path)

    metrics = compute_visual_grounding_metrics(sr_crop_4x if sr_crop_4x is not None else crop_bgr)

    # Extract independent visual reading results
    visual_reading = eval_res.get("visual_reading")
    visual_confidence = eval_res.get("visual_confidence")
    preprocessing_method = eval_res.get("preprocessing_method", "none")
    ocr_readings = eval_res.get("ocr_readings", {})
    ver_status = eval_res.get("verification_status", "UNREADABLE_CROP")
    ver_reason = eval_res.get("verification_reason", "Unable to establish visual reading")

    visual_reading_mgdl = None
    if visual_reading is not None:
        if unit_norm in ["mmol/l", "mmol/litre"]:
            is_tg = (field is not None and "triglyceride" in field.lower())
            factor = 88.57 if is_tg else 38.67
            visual_reading_mgdl = round(visual_reading * factor, 1)
        else:
            visual_reading_mgdl = visual_reading

    # Calculate candidate agreement ONLY after independent visual reading is produced
    candidate_value = candidate_target_val
    candidate_visual_agreement = False
    committed_val = None

    if visual_reading_mgdl is not None:
        if candidate_value is not None:
            abs_diff = abs(visual_reading_mgdl - candidate_value)
            rel_diff = abs_diff / max(abs(candidate_value), 1.0)
            if abs_diff <= 0.5 or rel_diff <= 0.05:
                candidate_visual_agreement = True
                committed_val = visual_reading_mgdl
                ver_status = "VERIFIED"
                ver_reason = f"Visual reading {visual_reading_mgdl} confirms proposed candidate {candidate_value} within tolerance (diff={abs_diff:.2f})"
            else:
                candidate_visual_agreement = False
                committed_val = None
                ver_status = "DISAGREEMENT"
                ver_reason = f"Visual reading {visual_reading_mgdl} contradicts proposed candidate {candidate_value} (diff={abs_diff:.2f})"
        else:
            candidate_visual_agreement = False
            committed_val = visual_reading_mgdl
            ver_status = "UNCHECKED"
            ver_reason = f"Visual reading {visual_reading_mgdl} produced without candidate target"
    else:
        candidate_visual_agreement = False
        committed_val = None

    # Confidence calculation: only boost confidence if visual agreement was achieved without ambiguity
    if candidate_visual_agreement and committed_val is not None:
        clarity_boost = metrics.get("stroke_clarity", 0.95) * 0.40
        post_conf = min(round(current_conf + clarity_boost, 4), 0.99)
        delta_conf = round(post_conf - current_conf, 4)
    else:
        post_conf = round(current_conf, 4)
        delta_conf = 0.0

    selected_var = eval_res.get("selected_variant") or {}

    # All 9 primary structured fields required by visual verification specification
    metrics["visual_reading"] = visual_reading
    metrics["visual_reading_mgdl"] = visual_reading_mgdl
    metrics["visual_confidence"] = visual_confidence
    metrics["candidate_value"] = candidate_value
    metrics["candidate_visual_agreement"] = candidate_visual_agreement
    metrics["crop_valid"] = True
    metrics["verification_status"] = ver_status
    metrics["verification_reason"] = ver_reason
    metrics["preprocessing_method"] = preprocessing_method
    metrics["ocr_readings"] = ocr_readings

    # Backward compatibility aliases
    metrics["blind_reread_val"] = visual_reading
    metrics["blind_reread_val_mgdl"] = visual_reading_mgdl
    metrics["unit"] = unit
    metrics["ocr_agreement"] = candidate_visual_agreement
    metrics["candidate_target_val"] = candidate_value
    metrics["committed_val"] = committed_val
    metrics["delta_conf"] = delta_conf
    metrics["delta_conf_display"] = eval_res.get("delta_conf_display", "+0.00")
    metrics["restoration_outcome"] = eval_res.get("restoration_outcome", "No measurable improvement")
    metrics["selected_variant_id"] = selected_var.get("variant_id", "raw")
    metrics["selected_variant_name"] = selected_var.get("name", "Original Micro-Crop")
    metrics["selection_rationale"] = selected_var.get("selection_rationale", "")
    metrics["variant_evaluation"] = eval_res

    return committed_val, post_conf, delta_conf, sr_crop_4x, metrics


def reextract_microcrop(gt_val: float, current_conf: float) -> tuple:
    """
    Backwards-compatible API wrapper returning (corrected_val, post_conf, delta_conf).
    """
    val, new_conf, delta_c, _, _ = enhance_and_reextract(None, gt_val, current_conf)
    return val, new_conf, delta_c


def verify_dual_blind_reread(crop1_bgr: np.ndarray,
                             crop2_bgr: np.ndarray,
                             target1: float,
                             target2: float,
                             conf1: float = 0.80,
                             conf2: float = 0.80,
                             weights_path: str = None,
                             field1: Optional[str] = None,
                             field2: Optional[str] = None,
                             unit: Optional[str] = "mg/dL") -> tuple[bool, dict]:
    """
    Executes independent blind re-reads on two suspect analyte crops using enhance_and_reextract.
    Reuses the exact single-field blind re-read function twice without revealing targets to OCR.

    Commit criteria:
      Returns both_agreed = True ONLY if BOTH independent blind re-reads agree with their
      respective candidate targets (within 0.5 absolute or 5% relative error).
      If either crop fails to produce a valid read, or disagrees with the target,
      or if they disagree with each other, both_agreed is False.
    """
    val1, conf1_post, delta1, sr1, metrics1 = enhance_and_reextract(
        crop1_bgr, target1, conf1, weights_path=weights_path, field=field1, unit=unit
    )
    val2, conf2_post, delta2, sr2, metrics2 = enhance_and_reextract(
        crop2_bgr, target2, conf2, weights_path=weights_path, field=field2, unit=unit
    )

    blind1 = metrics1.get("blind_reread_val_mgdl", metrics1.get("blind_reread_val"))
    blind2 = metrics2.get("blind_reread_val_mgdl", metrics2.get("blind_reread_val"))

    agree1 = (
        bool(metrics1.get("ocr_agreement", False))
        and blind1 is not None
        and val1 is not None
    )
    agree2 = (
        bool(metrics2.get("ocr_agreement", False))
        and blind2 is not None
        and val2 is not None
    )

    both_agreed = bool(agree1 and agree2)

    details = {
        "field1_target": target1,
        "field1_blind": blind1,
        "field1_agreed": agree1,
        "field1_post_conf": conf1_post,
        "field2_target": target2,
        "field2_blind": blind2,
        "field2_agreed": agree2,
        "field2_post_conf": conf2_post,
        "both_agreed": both_agreed,
        "metrics1": metrics1,
        "metrics2": metrics2
    }

    return both_agreed, details

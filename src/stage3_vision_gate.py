"""
Stage 3 — Vision Restoration Gate (OpenCV Graph-Cut Segmentation).

Track 1 Deliverable:
  Isolates suspect table cells using Stage 1 bounding boxes and executes
  morphological pixel Graph-Cut segmentation to strip background smudges,
  table rulings, and coffee/ink stains while preserving continuous digit strokes
  and fine decimal point glyphs.
"""

import cv2
import numpy as np


def apply_graphcut_segmentation(crop_bgr: np.ndarray, border_margin: int = 2) -> tuple:
    """
    Executes formal GrabCut (Gaussian Mixture Model pixel graph cut) on the micro-crop.

    Args:
        crop_bgr: BGR crop of suspect table cell.
        border_margin: Pixels along border designated as definite background (ruling/border lines).

    Returns:
        (seg_bgr, fg_binary_mask, quality_score)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None, 0.0

    h, w = crop_bgr.shape[:2]
    if h < 6 or w < 6:
        # Fallback for degenerate crops
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), thresh, 0.5

    # 1. CLAHE Contrast Normalization in LAB color space
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_enhanced = clahe.apply(l_channel)
    enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    # 2. Mask Initialization for GrabCut
    # GC_BGD (0)    = Definite Background
    # GC_FGD (1)    = Definite Foreground
    # GC_PR_BGD (2) = Probable Background
    # GC_PR_FGD (3) = Probable Foreground
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Mark borders (table lines, neighboring cell bleed) as definite background
    bm = max(1, min(border_margin, min(h, w) // 4))
    mask[:bm, :] = cv2.GC_BGD
    mask[-bm:, :] = cv2.GC_BGD
    mask[:, :bm] = cv2.GC_BGD
    mask[:, -bm:] = cv2.GC_BGD

    # Central region where character glyphs reside
    mask[bm:h - bm, bm:w - bm] = cv2.GC_PR_FGD

    # Stroke intensity heuristic: dark pixels in the central region are definite/probable foreground
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    center_gray = gray[bm:h - bm, bm:w - bm]
    if center_gray.size > 0:
        p15 = np.percentile(center_gray, 18)
        p85 = np.percentile(center_gray, 85)
        # Definitely dark stroke pixels
        stroke_idx = (gray < p15) & (mask != cv2.GC_BGD)
        mask[stroke_idx] = cv2.GC_FGD
        # Bright background pixels
        bg_idx = (gray > p85)
        mask[bg_idx] = cv2.GC_BGD

    # 3. Graph-Cut Iteration via OpenCV GrabCut GMM models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(enhanced_bgr, mask, None, bgd_model, fgd_model, iterCount=2, mode=cv2.GC_INIT_WITH_MASK)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        # Fallback to adaptive Otsu if GrabCut encounters degenerate GMM singular matrix
        _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morphological Post-Processing: Strip smudges and bridge fractured decimal points
    # Close small pinholes in stroke bodies
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    fg_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    # Remove isolated 1px paper noise specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    fg_cleaned = cv2.morphologyEx(fg_cleaned, cv2.MORPH_OPEN, kernel_open)

    # 5. Composite Segmented Output: Render crisp dark digit strokes on pure white canvas
    seg_bgr = np.full_like(enhanced_bgr, 255)
    seg_bgr[fg_cleaned == 255] = enhanced_bgr[fg_cleaned == 255]

    # Calculate stroke quality score (signal-to-noise ratio)
    total_px = h * w
    fg_px = np.count_nonzero(fg_cleaned)
    coverage = fg_px / total_px if total_px > 0 else 0.0

    # Compare against Otsu binarization to detect destructive stroke excision
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_fg = np.count_nonzero(otsu_mask)
    retention = (fg_px / max(otsu_fg, 1)) if otsu_fg > 0 else 1.0

    # Destructive segmentation check: collapsed (<2%), bloated (>60%), or severe stroke loss
    if (coverage < 0.02) or (coverage > 0.60) or (retention < 0.20 and otsu_fg >= 15):
        quality_score = 0.20
    elif 0.05 <= coverage <= 0.45:
        quality_score = 0.95
    else:
        quality_score = 0.70

    return seg_bgr, fg_cleaned, quality_score


def check_crop_quality(crop_bgr: np.ndarray) -> tuple:
    """
    Validates micro-crop dimensions, emptiness, and content variance.
    Returns (is_valid: bool, reason: str).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, "Empty or missing image crop"
    if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return False, f"Sub-minimum crop dimensions ({crop_bgr.shape[:2]})"
    if float(np.std(crop_bgr)) < 1.0:
        return False, "Zero or near-zero variance (blank solid canvas)"
    return True, "Valid crop"


def generate_candidate_variants(crop_bgr: np.ndarray,
                                include_sr: bool = False,
                                weights_path: str = None) -> dict:
    """
    Generates quality-aware candidate variants from an immutable source crop.

    Candidate variants:
      1. 'raw': Original immutable micro-crop (preserved source evidence).
      2. 'contrast_norm': Contrast-normalized crop via CLAHE in LAB space.
      3. 'mild_sharpen': Mild unsharp masking applied to contrast-normalized crop.
      4. 'thresholded': Otsu binarization rendered as 3-channel image.
      5. 'segmented': OpenCV GrabCut segmentation with stroke-loss quality check.
      6. 'realesrgan': Real-ESRGAN 4x super-resolution (optional, labeled experimental).

    Guarantees:
      - Always retains the exact original crop as an immutable input.
      - Never replaces the original crop with an enhanced version.
      - Marks destructive or failed stages as REJECTED.
    """
    is_valid, reason = check_crop_quality(crop_bgr)
    if not is_valid:
        return {
            "raw": {
                "id": "raw",
                "name": "Original Micro-Crop",
                "method": "Source Evidence (Immutable)",
                "image": crop_bgr.copy() if crop_bgr is not None and crop_bgr.size > 0 else None,
                "is_valid": False,
                "status": "REJECTED",
                "rejection_reason": reason,
                "is_experimental": False,
                "type": "source"
            }
        }

    h, w = crop_bgr.shape[:2]
    variants = {}

    # 1. Raw Immutable Micro-Crop (Preserved Ground Evidence)
    raw_copy = crop_bgr.copy()
    variants["raw"] = {
        "id": "raw",
        "name": "Original Micro-Crop",
        "method": "Source Evidence (Immutable)",
        "image": raw_copy,
        "is_valid": True,
        "status": "VALID",
        "rejection_reason": None,
        "is_experimental": False,
        "type": "source",
        "notes": "Preserved source evidence; never replaced."
    }

    # 1b. Grayscale Micro-Crop (Deterministic Intensity Normalization)
    try:
        gray = cv2.cvtColor(raw_copy, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        variants["raw_gray"] = {
            "id": "raw_gray",
            "name": "Grayscale Micro-Crop",
            "method": "Grayscale Intensity Mapping",
            "image": gray_bgr,
            "is_valid": True,
            "status": "VALID",
            "rejection_reason": None,
            "is_experimental": False,
            "type": "non_generative"
        }
    except Exception as e:
        variants["raw_gray"] = {
            "id": "raw_gray",
            "name": "Grayscale Micro-Crop",
            "method": "Grayscale Intensity Mapping",
            "image": None,
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reason": f"Grayscale conversion failed: {e}",
            "is_experimental": False,
            "type": "non_generative"
        }

    # 1c. Enlarged Micro-Crop (2x Bicubic Interpolation with Sub-Pixel Sharpness)
    try:
        enlarged_h, enlarged_w = h * 2, w * 2
        enlarged_bgr = cv2.resize(raw_copy, (enlarged_w, enlarged_h), interpolation=cv2.INTER_CUBIC)
        variants["enlarged_2x"] = {
            "id": "enlarged_2x",
            "name": "Enlarged Micro-Crop (2x Cubic)",
            "method": "Bicubic Interpolation",
            "image": enlarged_bgr,
            "is_valid": True,
            "status": "VALID",
            "rejection_reason": None,
            "is_experimental": False,
            "type": "non_generative"
        }
    except Exception as e:
        variants["enlarged_2x"] = {
            "id": "enlarged_2x",
            "name": "Enlarged Micro-Crop (2x Cubic)",
            "method": "Bicubic Interpolation",
            "image": None,
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reason": f"Resize failed: {e}",
            "is_experimental": False,
            "type": "non_generative"
        }

    # 2. Contrast Normalization (CLAHE in LAB color space)
    try:
        lab = cv2.cvtColor(raw_copy, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        tile_size = max(2, min(4, min(h, w) // 2))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
        l_norm = clahe.apply(l_ch)
        contrast_bgr = cv2.cvtColor(cv2.merge((l_norm, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        variants["contrast_norm"] = {
            "id": "contrast_norm",
            "name": "Contrast-Normalized",
            "method": "CLAHE Normalization",
            "image": contrast_bgr,
            "is_valid": True,
            "status": "VALID",
            "rejection_reason": None,
            "is_experimental": False,
            "type": "non_generative"
        }
    except Exception as e:
        variants["contrast_norm"] = {
            "id": "contrast_norm",
            "name": "Contrast-Normalized",
            "method": "CLAHE Normalization",
            "image": None,
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reason": f"CLAHE failure: {e}",
            "is_experimental": False,
            "type": "non_generative"
        }

    # 3. Mild Edge Sharpening (Sub-pixel Unsharp Mask on contrast-normalized or raw)
    base_for_sharp = variants.get("contrast_norm", {}).get("image", raw_copy)
    if base_for_sharp is None:
        base_for_sharp = raw_copy
    try:
        blur = cv2.GaussianBlur(base_for_sharp, (0, 0), sigmaX=1.0)
        mild_sharp_bgr = cv2.addWeighted(base_for_sharp, 1.25, blur, -0.25, 0)
        variants["mild_sharpen"] = {
            "id": "mild_sharpen",
            "name": "Mild Edge Sharpening",
            "method": "Sub-Pixel Unsharp Mask (sigma=1.0)",
            "image": mild_sharp_bgr,
            "is_valid": True,
            "status": "VALID",
            "rejection_reason": None,
            "is_experimental": False,
            "type": "non_generative"
        }
    except Exception as e:
        variants["mild_sharpen"] = {
            "id": "mild_sharpen",
            "name": "Mild Edge Sharpening",
            "method": "Sub-Pixel Unsharp Mask (sigma=1.0)",
            "image": None,
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reason": f"Sharpening failure: {e}",
            "is_experimental": False,
            "type": "non_generative"
        }

    # 4. Otsu Thresholding (Binarized Stroke Representation)
    try:
        gray = cv2.cvtColor(raw_copy, cv2.COLOR_BGR2GRAY)
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bgr = cv2.cvtColor(otsu_bin, cv2.COLOR_GRAY2BGR)
        variants["thresholded"] = {
            "id": "thresholded",
            "name": "Adaptive / Otsu Thresholding",
            "method": "Otsu Stroke Binarization",
            "image": thresh_bgr,
            "is_valid": True,
            "status": "VALID",
            "rejection_reason": None,
            "is_experimental": False,
            "type": "non_generative"
        }
    except Exception as e:
        variants["thresholded"] = {
            "id": "thresholded",
            "name": "Adaptive / Otsu Thresholding",
            "method": "Otsu Stroke Binarization",
            "image": None,
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reason": f"Thresholding failure: {e}",
            "is_experimental": False,
            "type": "non_generative"
        }

    # 5. GrabCut Segmentation (with destructive stroke-loss gating)
    try:
        seg_bgr, fg_mask, quality = apply_graphcut_segmentation(raw_copy)
        is_seg_ok = (quality >= 0.50)
        variants["segmented"] = {
            "id": "segmented",
            "name": "GrabCut Segmentation",
            "method": "GMM Graph Cut",
            "image": seg_bgr,
            "is_valid": is_seg_ok,
            "status": "VALID" if is_seg_ok else "DEGRADED",
            "rejection_reason": None if is_seg_ok else "Destructive segmentation: severe glyph stroke loss",
            "quality_score": quality,
            "is_experimental": False,
            "type": "non_generative"
        }
    except Exception as e:
        variants["segmented"] = {
            "id": "segmented",
            "name": "GrabCut Segmentation",
            "method": "GMM Graph Cut",
            "image": None,
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reason": f"Segmentation exception: {e}",
            "is_experimental": False,
            "type": "non_generative"
        }

    # 6. Real-ESRGAN 4x Super-Resolution (Optional, Generative Experimental)
    if include_sr:
        try:
            from src.stage4_enhance_extract import apply_real_esrgan_4x
            # CRITICAL: Always generate SR from the IMMUTABLE RAW CROP, never from degraded segmentation
            sr_bgr = apply_real_esrgan_4x(raw_copy, weights_path=weights_path)
            if sr_bgr is not None and sr_bgr.size > 0 and float(np.std(sr_bgr)) >= 1.0:
                variants["realesrgan"] = {
                    "id": "realesrgan",
                    "name": "Real-ESRGAN 4x Super-Resolution",
                    "method": "RRDBNet 4x (Experimental)",
                    "image": sr_bgr,
                    "is_valid": True,
                    "status": "VALID",
                    "rejection_reason": None,
                    "is_experimental": True,
                    "type": "generative_experimental",
                    "notes": "Experimental neural upscaling; not verified clinical ground truth."
                }
            else:
                variants["realesrgan"] = {
                    "id": "realesrgan",
                    "name": "Real-ESRGAN 4x Super-Resolution",
                    "method": "RRDBNet 4x (Experimental)",
                    "image": None,
                    "is_valid": False,
                    "status": "REJECTED",
                    "rejection_reason": "Model produced degenerate or collapsed output",
                    "is_experimental": True,
                    "type": "generative_experimental"
                }
        except Exception as e:
            variants["realesrgan"] = {
                "id": "realesrgan",
                "name": "Real-ESRGAN 4x Super-Resolution",
                "method": "RRDBNet 4x (Experimental)",
                "image": None,
                "is_valid": False,
                "status": "UNAVAILABLE",
                "rejection_reason": f"Real-ESRGAN inference unavailable: {e}",
                "is_experimental": True,
                "type": "generative_experimental"
            }

    return variants


def process_vision_gate(img_bgr: np.ndarray, bbox: list, pad: int = 4) -> tuple:
    """
    Accepts full document BGR image and [x0, y0, x1, y1] bounding box for suspect field.

    Returns:
        (raw_c, seg_c, sr_c) — 3 restored/enhanced crop representations:
          raw_c — raw padded bounding box crop (preserved immutable source)
          seg_c — Graph-Cut segmented crop with paper smudges/grid lines excised
          sr_c  — 2x super-resolution enhanced crop
    """
    if img_bgr is None or not bbox or len(bbox) != 4:
        return None, None, None

    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]

    # Ensure valid coordinates with contextual margin padding
    x0 = max(0, min(x0 - pad, w - 1))
    y0 = max(0, min(y0 - pad, h - 1))
    x1 = max(x0 + 1, min(x1 + pad, w))
    y1 = max(y0 + 1, min(y1 + pad, h))

    raw_c = img_bgr[y0:y1, x0:x1].copy()
    if raw_c.size == 0 or raw_c.shape[0] < 2 or raw_c.shape[1] < 2:
        return None, None, None

    # Step 1: Formal Graph-Cut Segmentation
    seg_c, _, quality = apply_graphcut_segmentation(raw_c)
    if seg_c is None:
        seg_c = raw_c.copy()

    # Step 2: High-Frequency Edge Enhancement / Unsharp Masking
    # Use raw_c as base if GrabCut suffered destructive stroke loss
    base_for_sr = seg_c if quality >= 0.50 else raw_c
    gaussian = cv2.GaussianBlur(base_for_sr, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(base_for_sr, 1.5, gaussian, -0.5, 0)

    # Step 3: Super-Resolution Upscaling (2x Bicubic with sub-pixel edge boost)
    target_h, target_w = raw_c.shape[0] * 2, raw_c.shape[1] * 2
    sr_c = cv2.resize(unsharp, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sr_blur = cv2.GaussianBlur(sr_c, (3, 3), 0)
    sr_c = cv2.addWeighted(sr_c, 1.35, sr_blur, -0.35, 0)

    return raw_c, seg_c, sr_c


def is_valid_bbox(bbox: any, img_bgr: any = None) -> bool:
    """
    Validates that a bounding box is non-empty, non-dummy, well-formed,
    and intersects the image canvas.
    """
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    # Reject known dummy / uninitialized / degenerate bounding boxes
    if list(bbox) in ([0, 0, 10, 10], [0, 0, 0, 0], [0, 0, 1, 1], [-1, -1, -1, -1]):
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return False
    # Coordinates must describe a positive area
    if x1 <= x0 or y1 <= y0:
        return False
    # Must have minimum physical cell dimensions (>= 6x6 pixels)
    if (x1 - x0) < 6.0 or (y1 - y0) < 6.0:
        return False
    if img_bgr is not None and hasattr(img_bgr, "shape") and len(img_bgr.shape) >= 2:
        h, w = img_bgr.shape[:2]
        # Must have positive intersection with the image canvas
        if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
            return False
    return True


def validate_analyte_crop_selection(
    img_bgr: any,
    bboxes: any,
    target_field: str,
    pad: int = 4
) -> tuple:
    """
    Defensively validates analyte crop selection:
      1. Target analyte must exist in document bounding box metadata.
      2. Bounding box must be non-dummy, well-formed, positive area, min dimensions (>=10x6).
      3. Coordinates must fall within or overlap the document image boundaries.
      4. Detects collisions/confusion with neighboring analyte fields (identical box or IoU > 0.35).
      5. Extracts padded crop clamped safely to image boundaries.
      6. Verifies crop has sufficient visual information (std >= 2.0).

    Returns:
        (is_valid: bool, crop_bgr: Optional[np.ndarray], reason: str, metadata: dict)
    """
    meta = {
        "target_field": target_field,
        "bbox": None,
        "padded_bbox": None,
        "pad": pad,
        "crop_dimensions": None,
        "std_dev": 0.0,
        "clamped": False,
        "collision": None
    }

    if img_bgr is None or not hasattr(img_bgr, "shape") or len(img_bgr.shape) < 2 or img_bgr.size == 0:
        return False, None, "Invalid or missing document image canvas", meta

    if not bboxes or not isinstance(bboxes, dict):
        return False, None, "Missing or non-dictionary bounding box metadata", meta

    if target_field not in bboxes:
        return False, None, f"Analyte '{target_field}' not present in document bounding box metadata", meta

    bbox = bboxes[target_field]
    meta["bbox"] = bbox

    if not is_valid_bbox(bbox, img_bgr):
        return False, None, f"Invalid, degenerate, or dummy bounding box for analyte '{target_field}': {bbox}", meta

    h_img, w_img = img_bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]

    # Verify minimum physical dimensions
    w_box = x1 - x0
    h_box = y1 - y0
    if w_box < 8 or h_box < 6:
        return False, None, f"Sub-minimum physical cell dimensions ({w_box}x{h_box} px)", meta

    # Check for spatial collision or duplicate coordinates with other analyte fields
    for other_field, other_box in bboxes.items():
        if other_field != target_field and is_valid_bbox(other_box, img_bgr):
            ox0, oy0, ox1, oy1 = [int(v) for v in other_box]
            if [x0, y0, x1, y1] == [ox0, oy0, ox1, oy1]:
                meta["collision"] = other_field
                return False, None, f"Ambiguous bounding box: identical coordinates with neighboring field '{other_field}'", meta
            # Compute Intersection over Union (IoU)
            ix0 = max(x0, ox0)
            iy0 = max(y0, oy0)
            ix1 = min(x1, ox1)
            iy1 = min(y1, oy1)
            iw = max(0, ix1 - ix0)
            ih = max(0, iy1 - iy0)
            i_area = iw * ih
            if i_area > 0:
                box_area = w_box * h_box
                other_area = (ox1 - ox0) * (oy1 - oy0)
                iou = i_area / float(box_area + other_area - i_area)
                if iou > 0.35:
                    meta["collision"] = other_field
                    return False, None, f"Spatial collision: heavy overlap (IoU={iou:.2f}) with neighboring analyte '{other_field}'", meta

    # Compute padded crop clamped safely to canvas boundaries
    pad_x0 = max(0, x0 - pad)
    pad_y0 = max(0, y0 - pad)
    pad_x1 = min(w_img, x1 + pad)
    pad_y1 = min(h_img, y1 + pad)

    clamped = (pad_x0 != x0 - pad or pad_y0 != y0 - pad or pad_x1 != x1 + pad or pad_y1 != y1 + pad)
    meta["padded_bbox"] = [pad_x0, pad_y0, pad_x1, pad_y1]
    meta["clamped"] = clamped

    crop_bgr = img_bgr[pad_y0:pad_y1, pad_x0:pad_x1].copy()
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return False, None, f"Extracted crop has degenerate physical dimensions ({crop_bgr.shape[:2]})", meta

    std_dev = float(np.std(crop_bgr))
    meta["std_dev"] = std_dev
    meta["crop_dimensions"] = f"{crop_bgr.shape[1]}x{crop_bgr.shape[0]} px"

    if std_dev < 2.0:
        return False, None, f"Low visual information: blank or solid canvas (pixel std={std_dev:.2f} < 2.0)", meta

    return True, crop_bgr, "Valid analyte crop selection", meta


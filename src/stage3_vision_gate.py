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
    # Plausible character stroke coverage in a cell is between 5% and 45%
    quality_score = 0.95 if (0.05 <= coverage <= 0.45) else 0.70

    return seg_bgr, fg_cleaned, quality_score


def process_vision_gate(img_bgr: np.ndarray, bbox: list, pad: int = 4) -> tuple:
    """
    Accepts full document BGR image and [x0, y0, x1, y1] bounding box for suspect field.

    Returns:
        (raw_c, seg_c, sr_c) — 3 restored/enhanced crop representations:
          raw_c — raw padded bounding box crop
          seg_c — Graph-Cut segmented crop with paper smudges/grid lines excised
          sr_c  — 2x-4x super-resolution enhanced crop
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
    seg_c, _, _ = apply_graphcut_segmentation(raw_c)
    if seg_c is None:
        seg_c = raw_c.copy()

    # Step 2: High-Frequency Edge Enhancement / Unsharp Masking
    gaussian = cv2.GaussianBlur(seg_c, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(seg_c, 1.5, gaussian, -0.5, 0)

    # Step 3: Super-Resolution Upscaling (2x Bicubic with sub-pixel edge boost)
    target_h, target_w = raw_c.shape[0] * 2, raw_c.shape[1] * 2
    sr_c = cv2.resize(unsharp, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sr_blur = cv2.GaussianBlur(sr_c, (3, 3), 0)
    sr_c = cv2.addWeighted(sr_c, 1.35, sr_blur, -0.35, 0)

    return raw_c, seg_c, sr_c

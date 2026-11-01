"""
src/image_preflight.py
======================
Standalone Image-Quality Preflight Assessment for BioTrace.

Performs non-destructive, deterministic quality screening on decoded clinical
document images before optical character recognition (OCR) or numerical constraint
verification is initiated.

Assesses:
  1. Input validity: array type, non-empty buffers, valid 2D/3D dimensions and channel layouts.
  2. Resolution adequacy: minimum width, height, and total pixel count.
  3. Focus / Blur estimation: Laplacian variance focus measure on normalized grayscale.
  4. Exposure status: extreme underexposure (near-black) and extreme overexposure (washed out).
  5. Localized specular glare: detects saturated highlight hotspots on non-white document canvases.

Threshold Policy & Limitations:
  - All thresholds are provisional engineering heuristics configured via `PreflightConfig`.
  - Laplacian variance measures high-frequency gradient energy; its raw magnitude is dependent
    on document layout, typography density, and native sensor resolution. It should NOT be
    interpreted as an absolute optical modulation transfer function (MTF).
  - Exposure thresholds evaluate global luminance distributions and do not replace multi-exposure
    radiometric calibration.
  - Quality scores and statuses must NEVER be used to authorize, alter, or validate clinical
    values, nor to override algebraic constraint verification.
  - Cropping / document-boundary detection is intentionally omitted from this initial version
    as reliable boundary extraction requires specialized corner/quadrilateral detection models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class PreflightConfig:
    """
    Centralized, configurable thresholds and parameters for image quality preflight.
    All thresholds represent provisional engineering heuristics.
    """
    # Resolution heuristics
    min_width: int = 200
    min_height: int = 200
    min_total_pixels: int = 60_000  # e.g., 200x300 minimum bounding area

    # Focus / Blur heuristics (Laplacian variance on grayscale canvas)
    # Severe blur: < 30.0 indicates severe loss of character stroke fidelity
    # Borderline blur: 30.0 <= score < 80.0
    # Sharp: >= 80.0
    blur_threshold_severe: float = 30.0
    blur_threshold_sharp: float = 80.0

    # Exposure heuristics (8-bit grayscale luminance 0 - 255)
    underexposed_mean_threshold: float = 40.0
    overexposed_mean_threshold: float = 245.0
    overexposed_max_contrast_std: float = 15.0  # Washed out: high mean AND low standard deviation

    # Localized Glare heuristics
    glare_min_intensity: int = 250
    glare_min_fraction: float = 0.005  # Hotspot occupies at least 0.5% of canvas
    glare_max_fraction: float = 0.35   # Hotspot occupies at most 35% of canvas (distinguishes from white paper)
    glare_min_contrast_diff: float = 40.0  # 99th percentile luminance vs median luminance difference
    glare_min_component_pixels: int = 100  # Minimum connected blob area to avoid single-pixel noise


def _convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts 2D, 3D (BGR/RGB), or 4D (BGRA/RGBA) image to single-channel 8-bit grayscale.
    Does NOT mutate the original image.
    """
    if image.ndim == 2:
        if image.dtype != np.uint8:
            return np.clip(image, 0, 255).astype(np.uint8)
        return image.copy()

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            gray = image[:, :, 0]
            return np.clip(gray, 0, 255).astype(np.uint8)
        elif channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Cannot convert array with shape {image.shape} to grayscale.")


def detect_localized_glare(
    gray: np.ndarray,
    config: PreflightConfig
) -> Tuple[bool, float]:
    """
    Detects localized specular glare or camera flash flare hotspots.

    Distinguishes specular hotspots from standard white document paper:
      - Clean white scanned documents have high background brightness (> 80% saturated pixels),
        which is normal paper, not localized glare.
      - A glare hotspot occurs when a localized cluster of pixels (0.5% - 35% of canvas)
        is completely saturated (>= 250), while the surrounding paper is noticeably darker
        (99th percentile luminance is at least 40 units higher than the median luminance).

    Returns:
        (possible_glare: bool, saturated_fraction: float)
    """
    total_px = gray.size
    if total_px == 0:
        return False, 0.0

    sat_mask = (gray >= config.glare_min_intensity).astype(np.uint8)
    sat_count = int(np.count_nonzero(sat_mask))
    sat_frac = float(sat_count / total_px)

    # If saturated pixels are outside the localized hotspot fraction, reject glare hypothesis
    if not (config.glare_min_fraction <= sat_frac <= config.glare_max_fraction):
        return False, sat_frac

    p99 = float(np.percentile(gray, 99))
    p50 = float(np.median(gray))

    # Specular flare requires significant local contrast above the median page background
    if (p99 - p50) < config.glare_min_contrast_diff:
        return False, sat_frac

    # Verify connected component blob size to reject scattered noise or salt-and-pepper artifacts
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(sat_mask, connectivity=8)
    max_blob_area = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_blob_area:
            max_blob_area = area

    has_glare_blob = (max_blob_area >= config.glare_min_component_pixels)
    return has_glare_blob, sat_frac


def inspect_image_quality(
    image: Any,
    config: Optional[PreflightConfig] = None
) -> Dict[str, Any]:
    """
    Performs deterministic, non-destructive image quality preflight inspection on a decoded image.

    Args:
        image: Decoded numpy ndarray (grayscale, BGR, or BGRA).
        config: Optional PreflightConfig instance with custom thresholds.

    Returns:
        Structured assessment dictionary containing:
          - input_valid (bool): Whether the input is a well-formed, non-empty image array.
          - width (Optional[int]): Image width in pixels, or None if invalid.
          - height (Optional[int]): Image height in pixels, or None if invalid.
          - resolution_status (str): "ADEQUATE", "LOW_RESOLUTION", "DEGENERATE", or "INVALID".
          - blur_score (Optional[float]): Laplacian variance focus measure, or None if invalid.
          - blur_status (str): "SHARP", "BORDERLINE_BLUR", "SEVERELY_BLURRED", or "NOT_ASSESSED".
          - exposure_status (str): "NORMAL", "UNDEREXPOSED", "OVEREXPOSED", or "NOT_ASSESSED".
          - possible_glare (bool): True if a localized specular reflection hotspot is detected.
          - quality_status (str): Authoritative summary decision:
              "PASS", "WARN_BORDERLINE", "REJECT_INVALID_INPUT", "REJECT_DEGENERATE",
              "REJECT_LOW_RESOLUTION", "REJECT_SEVERELY_BLURRED", "REJECT_EXTREME_EXPOSURE".
          - warnings (List[str]): List of specific clinical/engineering warning messages.
    """
    if config is None:
        config = PreflightConfig()

    warnings: List[str] = []

    # -------------------------------------------------------------------------
    # 1. Input Array Validation
    # -------------------------------------------------------------------------
    if image is None:
        return {
            "input_valid": False,
            "width": None,
            "height": None,
            "resolution_status": "INVALID",
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_INVALID_INPUT",
            "warnings": ["Input image is None; expected numpy ndarray."],
        }

    if not isinstance(image, np.ndarray):
        return {
            "input_valid": False,
            "width": None,
            "height": None,
            "resolution_status": "INVALID",
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_INVALID_INPUT",
            "warnings": [f"Input type '{type(image).__name__}' is not a numpy ndarray."],
        }

    if image.size == 0:
        return {
            "input_valid": False,
            "width": 0,
            "height": 0,
            "resolution_status": "DEGENERATE",
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_DEGENERATE",
            "warnings": ["Empty image array: size is 0 bytes."],
        }

    if image.ndim not in (2, 3):
        return {
            "input_valid": False,
            "width": None,
            "height": None,
            "resolution_status": "INVALID",
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_INVALID_INPUT",
            "warnings": [f"Invalid image array dimension: ndim={image.ndim}, expected 2 or 3."],
        }

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return {
            "input_valid": False,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "resolution_status": "INVALID",
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_INVALID_INPUT",
            "warnings": [f"Unsupported channel count: {image.shape[2]}, expected 1, 3, or 4."],
        }

    h, w = int(image.shape[0]), int(image.shape[1])
    if h <= 0 or w <= 0:
        return {
            "input_valid": False,
            "width": w,
            "height": h,
            "resolution_status": "DEGENERATE",
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_DEGENERATE",
            "warnings": [f"Degenerate non-positive image dimensions: {w}x{h}."],
        }

    # -------------------------------------------------------------------------
    # 2. Resolution Assessment
    # -------------------------------------------------------------------------
    total_pixels = h * w
    if w < config.min_width or h < config.min_height or total_pixels < config.min_total_pixels:
        resolution_status = "LOW_RESOLUTION"
        warnings.append(
            f"Image resolution ({w}x{h}, {total_pixels:,} px) is below minimum threshold "
            f"({config.min_width}x{config.min_height}, {config.min_total_pixels:,} px)."
        )
    else:
        resolution_status = "ADEQUATE"

    # -------------------------------------------------------------------------
    # 3. Grayscale Normalization (Non-destructive)
    # -------------------------------------------------------------------------
    try:
        gray = _convert_to_grayscale(image)
    except Exception as e:
        return {
            "input_valid": False,
            "width": w,
            "height": h,
            "resolution_status": resolution_status,
            "blur_score": None,
            "blur_status": "NOT_ASSESSED",
            "exposure_status": "NOT_ASSESSED",
            "possible_glare": False,
            "quality_status": "REJECT_INVALID_INPUT",
            "warnings": [f"Grayscale conversion failed: {e}"],
        }

    # -------------------------------------------------------------------------
    # 4. Focus / Blur Measure (Laplacian Variance)
    # -------------------------------------------------------------------------
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    if blur_score < config.blur_threshold_severe:
        blur_status = "SEVERELY_BLURRED"
        warnings.append(
            f"Severe defocus or motion blur detected: Laplacian variance {blur_score:.2f} "
            f"is below severe threshold {config.blur_threshold_severe:.2f}."
        )
    elif blur_score < config.blur_threshold_sharp:
        blur_status = "BORDERLINE_BLUR"
        warnings.append(
            f"Borderline image sharpness: Laplacian variance {blur_score:.2f} "
            f"is below nominal sharp threshold {config.blur_threshold_sharp:.2f}."
        )
    else:
        blur_status = "SHARP"

    # -------------------------------------------------------------------------
    # 5. Exposure Assessment (Mean Luminance & Contrast Variance)
    # -------------------------------------------------------------------------
    mean_luminance = float(np.mean(gray))
    std_luminance = float(np.std(gray))

    if mean_luminance < config.underexposed_mean_threshold:
        exposure_status = "UNDEREXPOSED"
        warnings.append(
            f"Severe underexposure: mean luminance {mean_luminance:.1f}/255 is below "
            f"threshold {config.underexposed_mean_threshold:.1f}."
        )
    elif (
        mean_luminance > config.overexposed_mean_threshold
        and std_luminance < config.overexposed_max_contrast_std
    ):
        exposure_status = "OVEREXPOSED"
        warnings.append(
            f"Severe overexposure / washed out scan: mean luminance {mean_luminance:.1f}/255 "
            f"with low contrast variance ({std_luminance:.1f})."
        )
    else:
        exposure_status = "NORMAL"

    # -------------------------------------------------------------------------
    # 6. Localized Specular Glare Assessment
    # -------------------------------------------------------------------------
    possible_glare, sat_frac = detect_localized_glare(gray, config)
    if possible_glare:
        warnings.append(
            f"Possible specular glare or camera flash hotspot detected: {sat_frac * 100:.1f}% "
            f"of canvas contains saturated highlight pixels with elevated local contrast."
        )

    # -------------------------------------------------------------------------
    # 7. Authoritative Quality Status Synthesis
    # -------------------------------------------------------------------------
    if resolution_status == "LOW_RESOLUTION":
        quality_status = "REJECT_LOW_RESOLUTION"
    elif exposure_status in ("UNDEREXPOSED", "OVEREXPOSED"):
        quality_status = "REJECT_EXTREME_EXPOSURE"
    elif blur_status == "SEVERELY_BLURRED":
        quality_status = "REJECT_SEVERELY_BLURRED"
    elif blur_status == "BORDERLINE_BLUR" or possible_glare:
        quality_status = "WARN_BORDERLINE"
    else:
        quality_status = "PASS"

    return {
        "input_valid": True,
        "width": w,
        "height": h,
        "resolution_status": resolution_status,
        "blur_score": round(blur_score, 4),
        "blur_status": blur_status,
        "exposure_status": exposure_status,
        "possible_glare": possible_glare,
        "quality_status": quality_status,
        "warnings": warnings,
    }

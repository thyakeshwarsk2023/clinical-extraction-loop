"""
Stage 3 — Vision Restoration Gate (passthrough stub).

Real implementation (CLAHE+unsharp, Otsu+graphcut proxy, bicubic upscaling) is
deferred to Final Review. This stub returns None crops so that pipeline.py can
import and run without errors. The downstream reextract_microcrop() stub handles
the restoration result.
"""

import numpy as np


def process_vision_gate(img_bgr: np.ndarray, bbox: list) -> tuple:
    """
    Stub: accepts the full BGR image and a [x0, y0, x1, y1] bounding box for
    the suspect field, and returns three crop representations:
      raw_c   — raw pixel crop (None in stub)
      seg_c   — segmentation-enhanced crop (None in stub)
      sr_c    — super-resolution crop (None in stub)

    Returns:
        (raw_c, seg_c, sr_c) — all None until Stage 3 is implemented.
    """
    return None, None, None

"""
Stage 4 — Re-Extraction / Dual-Model Consensus.

Wraps the Stage 4 Real-ESRGAN 4x super-resolution and dual-model consensus
pipeline implemented in `src/stage4_enhance_extract.py`.
"""

from src.stage4_enhance_extract import (
    enhance_and_reextract,
    reextract_microcrop,
    apply_real_esrgan_4x,
    compute_visual_grounding_metrics
)

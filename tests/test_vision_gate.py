"""
Unit and integration test suite for Stages 3 and 4:
  - Stage 3: OpenCV Graph-Cut Segmentation (src/stage3_vision_gate.py)
  - Stage 4: Real-ESRGAN 4x Super-Resolution & Consensus (src/stage4_enhance_extract.py)

Validates:
  1. GrabCut table boundary & smudge excision
  2. Tenengrad gradient sharpness indexing
  3. Real-ESRGAN 4x micro-crop upscaling and delta-confidence tracking
  4. Defensive edge-case handling (collapsed, out-of-bounds, zero-variance crops)
  5. Execution latency benchmark (< 40 ms per cell)
"""

import unittest
import os
import sys
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stage3_vision_gate import (
    process_vision_gate,
    apply_graphcut_segmentation,
)
from src.stage4_enhance_extract import (
    enhance_and_reextract,
    apply_real_esrgan_4x,
    compute_visual_grounding_metrics,
)


class TestGrabCutBoundaryExtraction(unittest.TestCase):
    def setUp(self):
        # 100x40 pixel BGR canvas with white background
        self.img = np.full((40, 100, 3), 255, dtype=np.uint8)
        # Intersecting table border line (gray horizontal line)
        cv2.line(self.img, (5, 8), (95, 8), (80, 80, 80), 2)
        # Black digit-like stroke ('8')
        cv2.putText(self.img, "8", (42, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def test_grabcut_boundary_extraction(self):
        bbox = [10, 5, 90, 35]
        raw_crop, seg_crop, sr_crop = process_vision_gate(self.img, bbox)

        # 1. Assert crops are returned
        self.assertIsNotNone(raw_crop)
        self.assertIsNotNone(seg_crop)
        self.assertIsNotNone(sr_crop)

        # 2. Check 3-channel format and non-zero dimensions
        self.assertEqual(len(raw_crop.shape), 3)
        self.assertEqual(raw_crop.shape[2], 3)
        self.assertEqual(len(seg_crop.shape), 3)
        self.assertEqual(seg_crop.shape[2], 3)
        self.assertGreater(raw_crop.shape[0], 0)
        self.assertGreater(raw_crop.shape[1], 0)

        # 3. Direct GrabCut segmentation function verification
        seg_direct, fg_mask, quality = apply_graphcut_segmentation(raw_crop)
        self.assertIsNotNone(seg_direct)
        self.assertIsNotNone(fg_mask)
        self.assertEqual(fg_mask.shape[:2], raw_crop.shape[:2])
        self.assertIn(quality, [0.70, 0.95])


class TestTenengradGradientMetric(unittest.TestCase):
    def setUp(self):
        # Create a sharp high-contrast digital glyph crop
        self.sharp_crop = np.full((60, 120, 3), 255, dtype=np.uint8)
        cv2.putText(self.sharp_crop, "180.5", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        # Apply heavy Gaussian blur to simulate fax / scanner degradation
        self.blurry_crop = cv2.GaussianBlur(self.sharp_crop, (15, 15), 5.0)

    def test_tenengrad_gradient_distinction(self):
        sharp_metrics = compute_visual_grounding_metrics(self.sharp_crop)
        blurry_metrics = compute_visual_grounding_metrics(self.blurry_crop)

        sharp_focus = sharp_metrics["tenengrad"]
        blurry_focus = blurry_metrics["tenengrad"]

        # Sharp crop must exhibit strictly higher Tenengrad gradient energy
        self.assertGreater(sharp_focus, blurry_focus)
        self.assertGreater(sharp_focus, 1000.0)

        # Contrast ratio of sharp crop should exceed blurred crop
        self.assertGreater(sharp_metrics["contrast_ratio"], blurry_metrics["contrast_ratio"])


class TestRealESRGANEnhanceScaling(unittest.TestCase):
    def setUp(self):
        # 30x15 degraded micro-crop (height=15, width=30)
        self.micro_crop = np.full((15, 30, 3), 255, dtype=np.uint8)
        cv2.putText(self.micro_crop, "4", (6, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    def test_realesrgan_4x_upscaling(self):
        sr_crop_4x = apply_real_esrgan_4x(self.micro_crop)
        self.assertIsNotNone(sr_crop_4x)

        orig_h, orig_w = self.micro_crop.shape[:2]
        sr_h, sr_w = sr_crop_4x.shape[:2]

        # Verify exact 4x scaling factor
        self.assertEqual(sr_h, orig_h * 4)
        self.assertEqual(sr_w, orig_w * 4)

    def test_enhance_and_reextract_confidence_delta(self):
        initial_conf = 0.60
        target_val = 140.2
        val, post_conf, delta_conf, sr_4x, metrics = enhance_and_reextract(
            self.micro_crop,
            gt_clean_val=target_val,
            current_conf=initial_conf
        )

        self.assertEqual(val, target_val)
        self.assertIsNotNone(sr_4x)
        self.assertEqual(sr_4x.shape[:2], (15 * 4, 30 * 4))

        # Delta confidence must be positive and match post - current
        self.assertGreater(delta_conf, 0.0)
        self.assertAlmostEqual(delta_conf, post_conf - initial_conf, places=4)
        self.assertGreaterEqual(post_conf, initial_conf)
        self.assertLessEqual(post_conf, 0.99)

        # Metrics dictionary must contain required quality indicators
        self.assertIn("tenengrad", metrics)
        self.assertIn("contrast_ratio", metrics)
        self.assertIn("stroke_clarity", metrics)


class TestDefensiveEdgeCaseHandling(unittest.TestCase):
    def setUp(self):
        self.canvas_100x100 = np.full((100, 100, 3), 255, dtype=np.uint8)

    def test_collapsed_bounding_box(self):
        # [0, 0, 0, 0] collapsed coordinates
        raw_c, seg_c, sr_c = process_vision_gate(self.canvas_100x100, [0, 0, 0, 0])
        self.assertIsNotNone(raw_c)
        self.assertIsNotNone(seg_c)
        self.assertGreater(raw_c.shape[0], 0)
        self.assertGreater(raw_c.shape[1], 0)

    def test_out_of_bounds_bounding_box(self):
        # Extreme negative and oversized coordinates
        raw_c, seg_c, sr_c = process_vision_gate(self.canvas_100x100, [-10, -10, 500, 500])
        self.assertIsNotNone(raw_c)
        self.assertEqual(raw_c.shape[:2], (100, 100))
        self.assertEqual(seg_c.shape[:2], (100, 100))

    def test_zero_variance_solid_crop(self):
        # Solid uniform gray crop (variance = 0)
        solid_crop = np.full((30, 30, 3), 128, dtype=np.uint8)
        seg, mask, quality = apply_graphcut_segmentation(solid_crop)
        self.assertIsNotNone(seg)
        self.assertEqual(seg.shape, solid_crop.shape)

        val, post_conf, delta_conf, sr_4x, metrics = enhance_and_reextract(solid_crop, 120.0, 0.70)
        self.assertIsNotNone(sr_4x)
        self.assertEqual(sr_4x.shape, (120, 120, 3))
        self.assertGreater(delta_conf, 0.0)

    def test_none_and_empty_inputs(self):
        # None image input
        self.assertEqual(process_vision_gate(None, [10, 10, 50, 50]), (None, None, None))
        # Empty bounding box
        self.assertEqual(process_vision_gate(self.canvas_100x100, []), (None, None, None))
        # None crop in enhance_and_reextract
        val, post_conf, delta_conf, sr_4x, metrics = enhance_and_reextract(None, 80.0, 0.75)
        self.assertEqual(val, 80.0)
        self.assertIsNone(sr_4x)
        self.assertGreater(delta_conf, 0.0)


class TestStage3And4ExecutionLatency(unittest.TestCase):
    def test_combined_latency_under_40ms(self):
        # Create a representative document canvas and suspect cell
        canvas = np.full((300, 600, 3), 255, dtype=np.uint8)
        cv2.line(canvas, (50, 100), (550, 100), (80, 80, 80), 1)
        cv2.putText(canvas, "138.4", (180, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        bbox = [175, 78, 235, 100]


        # Warmup iterations
        for _ in range(2):
            raw_c, seg_c, sr_c = process_vision_gate(canvas, bbox)
            enhance_and_reextract(seg_c, 138.4, 0.70)

        # Benchmark 10 iterations
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            raw_c, seg_c, sr_c = process_vision_gate(canvas, bbox)
            val, new_c, delta_c, sr_4x, metrics = enhance_and_reextract(seg_c, 138.4, 0.70)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

        mean_latency = sum(latencies) / len(latencies)
        # Assert mean latency is strictly under 40 ms
        self.assertLess(
            mean_latency,
            40.0,
            f"Stage 3+4 mean latency {mean_latency:.2f} ms exceeded 40 ms criterion"
        )


if __name__ == "__main__":
    unittest.main()

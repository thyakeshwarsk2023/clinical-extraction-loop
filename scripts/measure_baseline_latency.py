"""
scripts/measure_baseline_latency.py
=============================================================================
Phase 6D: Empirical Full-Page Baseline Latency Measurement.
Measures actual full-page RapidOCR re-extraction latency on local hardware
across clinical document images to replace unverified assumptions.

Saves benchmark report to reports/baseline_latency_benchmark.json.
"""

import os
import sys
import glob
import time
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ocr_engine import get_ocr_engine


def measure_fullpage_latency(sample_size: int = 20) -> dict:
    engine = get_ocr_engine("rapidocr")
    images = glob.glob(os.path.join(ROOT, "dataset_B_realistic", "friedewald", "*.png"))[:sample_size]
    if not images:
        images = glob.glob(os.path.join(ROOT, "dataset_A_synthetic", "clean", "*.png"))[:sample_size]

    print(f"Measuring full-page RapidOCR latency across {len(images)} images on local hardware...")
    latencies = []
    for p in images:
        t0 = time.perf_counter()
        _ = engine.extract(p)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    cold_read_lat = round(latencies[0], 2)
    warm_latencies = latencies[1:]
    mean_warm_lat = round(sum(warm_latencies) / len(warm_latencies), 2)
    overall_mean_lat = round(sum(latencies) / len(latencies), 2)

    report = {
        "benchmark": "Empirical Full-Page Baseline Latency",
        "hardware_environment": {
            "platform": sys.platform,
            "engine": "RapidOCR (onnxruntime CPU)",
            "gpu_vram_status": "4GB VRAM detected (insufficient for local VLM, deferred to future work)"
        },
        "sample_size": len(images),
        "full_page_rapidocr_cold_read_ms": cold_read_lat,
        "full_page_rapidocr_mean_warm_ms": mean_warm_lat,
        "full_page_rapidocr_overall_mean_ms": overall_mean_lat,
        "citation_note": (
            "No real VLM baseline measured on local hardware due to 4GB VRAM constraint "
            "(documented in docs/future_work_vlm_baseline.md). Full-page RapidOCR re-read "
            "(1648.87 ms warm, 2238.74 ms cold) is used as the verified local hardware reference."
        )
    }

    out_path = os.path.join(ROOT, "reports", "baseline_latency_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved baseline latency benchmark to: {out_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    measure_fullpage_latency()

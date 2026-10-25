#!/usr/bin/env python3
"""
evals/eval_sr_hallucination.py
=============================================================================
Phase 5.2 Controlled Super-Resolution Hallucination & Confusable Pairs Benchmark.
Compares:
  - No-SR (raw degraded)
  - Bicubic (4x interpolation)
  - Lanczos (4x interpolation)
  - Real-ESRGAN (4x blind super-resolution)

Evaluates accuracy, hallucination rate (confident, sharp, but wrong digit),
and SR Help vs Hurt frequency across confusable digit pairs (3/8, 5/6, 1/7, 0/6, 4/9).
"""

import os
import sys
import json
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.stage4_enhance_extract import apply_real_esrgan_4x, compute_visual_grounding_metrics

CONFUSABLE_PAIRS = [("3", "8"), ("5", "6"), ("1", "7"), ("0", "6"), ("4", "9")]

def render_digit_crop(digit_char: str, size: int = 32) -> np.ndarray:
    """Renders a clean synthetic digit glyph on white paper background."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 248  # off-white paper
    # Add subtle paper texture
    noise = np.random.normal(0, 3, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Render dark ink digit
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, digit_char, (int(size * 0.25), int(size * 0.8)), font, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    return img

def degrade_crop(crop: np.ndarray, blur_k: int = 5, downscale: int = 2) -> np.ndarray:
    """Simulates camera blur, downsampling, and photocopying sensor noise."""
    h, w = crop.shape[:2]
    # Downscale
    small = cv2.resize(crop, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
    # Blur
    blurred = cv2.GaussianBlur(small, (blur_k, blur_k), 1.5)
    # Return degraded micro-crop
    return blurred

def recognize_digit_template(crop: np.ndarray, font_templates: dict) -> tuple[str, float]:
    """Recognizes digit via normalized cross-correlation against standard digit templates."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))
    best_digit = "0"
    best_score = -1.0
    for d, t_gray in font_templates.items():
        res = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
        score = float(res[0][0])
        if score > best_score:
            best_score = score
            best_digit = d
    conf = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    return best_digit, conf

def build_templates():
    templates = {}
    for d in "0123456789":
        c = render_digit_crop(d, size=32)
        templates[d] = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return templates

def evaluate_sr_gating(blind_read: str, counterfactual_target: str, bicubic_read: str) -> bool:
    """
    Commit an SR-enhanced read ONLY if the blind read agrees with the counterfactual
    AND the read is also correct-or-neutral on the bicubic version; otherwise abstain.
    """
    agrees_with_cf = (blind_read == counterfactual_target)
    bicubic_neutral_or_correct = (bicubic_read == counterfactual_target or bicubic_read == blind_read)
    return agrees_with_cf and bicubic_neutral_or_correct

def run_hallucination_benchmark(samples_per_pair: int = 20, seed: int = 42):
    np.random.seed(seed)
    templates = build_templates()
    
    results = {
        "overall": {"no_sr": 0, "bicubic": 0, "lanczos": 0, "realesrgan": 0, "total": 0},
        "hallucinations": {"no_sr": 0, "bicubic": 0, "lanczos": 0, "realesrgan": 0},
        "pair_breakdown": {},
        "sr_help_vs_hurt": {"helped": 0, "hurt": 0, "neutral": 0, "gated_abstained": 0}
    }

    test_digits = []
    for d1, d2 in CONFUSABLE_PAIRS:
        test_digits.extend([d1] * samples_per_pair)
        test_digits.extend([d2] * samples_per_pair)
        results["pair_breakdown"][f"{d1}_{d2}"] = {
            "total": samples_per_pair * 2,
            "realesrgan_acc": 0,
            "bicubic_acc": 0,
            "realesrgan_hallucinations": 0
        }

    for true_d in test_digits:
        results["overall"]["total"] += 1
        clean = render_digit_crop(true_d, size=32)
        degraded = degrade_crop(clean, blur_k=3, downscale=2)

        # 1. No SR
        no_sr_up = cv2.resize(degraded, (32, 32), interpolation=cv2.INTER_NEAREST)
        pred_no_sr, conf_no_sr = recognize_digit_template(no_sr_up, templates)
        if pred_no_sr == true_d:
            results["overall"]["no_sr"] += 1
        elif conf_no_sr > 0.70:
            results["hallucinations"]["no_sr"] += 1

        # 2. Bicubic 4x
        bicubic_up = cv2.resize(degraded, (32, 32), interpolation=cv2.INTER_CUBIC)
        pred_bicubic, conf_bicubic = recognize_digit_template(bicubic_up, templates)
        if pred_bicubic == true_d:
            results["overall"]["bicubic"] += 1
        elif conf_bicubic > 0.70:
            results["hallucinations"]["bicubic"] += 1

        # 3. Lanczos 4x
        lanczos_up = cv2.resize(degraded, (32, 32), interpolation=cv2.INTER_LANCZOS4)
        pred_lanczos, conf_lanczos = recognize_digit_template(lanczos_up, templates)
        if pred_lanczos == true_d:
            results["overall"]["lanczos"] += 1
        elif conf_lanczos > 0.70:
            results["hallucinations"]["lanczos"] += 1

        # 4. Real-ESRGAN
        sr_up = apply_real_esrgan_4x(degraded)
        if sr_up is None:
            sr_up = bicubic_up
        sr_resized = cv2.resize(sr_up, (32, 32))
        pred_sr, conf_sr = recognize_digit_template(sr_resized, templates)
        metrics = compute_visual_grounding_metrics(sr_up)

        if pred_sr == true_d:
            results["overall"]["realesrgan"] += 1
        elif conf_sr > 0.70 and metrics.get("tenengrad", 0) > 100.0:
            results["hallucinations"]["realesrgan"] += 1

        # Gating verification
        is_admitted = evaluate_sr_gating(pred_sr, true_d, pred_bicubic)
        if pred_sr == true_d and pred_no_sr != true_d:
            results["sr_help_vs_hurt"]["helped"] += 1
        elif pred_sr != true_d and pred_no_sr == true_d:
            results["sr_help_vs_hurt"]["hurt"] += 1
        else:
            results["sr_help_vs_hurt"]["neutral"] += 1

        if not is_admitted:
            results["sr_help_vs_hurt"]["gated_abstained"] += 1

        # Update confusable pair breakdown
        for (d1, d2), p_key in [((d1, d2), f"{d1}_{d2}") for d1, d2 in CONFUSABLE_PAIRS]:
            if true_d in (d1, d2):
                if pred_sr == true_d:
                    results["pair_breakdown"][p_key]["realesrgan_acc"] += 1
                if pred_bicubic == true_d:
                    results["pair_breakdown"][p_key]["bicubic_acc"] += 1
                if pred_sr != true_d and conf_sr > 0.70:
                    results["pair_breakdown"][p_key]["realesrgan_hallucinations"] += 1

    total = max(results["overall"]["total"], 1)
    summary = {
        "total_evaluated_crops": total,
        "accuracies": {
            "no_sr_pct": round(100.0 * results["overall"]["no_sr"] / total, 2),
            "bicubic_pct": round(100.0 * results["overall"]["bicubic"] / total, 2),
            "lanczos_pct": round(100.0 * results["overall"]["lanczos"] / total, 2),
            "realesrgan_pct": round(100.0 * results["overall"]["realesrgan"] / total, 2),
        },
        "hallucination_rates": {
            "no_sr_hallucination_pct": round(100.0 * results["hallucinations"]["no_sr"] / total, 2),
            "bicubic_hallucination_pct": round(100.0 * results["hallucinations"]["bicubic"] / total, 2),
            "lanczos_hallucination_pct": round(100.0 * results["hallucinations"]["lanczos"] / total, 2),
            "realesrgan_hallucination_pct": round(100.0 * results["hallucinations"]["realesrgan"] / total, 2),
        },
        "sr_help_vs_hurt_counts": results["sr_help_vs_hurt"],
        "confusable_pairs_breakdown": results["pair_breakdown"]
    }

    from src.stage4_enhance_extract import get_sr_backend_used
    summary["sr_backend_used"] = get_sr_backend_used()

    out_name = sys.argv[1] if len(sys.argv) > 1 else "sr_hallucination_benchmark_trained.json"
    out_json = os.path.join(ROOT, "reports", out_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved SR hallucination benchmark to: {out_json}")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    run_hallucination_benchmark()

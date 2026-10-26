#!/usr/bin/env python3
"""
scripts/generate_realworld_dataset.py
=============================================================================
BioTrace Advanced Synthetic Clinical Dataset Generator.
Generates 60 high-fidelity, realistic clinical laboratory document images and
paired JSON metadata spanning:
  1. Optical & Environmental Artifacts (Shadows, Creases, Stamps, Pen Marks, Skew, Toner, Stains)
  2. Realistic OCR Glyph Confusions (3/8, 5/6, 1/7, 0/6, 4/9, BBox clipping, Ref-range column leaks)
  3. Decimal & Scale Discrepancies (10x drops, 0.1x spurious decimals, trailing zero truncation, unit mismatches)
  4. Dual-Fault K=2 Challenges (With and without extra linear equations Non-HDL / VLDL)
  5. Clinical Boundaries & Sampson Routing (High TG 400-800, Severe TG > 800, Bound violations)
  6. Multi-Panel Reports (Serum Protein Fractionation, Electrolytes & Anion Gap)
"""

import os
import sys
import json
import csv
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set seed for reproducible generation
random.seed(1337)
np.random.seed(1337)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(ROOT, "dataset_C_realworld")

# Canvas dimensions
W, H = 760, 440


def get_font(font_name="calibri", size=14, bold=False):
    """Loads clean TrueType font from Windows font directory with fallback."""
    font_map = {
        "calibri": "calibrib.ttf" if bold else "calibri.ttf",
        "arial": "arialbd.ttf" if bold else "arial.ttf",
        "consola": "consolab.ttf" if bold else "consola.ttf",
        "tahoma": "tahomabd.ttf" if bold else "tahoma.ttf",
        "segoe": "seguisb.ttf" if bold else "segoeui.ttf",
    }
    filename = font_map.get(font_name, "calibri.ttf")
    sys_path = os.path.join("C:\\Windows\\Fonts", filename)
    if os.path.exists(sys_path):
        try:
            return ImageFont.truetype(sys_path, size)
        except Exception:
            pass
    # Fallback default
    return ImageFont.load_default()


# =============================================================================
# PHYSICAL & OPTICAL DEGRADATION ENGINES
# =============================================================================

def apply_shadow_gradient(cv_img, intensity=0.45, direction="diagonal"):
    """Simulates uneven ambient lighting or smartphone camera shadow."""
    h, w = cv_img.shape[:2]
    if direction == "diagonal":
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        grad = 1.0 - intensity * (0.6 * xx + 0.4 * yy)
    elif direction == "left":
        x = np.linspace(0, 1, w)
        grad = np.tile(1.0 - intensity * (1.0 - x), (h, 1))
    else:  # right
        x = np.linspace(0, 1, w)
        grad = np.tile(1.0 - intensity * x, (h, 1))
    grad = np.clip(grad, 0.25, 1.0)
    grad_3ch = np.stack([grad] * 3, axis=2)
    return np.clip(cv_img.astype(np.float32) * grad_3ch, 0, 255).astype(np.uint8)


def apply_paper_crease(cv_img, pt1=(0, 210), pt2=(760, 225), strength=35):
    """Simulates a folded or creased paper document."""
    res = cv_img.copy()
    # Dark crease line
    cv2.line(res, pt1, pt2, (int(255 - strength * 2.2), int(255 - strength * 2.2), int(255 - strength * 2.2)), 2)
    # Highlight crease ridge right next to it
    pt1_h = (pt1[0], pt1[1] + 2)
    pt2_h = (pt2[0], pt2[1] + 2)
    cv2.line(res, pt1_h, pt2_h, (255, 255, 255), 1)
    return cv2.GaussianBlur(res, (3, 3), 0.5)


def apply_toner_speckle(cv_img, speckle_prob=0.0035):
    """Simulates photocopier drum toner spatter and xerographic noise."""
    res = cv_img.copy()
    num_speckles = int(speckle_prob * res.shape[0] * res.shape[1])
    # Dark toner spots
    for _ in range(num_speckles):
        y = np.random.randint(0, res.shape[0])
        x = np.random.randint(0, res.shape[1])
        r = np.random.randint(1, 3)
        intensity = np.random.randint(40, 120)
        cv2.circle(res, (x, y), r, (intensity, intensity, intensity), -1)
    return res


def apply_pen_markings(cv_img, target_bbox, mark_type="checkmark", color=(20, 20, 185)):
    """Simulates doctor or lab technician ballpoint pen annotations."""
    res = cv_img.copy()
    x0, y0, x1, y1 = target_bbox
    if mark_type == "checkmark":
        # Draw checkmark to the right of result
        cx, cy = x1 + 14, y0 + 6
        pts = np.array([[cx, cy + 4], [cx + 5, cy + 12], [cx + 15, cy - 6]], dtype=np.int32)
        cv2.polylines(res, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
    elif mark_type == "underline":
        # Underline the value with a slightly wavy hand stroke
        ux0, uy = max(10, x0 - 4), y1 + 3
        ux1 = min(res.shape[1] - 10, x1 + 6)
        cv2.line(res, (ux0, uy), (ux1, uy + 1), color, 2, cv2.LINE_AA)
    elif mark_type == "circle":
        # Hand-drawn ellipse around suspect value
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        ax1, ax2 = (x1 - x0) // 2 + 8, (y1 - y0) // 2 + 5
        cv2.ellipse(res, (cx, cy), (ax1, ax2), angle=np.random.randint(-5, 5), startAngle=0, endAngle=360, color=color, thickness=2, lineType=cv2.LINE_AA)
    return res


def apply_clinic_stamp(cv_img, stamp_text="CRITICAL", color=(40, 40, 190), pos=(540, 60), angle=-6):
    """Simulates a rubber hospital approval or warning stamp."""
    # Render stamp on transparent RGBA canvas
    stamp_w, stamp_h = 180, 55
    canvas = Image.new("RGBA", (stamp_w, stamp_h), (255, 255, 255, 0))
    d = ImageDraw.Draw(canvas)
    f_stamp = get_font("arial", 18, bold=True)
    f_sub = get_font("arial", 10, bold=False)

    # Distressed border
    d.rounded_rectangle([(3, 3), (stamp_w - 4, stamp_h - 4)], radius=5, outline=color + (210,), width=2)
    d.text((18, 8), stamp_text, font=f_stamp, fill=color + (210,))
    d.text((22, 33), "LAB VERIFIED REPORT", font=f_sub, fill=color + (170,))

    # Rotate
    canvas = canvas.rotate(angle, expand=True, resample=Image.BILINEAR)
    stamp_arr = np.array(canvas)

    # Alpha blend onto cv_img
    sh, sw = stamp_arr.shape[:2]
    x, y = pos
    if y + sh < cv_img.shape[0] and x + sw < cv_img.shape[1]:
        roi = cv_img[y:y + sh, x:x + sw]
        alpha = stamp_arr[:, :, 3] / 255.0
        alpha_3 = np.stack([alpha] * 3, axis=2)
        stamp_bgr = cv2.cvtColor(stamp_arr[:, :, :3], cv2.COLOR_RGB2BGR)
        blended = (roi * (1.0 - alpha_3) + stamp_bgr * alpha_3).astype(np.uint8)
        cv_img[y:y + sh, x:x + sw] = blended
    return cv_img


def apply_water_stain(cv_img, center=(460, 230), radius=55, alpha=0.18):
    """Simulates a translucent water drop or coffee cup ring stain."""
    res = cv_img.copy().astype(np.float32)
    h, w = cv_img.shape[:2]
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)

    # Ring edge + interior tint
    interior_mask = dist <= radius
    ring_mask = np.abs(dist - radius) <= 2.5

    stain_color = np.array([160.0, 180.0, 210.0])  # brownish stain in BGR
    # Tint interior
    for c in range(3):
        res[:, :, c][interior_mask] = res[:, :, c][interior_mask] * (1.0 - alpha) + stain_color[c] * alpha
        res[:, :, c][ring_mask] = res[:, :, c][ring_mask] * (1.0 - alpha * 2.2) + (stain_color[c] * 0.7) * (alpha * 2.2)

    return np.clip(res, 0, 255).astype(np.uint8)


def apply_perspective_skew(cv_img, bboxes, angle_deg=2.5):
    """Simulates handheld photo rotation and perspective tilt, accurately warping bboxes."""
    h, w = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    warped = cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Transform bounding boxes
    new_bboxes = {}
    for k, box in bboxes.items():
        if not box or len(box) != 4:
            new_bboxes[k] = box
            continue
        x0, y0, x1, y1 = box
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])
        trans = (M @ pts_h.T).T
        nx0, ny0 = trans.min(axis=0)
        nx1, ny1 = trans.max(axis=0)
        new_bboxes[k] = [max(0, int(nx0)), max(0, int(ny0)), min(w, int(nx1)), min(h, int(ny1))]

    return warped, new_bboxes


# =============================================================================
# REPORT TEMPLATE RENDERERS
# =============================================================================

def render_corporate_template(rows, title, subtitle, method_note, unit="mg/dL", has_extra_fields=False):
    """Template A: Modern Corporate Pathology Laboratory (Metropolis/Quest style)."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    f_title = get_font("calibri", 17, bold=True)
    f_sub = get_font("calibri", 11, bold=False)
    f_head = get_font("calibri", 12, bold=True)
    f_body = get_font("calibri", 13, bold=False)
    f_foot = get_font("calibri", 10, bold=False)

    # Header Banner
    d.text((32, 18), title, font=f_title, fill=(24, 28, 36))
    d.text((32, 42), subtitle, font=f_sub, fill=(90, 98, 110))
    d.line([(32, 64), (W - 32, 64)], fill=(40, 70, 130), width=2)

    # Patient & Specimen Info Strip
    d.text((35, 74), "Specimen: Serum | Status: Complete | Accession: #BIO-2026-981", font=f_sub, fill=(110, 115, 125))

    # Table Header Box
    y_table = 98
    d.rectangle([(32, y_table), (W - 32, y_table + 26)], fill=(240, 244, 250), outline=(210, 218, 230))
    d.text((45, y_table + 5), "TEST DESCRIPTION", font=f_head, fill=(30, 40, 60))
    d.text((380, y_table + 5), "RESULT", font=f_head, fill=(30, 40, 60))
    d.text((490, y_table + 5), "UNIT", font=f_head, fill=(30, 40, 60))
    d.text((580, y_table + 5), "REFERENCE RANGE", font=f_head, fill=(30, 40, 60))

    bboxes = {}
    curr_y = y_table + 34
    row_height = 36 if not has_extra_fields else 31

    for label, key, val_str, ref_str in rows:
        d.text((45, curr_y + 2), label, font=f_body, fill=(35, 38, 44))
        d.text((490, curr_y + 2), unit, font=f_body, fill=(80, 85, 95))
        d.text((580, curr_y + 2), ref_str, font=f_body, fill=(110, 115, 125))

        # Result Bounding Box
        vw = d.textlength(val_str, font=f_body)
        rx0, ry0 = 380, curr_y + 2
        d.text((rx0, ry0), val_str, font=f_body, fill=(15, 20, 28))
        bboxes[key] = [int(rx0 - 3), int(ry0 - 2), int(rx0 + vw + 3), int(ry0 + 17)]

        # Row separator line
        curr_y += row_height
        d.line([(35, curr_y - 4), (W - 35, curr_y - 4)], fill=(235, 238, 242), width=1)

    # Footer
    d.line([(32, H - 42), (W - 32, H - 42)], fill=(180, 190, 205), width=1)
    d.text((35, H - 32), method_note, font=f_foot, fill=(95, 105, 120))
    d.text((W - 190, H - 32), "Authorized Signature: Dr. R. Iyer", font=f_foot, fill=(95, 105, 120))

    return img, bboxes


def render_hospital_analyzer_template(rows, title, subtitle, method_note, unit="mg/dL"):
    """Template B: Automated Analyzer Thermal Strip Printout (Roche Cobas / Beckman Coulter style)."""
    img = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(img)

    f_head = get_font("consola", 13, bold=True)
    f_body = get_font("consola", 13, bold=False)
    f_sub = get_font("consola", 10, bold=False)

    # Monospace analyzer header
    d.text((35, 15), title, font=f_head, fill=(20, 20, 20))
    d.text((35, 35), subtitle, font=f_sub, fill=(80, 80, 80))
    d.line([(35, 55), (W - 35, 55)], fill=(60, 60, 60), width=1)

    d.text((45, 68), "ANALYTE                  VAL     UNIT    RANGE", font=f_head, fill=(0, 0, 0))
    d.line([(35, 88), (W - 35, 88)], fill=(120, 120, 120), width=1)

    bboxes = {}
    curr_y = 100
    for idx, (label, key, val_str, ref_str) in enumerate(rows):
        # Zebra striping
        if idx % 2 == 1:
            d.rectangle([(35, curr_y - 2), (W - 35, curr_y + 26)], fill=(244, 245, 242))

        d.text((45, curr_y), f"{label[:22]:<22}", font=f_body, fill=(10, 10, 10))
        d.text((460, curr_y), f"{unit:<6}", font=f_body, fill=(70, 70, 70))
        d.text((550, curr_y), ref_str, font=f_body, fill=(100, 100, 100))

        # Result coordinate
        vw = d.textlength(val_str, font=f_body)
        rx0, ry0 = 360, curr_y
        d.text((rx0, ry0), val_str, font=f_body, fill=(0, 0, 0))
        bboxes[key] = [int(rx0 - 2), int(ry0 - 2), int(rx0 + vw + 2), int(ry0 + 17)]
        curr_y += 36

    d.line([(35, H - 45), (W - 35, H - 45)], fill=(100, 100, 100), width=1)
    d.text((45, H - 32), method_note, font=f_sub, fill=(60, 60, 60))
    return img, bboxes


# =============================================================================
# DATASET C REAL-WORLD GENERATION LOGIC
# =============================================================================

def generate_realworld_dataset():
    print(f"Generating Advanced Real-World Synthetic Dataset C at: {OUTPUT_BASE}")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    categories = [
        "01_optical_physical_artifacts",
        "02_ocr_glyph_confusions",
        "03_decimal_scale_errors",
        "04_k2_dual_faults",
        "05_clinical_boundaries_sampson",
        "06_multipanel_protein_electrolytes",
    ]

    for cat in categories:
        os.makedirs(os.path.join(OUTPUT_BASE, cat), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE, "splits"), exist_ok=True)

    catalog = []
    csv_rows = []

    # -------------------------------------------------------------------------
    # SUBSET 1: OPTICAL & PHYSICAL REAL-WORLD ARTIFACTS (rw_0001 to rw_0010)
    # -------------------------------------------------------------------------
    optical_cases = [
        ("rw_0001", "Smartphone Shadow Gradient across Total Cholesterol", "shadow_gradient", {"tc": 218.4, "hdl": 44.2, "ldl": 142.0, "tg": 161.0}, {"tc": 218.4, "hdl": 44.2, "ldl": 142.0, "tg": 161.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0002", "Paper Crease/Fold bisecting HDL numeric field", "crease", {"tc": 194.0, "hdl": 52.0, "ldl": 114.0, "tg": 140.0}, {"tc": 194.0, "hdl": 52.0, "ldl": 114.0, "tg": 140.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0003", "Photocopier Toner Speckle / Xerographic Dropout", "toner_speckle", {"tc": 242.0, "hdl": 48.0, "ldl": 158.0, "tg": 180.0}, {"tc": 242.0, "hdl": 48.0, "ldl": 158.0, "tg": 180.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0004", "Technician Red Pen Checkmark touching LDL stroke", "pen_checkmark", {"tc": 206.0, "hdl": 38.0, "ldl": 136.0, "tg": 160.0}, {"tc": 206.0, "hdl": 38.0, "ldl": 136.0, "tg": 160.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0005", "Rubber Clinic Stamp ('CRITICAL') over Triglycerides", "clinic_stamp", {"tc": 230.0, "hdl": 45.0, "ldl": 135.0, "tg": 250.0}, {"tc": 230.0, "hdl": 45.0, "ldl": 135.0, "tg": 250.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0006", "Handheld Perspective Skew (4 deg) with ambient vignetting", "perspective_skew", {"tc": 188.0, "hdl": 50.0, "ldl": 110.0, "tg": 140.0}, {"tc": 188.0, "hdl": 50.0, "ldl": 110.0, "tg": 140.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0007", "Water Drop / Coffee Ring Stain over result table", "water_stain", {"tc": 222.0, "hdl": 46.0, "ldl": 142.0, "tg": 170.0}, {"tc": 222.0, "hdl": 46.0, "ldl": 142.0, "tg": 170.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0008", "Thermal Analyzer Printout with Faded Ink Contrast", "faded_thermal", {"tc": 196.0, "hdl": 54.0, "ldl": 118.0, "tg": 120.0}, {"tc": 196.0, "hdl": 54.0, "ldl": 118.0, "tg": 120.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0009", "Directional Motion Blur across patient row numbers", "motion_blur", {"tc": 215.0, "hdl": 41.0, "ldl": 138.0, "tg": 180.0}, {"tc": 215.0, "hdl": 41.0, "ldl": 138.0, "tg": 180.0}, [], "FRIEDEWALD", "mg/dL"),
        ("rw_0010", "Multi-Artifact: Crease + Blue Pen Underline on HDL", "crease_and_pen", {"tc": 180.0, "hdl": 40.0, "ldl": 110.0, "tg": 150.0}, {"tc": 180.0, "hdl": 40.0, "ldl": 110.0, "tg": 150.0}, [], "FRIEDEWALD", "mg/dL"),
    ]

    for doc_id, desc, artifact, clean, disp, corrupt_fields, assay, unit in optical_cases:
        rows = [
            ("Total Cholesterol", "total_cholesterol", f"{disp['tc']:.1f}", "< 200 mg/dL"),
            ("HDL Cholesterol", "hdl", f"{disp['hdl']:.1f}", ">= 40 mg/dL"),
            ("LDL Cholesterol", "ldl", f"{disp['ldl']:.1f}", "< 100 mg/dL"),
            ("Triglycerides", "triglycerides", f"{disp['tg']:.1f}", "< 150 mg/dL"),
        ]
        img, bboxes = render_corporate_template(
            rows, "METROPOLIS DIAGNOSTIC SERVICES", "DEPARTMENT OF CLINICAL BIOCHEMISTRY",
            "Method: Calculated by Friedewald Equation", unit=unit
        )
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Apply specific physical artifact
        if artifact == "shadow_gradient":
            cv_img = apply_shadow_gradient(cv_img, intensity=0.55, direction="diagonal")
        elif artifact == "crease":
            cv_img = apply_paper_crease(cv_img, (0, 185), (W, 205), strength=40)
        elif artifact == "toner_speckle":
            cv_img = apply_toner_speckle(cv_img, speckle_prob=0.005)
        elif artifact == "pen_checkmark":
            cv_img = apply_pen_markings(cv_img, bboxes["ldl"], mark_type="checkmark", color=(20, 20, 190))
        elif artifact == "clinic_stamp":
            cv_img = apply_clinic_stamp(cv_img, "CRITICAL", color=(30, 30, 190), pos=(530, 140), angle=-8)
        elif artifact == "perspective_skew":
            cv_img, bboxes = apply_perspective_skew(cv_img, bboxes, angle_deg=3.5)
            cv_img = apply_shadow_gradient(cv_img, intensity=0.35, direction="left")
        elif artifact == "water_stain":
            cv_img = apply_water_stain(cv_img, center=(420, 190), radius=65, alpha=0.22)
        elif artifact == "faded_thermal":
            cv_img = cv2.cvtColor(np.array(render_hospital_analyzer_template(rows, "ROCHE COBAS 6000 ANALYZER", "CENTRAL LAB AUTOMATION", "Method: Friedewald Formula")[0]), cv2.COLOR_RGB2BGR)
            # Fade contrast
            cv_img = np.clip(cv_img.astype(np.float32) * 0.75 + 65, 0, 255).astype(np.uint8)
        elif artifact == "motion_blur":
            kernel = np.zeros((5, 5))
            kernel[2, :] = 0.2
            cv_img = cv2.filter2D(cv_img, -1, kernel)
        elif artifact == "crease_and_pen":
            cv_img = apply_paper_crease(cv_img, (0, 165), (W, 180), strength=30)
            cv_img = apply_pen_markings(cv_img, bboxes["hdl"], mark_type="underline", color=(180, 25, 25))

        out_path = os.path.join(OUTPUT_BASE, "01_optical_physical_artifacts", f"{doc_id}.png")
        cv2.imwrite(out_path, cv_img)

        clean_panel = {"total_cholesterol": clean["tc"], "hdl": clean["hdl"], "ldl": clean["ldl"], "triglycerides": clean["tg"]}
        disp_panel = {"total_cholesterol": disp["tc"], "hdl": disp["hdl"], "ldl": disp["ldl"], "triglycerides": disp["tg"]}
        meta = {
            "doc_id": doc_id,
            "category": "optical_physical_artifacts",
            "error_scenario": desc,
            "artifact_applied": artifact,
            "panel_type": "lipid",
            "unit": unit,
            "method_text": "Method: Calculated by Friedewald Equation",
            "ground_truth_clean": clean_panel,
            "injected_display_data": disp_panel,
            "bounding_boxes": bboxes,
            "corrupted_fields": corrupt_fields,
            "expected_action": "VERIFIED_CLEAN" if not corrupt_fields else "CORRECT_K1",
        }
        with open(os.path.join(OUTPUT_BASE, "01_optical_physical_artifacts", f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        catalog.append(meta)
        csv_rows.append([doc_id, "01_optical_physical_artifacts", desc, clean["tc"], clean["hdl"], clean["ldl"], clean["tg"], disp["tc"], disp["hdl"], disp["ldl"], disp["tg"], unit, "VERIFIED_CLEAN"])

    # -------------------------------------------------------------------------
    # SUBSET 2: REAL-WORLD OCR GLYPH CONFUSIONS (rw_0011 to rw_0020)
    # -------------------------------------------------------------------------
    glyph_cases = [
        ("rw_0011", "3 <-> 8 Glyph Confusion on LDL (138.0 misread as 188.0)", {"tc": 218.0, "hdl": 42.0, "ldl": 138.0, "tg": 190.0}, {"tc": 218.0, "hdl": 42.0, "ldl": 188.0, "tg": 190.0}, ["ldl"], "CORRECT_K1"),
        ("rw_0012", "5 <-> 6 Glyph Confusion on Total Cholesterol (155.0 -> 165.0)", {"tc": 155.0, "hdl": 45.0, "ldl": 86.0, "tg": 120.0}, {"tc": 165.0, "hdl": 45.0, "ldl": 86.0, "tg": 120.0}, ["total_cholesterol"], "CORRECT_K1"),
        ("rw_0013", "1 <-> 7 Serif Confusion on HDL (41.0 misread as 47.0)", {"tc": 195.0, "hdl": 41.0, "ldl": 124.0, "tg": 150.0}, {"tc": 195.0, "hdl": 47.0, "ldl": 124.0, "tg": 150.0}, ["hdl"], "CORRECT_K1"),
        ("rw_0014", "0 <-> 6 Loop Closure on Triglycerides (105.0 -> 165.0)", {"tc": 191.0, "hdl": 50.0, "ldl": 120.0, "tg": 105.0}, {"tc": 191.0, "hdl": 50.0, "ldl": 120.0, "tg": 165.0}, ["triglycerides"], "CORRECT_K1"),
        ("rw_0015", "4 <-> 9 Top Closure Confusion on LDL (142.0 -> 192.0)", {"tc": 214.0, "hdl": 40.0, "ldl": 142.0, "tg": 160.0}, {"tc": 214.0, "hdl": 40.0, "ldl": 192.0, "tg": 160.0}, ["ldl"], "CORRECT_K1"),
        ("rw_0016", "Leading Digit Bounding Box Clipped (175.0 clipped to 75.0)", {"tc": 245.0, "hdl": 42.0, "ldl": 175.0, "tg": 140.0}, {"tc": 245.0, "hdl": 42.0, "ldl": 75.0, "tg": 140.0}, ["ldl"], "CORRECT_K1"),
        ("rw_0017", "Reference Range Column Leak (Detector grabs '< 100' instead of 145.0)", {"tc": 225.0, "hdl": 50.0, "ldl": 145.0, "tg": 150.0}, {"tc": 225.0, "hdl": 50.0, "ldl": 100.0, "tg": 150.0}, ["ldl"], "CORRECT_K1"),
        ("rw_0018", "Faint Stroke Dropped: 8 read as 3 on HDL (48.0 -> 43.0)", {"tc": 188.0, "hdl": 48.0, "ldl": 116.0, "tg": 120.0}, {"tc": 188.0, "hdl": 43.0, "ldl": 116.0, "tg": 120.0}, ["hdl"], "CORRECT_K1"),
        ("rw_0019", "Adjacent Transposition on Total Cholesterol (246.0 -> 264.0)", {"tc": 246.0, "hdl": 46.0, "ldl": 166.0, "tg": 170.0}, {"tc": 264.0, "hdl": 46.0, "ldl": 166.0, "tg": 170.0}, ["total_cholesterol"], "CORRECT_K1"),
        ("rw_0020", "Confusable Pair 0 <-> 8 on TC (208.0 read as 288.0)", {"tc": 208.0, "hdl": 44.0, "ldl": 132.0, "tg": 160.0}, {"tc": 288.0, "hdl": 44.0, "ldl": 132.0, "tg": 160.0}, ["total_cholesterol"], "CORRECT_K1"),
    ]

    for doc_id, desc, clean, disp, corrupt_fields, expected_action in glyph_cases:
        rows = [
            ("Total Cholesterol", "total_cholesterol", f"{disp['tc']:.1f}", "< 200 mg/dL"),
            ("HDL Cholesterol", "hdl", f"{disp['hdl']:.1f}", ">= 40 mg/dL"),
            ("LDL Cholesterol", "ldl", f"{disp['ldl']:.1f}", "< 100 mg/dL"),
            ("Triglycerides", "triglycerides", f"{disp['tg']:.1f}", "< 150 mg/dL"),
        ]
        img, bboxes = render_corporate_template(
            rows, "SRL CLINICAL REFERENCE LABS", "AUTOMATED BIOCHEMISTRY DIVISION",
            "Method: Calculated by Friedewald Equation", unit="mg/dL"
        )
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0.3)

        out_path = os.path.join(OUTPUT_BASE, "02_ocr_glyph_confusions", f"{doc_id}.png")
        cv2.imwrite(out_path, cv_img)

        clean_panel = {"total_cholesterol": clean["tc"], "hdl": clean["hdl"], "ldl": clean["ldl"], "triglycerides": clean["tg"]}
        disp_panel = {"total_cholesterol": disp["tc"], "hdl": disp["hdl"], "ldl": disp["ldl"], "triglycerides": disp["tg"]}
        meta = {
            "doc_id": doc_id,
            "category": "ocr_glyph_confusions",
            "error_scenario": desc,
            "panel_type": "lipid",
            "unit": "mg/dL",
            "method_text": "Method: Calculated by Friedewald Equation",
            "ground_truth_clean": clean_panel,
            "injected_display_data": disp_panel,
            "bounding_boxes": bboxes,
            "corrupted_fields": corrupt_fields,
            "expected_action": expected_action,
        }
        with open(os.path.join(OUTPUT_BASE, "02_ocr_glyph_confusions", f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        catalog.append(meta)
        csv_rows.append([doc_id, "02_ocr_glyph_confusions", desc, clean["tc"], clean["hdl"], clean["ldl"], clean["tg"], disp["tc"], disp["hdl"], disp["ldl"], disp["tg"], "mg/dL", expected_action])

    # -------------------------------------------------------------------------
    # SUBSET 3: DECIMAL & SCALE ERRORS (rw_0021 to rw_0030)
    # -------------------------------------------------------------------------
    decimal_cases = [
        ("rw_0021", "Missing Decimal Point (10x Error): Total Cholesterol 18.2 -> 182.0 mg/dL (clean mmol/L)", {"tc": 4.8, "hdl": 1.2, "ldl": 2.8, "tg": 1.8}, {"tc": 48.0, "hdl": 1.2, "ldl": 2.8, "tg": 1.8}, ["total_cholesterol"], "mmol/L", "CORRECT_K1"),
        ("rw_0022", "Missing Decimal Point (10x Error): HDL Cholesterol 42.5 -> 425.0 mg/dL", {"tc": 210.0, "hdl": 42.5, "ldl": 135.5, "tg": 160.0}, {"tc": 210.0, "hdl": 425.0, "ldl": 135.5, "tg": 160.0}, ["hdl"], "mg/dL", "CORRECT_K1"),
        ("rw_0023", "Premature Decimal Point (0.1x Error): LDL 140.0 -> 14.0 mg/dL", {"tc": 216.0, "hdl": 46.0, "ldl": 140.0, "tg": 150.0}, {"tc": 216.0, "hdl": 46.0, "ldl": 14.0, "tg": 150.0}, ["ldl"], "mg/dL", "CORRECT_K1"),
        ("rw_0024", "Premature Decimal Point (0.1x Error): Triglycerides 175.0 -> 17.5 mg/dL", {"tc": 205.0, "hdl": 45.0, "ldl": 125.0, "tg": 175.0}, {"tc": 205.0, "hdl": 45.0, "ldl": 125.0, "tg": 17.5}, ["triglycerides"], "mg/dL", "CORRECT_K1"),
        ("rw_0025", "Trailing Zero Truncation: Total Cholesterol 210.0 -> 21.0 mg/dL", {"tc": 210.0, "hdl": 50.0, "ldl": 130.0, "tg": 150.0}, {"tc": 21.0, "hdl": 50.0, "ldl": 130.0, "tg": 150.0}, ["total_cholesterol"], "mg/dL", "CORRECT_K1"),
        ("rw_0026", "Trailing Zero Truncation: Triglycerides 160.0 -> 16.0 mg/dL", {"tc": 192.0, "hdl": 42.0, "ldl": 118.0, "tg": 160.0}, {"tc": 192.0, "hdl": 42.0, "ldl": 118.0, "tg": 16.0}, ["triglycerides"], "mg/dL", "CORRECT_K1"),
        ("rw_0027", "Decimal Shift on TC (10x jump): 166.0 -> 1660.0 mg/dL", {"tc": 166.0, "hdl": 42.0, "ldl": 98.0, "tg": 130.0}, {"tc": 1660.0, "hdl": 42.0, "ldl": 98.0, "tg": 130.0}, ["total_cholesterol"], "mg/dL", "CORRECT_K1"),
        ("rw_0028", "Decimal Shift on LDL (10x jump): 51.2 -> 512.0 mg/dL", {"tc": 181.2, "hdl": 48.0, "ldl": 51.2, "tg": 410.0}, {"tc": 181.2, "hdl": 48.0, "ldl": 512.0, "tg": 410.0}, ["ldl"], "mg/dL", "CORRECT_K1"),
        ("rw_0029", "Premature Decimal on HDL (0.1x drop): 48.0 -> 4.8 mg/dL", {"tc": 190.0, "hdl": 48.0, "ldl": 116.0, "tg": 130.0}, {"tc": 190.0, "hdl": 4.8, "ldl": 116.0, "tg": 130.0}, ["hdl"], "mg/dL", "CORRECT_K1"),
        ("rw_0030", "Unit Normalization Test: mmol/L values correctly recognized", {"tc": 5.2, "hdl": 1.3, "ldl": 3.1, "tg": 1.8}, {"tc": 5.2, "hdl": 1.3, "ldl": 3.1, "tg": 1.8}, [], "mmol/L", "VERIFIED_CLEAN"),
    ]

    for doc_id, desc, clean, disp, corrupt_fields, unit, expected_action in decimal_cases:
        rows = [
            ("Total Cholesterol", "total_cholesterol", f"{disp['tc']:.1f}", "< 200 mg/dL" if unit == "mg/dL" else "< 5.2 mmol/L"),
            ("HDL Cholesterol", "hdl", f"{disp['hdl']:.1f}", ">= 40 mg/dL" if unit == "mg/dL" else ">= 1.0 mmol/L"),
            ("LDL Cholesterol", "ldl", f"{disp['ldl']:.1f}", "< 100 mg/dL" if unit == "mg/dL" else "< 2.6 mmol/L"),
            ("Triglycerides", "triglycerides", f"{disp['tg']:.1f}", "< 150 mg/dL" if unit == "mg/dL" else "< 1.7 mmol/L"),
        ]
        img, bboxes = render_corporate_template(
            rows, "PATHKIND DIAGNOSTICS", "ENDOCRINOLOGY & LIPID PANEL",
            "Method: Calculated by Friedewald Equation", unit=unit
        )
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        out_path = os.path.join(OUTPUT_BASE, "03_decimal_scale_errors", f"{doc_id}.png")
        cv2.imwrite(out_path, cv_img)

        clean_panel = {"total_cholesterol": clean["tc"], "hdl": clean["hdl"], "ldl": clean["ldl"], "triglycerides": clean["tg"]}
        disp_panel = {"total_cholesterol": disp["tc"], "hdl": disp["hdl"], "ldl": disp["ldl"], "triglycerides": disp["tg"]}
        meta = {
            "doc_id": doc_id,
            "category": "decimal_scale_errors",
            "error_scenario": desc,
            "panel_type": "lipid",
            "unit": unit,
            "method_text": "Method: Calculated by Friedewald Equation",
            "ground_truth_clean": clean_panel,
            "injected_display_data": disp_panel,
            "bounding_boxes": bboxes,
            "corrupted_fields": corrupt_fields,
            "expected_action": expected_action,
        }
        with open(os.path.join(OUTPUT_BASE, "03_decimal_scale_errors", f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        catalog.append(meta)
        csv_rows.append([doc_id, "03_decimal_scale_errors", desc, clean["tc"], clean["hdl"], clean["ldl"], clean["tg"], disp["tc"], disp["hdl"], disp["ldl"], disp["tg"], unit, expected_action])

    # -------------------------------------------------------------------------
    # SUBSET 4: K=2 DUAL FAULTS & EXTRA EQUATIONS (rw_0031 to rw_0040)
    # -------------------------------------------------------------------------
    k2_cases = [
        ("rw_0031", "K=2 Dual Fault with Extra Equations: Non-HDL & VLDL Present (2x2 Closed-Form)", {"tc": 220.0, "hdl": 45.0, "ldl": 143.0, "tg": 160.0, "non_hdl": 175.0, "vldl": 32.0}, {"tc": 250.0, "hdl": 45.0, "ldl": 120.0, "tg": 160.0, "non_hdl": 175.0, "vldl": 32.0}, ["total_cholesterol", "ldl"], True, "CORRECT_K2_EXTRA_FIELD"),
        ("rw_0032", "K=2 Dual Fault with Extra Non-HDL: HDL and TG corrupted", {"tc": 204.0, "hdl": 44.0, "ldl": 132.0, "tg": 140.0, "non_hdl": 160.0, "vldl": 28.0}, {"tc": 204.0, "hdl": 64.0, "ldl": 132.0, "tg": 190.0, "non_hdl": 160.0, "vldl": 28.0}, ["hdl", "triglycerides"], True, "CORRECT_K2_EXTRA_FIELD"),
        ("rw_0033", "K=2 Dual Fault with Extra VLDL: TC and TG corrupted", {"tc": 235.0, "hdl": 42.0, "ldl": 155.0, "tg": 190.0, "non_hdl": 193.0, "vldl": 38.0}, {"tc": 285.0, "hdl": 42.0, "ldl": 155.0, "tg": 240.0, "non_hdl": 193.0, "vldl": 38.0}, ["total_cholesterol", "triglycerides"], True, "CORRECT_K2_EXTRA_FIELD"),
        ("rw_0034", "K=2 Dual Fault (No Extra Fields): HDL Decimal Shift + LDL Digit Swap", {"tc": 218.0, "hdl": 42.0, "ldl": 142.0, "tg": 170.0}, {"tc": 218.0, "hdl": 420.0, "ldl": 124.0, "tg": 170.0}, ["hdl", "ldl"], False, "CORRECT_K2_DUAL_BLIND_REREAD"),
        ("rw_0035", "K=2 Dual Fault (No Extra Fields): TC 3/8 confusion + TG smudge", {"tc": 186.0, "hdl": 48.0, "ldl": 110.0, "tg": 140.0}, {"tc": 136.0, "hdl": 48.0, "ldl": 110.0, "tg": 190.0}, ["total_cholesterol", "triglycerides"], False, "CORRECT_K2_DUAL_BLIND_REREAD"),
        ("rw_0036", "K=2 Dual Fault with Crease intersecting HDL & LDL rows", {"tc": 202.0, "hdl": 40.0, "ldl": 130.0, "tg": 160.0}, {"tc": 202.0, "hdl": 60.0, "ldl": 110.0, "tg": 160.0}, ["hdl", "ldl"], False, "CORRECT_K2_DUAL_BLIND_REREAD"),
        ("rw_0037", "K=2 Dual Fault with Lateral Shadow affecting right column (LDL + TG)", {"tc": 210.0, "hdl": 45.0, "ldl": 135.0, "tg": 150.0}, {"tc": 210.0, "hdl": 45.0, "ldl": 115.0, "tg": 220.0}, ["ldl", "triglycerides"], False, "CORRECT_K2_DUAL_BLIND_REREAD"),
        ("rw_0038", "K=2 Dual Fault: 5/6 swap on TC + Trailing Zero Drop on TG (180 -> 18)", {"tc": 215.0, "hdl": 43.0, "ldl": 136.0, "tg": 180.0}, {"tc": 216.0, "hdl": 43.0, "ldl": 136.0, "tg": 18.0}, ["total_cholesterol", "triglycerides"], False, "CORRECT_K2_DUAL_BLIND_REREAD"),
        ("rw_0039", "K=2 Severely Degraded Ink: Safe Abstention (Algebraic Indeterminate)", {"tc": 198.0, "hdl": 38.0, "ldl": 128.0, "tg": 160.0}, {"tc": 260.0, "hdl": 18.0, "ldl": 128.0, "tg": 160.0}, ["total_cholesterol", "hdl"], False, "ABSTAIN_ALGEBRAIC_INDETERMINATE"),
        ("rw_0040", "K=2 Dual Transposition: TC (195 -> 159) + HDL (48 -> 84)", {"tc": 195.0, "hdl": 48.0, "ldl": 117.0, "tg": 150.0}, {"tc": 159.0, "hdl": 84.0, "ldl": 117.0, "tg": 150.0}, ["total_cholesterol", "hdl"], False, "CORRECT_K2_DUAL_BLIND_REREAD"),
    ]

    for doc_id, desc, clean, disp, corrupt_fields, has_extra, expected_action in k2_cases:
        rows = [
            ("Total Cholesterol", "total_cholesterol", f"{disp['tc']:.1f}", "< 200 mg/dL"),
            ("HDL Cholesterol", "hdl", f"{disp['hdl']:.1f}", ">= 40 mg/dL"),
            ("LDL Cholesterol", "ldl", f"{disp['ldl']:.1f}", "< 100 mg/dL"),
            ("Triglycerides", "triglycerides", f"{disp['tg']:.1f}", "< 150 mg/dL"),
        ]
        if has_extra:
            rows.append(("Non-HDL Cholesterol", "non_hdl_cholesterol", f"{disp['non_hdl']:.1f}", "< 130 mg/dL"))
            rows.append(("VLDL Cholesterol", "vldl", f"{disp['vldl']:.1f}", "< 30 mg/dL"))

        img, bboxes = render_corporate_template(
            rows, "APOLLO HOSPITALS DIAGNOSTICS", "DEPARTMENT OF CLINICAL BIOCHEMISTRY",
            "Method: Calculated by Friedewald Equation", unit="mg/dL", has_extra_fields=has_extra
        )
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if "Crease" in desc:
            cv_img = apply_paper_crease(cv_img, (0, 160), (W, 180), strength=35)
        elif "Shadow" in desc:
            cv_img = apply_shadow_gradient(cv_img, intensity=0.50, direction="right")

        out_path = os.path.join(OUTPUT_BASE, "04_k2_dual_faults", f"{doc_id}.png")
        cv2.imwrite(out_path, cv_img)

        clean_panel = {k: clean[k] for k in clean}
        disp_panel = {k: disp[k] for k in disp}
        meta = {
            "doc_id": doc_id,
            "category": "k2_dual_faults",
            "error_scenario": desc,
            "has_extra_fields": has_extra,
            "panel_type": "lipid",
            "unit": "mg/dL",
            "method_text": "Method: Calculated by Friedewald Equation",
            "ground_truth_clean": clean_panel,
            "injected_display_data": disp_panel,
            "bounding_boxes": bboxes,
            "corrupted_fields": corrupt_fields,
            "expected_action": expected_action,
        }
        with open(os.path.join(OUTPUT_BASE, "04_k2_dual_faults", f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        catalog.append(meta)
        csv_rows.append([doc_id, "04_k2_dual_faults", desc, clean["tc"], clean["hdl"], clean["ldl"], clean["tg"], disp["tc"], disp["hdl"], disp["ldl"], disp["tg"], "mg/dL", expected_action])

    # -------------------------------------------------------------------------
    # SUBSET 5: CLINICAL BOUNDARIES & SAMPSON HIGH-TG (rw_0041 to rw_0050)
    # -------------------------------------------------------------------------
    sampson_cases = [
        ("rw_0041", "High Triglycerides (TG=485 mg/dL): Sampson Eq 2 Clean Routing", {"tc": 215.0, "hdl": 38.0, "ldl": 115.6, "tg": 485.0}, {"tc": 215.0, "hdl": 38.0, "ldl": 115.6, "tg": 485.0}, [], "SAMPSON_ROUTING"),
        ("rw_0042", "High Triglycerides (TG=560 mg/dL): Sampson LDL Digit Swap (102 -> 120)", {"tc": 210.0, "hdl": 35.0, "ldl": 102.0, "tg": 560.0}, {"tc": 210.0, "hdl": 35.0, "ldl": 120.0, "tg": 560.0}, ["ldl"], "SAMPSON_ROUTING_REPAIR"),
        ("rw_0043", "High Triglycerides (TG=640 mg/dL): Smudge on TC (228 -> 268)", {"tc": 228.0, "hdl": 34.0, "ldl": 112.5, "tg": 640.0}, {"tc": 268.0, "hdl": 34.0, "ldl": 112.5, "tg": 640.0}, ["total_cholesterol"], "SAMPSON_ROUTING_REPAIR"),
        ("rw_0044", "High Triglycerides (TG=720 mg/dL): Sampson Decimal Shift on HDL (3.8 -> 38.0)", {"tc": 220.0, "hdl": 32.0, "ldl": 103.8, "tg": 720.0}, {"tc": 220.0, "hdl": 3.2, "ldl": 103.8, "tg": 720.0}, ["hdl"], "SAMPSON_ROUTING_REPAIR"),
        ("rw_0045", "Severe Hypertriglyceridemia (TG=920 mg/dL > 800): Safe Principled Abstention", {"tc": 250.0, "hdl": 30.0, "ldl": 80.0, "tg": 920.0}, {"tc": 250.0, "hdl": 30.0, "ldl": 80.0, "tg": 920.0}, [], "ABSTAIN_TG800"),
        ("rw_0046", "Severe Hypertriglyceridemia (TG=1150 mg/dL): Safe Principled Abstention with Stamp", {"tc": 280.0, "hdl": 28.0, "ldl": 75.0, "tg": 1150.0}, {"tc": 280.0, "hdl": 28.0, "ldl": 75.0, "tg": 1150.0}, [], "ABSTAIN_TG800"),
        ("rw_0047", "Biological Bound Violation: HDL + LDL >= TC (HDL 65 + LDL 180 >= TC 210)", {"tc": 210.0, "hdl": 65.0, "ldl": 180.0, "tg": 160.0}, {"tc": 210.0, "hdl": 65.0, "ldl": 180.0, "tg": 160.0}, ["ldl"], "ABSTAIN_BOUNDS_VIOLATION"),
        ("rw_0048", "Biological Bound Violation: HDL >= Total Cholesterol (HDL 195 >= TC 180)", {"tc": 180.0, "hdl": 195.0, "ldl": 110.0, "tg": 150.0}, {"tc": 180.0, "hdl": 195.0, "ldl": 110.0, "tg": 150.0}, ["hdl"], "ABSTAIN_BOUNDS_VIOLATION"),
        ("rw_0049", "Martin-Hopkins Boundary: Low TG (< 50 mg/dL) Testing Divisor f=3.5", {"tc": 165.0, "hdl": 55.0, "ldl": 97.1, "tg": 45.0}, {"tc": 165.0, "hdl": 55.0, "ldl": 97.1, "tg": 45.0}, [], "MARTIN_HOPKINS_ROUTING"),
        ("rw_0050", "Martin-Hopkins Boundary: TG=380 mg/dL, Non-HDL=190 mg/dL, Divisor f=6.0", {"tc": 235.0, "hdl": 45.0, "ldl": 126.7, "tg": 380.0}, {"tc": 235.0, "hdl": 45.0, "ldl": 126.7, "tg": 380.0}, [], "MARTIN_HOPKINS_ROUTING"),
    ]

    for doc_id, desc, clean, disp, corrupt_fields, expected_action in sampson_cases:
        method = "Method: Calculated by NIH Sampson 2020 Equation" if "Sampson" in desc or disp["tg"] > 400 else ("Method: Calculated by Martin-Hopkins Equation" if "Martin-Hopkins" in desc else "Method: Calculated by Friedewald Equation")
        rows = [
            ("Total Cholesterol", "total_cholesterol", f"{disp['tc']:.1f}", "< 200 mg/dL"),
            ("HDL Cholesterol", "hdl", f"{disp['hdl']:.1f}", ">= 40 mg/dL"),
            ("LDL Cholesterol", "ldl", f"{disp['ldl']:.1f}", "< 100 mg/dL"),
            ("Triglycerides", "triglycerides", f"{disp['tg']:.1f}", "< 150 mg/dL"),
        ]
        img, bboxes = render_corporate_template(
            rows, "LAL PATHLABS CLINICAL SPECIALTIES", "LIPID FRACTIONATION & RISK REPORT",
            method, unit="mg/dL"
        )
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if "Stamp" in desc:
            cv_img = apply_clinic_stamp(cv_img, "CHYLOMICRON", color=(40, 40, 200), pos=(520, 150), angle=-6)

        out_path = os.path.join(OUTPUT_BASE, "05_clinical_boundaries_sampson", f"{doc_id}.png")
        cv2.imwrite(out_path, cv_img)

        clean_panel = {"total_cholesterol": clean["tc"], "hdl": clean["hdl"], "ldl": clean["ldl"], "triglycerides": clean["tg"]}
        disp_panel = {"total_cholesterol": disp["tc"], "hdl": disp["hdl"], "ldl": disp["ldl"], "triglycerides": disp["tg"]}
        meta = {
            "doc_id": doc_id,
            "category": "clinical_boundaries_sampson",
            "error_scenario": desc,
            "panel_type": "lipid",
            "unit": "mg/dL",
            "method_text": method,
            "ground_truth_clean": clean_panel,
            "injected_display_data": disp_panel,
            "bounding_boxes": bboxes,
            "corrupted_fields": corrupt_fields,
            "expected_action": expected_action,
        }
        with open(os.path.join(OUTPUT_BASE, "05_clinical_boundaries_sampson", f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        catalog.append(meta)
        csv_rows.append([doc_id, "05_clinical_boundaries_sampson", desc, clean["tc"], clean["hdl"], clean["ldl"], clean["tg"], disp["tc"], disp["hdl"], disp["ldl"], disp["tg"], "mg/dL", expected_action])

    # -------------------------------------------------------------------------
    # SUBSET 6: MULTI-PANEL PROTEIN & ELECTROLYTES (rw_0051 to rw_0060)
    # -------------------------------------------------------------------------
    # Protein panel: Total Protein = Albumin + Globulin (g/dL)
    # Electrolytes: Na - (Cl + HCO3) = Anion Gap (mEq/L)
    multipanel_cases = [
        ("rw_0051", "Protein Panel: Clean Normal Reference (TP = Alb + Glob)", "protein", {"total_protein": 7.2, "albumin": 4.2, "globulin": 3.0}, {"total_protein": 7.2, "albumin": 4.2, "globulin": 3.0}, [], "g/dL", "VERIFIED_CLEAN"),
        ("rw_0052", "Protein Panel: Albumin Digit Swap (4.2 -> 2.4 g/dL) with Paper Crease", "protein", {"total_protein": 7.4, "albumin": 4.2, "globulin": 3.2}, {"total_protein": 7.4, "albumin": 2.4, "globulin": 3.2}, ["albumin"], "g/dL", "CORRECT_K1"),
        ("rw_0053", "Protein Panel: Decimal Shift on Total Protein (7.1 -> 71.0 g/dL)", "protein", {"total_protein": 7.1, "albumin": 4.0, "globulin": 3.1}, {"total_protein": 71.0, "albumin": 4.0, "globulin": 3.1}, ["total_protein"], "g/dL", "CORRECT_K1"),
        ("rw_0054", "Protein Panel: Bound Violation (Albumin 7.5 >= Total Protein 7.0 g/dL)", "protein", {"total_protein": 7.0, "albumin": 4.1, "globulin": 2.9}, {"total_protein": 7.0, "albumin": 7.5, "globulin": 2.9}, ["albumin"], "g/dL", "ABSTAIN_BOUNDS_VIOLATION"),
        ("rw_0055", "Protein Panel: 3 <-> 8 Confusion on Globulin (3.3 -> 3.8 g/dL)", "protein", {"total_protein": 7.5, "albumin": 4.2, "globulin": 3.3}, {"total_protein": 7.5, "albumin": 4.2, "globulin": 3.8}, ["globulin"], "g/dL", "CORRECT_K1"),
        ("rw_0056", "Electrolyte Panel: Clean Normal Anion Gap (140 - (102 + 24) = 14 mEq/L)", "electrolytes", {"sodium": 140.0, "chloride": 102.0, "bicarbonate": 24.0, "potassium": 4.2, "anion_gap": 14.0}, {"sodium": 140.0, "chloride": 102.0, "bicarbonate": 24.0, "potassium": 4.2, "anion_gap": 14.0}, [], "mEq/L", "VERIFIED_CLEAN"),
        ("rw_0057", "Electrolyte Panel: Decimal Shift on Potassium (4.2 -> 42.0 mEq/L)", "electrolytes", {"sodium": 142.0, "chloride": 104.0, "bicarbonate": 25.0, "potassium": 4.2, "anion_gap": 13.0}, {"sodium": 142.0, "chloride": 104.0, "bicarbonate": 25.0, "potassium": 42.0, "anion_gap": 13.0}, ["potassium"], "mEq/L", "CORRECT_K1"),
        ("rw_0058", "Electrolyte Panel: Digit Swap on Sodium (142 -> 124 mEq/L)", "electrolytes", {"sodium": 142.0, "chloride": 103.0, "bicarbonate": 24.0, "potassium": 4.1, "anion_gap": 15.0}, {"sodium": 124.0, "chloride": 103.0, "bicarbonate": 24.0, "potassium": 4.1, "anion_gap": 15.0}, ["sodium"], "mEq/L", "CORRECT_K1"),
        ("rw_0059", "Electrolyte Panel: High Anion Gap Metabolic Acidosis (AG = 24.0 mEq/L)", "electrolytes", {"sodium": 138.0, "chloride": 98.0, "bicarbonate": 16.0, "potassium": 4.5, "anion_gap": 24.0}, {"sodium": 138.0, "chloride": 98.0, "bicarbonate": 16.0, "potassium": 4.5, "anion_gap": 24.0}, [], "mEq/L", "VERIFIED_CLEAN"),
        ("rw_0060", "Electrolyte Panel: Osmolar Gap Advisory Alert (Measured 315 vs Calc 290)", "electrolytes", {"sodium": 140.0, "chloride": 102.0, "bicarbonate": 24.0, "glucose": 95.0, "bun": 14.0, "measured_osmolality": 315.0, "anion_gap": 14.0}, {"sodium": 140.0, "chloride": 102.0, "bicarbonate": 24.0, "glucose": 95.0, "bun": 14.0, "measured_osmolality": 315.0, "anion_gap": 14.0}, [], "mEq/L", "OSMOLAR_GAP_ADVISORY"),
    ]

    for doc_id, desc, p_type, clean, disp, corrupt_fields, unit, expected_action in multipanel_cases:
        if p_type == "protein":
            rows = [
                ("Total Protein", "total_protein", f"{disp['total_protein']:.1f}", "6.0 - 8.3 g/dL"),
                ("Albumin", "albumin", f"{disp['albumin']:.1f}", "3.5 - 5.0 g/dL"),
                ("Globulin", "globulin", f"{disp['globulin']:.1f}", "2.3 - 3.5 g/dL"),
            ]
            header = "SERUM PROTEIN FRACTIONATION REPORT"
            sub = "Total Protein = Albumin + Globulin (Biuret & BCG Dye Binding)"
            meth = "Method: Photometric Absorbance (Biuret / BCG)"
        else:
            rows = [
                ("Sodium (Na)", "sodium", f"{disp['sodium']:.1f}", "136 - 145 mEq/L"),
                ("Chloride (Cl)", "chloride", f"{disp['chloride']:.1f}", "96 - 106 mEq/L"),
                ("Bicarbonate (HCO3)", "bicarbonate", f"{disp['bicarbonate']:.1f}", "22 - 29 mEq/L"),
                ("Anion Gap", "anion_gap", f"{disp['anion_gap']:.1f}", "8.0 - 16.0 mEq/L"),
            ]
            if "potassium" in disp:
                rows.insert(1, ("Potassium (K)", "potassium", f"{disp['potassium']:.1f}", "3.5 - 5.0 mEq/L"))
            if "glucose" in disp:
                rows.append(("Glucose", "glucose", f"{disp['glucose']:.1f}", "70 - 100 mg/dL"))
                rows.append(("Blood Urea Nitrogen", "bun", f"{disp['bun']:.1f}", "7 - 20 mg/dL"))
                rows.append(("Measured Osmolality", "measured_osmolality", f"{disp['measured_osmolality']:.1f}", "275 - 295 mOsm/kg"))
            header = "ELECTROLYTE & ACID-BASE STATUS PANEL"
            sub = "Anion Gap = Na - (Cl + HCO3) | Ion-Selective Electrode"
            meth = "Method: Indirect Ion-Selective Electrode (ISE)"

        img, bboxes = render_corporate_template(
            rows, header, sub, meth, unit=unit, has_extra_fields=(len(rows) > 4)
        )
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if "Crease" in desc:
            cv_img = apply_paper_crease(cv_img, (0, 175), (W, 190), strength=35)
        elif "Shadow" in desc:
            cv_img = apply_shadow_gradient(cv_img, intensity=0.45, direction="left")

        out_path = os.path.join(OUTPUT_BASE, "06_multipanel_protein_electrolytes", f"{doc_id}.png")
        cv2.imwrite(out_path, cv_img)

        meta = {
            "doc_id": doc_id,
            "category": "multipanel_protein_electrolytes",
            "error_scenario": desc,
            "panel_type": p_type,
            "unit": unit,
            "method_text": meth,
            "ground_truth_clean": clean,
            "injected_display_data": disp,
            "bounding_boxes": bboxes,
            "corrupted_fields": corrupt_fields,
            "expected_action": expected_action,
        }
        with open(os.path.join(OUTPUT_BASE, "06_multipanel_protein_electrolytes", f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        catalog.append(meta)
        tc_val = clean.get("total_protein", clean.get("sodium", 0.0))
        hdl_val = clean.get("albumin", clean.get("chloride", 0.0))
        ldl_val = clean.get("globulin", clean.get("bicarbonate", 0.0))
        tg_val = clean.get("anion_gap", 0.0)
        csv_rows.append([doc_id, "06_multipanel_protein_electrolytes", desc, tc_val, hdl_val, ldl_val, tg_val, disp.get("total_protein", disp.get("sodium", 0.0)), disp.get("albumin", disp.get("chloride", 0.0)), disp.get("globulin", disp.get("bicarbonate", 0.0)), disp.get("anion_gap", 0.0), unit, expected_action])

    # -------------------------------------------------------------------------
    # WRITE MASTER CATALOG & CSV
    # -------------------------------------------------------------------------
    catalog_path = os.path.join(OUTPUT_BASE, "dataset_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "dataset_C_realworld",
            "total_samples": len(catalog),
            "description": "Advanced synthetic clinical benchmark modeling realistic optical degradations, OCR confusions, decimal shifts, dual faults, clinical boundaries, and multi-panel reports.",
            "subsets": categories,
            "samples": catalog,
        }, f, indent=2)

    csv_path = os.path.join(OUTPUT_BASE, "ground_truth.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "category", "description", "clean_f1", "clean_f2", "clean_f3", "clean_f4", "disp_f1", "disp_f2", "disp_f3", "disp_f4", "unit", "expected_action"])
        writer.writerows(csv_rows)

    # -------------------------------------------------------------------------
    # WRITE DATASET SPLITS (train 60%, val 20%, test 20%)
    # -------------------------------------------------------------------------
    all_ids = [c["doc_id"] for c in catalog]
    random.shuffle(all_ids)
    n_total = len(all_ids)
    n_train = int(0.60 * n_total)
    n_val = int(0.20 * n_total)

    train_ids = sorted(all_ids[:n_train])
    val_ids = sorted(all_ids[n_train:n_train + n_val])
    test_ids = sorted(all_ids[n_train + n_val:])

    with open(os.path.join(OUTPUT_BASE, "splits", "train.txt"), "w") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(OUTPUT_BASE, "splits", "val.txt"), "w") as f:
        f.write("\n".join(val_ids))
    with open(os.path.join(OUTPUT_BASE, "splits", "test.txt"), "w") as f:
        f.write("\n".join(test_ids))

    print(f"Successfully generated {len(catalog)} real-world clinical documents in {OUTPUT_BASE}!")
    print(f"  - Master Catalog: {catalog_path}")
    print(f"  - Ground Truth CSV: {csv_path}")
    print(f"  - Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")


if __name__ == "__main__":
    generate_realworld_dataset()

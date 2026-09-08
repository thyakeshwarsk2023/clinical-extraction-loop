import os
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_font(size=18):
    for f in ["C:\\Windows\\Fonts\\calibri.ttf", "calibri.ttf", "arial.ttf"]:
        try: return ImageFont.truetype(f, size)
        except Exception: continue
    return ImageFont.load_default()

def generate_sample(sample_id: int):
    # Error modes including multi-error k=2 and assay variations
    modes = ["CLEAN", "DIGIT_CONFUSION_K1", "DECIMAL_SHIFT_K1", "DOUBLE_ERROR_K2", "DIRECT_ASSAY_CLEAN", "ABSTAIN_HIGH_TG"]
    mode = modes[(sample_id - 1) % len(modes)]
    
    # 1. Base Numbers
    trig = round(random.uniform(410.0, 600.0) if mode == "ABSTAIN_HIGH_TG" else random.uniform(80.0, 220.0), 1)
    hdl = round(random.uniform(35.0, 65.0), 1)
    ldl = round(random.uniform(70.0, 180.0), 1)
    tc = round(hdl + ldl + (trig / 5.0), 1)
    
    clean_data = {"total_cholesterol": tc, "hdl": hdl, "ldl": ldl, "triglycerides": trig}
    display_data = clean_data.copy()
    
    # 2. Corruptions
    corrupted_fields = []
    if mode == "DIGIT_CONFUSION_K1":
        display_data["ldl"] = 10.0 if clean_data["ldl"] > 100 else 75.0
        corrupted_fields.append("ldl")
    elif mode == "DECIMAL_SHIFT_K1":
        display_data["hdl"] = clean_data["hdl"] * 10.0
        corrupted_fields.append("hdl")
    elif mode == "DOUBLE_ERROR_K2":
        display_data["ldl"] = 10.0 if clean_data["ldl"] > 100 else 75.0
        display_data["triglycerides"] = clean_data["triglycerides"] + 50.0
        corrupted_fields.extend(["ldl", "triglycerides"])
        
    # 3. Assay Header Metadata
    if mode == "DIRECT_ASSAY_CLEAN":
        method_str = "Method: Direct Immunoinhibition Assay (Direct LDL-C)"
        assay_truth = "DIRECT_MEASURED"
    elif mode == "ABSTAIN_HIGH_TG":
        method_str = "Method: Calculated by Friedewald Formula"
        assay_truth = "FRIEDEWALD"
    else:
        method_str = random.choice([
            "Method: Calculated Friedewald Equation",
            "Method: Enzymatic Colorimetric Assay",  # Ambiguous fallback
            "Calculated Panel (TC - HDL - TRIG/5)"
        ])
        assay_truth = "FRIEDEWALD" if "calculated" in method_str.lower() or "friedewald" in method_str.lower() else "AMBIGUOUS_FALLBACK"

    # 4. Render Canvas
    img = Image.new("RGB", (750, 420), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_b, font_h = get_font(18), get_font(20)
    
    draw.text((30, 25), "METROPOLIS DIAGNOSTIC SERVICES", fill=(20, 20, 20), font=font_h)
    draw.text((30, 55), "DEPARTMENT OF CLINICAL BIOCHEMISTRY", fill=(80, 80, 80), font=font_b)
    draw.line([(30, 85), (720, 85)], fill=(120, 120, 120), width=2)
    
    # Table Header
    draw.text((35, 105), "TEST PARAMETER", fill=(0, 0, 0), font=font_b)
    draw.text((320, 105), "RESULT", fill=(0, 0, 0), font=font_b)
    draw.text((450, 105), "UNIT", fill=(0, 0, 0), font=font_b)
    draw.line([(30, 130), (720, 130)], fill=(180, 180, 180), width=1)
    
    rows = [
        ("Total Cholesterol", "total_cholesterol", display_data["total_cholesterol"]),
        ("HDL Cholesterol", "hdl", display_data["hdl"]),
        ("LDL Cholesterol", "ldl", display_data["ldl"]),
        ("Triglycerides", "triglycerides", display_data["triglycerides"])
    ]
    
    bboxes = {}
    y = 145
    for label, key, val in rows:
        val_str = f"{val:.1f}"
        draw.text((35, y), label, fill=(30, 30, 30), font=font_b)
        draw.text((450, y), "mg/dL", fill=(60, 60, 60), font=font_b)
        
        bbox_raw = draw.textbbox((320, y), val_str, font=font_b)
        bboxes[key] = [max(0, bbox_raw[0]-6), max(0, bbox_raw[1]-4), min(750, bbox_raw[2]+6), min(420, bbox_raw[3]+4)]
        draw.text((320, y), val_str, fill=(0, 0, 0), font=font_b)
        y += 40
        
    draw.line([(30, y+10), (720, y+10)], fill=(120, 120, 120), width=1)
    draw.text((35, y+25), method_str, fill=(100, 100, 100), font=font_b)
    
    # Degradation
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0)
    
    # Save Artifacts
    base_name = f"sample_{sample_id:03d}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}.png"), cv_img)
    
    with open(os.path.join(OUTPUT_DIR, f"{base_name}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sample_id": sample_id,
            "mode": mode,
            "assay_ground_truth": assay_truth,
            "method_text": method_str,
            "corrupted_fields": corrupted_fields,
            "ground_truth_clean": clean_data,
            "injected_display_data": display_data,
            "bounding_boxes": bboxes
        }, f, indent=2)

if __name__ == "__main__":
    for i in range(1, 41):
        generate_sample(i)
    print("Dataset generated: 40 multi-assay and multi-error samples ready.")
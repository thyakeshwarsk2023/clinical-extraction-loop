import os
import sys
import json
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evals.run_final_ablation import (
    _load_clean_map,
    _load_corrupted_double,
    _write_sidecar,
    _remove_sidecar,
    TOL
)
from src.pipeline import run_pipeline
from evals.eval_k2_ablation import evaluate_doc_accuracy
from src.clinical_risk import classify_ldl_risk, evaluate_clinical_risk_flip

out_dir = os.path.join(ROOT, "reports", "k2_wrong_case_crops")
os.makedirs(out_dir, exist_ok=True)

clean_map = _load_clean_map()
k2_records = _load_corrupted_double(clean_map)

wrong_cases = []
for idx, doc in enumerate(k2_records):
    doc_id = doc['doc_id']
    gt_clean = doc['gt_clean_values']
    native_disp = doc['display_values'].copy()
    bboxes = doc['bboxes_long']
    has_extended = (idx % 2 == 0)
    if has_extended:
        native_disp['non_hdl_cholesterol'] = round(gt_clean['total_cholesterol'] - gt_clean['hdl'], 1)
    
    _write_sidecar(doc['img_path'], native_disp, gt_clean, doc['unit'], bboxes, doc['method_text'])
    try:
        res = run_pipeline(doc['img_path'], max_attempts=2, tolerance=TOL)
    finally:
        _remove_sidecar(doc['img_path'])
    
    status = res.get('status', '')
    extracted = res.get('data', native_disp)
    is_correct = evaluate_doc_accuracy(extracted, gt_clean, tol=TOL)
    was_abstained = status.startswith('ABSTAIN') or 'ABSTAIN' in status
    
    if not was_abstained and not is_correct:
        wrong_cases.append({
            'idx': idx,
            'doc_id': doc_id,
            'status': status,
            'gt': gt_clean,
            'disp': native_disp,
            'extracted': extracted,
            'audit': res.get('audit_trail', []),
            'corrupt_types': doc['corruption_type'],
            'corrupted_fields': doc['corrupted_fields'],
            'bboxes': bboxes,
            'img_path': doc['img_path'],
            'unit': doc['unit']
        })

print(f"Total wrong cases found: {len(wrong_cases)}")
results_report = []

for w in wrong_cases:
    doc_id = w['doc_id']
    print("=" * 70)
    print(f"Document: {doc_id} | Status: {w['status']}")
    print(f"Corrupted Fields: {w['corrupted_fields']} | Types: {w['corrupt_types']}")
    print(f"Ground Truth: {w['gt']}")
    print(f"Display (Corrupted Input): {w['disp']}")
    print(f"Extracted (Committed): {w['extracted']}")
    
    # Save crops
    img = cv2.imread(w['img_path'])
    crops_saved = {}
    if img is not None:
        h, wid = img.shape[:2]
        for fld in w['corrupted_fields']:
            box = w['bboxes'].get(fld, [0, 0, 10, 10])
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = 4
            x0 = max(0, min(x0 - pad, wid - 1))
            y0 = max(0, min(y0 - pad, h - 1))
            x1 = max(x0 + 1, min(x1 + pad, wid))
            y1 = max(y0 + 1, min(y1 + pad, h))
            crop = img[y0:y1, x0:x1]
            crop_filename = f"{doc_id}_{fld}_crop.png"
            crop_path = os.path.join(out_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
            crops_saved[fld] = crop_path
            print(f"Saved crop for {fld} to {crop_path} (shape: {crop.shape})")
    # Run independent blind re-reads on both crops to report exact re-read values and confidences
    from src.stage4_enhance_extract import enhance_and_reextract
    reread_info = {}
    for fld in w['corrupted_fields']:
        crop_path = crops_saved.get(fld)
        if crop_path and os.path.exists(crop_path):
            crop_bgr = cv2.imread(crop_path)
            _, conf_post, _, _, metrics = enhance_and_reextract(crop_bgr, gt_clean_val=0.0, current_conf=0.50)
            blind_val = metrics.get("blind_reread_val")
            reread_info[fld] = {"blind_val": blind_val, "confidence": conf_post, "metrics": metrics}
            print(f"Blind re-read for {fld}: blind_val={blind_val} (confidence: {conf_post})")

    # Risk flip analysis on LDL
    gt_ldl = w['gt']['ldl']
    com_ldl = w['extracted']['ldl']
    raw_ldl = w['disp']['ldl']
    risk_flip_vs_gt = evaluate_clinical_risk_flip(com_ldl, gt_ldl)
    print(f"LDL GT: {gt_ldl:.1f} mg/dL ({risk_flip_vs_gt['verified_tier']['category']})")
    print(f"LDL Committed: {com_ldl:.1f} mg/dL ({risk_flip_vs_gt['raw_tier']['category']})")
    print(f"Risk Flip: {risk_flip_vs_gt['has_risk_flip']} | Direction: {risk_flip_vs_gt['direction']} | Severity: {risk_flip_vs_gt['severity']}")
    
    print("\nAudit trail stages:")
    for a in w['audit']:
        print(f"  Stage: {a.get('stage')}, Action: {a.get('action')}, Suspects: {a.get('suspect_fields')}, Candidates: {a.get('candidates')}")

    results_report.append({
        'doc_id': doc_id,
        'status': w['status'],
        'corrupted_fields': w['corrupted_fields'],
        'corruption_types': w['corrupt_types'],
        'ground_truth': w['gt'],
        'display_input': w['disp'],
        'committed': w['extracted'],
        'risk_flip': risk_flip_vs_gt,
        'crops': crops_saved,
        'blind_reread': reread_info,
        'audit': w['audit']
    })

with open(os.path.join(ROOT, "reports", "k2_wrong_case_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(results_report, f, indent=2)
print("\nWrote reports/k2_wrong_case_analysis.json")

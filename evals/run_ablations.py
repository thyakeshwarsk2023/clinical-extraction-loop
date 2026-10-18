"""
evals/run_ablations.py -- Evaluation harness for the lipid-panel
self-correction pipeline over the dataset_A_synthetic benchmark.

Loads four splits:
  dataset_A_synthetic/clean/
  dataset_A_synthetic/clean_control/
  dataset_A_synthetic/corrupted_single/
  dataset_A_synthetic/corrupted_double/

Each split has a ground_truth.csv. Because there are no .json sidecars,
this harness builds them on-the-fly in a temp directory, then calls
extract_document_state() -> run_pipeline() for each document.

Outputs:
  audit/<split>_<doc_id>.json  -- per-doc audit record
  Summary table printed to stdout

Usage:
    python -m evals.run_ablations
    python -m evals.run_ablations -v   (verbose: one line per doc)
"""

import os
import sys
import csv
import json
import ast
import tempfile
import traceback
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline
from src.stage2_constraints import normalize_to_mgdl
from evals.metrics import (
    FIELDS, CLEAN_MODES, EXPECTED_ABSTAIN_MODES,
    field_accuracy, isolation_accuracy,
    status_distribution, false_trigger_rate, correction_precision,
)

DATASET_ROOT = os.path.join(ROOT, 'dataset_A_synthetic')
AUDIT_DIR    = os.path.join(ROOT, 'audit')
os.makedirs(AUDIT_DIR, exist_ok=True)

# Map short CSV keys to pipeline long keys (never rename these)
SHORT_TO_LONG = {
    'TC':  'total_cholesterol',
    'HDL': 'hdl',
    'LDL': 'ldl',
    'TG':  'triglycerides',
}
LONG_TO_SHORT = {v: k for k, v in SHORT_TO_LONG.items()}

# Dummy bbox used when the CSV provides no bbox for a field.
# stage3_vision_gate and stage4_reextract are stubs that ignore the crop,
# so any value is safe here.
DUMMY_BBOX = [0, 0, 10, 10]

METHOD_TEXT_FRIEDEWALD = 'Method: Calculated by Friedewald Equation'


# ---------------------------------------------------------------------------
# Sidecar builder
# ---------------------------------------------------------------------------

def parse_bbox_str(s):
    '''Parse (x0, y0, x1, y1) string to [x0, y0, x1, y1] list.'''
    try:
        t = ast.literal_eval(s.strip())
        return list(t)
    except Exception:
        return DUMMY_BBOX


def build_sidecar(img_path, display_values, gt_clean_values,
                  unit, corrupted_long_keys, bboxes_long, method_text):
    '''
    Writes a temporary .json sidecar next to img_path so that
    extract_document_state() can load it.

    display_values  -- {long_key: float}  (values shown in the document,
                        possibly corrupted)
    gt_clean_values -- {long_key: float}  (ground-truth clean values)
    unit            -- 'mg/dL' or 'mmol/L'
    corrupted_long_keys -- list of long-key strings that were corrupted
    bboxes_long     -- {long_key: [x0,y0,x1,y1]} (may contain DUMMY_BBOX)
    method_text     -- string for assay classification
    '''
    payload = {
        'method_text':           method_text,
        'unit':                  unit,
        'injected_display_data': display_values,
        'bounding_boxes':        bboxes_long,
        'corrupted_fields':      corrupted_long_keys,
        'ground_truth_clean':    gt_clean_values,
        'assay_ground_truth':    'FRIEDEWALD',
    }
    json_path = os.path.splitext(img_path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh)
    return json_path


def cleanup_sidecar(img_path):
    '''Remove the temporary .json sidecar after the run.'''
    json_path = os.path.splitext(img_path)[0] + '.json'
    if os.path.exists(json_path):
        os.remove(json_path)


# ---------------------------------------------------------------------------
# CSV loaders for each split
# ---------------------------------------------------------------------------

def load_clean_rows(split_name):
    '''
    Returns list of dicts for clean or clean_control split.
    All values are normalised to mg/dL so field_accuracy() always
    compares in the same unit as pipeline output.
    '''
    split_dir = os.path.join(DATASET_ROOT, split_name)
    csv_path  = os.path.join(split_dir, 'ground_truth.csv')
    records = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            doc_id = row['doc_id']
            unit   = row['unit']
            native_vals = {
                SHORT_TO_LONG[k]: float(row[k])
                for k in ('TC', 'HDL', 'LDL', 'TG')
            }
            # Normalise BOTH display and gt to mg/dL for apples-to-apples
            # comparison with pipeline output (which always returns mg/dL).
            gt_vals = normalize_to_mgdl(native_vals, unit)
            records.append({
                'doc_id':              doc_id,
                'split':               split_name,
                'display_values':      dict(native_vals),  # sidecar keeps native unit
                'gt_clean_values':     dict(gt_vals),      # comparison in mg/dL
                'unit':                unit,
                'corrupted_long_keys': [],
                'bboxes_long':         {lk: list(DUMMY_BBOX) for lk in SHORT_TO_LONG.values()},
                'method_text':         METHOD_TEXT_FRIEDEWALD,
                'img_path':            os.path.join(split_dir, doc_id + '.png'),
                'corruption_type':     None,
            })
    return records


def load_corrupted_single_rows(clean_map):
    '''
    Returns list of dicts for corrupted_single split.
    clean_map: {doc_id: clean_csv_row} from the clean split CSV.
    '''
    split_dir = os.path.join(DATASET_ROOT, 'corrupted_single')
    csv_path  = os.path.join(split_dir, 'ground_truth.csv')
    records = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            doc_id     = row['doc_id']
            short_key  = row['corrupted_field']       # e.g. 'HDL'
            long_key   = SHORT_TO_LONG[short_key]
            orig_val   = float(row['orig_val'])
            corrupt_val = float(row['corrupt_val'])
            corr_type  = row['corruption_type']

            clean_row  = clean_map[doc_id]
            unit       = clean_row['unit']

            # Ground-truth clean values normalised to mg/dL
            native_gt = {
                SHORT_TO_LONG[k]: float(clean_row[k])
                for k in ('TC', 'HDL', 'LDL', 'TG')
            }
            gt_vals = normalize_to_mgdl(native_gt, unit)
            # Display values: swap the corrupted field (still in native unit)
            native_display = dict(native_gt)
            native_display[long_key] = corrupt_val

            # Bboxes: use the annotated one for the corrupted field,
            # dummy for the rest
            bboxes = {lk: list(DUMMY_BBOX) for lk in SHORT_TO_LONG.values()}
            bboxes[long_key] = parse_bbox_str(row['true_bbox'])

            records.append({
                'doc_id':              doc_id,
                'split':               'corrupted_single',
                'display_values':      native_display,
                'gt_clean_values':     gt_vals,       # mg/dL for comparison
                'unit':                unit,
                'corrupted_long_keys': [long_key],
                'bboxes_long':         bboxes,
                'method_text':         METHOD_TEXT_FRIEDEWALD,
                'img_path':            os.path.join(split_dir, doc_id + '.png'),
                'corruption_type':     corr_type,
            })
    return records


def load_corrupted_double_rows(clean_map):
    '''
    Returns list of dicts for corrupted_double split.
    '''
    split_dir = os.path.join(DATASET_ROOT, 'corrupted_double')
    csv_path  = os.path.join(split_dir, 'ground_truth.csv')
    records = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            doc_id  = row['doc_id']
            sk1     = row['field1']
            sk2     = row['field2']
            lk1     = SHORT_TO_LONG[sk1]
            lk2     = SHORT_TO_LONG[sk2]
            c1      = float(row['corrupt1'])
            c2      = float(row['corrupt2'])
            corr_type = row['types']

            clean_row = clean_map[doc_id]
            unit      = clean_row['unit']
            native_gt = {
                SHORT_TO_LONG[k]: float(clean_row[k])
                for k in ('TC', 'HDL', 'LDL', 'TG')
            }
            gt_vals = normalize_to_mgdl(native_gt, unit)
            native_display = dict(native_gt)
            native_display[lk1] = c1
            native_display[lk2] = c2

            # Parse two bboxes from the pipe-separated string
            bboxes = {lk: list(DUMMY_BBOX) for lk in SHORT_TO_LONG.values()}
            bbox_parts = row['bboxes'].split('|')
            if len(bbox_parts) >= 1:
                bboxes[lk1] = parse_bbox_str(bbox_parts[0])
            if len(bbox_parts) >= 2:
                bboxes[lk2] = parse_bbox_str(bbox_parts[1])

            records.append({
                'doc_id':              doc_id,
                'split':               'corrupted_double',
                'display_values':      native_display,
                'gt_clean_values':     gt_vals,       # mg/dL for comparison
                'unit':                unit,
                'corrupted_long_keys': [lk1, lk2],
                'bboxes_long':         bboxes,
                'method_text':         METHOD_TEXT_FRIEDEWALD,
                'img_path':            os.path.join(split_dir, doc_id + '.png'),
                'corruption_type':     corr_type,
            })
    return records


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

def run_eval(verbose=False):
    # Load the clean split CSV (needed as reference for corrupted splits)
    clean_csv = os.path.join(DATASET_ROOT, 'clean', 'ground_truth.csv')
    with open(clean_csv, newline='', encoding='utf-8') as fh:
        clean_map = {r['doc_id']: r for r in csv.DictReader(fh)}

    records = []
    records += load_clean_rows('clean')
    records += load_clean_rows('clean_control')
    records += load_corrupted_single_rows(clean_map)
    records += load_corrupted_double_rows(clean_map)

    print('Loaded {} records across 4 splits:'.format(len(records)))
    split_counts = defaultdict(int)
    for r in records:
        split_counts[r['split']] += 1
    for sp, cnt in sorted(split_counts.items()):
        print('  {:<25} {:>4} docs'.format(sp, cnt))
    print('Audit JSONs -> {}\n'.format(AUDIT_DIR))

    all_results = []
    errors = []

    for rec in records:
        img_path = rec['img_path']
        if not os.path.exists(img_path):
            print('  [WARN] Image missing: {}'.format(img_path))
            continue

        # Write temporary sidecar so extract_document_state() can load it
        build_sidecar(
            img_path           = img_path,
            display_values     = rec['display_values'],
            gt_clean_values    = rec['gt_clean_values'],
            unit               = rec['unit'],
            corrupted_long_keys = rec['corrupted_long_keys'],
            bboxes_long        = rec['bboxes_long'],
            method_text        = rec['method_text'],
        )

        try:
            result = run_pipeline(img_path)
        except Exception as exc:
            err_msg = '{}/{}: {}'.format(rec['split'], rec['doc_id'], exc)
            print('  [ERROR] ' + err_msg)
            if verbose:
                traceback.print_exc()
            errors.append(err_msg)
            cleanup_sidecar(img_path)
            continue
        finally:
            cleanup_sidecar(img_path)

        corrected_data   = result.get('data', {})
        audit_trail      = result.get('audit_trail', [])
        corrupted_fields = rec['corrupted_long_keys']

        fa  = field_accuracy(corrected_data, rec['gt_clean_values'])
        iso = isolation_accuracy(audit_trail, corrupted_fields)

        row = {
            'doc_id':            rec['doc_id'],
            'split':             rec['split'],
            'status':            result['status'],
            'unit':              rec['unit'],
            'corruption_type':   rec['corruption_type'],
            'corrupted_fields':  corrupted_fields,
            'corrected_data':    corrected_data,
            'gt_clean_values':   rec['gt_clean_values'],
            'field_accuracy':    fa,
            'isolation_correct': iso,
            'audit_trail':       audit_trail,
            'residual':          result.get('residual'),
            # evals/metrics.py uses 'mode' key for CLEAN_MODES check;
            # map split names to the expected mode values
            'mode':              ('CLEAN' if rec['split'] in ('clean', 'clean_control')
                                  else rec['split']),
        }
        all_results.append(row)

        audit_key = '{}_{}'.format(rec['split'], rec['doc_id'])
        audit_path = os.path.join(AUDIT_DIR, audit_key + '.json')
        with open(audit_path, 'w', encoding='utf-8') as fh:
            json.dump({
                'doc_id':            rec['doc_id'],
                'split':             rec['split'],
                'unit':              rec['unit'],
                'corruption_type':   rec['corruption_type'],
                'corrupted_fields':  corrupted_fields,
                'gt_clean_values':   rec['gt_clean_values'],
                'pipeline_result':   result,
                'metrics': {
                    'field_accuracy':    fa,
                    'isolation_correct': iso,
                },
            }, fh, indent=2, default=str)

        if verbose:
            iso_str = {True: 'OK', False: 'MISS', None: 'N/A'}.get(iso, '?')
            print('  {:<17} {:<12} {:<32} fa={:.0%} iso={}'.format(
                rec['split'], rec['doc_id'], result['status'],
                fa.get('overall', 0.0), iso_str))

    if errors:
        print('\n[WARN] {} doc(s) failed -- see above.\n'.format(len(errors)))

    return all_results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results):
    sep  = '=' * 74
    thin = '-' * 74
    n    = len(results)

    print('\n' + sep)
    print('  BENCHMARK EVALUATION SUMMARY  (dataset_A_synthetic)')
    print(sep)
    print('  Total documents evaluated : {}'.format(n))

    # Split breakdown
    by_split = defaultdict(list)
    for r in results:
        by_split[r['split']].append(r)
    print('  Split breakdown:')
    for sp in ['clean', 'clean_control', 'corrupted_single', 'corrupted_double']:
        print('    {:<25} {:>4}'.format(sp, len(by_split.get(sp, []))))

    # --- 1. Status distribution ---
    sd = status_distribution(results)
    print('\nSTATUS DISTRIBUTION')
    print('  {:<48} {:>5} {:>7}'.format('Status', 'Count', '%'))
    print('  ' + thin)
    for status, count in sd.items():
        print('  {:<48} {:>5} {:>6.1%}'.format(status, count, count / n))

    # --- 2. False-trigger rate on clean_control ---
    ctrl_rows  = by_split.get('clean_control', [])
    clean_rows = by_split.get('clean', [])
    if ctrl_rows:
        triggered = [r for r in ctrl_rows if r.get('audit_trail')]
        ftr = len(triggered) / len(ctrl_rows)
        print('\nFALSE-TRIGGER RATE on clean_control  (n={})'.format(len(ctrl_rows)))
        print('  Docs that triggered a correction attempt : {}/{} = {:.1%}'.format(
            len(triggered), len(ctrl_rows), ftr))
    if clean_rows:
        triggered2 = [r for r in clean_rows if r.get('audit_trail')]
        ftr2 = len(triggered2) / len(clean_rows)
        print('  False-trigger rate on clean split        : {}/{} = {:.1%}'.format(
            len(triggered2), len(clean_rows), ftr2))

    # --- 3. Field accuracy (corrupted subsets) ---
    for split_name in ['corrupted_single', 'corrupted_double']:
        rows = by_split.get(split_name, [])
        if not rows:
            continue
        print('\nFIELD ACCURACY -- POST-CORRECTION  ({}, n={})'.format(split_name, len(rows)))
        print('  {:<25} {:>7} {:>7} {:>9}'.format('Field', 'Correct', 'Total', 'Accuracy'))
        print('  ' + thin)
        for f in FIELDS:
            correct = sum(1 for r in rows if r['field_accuracy'].get(f) is True)
            total   = sum(1 for r in rows if r['field_accuracy'].get(f) is not None)
            acc     = correct / total if total else 0.0
            print('  {:<25} {:>7} {:>7} {:>8.1%}'.format(f, correct, total, acc))
        oa = sum(r['field_accuracy'].get('overall', 0.0) for r in rows) / len(rows)
        print('  {:<25} {:>7} {:>7} {:>8.1%}'.format('OVERALL AVERAGE', '', '', oa))

    # --- 4. Field accuracy -- pre-correction baseline ---
    # 'pre-correction' means: what would field accuracy be if we just
    # passed the raw corrupted values (corrupt_val) through without any fix?
    # We approximate this: CONVERGED on corrupted = the pipeline accepted
    # (and possibly fixed) the value; the corrected_data is the pipeline output.
    # True pre-correction accuracy = fraction of corrupted fields where
    # corrupt_val == orig_val (i.e. smudge -- value unchanged).
    for split_name in ['corrupted_single', 'corrupted_double']:
        rows = by_split.get(split_name, [])
        if not rows:
            continue
        # smudge rows already have correct values, so pre and post are same
        smudge_rows = [r for r in rows
                       if r.get('corruption_type') and 'smudge' in r['corruption_type']]
        non_smudge  = [r for r in rows
                       if not (r.get('corruption_type') and 'smudge' in r['corruption_type'])]
        # Pre-correction field accuracy on non-smudge = 0% on corrupted fields
        # (by construction -- corrupt_val != orig_val).
        # We report post-correction on all, and break out by type.
        print('\nBREAKDOWN BY CORRUPTION TYPE  ({})'.format(split_name))
        print('  {:<35} {:>5} {:>10} {:>11}'.format(
            'corruption_type', 'N', 'Converged', 'AvgFieldAcc'))
        print('  ' + thin)
        type_groups = defaultdict(list)
        for r in rows:
            type_groups[r.get('corruption_type', 'unknown')].append(r)
        for ctype, crows in sorted(type_groups.items()):
            conv  = sum(1 for r in crows if 'CONVERGED' in r['status'])
            avg   = sum(r['field_accuracy'].get('overall', 0.0) for r in crows) / len(crows)
            print('  {:<35} {:>5} {:>10} {:>10.1%}'.format(ctype, len(crows), conv, avg))

    # --- 5. Isolation accuracy ---
    for split_name in ['corrupted_single', 'corrupted_double']:
        rows = by_split.get(split_name, [])
        isolatable = [r for r in rows if r['isolation_correct'] is not None]
        if not isolatable:
            continue
        iso_correct = sum(1 for r in isolatable if r['isolation_correct'])
        iso_total   = len(isolatable)
        print('\nISOLATION ACCURACY  ({}, n={})'.format(split_name, iso_total))
        print('  Overall: {}/{} = {:.1%}'.format(
            iso_correct, iso_total, iso_correct / iso_total))
        # Break down by corruption type
        type_groups = defaultdict(list)
        for r in isolatable:
            type_groups[r.get('corruption_type', 'unknown')].append(r['isolation_correct'])
        print('  {:<35} {:>7} {:>7} {:>9}'.format('corruption_type', 'Correct', 'Total', 'Accuracy'))
        print('  ' + thin)
        for ctype, vals in sorted(type_groups.items()):
            nc = sum(vals)
            print('  {:<35} {:>7} {:>7} {:>8.1%}'.format(ctype, nc, len(vals), nc / len(vals)))

    # --- 6. Per-split summary row ---
    print('\nPER-SPLIT SUMMARY')
    print('  {:<25} {:>4} {:>10} {:>12} {:>8} {:>10}'.format(
        'Split', 'N', 'Converged', 'AvgFieldAcc', 'IsoAcc', 'Dominant Status'))
    print('  ' + thin)
    for sp in ['clean', 'clean_control', 'corrupted_single', 'corrupted_double']:
        rows = by_split.get(sp, [])
        if not rows:
            continue
        conv  = sum(1 for r in rows if 'CONVERGED' in r['status'])
        avg   = sum(r['field_accuracy'].get('overall', 0.0) for r in rows) / len(rows)
        iso_r = [r for r in rows if r['isolation_correct'] is not None]
        iso_s = ('{}/{}'.format(sum(r['isolation_correct'] for r in iso_r), len(iso_r))
                 if iso_r else 'N/A')
        dom_sd = status_distribution(rows)
        dom = max(dom_sd, key=dom_sd.get)
        print('  {:<25} {:>4} {:>10} {:>11.1%} {:>8} {:>10}'.format(
            sp, len(rows), conv, avg, iso_s, dom))

    print('\n' + sep)
    print('  Audit JSONs written to : ' + AUDIT_DIR)
    print(sep + '\n')


if __name__ == '__main__':
    verbose_flag = '--verbose' in sys.argv or '-v' in sys.argv
    results = run_eval(verbose=verbose_flag)
    print_summary(results)

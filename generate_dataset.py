#!/usr/bin/env python3
"""
Corrected Dataset A/B generator for the lipid-panel self-correction project.

Fixes applied vs. the previous delivery (see audit):
  1. ground_truth.csv now stores the SAME value/unit actually rendered on the
     image (previously mmol/L-labeled rows silently kept the raw mg/dL number).
  2. clean_control subset (~15% of Dataset A) is now generated and logged.
  3. hard_visual_subset now has a ground-truth row for every image (previously
     66/102 images were unlabeled orphans).
  4. Dataset B splits are deduplicated (previously each doc_id was repeated 5x).
  5. Dataset A splits are validated against actual rendered files (no orphan
     doc_ids referencing images that don't exist in any subfolder).
  6. 3 distinct templates instead of 2 (spec asked for 3-5).

NOTE on value pool: this sandbox has no network access to wwwn.cdc.gov, so a
physiologically-plausible synthetic pool is used instead of real NHANES data
(clearly marked in nhanes_notes.md). Swap `build_value_pool()` for a real
`pandas.read_sas()` NHANES loader when you have that access.
"""
import os, csv, random, shutil, json
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

ROOT = "/home/claude/gen/output"
A_DIR = os.path.join(ROOT, "dataset_A_synthetic")
B_DIR = os.path.join(ROOT, "dataset_B_realistic")
W, H = 750, 420

def font(size=13, bold=False):
    # Use PIL default font scaled; keep dependency-free.
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
            size,
        )
    except Exception:
        return ImageFont.load_default()

FIELDS = ["Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Triglycerides"]
FIELD_KEYS = ["TC", "HDL", "LDL", "TG"]

# ---------------------------------------------------------------- value pool
def build_value_pool(n=260):
    """Physiologically-plausible synthetic lipid tuples (stand-in for NHANES)."""
    pool = []
    for i in range(n):
        hdl = round(random.uniform(25, 90), 1)
        tg = round(random.uniform(40, 395), 1)  # keep <=400 -> Friedewald-applicable pool
        tc = round(random.uniform(120, 260), 1)
        ldl = round(tc - hdl - tg / 5.0, 1)
        if ldl <= 0:
            continue
        pool.append({"id": i, "TC": tc, "HDL": hdl, "LDL": ldl, "TG": tg, "cycle": "synthetic_v1"})
    return pool

def write_nhanes_notes():
    with open(os.path.join(ROOT, "nhanes_notes.md"), "w") as f:
        f.write(
            "# Value Pool Notes\n\n"
            "This pool is SYNTHETIC, not real NHANES data — this environment has no "
            "network access to wwwn.cdc.gov. All rows use `assay_method=friedewald` "
            "by construction (LDL = TC - HDL - TG/5) before any per-dataset perturbation "
            "is applied downstream.\n\n"
            "To move to paper-grade results: replace `build_value_pool()` with a "
            "`pandas.read_sas()` loader over TCHOL_*/HDL_*/TRIGLY_* NHANES cycles, "
            "merged on SEQN, and re-run this script unchanged — the rest of the "
            "pipeline only depends on the TC/HDL/LDL/TG/id schema.\n"
        )

# --------------------------------------------------------------- unit helper
def to_display(tc, hdl, ldl, tg, unit):
    """Return the values AS THEY WILL BE RENDERED, in the given display unit.
    This is the single source of truth used both for drawing and for the
    ground-truth CSV, which is exactly the bug this script fixes."""
    if unit == "mg/dL":
        return round(tc, 2), round(hdl, 2), round(ldl, 2), round(tg, 2)
    # mmol/L
    return (
        round(tc / 38.67, 2),
        round(hdl / 38.67, 2),
        round(ldl / 38.67, 2),
        round(tg / 88.57, 2),
    )

# --------------------------------------------------------- reference ranges
# Canonical adult ranges in mg/dL; convert per-field when unit is mmol/L.
REF_RANGES_MGDL = {
    "Total Cholesterol": ("<", 200),
    "HDL Cholesterol": (">", 40),
    "LDL Cholesterol": ("<", 100),
    "Triglycerides": ("<", 150),
}
MGDL_TO_MMOL = {"Total Cholesterol": 38.67, "HDL Cholesterol": 38.67,
                "LDL Cholesterol": 38.67, "Triglycerides": 88.57}

def ref_range_text(label, unit, force_mgdl_scale=False):
    """Reference range string for a field. If force_mgdl_scale=True, always
    print the mg/dL-scale threshold even when unit is mmol/L — this is the
    deliberate reference-range/unit mismatch used to exercise Stage 0's new
    sanity check."""
    op, thresh = REF_RANGES_MGDL[label]
    if unit == "mmol/L" and not force_mgdl_scale:
        thresh = round(thresh / MGDL_TO_MMOL[label], 1)
    return f"{op} {thresh}"

# ------------------------------------------------------------------ templates
def render_template(template_id, tc, hdl, ldl, tg, unit, assay_note=None, ref_ranges=None):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    labels = ["Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Triglycerides"]
    vals = [tc, hdl, ldl, tg]
    bboxes = {}
    ref_ranges = ref_ranges or {}

    if template_id == 0:
        d.text((30, 20), "LIPID PROFILE REPORT", font=font(16, True), fill="black")
        d.line((30, 55, 720, 55), fill="black", width=2)
        headers = ["Test", "Result", "Unit", "Reference Range"]
        xs = [30, 400, 500, 600]
        for x, htext in zip(xs, headers):
            d.text((x, 90), htext, font=font(12, True), fill="black")
        d.line((30, 115, 720, 115), fill="gray", width=1)
        y = 135
        for lab, val in zip(labels, vals):
            d.text((30, y), lab, font=font(12), fill="black")
            vx, vy = 400, y
            vw = d.textlength(f"{val}", font=font(12))
            d.text((vx, vy), f"{val}", font=font(12), fill="black")
            bboxes[lab] = (int(vx) - 2, int(vy) - 2, int(vx + vw) + 2, int(vy + 16))
            d.text((500, y), unit, font=font(12), fill="black")
            if lab in ref_ranges:
                d.text((600, y), ref_ranges[lab], font=font(12), fill="black")
            y += 45
        if assay_note:
            d.text((30, H - 25), assay_note, font=font(10), fill=(90, 90, 90))

    elif template_id == 1:
        d.rectangle((0, 0, W, 55), fill=(230, 230, 230))
        d.text((30, 18), "Lipid Panel Results", font=font(15, True), fill="black")
        y = 90
        for lab, val in zip(labels, vals):
            d.rectangle((30, y, 720, y + 55), outline="black", width=1)
            d.text((45, y + 20), lab, font=font(12), fill="black")
            txt = f"{val} {unit}"
            vx, vy = 560, y + 20
            vw = d.textlength(txt, font=font(12))
            d.text((vx, vy), txt, font=font(12), fill="black")
            bboxes[lab] = (int(vx) - 2, int(vy) - 2, int(vx + vw) + 2, int(vy + 16))
            if lab in ref_ranges:
                d.text((45, y + 38), f"Ref: {ref_ranges[lab]}", font=font(9), fill=(110, 110, 110))
            y += 65
        if assay_note:
            d.text((30, H - 25), assay_note, font=font(10), fill=(90, 90, 90))

    else:  # template_id == 2
        d.text((30, 25), "CLINICAL CHEMISTRY - LIPID PANEL", font=font(14, True), fill="black")
        d.text((30, 45), "Specimen: Serum", font=font(10), fill=(90, 90, 90))
        d.line((30, 70, 720, 70), fill="black", width=1)
        y = 100
        for idx, (lab, val) in enumerate(zip(labels, vals)):
            bg = (245, 245, 245) if idx % 2 == 0 else (255, 255, 255)
            d.rectangle((30, y, 720, y + 50), fill=bg)
            d.text((45, y + 15), lab.upper(), font=font(11, True), fill="black")
            txt = f"{val}  {unit}"
            vx, vy = 480, y + 15
            vw = d.textlength(txt, font=font(11))
            d.text((vx, vy), txt, font=font(11), fill="black")
            bboxes[lab] = (int(vx) - 2, int(vy) - 2, int(vx + vw) + 2, int(vy + 16))
            if lab in ref_ranges:
                d.text((45, y + 32), f"Ref: {ref_ranges[lab]}", font=font(9), fill=(110, 110, 110))
            y += 50
        if assay_note:
            d.text((30, H - 25), assay_note, font=font(10), fill=(90, 90, 90))

    return img, bboxes

# ---------------------------------------------------------------- corruption
def corrupt_value(val, ctype):
    if ctype == "digit_swap":
        s = f"{val:.1f}"
        digs = [c for c in s if c.isdigit()]
        if len(digs) >= 2:
            i, j = random.sample(range(len(digs)), 2)
            digs[i], digs[j] = digs[j], digs[i]
        new_s, di = "", 0
        for c in s:
            if c.isdigit():
                new_s += digs[di]; di += 1
            else:
                new_s += c
        try:
            return float(new_s)
        except ValueError:
            return val * 10
    if ctype == "decimal_shift":
        return round(val * 10, 1) if random.random() < 0.5 else round(val / 10, 1)
    if ctype == "smudge":
        return val  # visual-only corruption; true value unchanged, image degraded
    return val

def apply_smudge(img, bbox):
    img = img.copy()
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    d.rectangle((x0, y0, x1, y1), fill=(210, 210, 210))
    d.rectangle((x0, y0, x1, y1), outline=(190, 190, 190))
    return img

def degrade_hard(img):
    """Heavier degradation for the restoration-comparison subset."""
    small = img.resize((W // 3, H // 3), Image.BILINEAR)
    return small.resize((W, H), Image.BILINEAR)

# ------------------------------------------------------- bbox mislocalization
def simulate_detected_bbox(true_bbox, status):
    """Simulate what a (fallible) Stage-1 detector would hand to Stage 3,
    so Stage 3's contour-area-threshold fallback has something real to
    trigger on. status: 'normal' | 'shrunk' | 'shifted'."""
    x0, y0, x1, y1 = true_bbox
    if status == "normal":
        return true_bbox
    if status == "shrunk":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0) * 0.15, (y1 - y0) * 0.15  # collapses well below theta
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    # shifted: same size, moved off the actual text (e.g. onto blank row space)
    dx, dy = random.randint(60, 120), random.randint(15, 30)
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

# --------------------------------------------------------------------- main A
def build_dataset_A(pool, n_clean=200):
    for sub in ["clean", "corrupted_single", "corrupted_double", "clean_control",
                "hard_visual_subset", "splits"]:
        os.makedirs(os.path.join(A_DIR, sub), exist_ok=True)

    sample = random.sample(pool, n_clean)
    clean_rows, single_rows, double_rows, control_rows = [], [], [], []
    doc_meta = {}  # doc_id -> (tuple, unit, template_id) for split validation

    # ~15% of docs get a deliberate reference-range/unit mismatch (Stage 0 test cases)
    mismatch_ids = set(random.sample(range(n_clean), int(0.15 * n_clean)))

    for i, tup in enumerate(sample):
        doc_id = f"doc_{i:04d}"
        template_id = i % 3
        unit = random.choice(["mg/dL", "mmol/L"])
        tc, hdl, ldl, tg = to_display(tup["TC"], tup["HDL"], tup["LDL"], tup["TG"], unit)
        force_wrong = i in mismatch_ids and unit == "mmol/L"
        ref_ranges = {lab: ref_range_text(lab, unit, force_mgdl_scale=force_wrong) for lab in FIELDS}
        expected_ranges = {lab: ref_range_text(lab, unit, force_mgdl_scale=False) for lab in FIELDS}
        img, bboxes = render_template(template_id, tc, hdl, ldl, tg, unit, ref_ranges=ref_ranges)
        img.save(os.path.join(A_DIR, "clean", f"{doc_id}.png"))
        clean_rows.append([doc_id, tc, hdl, ldl, tg, unit, template_id,
                            force_wrong, json.dumps(ref_ranges), json.dumps(expected_ranges)])
        doc_meta[doc_id] = dict(tc=tc, hdl=hdl, ldl=ldl, tg=tg, unit=unit,
                                 template_id=template_id, bboxes=bboxes)

    # partition indices: 60% single, 25% double, 15% clean_control
    idxs = list(range(n_clean)); random.shuffle(idxs)
    n_single = int(0.60 * n_clean)
    n_double = int(0.25 * n_clean)
    single_idx = idxs[:n_single]
    double_idx = idxs[n_single:n_single + n_double]
    control_idx = idxs[n_single + n_double:]

    for i in single_idx:
        doc_id = f"doc_{i:04d}"
        m = doc_meta[doc_id]
        field = random.choice(FIELDS)
        ctype = random.choice(["digit_swap", "decimal_shift", "smudge"])
        vals = {"Total Cholesterol": m["tc"], "HDL Cholesterol": m["hdl"],
                "LDL Cholesterol": m["ldl"], "Triglycerides": m["tg"]}
        orig = vals[field]
        corrupt = corrupt_value(orig, ctype)
        ref_ranges = {lab: ref_range_text(lab, m["unit"]) for lab in FIELDS}
        img, bboxes = render_template(m["template_id"], *(vals[f] if f != field else corrupt
                                       for f in ["Total Cholesterol", "HDL Cholesterol",
                                                 "LDL Cholesterol", "Triglycerides"]),
                                       m["unit"], ref_ranges=ref_ranges)
        bbox = bboxes[field]
        if ctype == "smudge":
            img = apply_smudge(img, bbox)
        img.save(os.path.join(A_DIR, "corrupted_single", f"{doc_id}.png"))
        fkey = FIELD_KEYS[FIELDS.index(field)]
        # ~20% of single-corruption docs simulate a Stage-1 bbox mislocalization,
        # so Stage 3's contour-area-threshold fallback has real cases to trigger on
        r = random.random()
        bbox_status = "shrunk" if r < 0.10 else ("shifted" if r < 0.20 else "normal")
        detected_bbox = simulate_detected_bbox(bbox, bbox_status)
        single_rows.append([doc_id, fkey, orig, corrupt, ctype, bbox, bbox_status, detected_bbox])

    for i in double_idx:
        doc_id = f"doc_{i:04d}"
        m = doc_meta[doc_id]
        f1, f2 = random.sample(FIELDS, 2)
        t1, t2 = random.choice(["digit_swap", "decimal_shift", "smudge"]), \
                  random.choice(["digit_swap", "decimal_shift", "smudge"])
        vals = {"Total Cholesterol": m["tc"], "HDL Cholesterol": m["hdl"],
                "LDL Cholesterol": m["ldl"], "Triglycerides": m["tg"]}
        o1, o2 = vals[f1], vals[f2]
        c1, c2 = corrupt_value(o1, t1), corrupt_value(o2, t2)
        newvals = dict(vals); newvals[f1] = c1; newvals[f2] = c2
        img, bboxes = render_template(
            m["template_id"], newvals["Total Cholesterol"], newvals["HDL Cholesterol"],
            newvals["LDL Cholesterol"], newvals["Triglycerides"], m["unit"])
        for f, t in [(f1, t1), (f2, t2)]:
            if t == "smudge":
                img = apply_smudge(img, bboxes[f])
        img.save(os.path.join(A_DIR, "corrupted_double", f"{doc_id}.png"))
        k1, k2 = FIELD_KEYS[FIELDS.index(f1)], FIELD_KEYS[FIELDS.index(f2)]
        double_rows.append([doc_id, k1, k2, o1, o2, c1, c2, f"{t1}+{t2}",
                             f"{bboxes[f1]}|{bboxes[f2]}"])

    for i in control_idx:
        doc_id = f"doc_{i:04d}"
        m = doc_meta[doc_id]
        shutil.copy(os.path.join(A_DIR, "clean", f"{doc_id}.png"),
                    os.path.join(A_DIR, "clean_control", f"{doc_id}.png"))
        control_rows.append([doc_id, m["tc"], m["hdl"], m["ldl"], m["tg"], m["unit"], m["template_id"]])

    # hard_visual_subset: heavier degradation of smudge docs, FULL paired ground truth
    smudge_docs = [r for r in single_rows if r[4] == "smudge"]
    hard_rows = []
    for doc_id, fkey, orig, corrupt, ctype, bbox, bbox_status, detected_bbox in smudge_docs:
        img = Image.open(os.path.join(A_DIR, "corrupted_single", f"{doc_id}.png"))
        hard = degrade_hard(img)
        hard.save(os.path.join(A_DIR, "hard_visual_subset", f"{doc_id}.jpg"), quality=70)
        hard_rows.append([doc_id, fkey, orig])

    # write ground truth CSVs
    def wcsv(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    wcsv(os.path.join(A_DIR, "clean", "ground_truth.csv"),
         ["doc_id", "TC", "HDL", "LDL", "TG", "unit", "template_id",
          "ref_range_mismatch", "printed_ranges", "expected_ranges"], clean_rows)
    wcsv(os.path.join(A_DIR, "corrupted_single", "ground_truth.csv"),
         ["doc_id", "corrupted_field", "orig_val", "corrupt_val", "corruption_type",
          "true_bbox", "bbox_status", "detected_bbox"], single_rows)
    wcsv(os.path.join(A_DIR, "corrupted_double", "ground_truth.csv"),
         ["doc_id", "field1", "field2", "orig1", "orig2", "corrupt1", "corrupt2", "types", "bboxes"], double_rows)
    wcsv(os.path.join(A_DIR, "clean_control", "ground_truth.csv"),
         ["doc_id", "TC", "HDL", "LDL", "TG", "unit", "template_id"], control_rows)
    wcsv(os.path.join(A_DIR, "hard_visual_subset", "ground_truth.csv"),
         ["doc_id", "field", "true_value"], hard_rows)

    # splits: stratified by tier, validated against what actually exists on disk
    def tier_of(i):
        if i in control_idx: return "clean_control"
        if i in double_idx: return "corrupted_double"
        if i in single_idx: return "corrupted_single"
        return "clean"

    by_tier = {"clean_control": [], "corrupted_double": [], "corrupted_single": [], "clean": []}
    for i in range(n_clean):
        by_tier[tier_of(i)].append(i)

    train, val, test = [], [], []
    for tier, ids in by_tier.items():
        random.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(0.6 * n), int(0.2 * n)
        train += ids[:n_tr]; val += ids[n_tr:n_tr + n_va]; test += ids[n_tr + n_va:]

    existing = set()
    for sub in ["clean", "corrupted_single", "corrupted_double", "clean_control"]:
        for fn in os.listdir(os.path.join(A_DIR, sub)):
            if fn.endswith(".png"):
                existing.add(fn[:-4])

    def write_split(name, ids):
        ids = [f"doc_{i:04d}" for i in ids]
        ids = [d for d in ids if d in existing]  # never write orphan references
        with open(os.path.join(A_DIR, "splits", name), "w") as f:
            f.write("\n".join(sorted(set(ids))))

    write_split("train.txt", train)
    write_split("val.txt", val)
    write_split("test.txt", test)

    print(f"Dataset A: clean={n_clean} single={len(single_rows)} double={len(double_rows)} "
          f"clean_control={len(control_rows)} hard_visual={len(hard_rows)} (all labeled)")

# --------------------------------------------------------------------- main B
def build_dataset_B(pool, n_per_subset=60):
    for sub in ["friedewald", "direct_ldl", "unlabeled_ambiguous", "splits"]:
        os.makedirs(os.path.join(B_DIR, sub), exist_ok=True)

    def make_subset(name, assay_method, sample, perturb=False, show_label=True):
        rows = []
        for i, tup in enumerate(sample):
            doc_id = f"doc_{i:04d}"
            template_id = i % 3
            unit = random.choice(["mg/dL", "mmol/L"])
            tc0, hdl0, ldl0, tg0 = tup["TC"], tup["HDL"], tup["LDL"], tup["TG"]
            if perturb:
                pct = random.uniform(0.10, 0.20) * random.choice([-1, 1])
                ldl0 = round(ldl0 * (1 + pct), 1)
            tc, hdl, ldl, tg = to_display(tc0, hdl0, ldl0, tg0, unit)
            note = None
            if show_label:
                if assay_method == "friedewald":
                    note = "LDL Cholesterol (Calculated, Friedewald)"
                elif assay_method == "direct":
                    note = "LDL Cholesterol (Direct Measurement)"
            img, _ = render_template(template_id, tc, hdl, ldl, tg, unit, assay_note=note)
            img.save(os.path.join(B_DIR, name, f"{doc_id}.png"))
            rows.append([doc_id, tc, hdl, ldl, tg, unit, template_id, assay_method])
        with open(os.path.join(B_DIR, name, "ground_truth.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "TC", "HDL", "LDL", "TG", "unit", "template_id", "assay_method"])
            w.writerows(rows)
        return rows

    fw_sample = random.sample(pool, n_per_subset)
    dl_sample = random.sample(pool, n_per_subset)
    amb_sample = random.sample(pool, n_per_subset)

    make_subset("friedewald", "friedewald", fw_sample, perturb=False, show_label=True)
    make_subset("direct_ldl", "direct", dl_sample, perturb=True, show_label=True)
    make_subset("unlabeled_ambiguous", "unknown", amb_sample, perturb=False, show_label=False)

    # splits: deduplicated, path-prefixed, 80/20 val/test, no repeats
    all_paths = []
    for sub in ["friedewald", "direct_ldl", "unlabeled_ambiguous"]:
        for i in range(n_per_subset):
            all_paths.append(f"{sub}/doc_{i:04d}")
    random.shuffle(all_paths)
    n_val = int(0.8 * len(all_paths))
    val_paths, test_paths = all_paths[:n_val], all_paths[n_val:]

    with open(os.path.join(B_DIR, "splits", "val.txt"), "w") as f:
        f.write("\n".join(sorted(set(val_paths))))
    with open(os.path.join(B_DIR, "splits", "test.txt"), "w") as f:
        f.write("\n".join(sorted(set(test_paths))))

    print(f"Dataset B: friedewald={n_per_subset} direct_ldl={n_per_subset} "
          f"unlabeled_ambiguous={n_per_subset}, splits deduplicated "
          f"(val={len(set(val_paths))}, test={len(set(test_paths))})")

# ------------------------------------------------------------------------ run
if __name__ == "__main__":
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT, exist_ok=True)

    pool = build_value_pool(n=260)
    write_nhanes_notes()
    build_dataset_A(pool, n_clean=200)
    build_dataset_B(pool, n_per_subset=60)

    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write(
            "# Regenerated Dataset A/B\n\n"
            "Generated by generate_dataset.py. Fixes applied vs. previous delivery:\n"
            "1. ground_truth.csv values now match the unit actually rendered on the image.\n"
            "2. clean_control subset (~15% of Dataset A) included.\n"
            "3. hard_visual_subset: every image has a ground-truth row.\n"
            "4. Dataset B splits deduplicated.\n"
            "5. Dataset A splits contain no orphan doc_id references.\n"
            "6. 3 distinct templates (was 2).\n\n"
            "Value pool is SYNTHETIC (no NHANES network access in this sandbox) — "
            "see nhanes_notes.md for how to swap in real NHANES data.\n"
        )
    print("\nDone. Output at:", ROOT)

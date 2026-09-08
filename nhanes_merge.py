"""
NHANES multi-panel merge script.

Downloads/reads .XPT files for:
  - Lipid panel:   TCHOL_*, HDL_*, TRIGLY_*   (reference-method values)
  - Biochemistry:  P_BIOPRO (electrolytes + calcium/albumin)

Merges everything on SEQN, computes two extra ground-truth invariant
columns (anion gap, corrected calcium), and writes one combined CSV
per cycle plus a single combined file across cycles.

Usage:
    1. Download the .XPT files you need from wwwn.cdc.gov/nchs/nhanes
       and put them in ./nhanes_raw/  (see FILES dict below for names).
    2. Run: python nhanes_merge.py
    3. Output: nhanes_combined.csv, friedewald_subset.csv,
       direct_ldl_subset.csv (empty stub - fill in perturbation logic)

Only standard library + pandas required.
"""

import os
import pandas as pd

RAW_DIR = "nhanes_raw"
OUT_DIR = "."

# cycle_label -> dict of component -> filename (without .XPT extension)
# Fill in / rename to match whatever you actually downloaded.
FILES = {
    "2017-2018": {
        "TCHOL": "TCHOL_J",
        "HDL": "HDL_J",
        "TRIGLY": "TRIGLY_J",
        "BIOPRO": None,  # 2017-2018 alone has its own BIOPRO_J if needed
    },
    "2017-March2020": {
        "TCHOL": "P_TCHOL",
        "HDL": "P_HDL",
        "TRIGLY": "P_TRIGLY",
        "BIOPRO": "P_BIOPRO",
    },
    "2021-2023": {
        "TCHOL": "TCHOL_L",
        "HDL": "HDL_L",
        "TRIGLY": "TRIGLY_L",
        "BIOPRO": "BIOPRO_L",
    },
}

# Column name -> standardized name, per component.
# NHANES reference-method lipid variable names (not the BIOPRO duplicates).
LIPID_COLS = {
    "TCHOL": {"LBXTC": "TC"},
    "HDL": {"LBDHDD": "HDL"},
    "TRIGLY": {"LBXTR": "TG", "LBDLDL": "LDL"},
}

BIOPRO_COLS = {
    "LBXSNASI": "sodium",
    "LBXSCLSI": "chloride",
    "LBXSC3SI": "bicarbonate",
    "LBXSCA": "calcium_total",
    "LBXSAL": "albumin",
}


def read_xpt(path):
    if not os.path.exists(path):
        print(f"  [skip] missing file: {path}")
        return None
    return pd.read_sas(path, format="xport")


def load_component(cycle_dir, filename, rename_map):
    if filename is None:
        return None
    path = os.path.join(cycle_dir, f"{filename}.XPT")
    df = read_xpt(path)
    if df is None:
        return None
    keep = ["SEQN"] + [c for c in rename_map if c in df.columns]
    df = df[keep].rename(columns=rename_map)
    return df


def merge_cycle(cycle_label, file_map):
    print(f"\nCycle: {cycle_label}")
    merged = None

    # lipid panel: TCHOL + HDL + TRIGLY (reference-method columns only)
    for component in ("TCHOL", "HDL", "TRIGLY"):
        fname = file_map.get(component)
        df = load_component(RAW_DIR, fname, LIPID_COLS[component])
        if df is None:
            continue
        merged = df if merged is None else merged.merge(df, on="SEQN", how="outer")

    # biochemistry profile: electrolytes + calcium/albumin only
    # (NOT total cholesterol/triglycerides from BIOPRO - NHANES docs say
    # those standard-battery values should not be used; reference-method
    # files above are the correct source for lipids)
    bio_fname = file_map.get("BIOPRO")
    bio_df = load_component(RAW_DIR, bio_fname, BIOPRO_COLS)
    if bio_df is not None:
        merged = bio_df if merged is None else merged.merge(bio_df, on="SEQN", how="outer")

    if merged is None:
        print("  no files found for this cycle, skipping")
        return None

    merged["cycle"] = cycle_label
    return merged


def add_invariants(df):
    # Anion gap = Na - (Cl + HCO3)
    if {"sodium", "chloride", "bicarbonate"}.issubset(df.columns):
        df["anion_gap"] = df["sodium"] - (df["chloride"] + df["bicarbonate"])

    # Corrected calcium = measured Ca + 0.8 * (4 - albumin[g/dL])
    if {"calcium_total", "albumin"}.issubset(df.columns):
        df["calcium_corrected"] = df["calcium_total"] + 0.8 * (4 - df["albumin"])

    return df


def clean(df):
    lipid_fields = [c for c in ("TC", "HDL", "LDL", "TG") if c in df.columns]
    # drop rows missing all lipid fields AND all biochem fields (keep partial coverage otherwise)
    bio_fields = [c for c in ("sodium", "chloride", "bicarbonate", "calcium_total", "albumin") if c in df.columns]
    if lipid_fields:
        df = df[~df[lipid_fields].isna().all(axis=1) | ~df[bio_fields].isna().all(axis=1)] if bio_fields else df.dropna(subset=lipid_fields)
    # drop obviously non-physiological negatives across any numeric field
    numeric_cols = [c for c in df.columns if c not in ("SEQN", "cycle") and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        df = df[(df[col].isna()) | (df[col] > 0)]
    return df


def main():
    all_cycles = []
    for cycle_label, file_map in FILES.items():
        merged = merge_cycle(cycle_label, file_map)
        if merged is not None:
            merged = add_invariants(merged)
            merged = clean(merged)
            all_cycles.append(merged)

    if not all_cycles:
        print("\nNo data loaded. Put .XPT files in ./nhanes_raw/ and check FILES dict.")
        return

    combined = pd.concat(all_cycles, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "nhanes_combined.csv"), index=False)
    print(f"\nWrote nhanes_combined.csv: {len(combined)} rows, columns: {list(combined.columns)}")

    # Friedewald applicability subset: TG <= 400, has all four lipid fields
    lipid_fields = ["TC", "HDL", "LDL", "TG"]
    if all(c in combined.columns for c in lipid_fields):
        friedewald = combined.dropna(subset=lipid_fields)
        friedewald = friedewald[friedewald["TG"] <= 400]
        friedewald.to_csv(os.path.join(OUT_DIR, "friedewald_subset.csv"), index=False)
        print(f"Wrote friedewald_subset.csv: {len(friedewald)} rows")

        # Direct LDL subset stub: perturb LDL +/-10-20% to simulate
        # direct-measured values that legitimately don't satisfy Friedewald.
        # Fill in / tune this before using for real - this is a starting point.
        import numpy as np
        rng = np.random.default_rng(42)
        direct = friedewald.copy()
        pct = rng.uniform(0.10, 0.20, size=len(direct)) * rng.choice([-1, 1], size=len(direct))
        direct["LDL_direct"] = direct["LDL"] * (1 + pct)
        direct["perturbation_pct"] = pct
        direct.to_csv(os.path.join(OUT_DIR, "direct_ldl_subset.csv"), index=False)
        print(f"Wrote direct_ldl_subset.csv: {len(direct)} rows (LDL perturbed +/-10-20%)")


if __name__ == "__main__":
    main()

"""
Step 1: Data preparation

Reads the corrected Python pipeline output (pipeline_results_v7.csv) --
NOT the original ResultsDataSheet.xlsx -- so that downstream modelling
uses the validated VAF/DMCI values (P33 fixed to its own file, P11
walkingincline excluded, dropout-flagged trials visible via `quality_*`
columns) rather than the uncorrected MATLAB-era numbers.

Cleans it, engineers ratio/mean/range features, and saves a clean CSV
for downstream modelling.
"""

import pandas as pd
import numpy as np

RAW_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\pipeline_outputs\pipeline_results_v7.csv"
OUT_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\participants_clean.csv"

EXERCISES = ["walking", "walkingincline", "stepforward", "steplateral"]

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
df.columns = [str(c).strip() for c in df.columns]

# pipeline_results_v7.csv already uses full exercise names in its column
# names (vaf_walking, dmci_walkingincline, ...) and already has clean
# "participant", "age", "group" columns -- no MATLAB-era renaming needed,
# unlike the old ResultsDataSheet.xlsx-based version of this script.
vaf_cols  = [f"vaf_{ex}"  for ex in EXERCISES]
dmci_cols = [f"dmci_{ex}" for ex in EXERCISES]
quality_cols = [f"quality_{ex}" for ex in EXERCISES]

core_cols = ["participant", "age", "group"] + vaf_cols + dmci_cols + quality_cols
missing_core = [c for c in core_cols if c not in df.columns]
if missing_core:
    raise ValueError(f"Expected columns missing from {RAW_PATH}: {missing_core}")
df = df[core_cols].copy()

# ── 2. Keep only rows with a valid age ───────────────────────────────────────
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df["group"] = pd.to_numeric(df["group"], errors="coerce")
df = df[df["age"].notna()].copy()

print(f"Participants loaded: {len(df)}")
print(f"  Young (group=1): {(df['group']==1).sum()}")
print(f"  Older (group=2): {(df['group']==2).sum()}")
print(f"  Age range: {df['age'].min():.0f} - {df['age'].max():.0f} yrs")

# ── 3. Feature engineering ───────────────────────────────────────────────────
# Ratio features (step-forward / walking) -- capture task difficulty contrast
df["vaf_ratio_sfwd_walk"]  = df["vaf_stepforward"]  / df["vaf_walking"]
df["dmci_ratio_sfwd_walk"] = df["dmci_stepforward"] / df["dmci_walking"]
df["vaf_ratio_sfwd_slat"]  = df["vaf_stepforward"]  / df["vaf_steplateral"]
df["dmci_ratio_sfwd_slat"] = df["dmci_stepforward"] / df["dmci_steplateral"]

# Mean and range of VAF/DMCI across all exercises (where available)
df["vaf_mean_all"]  = df[vaf_cols].mean(axis=1)
df["dmci_mean_all"] = df[dmci_cols].mean(axis=1)
df["vaf_range"]  = df[vaf_cols].max(axis=1)  - df[vaf_cols].min(axis=1)
df["dmci_range"] = df[dmci_cols].max(axis=1) - df[dmci_cols].min(axis=1)

# ── 4. Report missingness ────────────────────────────────────────────────────
print("\nMissing values per feature:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nQuality flags (non-'ok' trials):")
for qc in quality_cols:
    bad = df[df[qc].notna() & (df[qc] != "ok")]
    if len(bad):
        print(f"  {qc}: {bad[['participant', qc]].to_dict('records')}")

# ── 5. Save ──────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")

# ── 6. Confirm load: shape + first 5 rows ────────────────────────────────────
print(f"\nShape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head(5).to_string())

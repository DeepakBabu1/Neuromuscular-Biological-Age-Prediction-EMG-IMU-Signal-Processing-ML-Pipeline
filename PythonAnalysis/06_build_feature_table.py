"""
Step 4: Build the full feature table.

Combines, per participant:
  - VAF / DMCI (existing pipeline output)
  - CCI knee/ankle (Step 2, Falconer & Winter)
  - IMU kinematic features (Step 3)
  - Cross-task ratio features (step-forward vs walking)
  - age (target) and group (reference only)

Reads pipeline_results_v7.csv directly (run_pipeline.py now writes VAF,
DMCI, CCI, and kinematic columns together in one pass) and saves
feature_table_full.csv for feature screening / modelling.
"""

import pandas as pd
import numpy as np

IN_PATH  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\pipeline_outputs\pipeline_results_v7.csv"
OUT_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"

df = pd.read_csv(IN_PATH)
df.columns = [str(c).strip() for c in df.columns]

# ── Column selection, exactly as specified ───────────────────────────────────
VAF_COLS = ["vaf_walking", "vaf_walkingincline", "vaf_stepforward", "vaf_steplateral"]
DMCI_COLS = ["dmci_walking", "dmci_walkingincline", "dmci_stepforward", "dmci_steplateral"]

CCI_COLS = [
    "cci_knee_walking", "cci_knee_walkingincline", "cci_knee_stepforward", "cci_knee_steplateral",
    "cci_ankle_walking", "cci_ankle_walkingincline", "cci_ankle_stepforward", "cci_ankle_steplateral",
]

# Walking / walkingincline: 5 kinematic features each (includes stride_timing_cv,
# meaningful only for a repetitive gait task).
KINEMATIC_COLS_GAIT = []
for ex in ["walking", "walkingincline"]:
    KINEMATIC_COLS_GAIT += [
        f"peak_roll_{ex}", f"rom_roll_{ex}", f"peak_pitch_{ex}", f"rom_pitch_{ex}",
        f"stride_timing_cv_{ex}",
    ]

# Step-forward / step-lateral: 4 kinematic features each (no stride_timing_cv --
# these are single-rep tasks, not a cyclic gait, so inter-stride timing
# variability isn't a meaningful descriptor).
KINEMATIC_COLS_STEP = []
for ex in ["stepforward", "steplateral"]:
    KINEMATIC_COLS_STEP += [
        f"peak_roll_{ex}", f"rom_roll_{ex}", f"peak_pitch_{ex}", f"rom_pitch_{ex}",
    ]

KINEMATIC_COLS = KINEMATIC_COLS_GAIT + KINEMATIC_COLS_STEP

base_cols = ["participant", "age", "group"] + VAF_COLS + DMCI_COLS + CCI_COLS + KINEMATIC_COLS
missing = [c for c in base_cols if c not in df.columns]
if missing:
    raise ValueError(f"Expected columns missing from {IN_PATH}: {missing}")

feat = df[base_cols].copy()

# ── Cross-task ratios ─────────────────────────────────────────────────────────
feat["vaf_ratio_stepforward_walk"] = feat["vaf_stepforward"] / feat["vaf_walking"]
feat["cci_knee_ratio_stepforward_walk"] = feat["cci_knee_stepforward"] / feat["cci_knee_walking"]
feat["rom_roll_ratio_stepforward_walk"] = feat["rom_roll_stepforward"] / feat["rom_roll_walking"]

# ── Save ──────────────────────────────────────────────────────────────────────
feat.to_csv(OUT_PATH, index=False)

print(f"Saved -> {OUT_PATH}")
print(f"\nShape: {feat.shape}")
print(f"\nColumns ({len(feat.columns)}):")
for c in feat.columns:
    print(f"  {c}")

"""
Main pipeline runner (mainfile.m equivalent).

Runs the full EMG + IMU analysis pipeline on all valid participants:
  1. Discover data files (create_struct.m)
  2. For each participant-exercise:
       - Load EMG and Xsens
       - Process EMG (filter + normalize + stride segment if walking)
       - Compute 1-synergy NMF VAF (PosturalData_NMFvsPCA GUI)
  3. Compute DMCI for each participant-exercise
     (using young participants, age <= 40, as the control group)
  4. Save results to pipeline_results.csv

Output columns match the structure of ResultsDataSheet.xlsx:
  participant, vaf_walk, vaf_walk_inc, vaf_step_fwd, vaf_step_lat,
  dmci_walk, dmci_walk_inc, dmci_step_fwd, dmci_step_lat

Run time: approximately 2-5 minutes per participant (NMF with 20 replicates).
Total: ~60-150 minutes for all 31 participants.
For a quick test, set N_REPLICATES = 3 at the top.
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# Add parent directory so we can import emg_pipeline
sys.path.insert(0, os.path.dirname(__file__))

from emg_pipeline import (
    discover_participant_files,
    process_walking,
    process_step,
    compute_nmf_vaf,
    compute_dmci,
    compute_cci,
    compute_kinematic_features,
    load_emg,
    load_xsens,
    detect_dropout,
    EXCLUDED_TRIALS,
    KNEE_AGONIST,
    KNEE_ANTAGONIST,
    ANKLE_AGONIST,
    ANKLE_ANTAGONIST,
)

# Extra per-exercise columns beyond vaf/dmci/quality, always present in the
# output row (NaN-filled) even for missing/excluded/errored trials, so the
# CSV has a consistent column set across all participants.
EXTRA_COLS = [
    "cci_knee", "cci_ankle",
    "peak_roll", "rom_roll", "peak_pitch", "rom_pitch",
    "stride_count", "stride_timing_cv",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\DATACOLLECTION"
PART_INFO = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\DATACOLLECTION\ParticipantInfo.xlsx"
# Fallback source for participants missing age/group in ParticipantInfo.xlsx
# (32, 33, 35, 36). Cross-validated: every age present in both files matches
# exactly, so this is a trustworthy supplement, not a conflicting source.
PART_INFO_FALLBACK = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\participants_clean.csv"
OUT_DIR   = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\pipeline_outputs"

N_REPLICATES = 50   # NMF random restarts (50 = full analysis matching paper quality)
N_SYNERGIES  = 1    # Always 1 for DMCI

# Participants explicitly discarded in MATLAB create_struct.m
DISCARDED_PARTS = {1, 23, 28, 34}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load participant info (age and group)
# ---------------------------------------------------------------------------

def load_participant_info(xlsx_path: str) -> pd.DataFrame:
    """Load age and group info from ParticipantInfo.xlsx."""
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        # Try to find participant number and age columns
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "participant" in cl or col.lower() in ("part", "id", "subject"):
                col_map.setdefault("part_num", col)
            elif "age" in cl:
                col_map.setdefault("age", col)
            # "YA = 1 OA=2" column: young adult=1, older adult=2
            elif "group" in cl or ("ya" in cl and "oa" in cl):
                col_map.setdefault("group", col)

        if "part_num" in col_map and "age" in col_map:
            out = df[[col_map["part_num"], col_map["age"]]].copy()
            out.columns = ["part_num", "age"]
            if "group" in col_map:
                out["group"] = df[col_map["group"]].values
            # Extract numeric participant number
            out["part_num"] = out["part_num"].astype(str).str.extract(r"(\d+)").astype(float)
            out = out.dropna(subset=["part_num"])
            out["part_num"] = out["part_num"].astype(int)
            out = out.set_index("part_num")

            # Fill in participants missing age/group (32, 33, 35, 36) from the
            # fallback source, where available.
            try:
                fb = pd.read_csv(PART_INFO_FALLBACK)
                fb["part_num"] = fb["participant"].astype(str).str.extract(r"(\d+)").astype(int)
                fb = fb.set_index("part_num")[["age", "group"]]
                for col in ("age", "group"):
                    if col not in out.columns:
                        out[col] = np.nan
                    missing = out[col].isna()
                    out.loc[missing, col] = fb.reindex(out.index)[col][missing]
            except Exception as e:
                print(f"  Warning: could not load fallback participant info: {e}")

            out = out.dropna(subset=["age"])
            return out
    except Exception as e:
        print(f"  Warning: could not load ParticipantInfo.xlsx: {e}")
    return pd.DataFrame(columns=["age"])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_all(test_mode: bool = False):
    """
    test_mode=True: process only participant 2, all 4 exercises, to verify outputs.
    test_mode=False: process all valid participants.
    """
    print("=" * 60)
    print("EMG Pipeline: MATLAB -> Python conversion")
    print("=" * 60)

    # Step 1: discover files
    print("\n[1/4] Scanning data directory...")
    registry = discover_participant_files(DATA_ROOT)
    all_parts = sorted(registry.keys())
    # Remove explicitly discarded participants
    valid_parts = [p for p in all_parts if p not in DISCARDED_PARTS]
    print(f"      Found {len(all_parts)} participant folders, "
          f"{len(valid_parts)} valid (after removing discarded).")

    if test_mode:
        valid_parts = [p for p in valid_parts if p == 2]
        print(f"      TEST MODE: processing participant(s): {valid_parts}")

    # Step 2: load participant ages
    print("\n[2/4] Loading participant info...")
    part_info = load_participant_info(PART_INFO)
    print(f"      Loaded info for {len(part_info)} participants.")

    # Step 3: compute VAF for each participant-exercise
    print("\n[3/4] Computing NMF VAF (1 synergy) for each participant-exercise...")
    print(f"      NMF replicates: {N_REPLICATES}")

    exercise_keys = ["walking", "walkingincline", "stepforward", "steplateral"]
    results = []

    for part_num in valid_parts:
        part_data = registry[part_num]
        row = {"participant": part_num}

        # Get age from ParticipantInfo if available
        if part_num in part_info.index:
            row["age"] = part_info.loc[part_num, "age"]
            if "group" in part_info.columns:
                row["group"] = part_info.loc[part_num, "group"]
        else:
            row["age"] = np.nan
            row["group"] = np.nan

        t0 = time.time()
        print(f"\n  Participant {part_num:3d}  (folder: {part_data['folder']})")

        for ex_key in exercise_keys:
            if ex_key not in part_data:
                if (part_num, ex_key) in EXCLUDED_TRIALS:
                    print(f"    {ex_key:20s}: EXCLUDED, no reliable data source")
                    row[f"quality_{ex_key}"] = "excluded_no_reliable_data"
                else:
                    print(f"    {ex_key:20s}: MISSING files, skipping")
                row[f"vaf_{ex_key}"] = np.nan
                for col in EXTRA_COLS:
                    row[f"{col}_{ex_key}"] = np.nan
                continue

            ex_files = part_data[ex_key]
            try:
                # Check for sensor dropout in the raw EMG (the same file
                # process_walking/process_step will load) -- flatlined
                # channels that can't be recovered, so we mark the trial
                # unreliable rather than try to fix it.
                raw_emg = load_emg(ex_files["emg"])
                dropout = detect_dropout(raw_emg)
                if dropout["n_severe"] > 0:
                    row[f"quality_{ex_key}"] = f"severe_dropout({dropout['n_severe']}_muscles)"
                else:
                    row[f"quality_{ex_key}"] = "ok"

                if ex_key in ("walking", "walkingincline"):
                    emg_processed, muscle_cols = process_walking(
                        ex_files["emg"], ex_files["xsens"], return_names=True)
                else:
                    emg_processed, muscle_cols = process_step(
                        ex_files["emg"], return_names=True)

                vaf = compute_nmf_vaf(emg_processed, n_synergies=N_SYNERGIES,
                                      n_replicates=N_REPLICATES)
                row[f"vaf_{ex_key}"] = round(vaf, 4)

                # Co-contraction indices (Falconer & Winter, 1985) on the same
                # filtered/normalized envelope used for VAF. NaN if a required
                # muscle is missing or flagged as severe dropout above.
                cci_knee = compute_cci(emg_processed, muscle_cols,
                                       KNEE_AGONIST, KNEE_ANTAGONIST,
                                       severe_dropout_idx=dropout["severe_muscles"])
                cci_ankle = compute_cci(emg_processed, muscle_cols,
                                        ANKLE_AGONIST, ANKLE_ANTAGONIST,
                                        severe_dropout_idx=dropout["severe_muscles"])
                row[f"cci_knee_{ex_key}"] = round(cci_knee, 4) if not np.isnan(cci_knee) else np.nan
                row[f"cci_ankle_{ex_key}"] = round(cci_ankle, 4) if not np.isnan(cci_ankle) else np.nan

                # IMU kinematic features from the foot sensor (Roll/Pitch),
                # native 100 Hz signal -- independent of the EMG segmentation
                # above, computed for every exercise (registry guarantees the
                # xsens file exists whenever the exercise entry does).
                roll_100hz, pitch_100hz = load_xsens(ex_files["xsens"], return_pitch=True)
                kin = compute_kinematic_features(roll_100hz, pitch_100hz)
                for k, v in kin.items():
                    row[f"{k}_{ex_key}"] = v

                flag = "  *** SEVERE DROPOUT, UNRELIABLE ***" if dropout["n_severe"] > 0 else ""
                print(f"    {ex_key:20s}: VAF = {vaf:.2f}%  CCI_knee = {cci_knee:.3f}  "
                      f"CCI_ankle = {cci_ankle:.3f}  "
                      f"[{emg_processed.shape[0]} muscles x {emg_processed.shape[1]} samples]{flag}")

            except Exception as e:
                print(f"    {ex_key:20s}: ERROR - {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()
                row[f"vaf_{ex_key}"] = np.nan
                row[f"quality_{ex_key}"] = "error"
                for col in EXTRA_COLS:
                    row[f"{col}_{ex_key}"] = np.nan

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        results.append(row)

    # Step 4: compute DMCI using young group as control
    print("\n[4/4] Computing DMCI...")

    df = pd.DataFrame(results)
    df = df.set_index("participant")

    # Identify young participants (group=1 or age <= 40)
    if "group" in df.columns and df["group"].notna().any():
        young_mask = df["group"] == 1
    elif "age" in df.columns:
        young_mask = df["age"] <= 40
    else:
        print("  WARNING: cannot identify young group; using all participants as control.")
        young_mask = pd.Series(True, index=df.index)

    young_parts = df.index[young_mask].tolist()
    print(f"  Young control group: {len(young_parts)} participants: {young_parts}")

    for ex_key in exercise_keys:
        vaf_col = f"vaf_{ex_key}"
        dmci_col = f"dmci_{ex_key}"
        quality_col = f"quality_{ex_key}"

        if vaf_col not in df.columns:
            df[dmci_col] = np.nan
            continue

        # Exclude trials with severe sensor dropout from the young-group
        # baseline -- their VAF is not a trustworthy measure of synergy
        # quality, and including them would bias the DMCI reference for
        # every other participant in this exercise.
        if quality_col in df.columns:
            reliable_mask = df[quality_col].apply(lambda q: not str(q).startswith("severe_dropout"))
        else:
            reliable_mask = pd.Series(True, index=df.index)

        baseline_mask = young_mask & reliable_mask
        excluded = df.index[young_mask & ~reliable_mask].tolist()
        if excluded:
            print(f"  {ex_key}: excluding {excluded} from young-group DMCI baseline "
                  "(severe sensor dropout)")

        young_vafs = df.loc[baseline_mask, vaf_col].dropna().values
        if len(young_vafs) < 2:
            print(f"  WARNING: fewer than 2 reliable young participants with VAF for {ex_key}; "
                  "DMCI will be NaN.")
            df[dmci_col] = np.nan
            continue

        dmci_vals = [
            compute_dmci(vaf, young_vafs) if not np.isnan(vaf) else np.nan
            for vaf in df[vaf_col]
        ]
        df[dmci_col] = dmci_vals

    # Reorder columns: base ResultsDataSheet.xlsx layout, then the new CCI
    # and kinematic columns per exercise.
    ordered_cols = ["age"]
    if "group" in df.columns:
        ordered_cols.append("group")
    for ex in exercise_keys:
        ordered_cols += [f"vaf_{ex}", f"dmci_{ex}", f"quality_{ex}"]
        ordered_cols += [f"{col}_{ex}" for col in EXTRA_COLS]
    df = df[[c for c in ordered_cols if c in df.columns]]
    df = df.reset_index()

    # Save
    out_csv = os.path.join(OUT_DIR, "pipeline_results_v7.csv")
    try:
        df.to_csv(out_csv, index=False)
        print(f"\nSaved results to: {out_csv}")
    except PermissionError:
        fallback_csv = os.path.join(OUT_DIR, f"pipeline_results_v7_{int(time.time())}.csv")
        df.to_csv(fallback_csv, index=False)
        print(f"\n{out_csv} was locked (open elsewhere?); saved to: {fallback_csv}")
    print("\nResults summary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return df


# ---------------------------------------------------------------------------
# Quick single-participant test
# ---------------------------------------------------------------------------

def test_participant(part_num: int = 3):
    """Quick test on one participant to validate the pipeline."""
    print(f"\n=== Quick test: participant {part_num} ===")

    registry = discover_participant_files(DATA_ROOT)
    if part_num not in registry:
        print(f"Participant {part_num} not found in registry.")
        return

    part_data = registry[part_num]
    print(f"Folder: {part_data['folder']}")

    for ex_key in ["walking", "walkingincline", "stepforward", "steplateral"]:
        if ex_key not in part_data:
            print(f"  {ex_key}: missing")
            continue

        files = part_data[ex_key]
        print(f"\n  Exercise: {ex_key}")
        print(f"  EMG:   {os.path.basename(files['emg'])}")
        print(f"  Xsens: {os.path.basename(files['xsens'])}")

        try:
            if ex_key in ("walking", "walkingincline"):
                emg = process_walking(files["emg"], files["xsens"])
            else:
                emg = process_step(files["emg"])

            vaf = compute_nmf_vaf(emg, n_synergies=1, n_replicates=5)
            print(f"  EMG shape: {emg.shape}  (muscles x samples)")
            print(f"  VAF (1 synergy, 5 replicates): {vaf:.2f}%")
            print(f"  EMG range: [{emg.min():.4f}, {emg.max():.4f}]")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_participant(3)
    elif "--full" in sys.argv:
        run_all(test_mode=False)
    else:
        # Default: run test on participant 3 first
        print("Usage:")
        print("  python run_pipeline.py --test    # quick test on participant 3")
        print("  python run_pipeline.py --full    # process all participants (~2 hrs)")
        print()
        print("Running quick test (participant 3)...")
        test_participant(3)

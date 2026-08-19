"""
Python conversion of MATLAB EMG + IMU analysis pipeline.

Converts:
  create_struct.m       -> discover_participant_files()
  resamplexsens.m       -> resample_xsens()
  process_data.m        -> process_walking_emg()
  process_stepdata.m    -> process_step_emg()
  PosturalData_NMFvsPCA -> compute_nmf_vaf()   [replaces the MATLAB GUI]
  calc_dmci.m           -> compute_dmci()

EMG signal processing pipeline (matches MATLAB process_data.m / process_stepdata.m exactly):
  raw EMG
    -> low-pass Butterworth filter (40 Hz, 4th order, zero-phase)
       NOTE: MATLAB code calls butter(4, Fcut_high/(Fs/2)) WITHOUT the 'high' type argument,
       which defaults to a LOW-PASS filter. The comment in the MATLAB code says "HIGHPASS
       FILTERING" but the code actually creates a low-pass filter at 40 Hz.
    -> detrend across muscles at each time point (cross-muscle detrend)
       NOTE: MATLAB calls detrend() on a (13 x n_samples) matrix. MATLAB's detrend()
       operates on columns by default, so it detrends each time-point's 13-muscle vector
       (axis=0 in Python), NOT each muscle across time (axis=1).
    -> rectify (abs)
    -> low-pass Butterworth filter (4 Hz, 4th order, zero-phase)
    -> peak normalize per muscle

Stride segmentation (walking only, MATLAB process_data.m):
  Xsens Roll signal (100 Hz) resampled to 1000 Hz
  -> findpeaks on negated signal (height > 20 deg) to find foot-strikes
  -> extract middle 30 strides (15 before + 15 after median trough)
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample, detrend

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS_EMG = 1000      # BTS FREEEMG sampling rate (Hz)
FS_XSENS = 100     # Xsens IMU sampling rate (Hz)
FS_XSENS_UP = 1000 # Upsampled Xsens rate (Hz)

FCUT_HIGH = 40     # High-pass filter cut-off (Hz)
FCUT_LOW = 4       # Low-pass filter cut-off (Hz)
FILTER_ORDER = 4   # Butterworth filter order

N_STRIDES = 30     # Number of strides to extract (middle N_STRIDES, i.e. ±15)

# TFL column name (removed from analysis per MATLAB create_struct.m)
TFL_COL = "Right Tensor fasciae latae"

# Exercises and their file patterns
EXERCISES = {
    "walking":      {"emg": "walking",       "xsens": ["walking_foot", "walkingflat_foot"]},
    "walkingincline": {"emg": "walkingincline", "xsens": ["walkingincline_foot"]},
    "stepforward":  {"emg": "stepforward",   "xsens": ["stepforward_foot"]},
    "steplateral":  {"emg": "steplateral",   "xsens": ["steplateral_foot"]},
}

# ---------------------------------------------------------------------------
# Known data substitutions baked into the original MATLAB create_struct.m.
#
# Each one was investigated individually by loading the participant's own
# file directly and checking its quality, rather than assuming the MATLAB
# substitution was either "correct" or "a bug" by default:
#
#   - Participant 33 steplateral (line 1015, read part3steplateral.xlsx):
#     CORRECTED below. Participant 33's own EMG file is clean (no sensor
#     dropout), so the participant-3 substitution was a pure copy-paste
#     error with no reason to keep it.
#
#   - Participant 11 walkingincline (line 321, read part1walkingincline.xlsx):
#     NOT corrected to its own file -- instead excluded entirely, see
#     EXCLUDED_TRIALS below. Participant 11's own EMG has severe sensor
#     dropout in 7 of its 13 muscles (verified via detect_dropout()), so
#     using it would just replace "wrong person's data" with "unusable
#     data." Neither source is trustworthy.
#
#   - Participant 13 / 29 walkingincline xsens (lines 391, 870/878): KEPT.
#     Their own foot-IMU files were checked directly -- participant 13's
#     Roll signal swings only +-8 deg (need >20 deg to detect a footstep)
#     over a 13-second recording; participant 29's has zero peaks above
#     20 deg. Both are genuinely unusable for stride segmentation, so
#     MATLAB's cross-trial substitution is the only way to get a proper
#     30-stride segmented trial comparable to every other participant --
#     not a bug to fix.
# ---------------------------------------------------------------------------
FILE_OVERRIDES = {
    # line 391, comment "(walkingincline was corrupted)": MATLAB substitutes
    # the WALKING trial's xsens foot file for stride segmentation instead of
    # participant 13's own (corrupted) walkingincline xsens file. Verified:
    # participant 13's own file has Roll amplitude of only +-8 deg (below
    # the 20 deg footstep-detection threshold) over just 13 seconds.
    (13, "walkingincline", "xsens"): "part13_walking_foot.txt",
    # lines 870/878, comment "mixed up read in xsens data": walkingincline
    # and stepforward xsens files are swapped for participant 29. Verified:
    # participant 29's own walkingincline xsens has zero peaks above the
    # 20 deg footstep-detection threshold, so the swap is a necessary
    # workaround, not a mistake to revert.
    (29, "walkingincline", "xsens"): "part29_stepforward_foot.txt",
    (29, "stepforward", "xsens"): "part29_walkingincline_foot.txt",
}

# Trials excluded outright because NEITHER the MATLAB substitute NOR the
# participant's own recording is usable.
EXCLUDED_TRIALS = {
    # MATLAB substituted participant 1's EMG file here (wrong person).
    # Participant 11's own file exists but has severe sensor dropout in
    # 7 of its 13 muscles (verified via detect_dropout()) -- not
    # recoverable. No trustworthy data exists for this participant-exercise,
    # so it is dropped rather than reported from either bad source.
    (11, "walkingincline"),
}


def _find_file_anywhere(data_root: str, filename: str) -> str:
    """Search the entire data_root tree (all participant folders) for a file
    matching filename (case-insensitive). Used to resolve FILE_OVERRIDES,
    which reference files belonging to a different participant's folder."""
    fn_low = filename.lower()
    for root, _dirs, fns in os.walk(data_root):
        for fn in fns:
            if fn.startswith("~$"):
                continue
            if fn.lower() == fn_low:
                return os.path.join(root, fn)
    return None


# ---------------------------------------------------------------------------
# 1. Data discovery  (create_struct.m equivalent)
# ---------------------------------------------------------------------------

def discover_participant_files(data_root: str) -> dict:
    """
    Scan DATACOLLECTION folder and build a registry of valid participant files.

    Uses recursive search within each participant folder because subfolder names
    are inconsistent: 'btsexported', 'bts exported', 'bts exported' (typo), etc.

    Returns a dict keyed by participant number (int) with structure:
        {
          2: {
            'folder': 'participant2 (lb)',
            'walking':       {'emg': <path>, 'xsens': <path>},
            'walkingincline': {'emg': <path>, 'xsens': <path>},
            'stepforward':   {'emg': <path>, 'xsens': <path>},
            'steplateral':   {'emg': <path>, 'xsens': <path>},
          }, ...
        }

    Entries are only included when BOTH emg and xsens files exist for that exercise.
    """
    registry = {}

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if "participant" not in folder.lower():
            continue

        # Extract participant number from folder name
        m = re.search(r"participant(\d+)", folder, re.IGNORECASE)
        if not m:
            continue
        part_num = int(m.group(1))

        # Recursively collect all xlsx and foot-txt files in this participant folder
        all_xlsx = {}  # filename (lowercase) -> full path
        all_txts = {}  # filename (lowercase) -> full path

        for root, dirs, fns in os.walk(folder_path):
            # Skip temp/lock files created by Excel
            fns = [f for f in fns if not f.startswith("~$")]
            for fn in fns:
                fn_low = fn.lower()
                full = os.path.join(root, fn)
                if fn_low.endswith(".xlsx"):
                    all_xlsx[fn_low] = full
                elif fn_low.endswith(".txt") and "foot" in fn_low:
                    all_txts[fn_low] = full

        exercises_found = {}
        for ex_key, ex_info in EXERCISES.items():
            # Match EMG file: part{N}{exercise}.xlsx
            emg_fn = f"part{part_num}{ex_info['emg']}.xlsx"
            emg_path = all_xlsx.get(emg_fn.lower())
            if emg_path is None:
                continue

            # Match Xsens file: part{N}_{pattern}_foot.txt
            xsens_path = None
            for xs_pattern in ex_info["xsens"]:
                xs_fn = f"part{part_num}_{xs_pattern}.txt"
                xsens_path = all_txts.get(xs_fn.lower())
                if xsens_path is not None:
                    break
            if xsens_path is None:
                continue

            exercises_found[ex_key] = {"emg": emg_path, "xsens": xsens_path}

        # Drop trials with no trustworthy data source at all (see
        # EXCLUDED_TRIALS above) before they ever enter the registry.
        for ex_key in list(exercises_found.keys()):
            if (part_num, ex_key) in EXCLUDED_TRIALS:
                del exercises_found[ex_key]

        if exercises_found:
            registry[part_num] = {"folder": folder, **exercises_found}

    # Apply known MATLAB create_struct.m data substitutions (see FILE_OVERRIDES)
    for (part_num, ex_key, file_type), override_fn in FILE_OVERRIDES.items():
        if part_num not in registry or ex_key not in registry[part_num]:
            continue
        override_path = _find_file_anywhere(data_root, override_fn)
        if override_path is not None:
            registry[part_num][ex_key][file_type] = override_path
        else:
            warnings.warn(
                f"Override file '{override_fn}' for participant {part_num} "
                f"{ex_key} ({file_type}) not found; using the file discovered "
                "by name-matching instead, which will NOT match the paper.",
                RuntimeWarning,
            )

    return registry


# ---------------------------------------------------------------------------
# 2. Load raw data
# ---------------------------------------------------------------------------

def _longest_true_run(bool_arr: np.ndarray) -> int:
    """Length of the longest run of consecutive True values in a 1-D boolean array."""
    if not bool_arr.any():
        return 0
    diff = np.diff(bool_arr.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if bool_arr[0]:
        starts = np.r_[0, starts]
    if bool_arr[-1]:
        ends = np.r_[ends, len(bool_arr)]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def detect_dropout(emg: np.ndarray, zero_pct_thresh: float = 15.0, run_thresh: int = 1000) -> dict:
    """
    Detect wireless sensor dropout: a muscle channel held flat at exactly zero
    for an extended stretch (the BTS FREEEMG system holds the last value, or
    reports zero, during a connection loss -- this is distinct from normal
    EMG zero-crossings, which never hold for more than 1-2 samples).

    A channel is flagged 'severe' if either:
      - more than zero_pct_thresh% of samples are exactly zero, OR
      - the longest single held-flat run exceeds run_thresh samples (1000
        samples = 1 second at 1000 Hz).

    Returns dict with 'n_severe', 'severe_muscles' (0-indexed list), 'details'
    (per-muscle zero_pct / max_run).
    """
    n_muscles = emg.shape[0]
    details = []
    severe = []
    for i in range(n_muscles):
        row = emg[i]
        zero_mask = row == 0
        zero_pct = 100.0 * zero_mask.sum() / len(row)
        max_run = _longest_true_run(zero_mask)
        is_severe = zero_pct > zero_pct_thresh or max_run > run_thresh
        details.append({"muscle_idx": i, "zero_pct": zero_pct, "max_run": max_run, "severe": is_severe})
        if is_severe:
            severe.append(i)
    return {"n_severe": len(severe), "severe_muscles": severe, "details": details}


def load_emg(xlsx_path: str, drop_tfl: bool = True, return_names: bool = False):
    """
    Load BTS FREEEMG xlsx file.

    Returns: (n_muscles, n_samples) float64 array, or (emg, muscle_cols) if
    return_names=True -- muscle_cols is the list of column names in row order,
    needed by compute_cci() to look up specific muscles by name.

    The xlsx has a metadata header; row 10 (0-indexed) is the column name row.
    Muscles are columns 2-15 (Frame=0, Time=1, 14 muscles).
    TFL (Right Tensor fasciae latae) is dropped if drop_tfl=True,
    leaving 13 muscles to match the MATLAB pipeline (nmus=13).
    """
    df = pd.read_excel(xlsx_path, header=10, engine="openpyxl")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Drop Frame and Time columns
    muscle_cols = [c for c in df.columns if c not in ("Frame", "Time")]

    if drop_tfl and TFL_COL in muscle_cols:
        muscle_cols = [c for c in muscle_cols if c != TFL_COL]

    emg = df[muscle_cols].to_numpy(dtype=np.float64).T  # (n_muscles, n_samples)

    # Clamp extreme values from corrupted channels (e.g. participant 11 stepforward VL)
    # Legitimate surface EMG is always well within ±100 mV.
    corrupt_mask = np.abs(emg) > 100.0
    if corrupt_mask.any():
        n_bad = int(corrupt_mask.sum())
        warnings.warn(
            f"{os.path.basename(xlsx_path)}: {n_bad} samples with |EMG| > 100 mV "
            "detected (corrupted channel). Clamping to 0.",
            RuntimeWarning,
        )
        emg[corrupt_mask] = 0.0

    # Detect wireless sensor dropout (held-flat zero runs). This cannot be
    # repaired computationally -- there is no valid signal to recover -- so
    # we only flag it; callers (run_pipeline.py) use this to exclude affected
    # trials from DMCI's young-group baseline.
    dropout = detect_dropout(emg)
    if dropout["n_severe"] > 0:
        bad_names = [
            [c for c in muscle_cols][i] for i in dropout["severe_muscles"]
        ]
        warnings.warn(
            f"{os.path.basename(xlsx_path)}: severe sensor dropout detected in "
            f"{dropout['n_severe']} muscle(s) ({', '.join(bad_names)}) -- held-flat "
            "zero runs of >1s or >15% of the trial. This data cannot be recovered; "
            "VAF computed from it should be treated as unreliable.",
            RuntimeWarning,
        )

    if return_names:
        return emg, muscle_cols
    return emg


def load_xsens(txt_path: str, return_pitch: bool = False):
    """
    Load exported Xsens IMU txt file (tab-separated, // comment lines).

    Returns the Roll column as a 1-D float64 array at 100 Hz, or (roll, pitch)
    if return_pitch=True -- pitch is NaN-filled if the Pitch column is absent.

    MATLAB create_struct.m reads rawxsens(:,3) which corresponds to
    the Roll column (SampleTimeFine is NaN/empty, so Roll is the 3rd
    non-NaN column in 1-indexed MATLAB = index 2 in 0-indexed Python).
    """
    df = pd.read_csv(txt_path, sep="\t", comment="/", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]

    # Roll = column index 2 (MATLAB rawxsens(:,3))
    roll = df["Roll"].to_numpy(dtype=np.float64)
    if return_pitch:
        if "Pitch" in df.columns:
            pitch = df["Pitch"].to_numpy(dtype=np.float64)
        else:
            pitch = np.full_like(roll, np.nan)
        return roll, pitch
    return roll


# ---------------------------------------------------------------------------
# 3. Xsens resampling  (resamplexsens.m equivalent)
# ---------------------------------------------------------------------------

def resample_xsens(roll_100hz: np.ndarray) -> np.ndarray:
    """
    Upsample Xsens Roll from 100 Hz to 1000 Hz using polyphase resampling
    (matches MATLAB resample(pitch, 1000, 100)).
    """
    n_out = len(roll_100hz) * (FS_XSENS_UP // FS_XSENS)
    resampled = resample(roll_100hz, n_out)
    return resampled


# ---------------------------------------------------------------------------
# 4. EMG signal processing  (process_data.m / process_stepdata.m equivalent)
# ---------------------------------------------------------------------------

def _butter_filter(signal_2d: np.ndarray, cutoff: float, btype: str) -> np.ndarray:
    """
    Apply zero-phase Butterworth filter to each row (muscle) of a 2-D array.
    signal_2d: (n_muscles, n_samples)
    """
    b, a = butter(FILTER_ORDER, cutoff / (FS_EMG / 2), btype=btype)
    return filtfilt(b, a, signal_2d, axis=1)


def filter_emg(emg: np.ndarray) -> np.ndarray:
    """
    Full EMG processing pipeline matching MATLAB process_data.m / process_stepdata.m exactly.

      1. Low-pass filter   (40 Hz, 4th order Butterworth, zero-phase)
         [MATLAB bug: code says 'high-pass' but butter() without 'high' flag = low-pass]
      2. Cross-muscle detrend at each time point (axis=0)
         [MATLAB bug: detrend() on (13 x n_samples) detrends each column = each time point,
          NOT each muscle across time]
      3. Rectify (abs)
      4. Low-pass filter   ( 4 Hz, 4th order Butterworth, zero-phase)
      5. Peak-normalize each muscle to its maximum value

    emg: (n_muscles, n_samples) raw signal
    Returns: (n_muscles, n_samples) processed signal, normalized 0-1
    """
    # Step 1: low-pass at 40 Hz (matches MATLAB butter(4, Fcut_high/(Fs/2)) without 'high')
    filt = _butter_filter(emg, FCUT_HIGH, "low")

    # Step 2: cross-muscle detrend (MATLAB: detrend on (13 x n_samples) = column-wise = axis=0)
    filt = detrend(filt, axis=0)

    # Step 3: rectify
    filt = np.abs(filt)

    # Step 4: low-pass at 4 Hz
    filt = _butter_filter(filt, FCUT_LOW, "low")

    # Clip filter ringing artifacts
    filt = np.maximum(filt, 0.0)

    # Step 5: peak normalize per muscle
    peak_vals = filt.max(axis=1, keepdims=True)
    peak_vals[peak_vals == 0] = 1.0  # avoid div-by-zero
    filt = filt / peak_vals

    return filt


# ---------------------------------------------------------------------------
# 5. Stride segmentation  (process_data.m stride extraction)
# ---------------------------------------------------------------------------

def segment_walking_strides(
    emg: np.ndarray,
    xsens_1000hz: np.ndarray,
    n_strides: int = N_STRIDES,
) -> np.ndarray:
    """
    Extract the middle n_strides from a walking trial using Xsens foot IMU.

    Exact replication of MATLAB process_data.m (lines 11-53):
      1. Truncate resampledxsens to the EMG length, but ONLY if xsens is
         longer (MATLAB never trims the EMG/time array itself).
      2. findpeaks(-resampledxsens, 'MinPeakHeight', 20) -> trough (peak
         values), troughloc (peak locations). Fixed threshold.
      3. If an even number of troughs found, append a sentinel 0 to
         troughloc ONLY (not to trough) — MATLAB's trick to make
         median(troughloc) land exactly on an existing element.
      4. anchor = index in troughloc where troughloc == median(troughloc).
      5. first/last stride = troughloc[anchor -+ n_strides/2], indexed into
         the ORIGINAL (un-appended) peak array.

    Note: when the peak count is even, this anchor calculation is shifted by
    one relative to a naive "middle element" — that's an artifact of
    MATLAB's append-then-median trick, replicated here intentionally rather
    than smoothed over.

    Sensor-orientation fallback (deviation from literal MATLAB): for a
    handful of participants the foot IMU appears to have been mounted with
    reversed polarity, so the gait signal shows up as POSITIVE Roll
    excursions rather than negative ones — findpeaks(-resampledxsens, ...)
    then finds near-zero troughs. MATLAB's process_data.m hardcodes the
    negative sign with no fallback, so it would have failed identically.
    Since the published results exist for these participants anyway (likely
    via manual intervention in the original analysis that isn't visible in
    the saved script), we try the negative sign first (literal MATLAB), and
    only if that doesn't find enough peaks, retry with the positive sign.

    emg:          (n_muscles, n_samples) processed EMG
    xsens_1000hz: (n_samples,) Xsens Roll at 1000 Hz, aligned to EMG time axis
    n_strides:    target number of strides to extract (default 30)

    Returns: (n_muscles, n_stride_samples)
    """
    half = n_strides // 2  # 15 for default N_STRIDES=30
    min_peaks = n_strides + 1

    # MATLAB only trims resampledxsens, and only if it's longer than the
    # EMG/time array; it never trims EMG itself.
    n_emg = emg.shape[1]
    if len(xsens_1000hz) > n_emg:
        xsens_1000hz = xsens_1000hz[:n_emg]

    # findpeaks(-resampledxsens, 'MinPeakHeight', 20) — literal MATLAB sign first.
    peak_idxs, _ = find_peaks(-xsens_1000hz, height=20)
    sign_used = "negative (MATLAB literal)"

    if len(peak_idxs) < min_peaks:
        # Sensor-orientation fallback: try the opposite polarity.
        peak_idxs_pos, _ = find_peaks(xsens_1000hz, height=20)
        if len(peak_idxs_pos) >= min_peaks:
            warnings.warn(
                "Foot IMU Roll signal has too few negative troughs >= 20 deg "
                f"({len(peak_idxs)} found); using positive excursions instead "
                f"({len(peak_idxs_pos)} found). This deviates from MATLAB's "
                "literal code (which hardcodes negative) but is needed for "
                "participants with reversed sensor orientation.",
                RuntimeWarning,
            )
            peak_idxs = peak_idxs_pos
            sign_used = "positive (sensor-orientation fallback)"

    n_peaks = len(peak_idxs)
    MIN_PEAKS_GRACEFUL = 11  # at least +-5 strides

    if n_peaks < MIN_PEAKS_GRACEFUL:
        warnings.warn(
            f"Only {n_peaks} foot-strike troughs found at MinPeakHeight=20 deg "
            f"({sign_used} sign). Too few even for a reduced stride window; "
            "using the full trial instead. MATLAB's literal code would error "
            "here -- the published result for this trial likely involved "
            "manual review not captured in the saved script.",
            RuntimeWarning,
        )
        return emg

    # MATLAB: if mod(length(troughloc),2)==0, troughloc(end+1)=0
    troughloc = peak_idxs.astype(np.int64)
    if n_peaks % 2 == 0:
        troughloc = np.append(troughloc, 0)

    median_val = np.median(troughloc)
    anchor_pos = int(np.where(troughloc == median_val)[0][0])  # MATLAB find(): first match

    if n_peaks < min_peaks:
        # Graceful degradation (deviation from literal MATLAB, which has no
        # such fallback): use as many strides as available, still anchored
        # on the same median position so the window stays centered.
        effective_half = min(half, anchor_pos, n_peaks - 1 - anchor_pos)
        warnings.warn(
            f"Only {n_peaks} foot-strike troughs found at MinPeakHeight=20 deg "
            f"({sign_used} sign; need >= {min_peaks} for {n_strides} strides). "
            f"Using {effective_half * 2} strides instead of {n_strides}.",
            RuntimeWarning,
        )
    else:
        effective_half = half

    first_sample = peak_idxs[anchor_pos - effective_half]
    last_sample  = peak_idxs[anchor_pos + effective_half]

    return emg[:, first_sample:last_sample]


# ---------------------------------------------------------------------------
# 6. Complete per-exercise processing
# ---------------------------------------------------------------------------

def process_walking(emg_path: str, xsens_path: str, return_names: bool = False):
    """
    Load + process a walking trial (with stride segmentation).
    Returns: (n_muscles, n_stride_samples) normalized EMG, or (emg, muscle_cols)
    if return_names=True.

    Order matches MATLAB process_data.m:
      1. Segment raw EMG to middle 30 strides FIRST
      2. THEN apply filter pipeline + peak normalisation on the segment only
    Peak normalisation must use segment peaks, not full-trial peaks.
    """
    if return_names:
        emg_raw, muscle_cols = load_emg(emg_path, return_names=True)
    else:
        emg_raw = load_emg(emg_path)
    roll_100hz = load_xsens(xsens_path)
    roll_1000hz = resample_xsens(roll_100hz)
    emg_raw_seg = segment_walking_strides(emg_raw, roll_1000hz)  # cut raw EMG first
    emg_filt = filter_emg(emg_raw_seg)                           # filter the segment only
    if return_names:
        return emg_filt, muscle_cols
    return emg_filt


def process_step(emg_path: str, return_names: bool = False):
    """
    Load + process a step exercise trial (NO stride segmentation — uses all data).
    Returns: (n_muscles, n_samples) normalized EMG, or (emg, muscle_cols) if
    return_names=True.
    """
    if return_names:
        emg_raw, muscle_cols = load_emg(emg_path, return_names=True)
    else:
        emg_raw = load_emg(emg_path)
    emg_filt = filter_emg(emg_raw)
    if return_names:
        return emg_filt, muscle_cols
    return emg_filt


# ---------------------------------------------------------------------------
# 6b. Co-contraction Index  (Falconer & Winter, 1985)
# ---------------------------------------------------------------------------

KNEE_AGONIST = [
    "Right Vastus medialis", "Right Vastus lateralis", "Right Rectus femoris",
]
KNEE_ANTAGONIST = [
    "Right Biceps femoris caput longus", "Right Semitendinosus",
]
ANKLE_AGONIST = [
    "Right Tibialis anterior",
]
ANKLE_ANTAGONIST = [
    "Right Gastrocnemius medialis", "Right Gastrocnemius lateralis", "Right Soleus",
]


def compute_cci(
    emg_processed: np.ndarray,
    muscle_names: list,
    agonist_muscles: list,
    antagonist_muscles: list,
    severe_dropout_idx=None,
) -> float:
    """
    Co-contraction Index (Falconer & Winter, 1985):
        CCI = 2 * sum(min(E_ag(t), E_ant(t))) / sum(E_ag(t) + E_ant(t))

    Each muscle group is averaged into a single envelope first (E_ag, E_ant),
    then the formula is applied to those two envelopes. emg_processed must be
    the same filtered/rectified/normalized envelope used for VAF (output of
    process_walking/process_step), and muscle_names must be in the same row
    order (returned alongside it via return_names=True).

    Returns NaN if:
      - any required muscle is absent from muscle_names (e.g. dropped upstream), or
      - any required muscle's row index appears in severe_dropout_idx (flatlined
        wireless dropout -- the same check used to exclude trials from the DMCI
        baseline), or
      - the combined envelope is uniformly zero (degenerate denominator).
    """
    name_to_idx = {name: i for i, name in enumerate(muscle_names)}

    ag_idx = [name_to_idx[m] for m in agonist_muscles if m in name_to_idx]
    ant_idx = [name_to_idx[m] for m in antagonist_muscles if m in name_to_idx]

    if len(ag_idx) != len(agonist_muscles) or len(ant_idx) != len(antagonist_muscles):
        return float("nan")

    if severe_dropout_idx:
        severe_set = set(severe_dropout_idx)
        if severe_set.intersection(ag_idx) or severe_set.intersection(ant_idx):
            return float("nan")

    e_ag = emg_processed[ag_idx].mean(axis=0)
    e_ant = emg_processed[ant_idx].mean(axis=0)

    denom = np.sum(e_ag + e_ant)
    if denom == 0:
        return float("nan")

    return float(2.0 * np.sum(np.minimum(e_ag, e_ant)) / denom)


# ---------------------------------------------------------------------------
# 6c. IMU kinematic features
# ---------------------------------------------------------------------------

def compute_kinematic_features(roll_100hz: np.ndarray, pitch_100hz: np.ndarray,
                                threshold: float = 20.0) -> dict:
    """
    Descriptive kinematic features from a single trial's foot IMU signal,
    computed on the native 100 Hz recording (not the 1000 Hz upsampled
    version used for EMG alignment, to avoid resample ringing affecting
    peak/ROM values).

    Foot-strike detection reuses the same convention as
    segment_walking_strides() (negative-Roll peaks above `threshold` degrees,
    no positive-sign fallback here since these are descriptive features, not
    used to cut EMG). For non-gait exercises (step-forward/step-lateral) this
    may legitimately find few or zero peaks -- it is not a true gait cadence
    for those tasks.

    Returns a dict with:
      peak_roll, rom_roll, peak_pitch, rom_pitch : float (NaN if no data)
      stride_count      : int, number of foot-strike peaks found
      stride_timing_cv  : float, SD/mean of inter-peak intervals (seconds);
                          NaN if fewer than 3 peaks (need >=2 intervals)
    """
    roll_100hz = np.asarray(roll_100hz, dtype=np.float64)
    peak_roll = float(np.max(np.abs(roll_100hz))) if len(roll_100hz) else float("nan")
    rom_roll = float(np.max(roll_100hz) - np.min(roll_100hz)) if len(roll_100hz) else float("nan")

    valid_pitch = pitch_100hz[~np.isnan(pitch_100hz)] if pitch_100hz is not None else np.array([])
    peak_pitch = float(np.max(np.abs(valid_pitch))) if len(valid_pitch) else float("nan")
    rom_pitch = float(np.max(valid_pitch) - np.min(valid_pitch)) if len(valid_pitch) else float("nan")

    peak_idxs, _ = find_peaks(-roll_100hz, height=threshold)
    stride_count = int(len(peak_idxs))
    if stride_count >= 3:
        intervals = np.diff(peak_idxs) / FS_XSENS  # seconds
        stride_timing_cv = float(np.std(intervals, ddof=1) / np.mean(intervals))
    else:
        stride_timing_cv = float("nan")

    return {
        "peak_roll": peak_roll,
        "rom_roll": rom_roll,
        "peak_pitch": peak_pitch,
        "rom_pitch": rom_pitch,
        "stride_count": stride_count,
        "stride_timing_cv": stride_timing_cv,
    }


# ---------------------------------------------------------------------------
# 7. NMF Muscle Synergy  (PosturalData_NMFvsPCA_GUI equivalent)
# ---------------------------------------------------------------------------

def compute_nmf_vaf(emg: np.ndarray, n_synergies: int = 1, n_replicates: int = 50) -> float:
    """
    Compute the Variance Accounted For (VAF) by NMF with n_synergies components.

    Equivalent to the Ting & Chvatal PosturalData_NMFvsPCA GUI.

    emg:          (n_muscles, n_samples) non-negative normalized EMG
    n_synergies:  number of synergy components (default 1 for DMCI)
    n_replicates: number of random restarts; takes the best result

    VAF formula — matches MATLAB PosturalData_NMFvsPCA_GUI funur() function:
        UR = (sum(M * WH))^2 / (sum(M^2) * sum(WH^2)) * 100

    This is the squared uncentered correlation (Zar method), NOT the residual
    formula (1 - ss_res/ss_tot). The Ting & Chvatal GUI uses this formula,
    which measures pattern similarity ignoring amplitude scaling, and gives
    higher VAF values than the residual formula.

    Returns: VAF as a percentage (0-100).
    """
    from sklearn.decomposition import NMF

    M = emg  # (n_muscles, n_samples)
    M_T = M.T  # sklearn NMF expects (n_samples, n_features) = (n_samples, n_muscles)

    best_vaf = -np.inf

    for rep in range(n_replicates):
        try:
            model = NMF(
                n_components=n_synergies,
                init="random",
                max_iter=1000,
                tol=1e-6,
                random_state=rep * 7 + 13,
            )
            W = model.fit_transform(M_T)   # (n_samples, n_syn)
            H = model.components_          # (n_syn, n_muscles)
            M_recon = (W @ H).T           # (n_muscles, n_samples)

            # MATLAB funur() squared uncentered correlation (Ting & Chvatal method)
            inner   = np.sum(M * M_recon)
            denom   = np.sum(M ** 2) * np.sum(M_recon ** 2)

            if denom == 0:
                vaf = 0.0
            else:
                vaf = (inner ** 2 / denom) * 100.0

            if vaf > best_vaf:
                best_vaf = vaf
        except Exception:
            continue

    return float(best_vaf)


# ---------------------------------------------------------------------------
# 8. DMCI calculation  (calc_dmci.m equivalent)
# ---------------------------------------------------------------------------

def compute_dmci(vaf_individual: float, vaf_young_group: np.ndarray) -> float:
    """
    Dynamic Motor Control Index (DMCI) for one participant.

    Formula (MATLAB calc_dmci.m):
        DMCI = 100 + 10 * (AVG_VAF_young - VAF_individual) / SD_VAF_young

    vaf_individual:  1-synergy VAF for this participant (%)
    vaf_young_group: array of 1-synergy VAFs for all young controls (%)

    Returns: DMCI value.
    """
    avg = float(np.mean(vaf_young_group))
    sd = float(np.std(vaf_young_group, ddof=1))  # MATLAB std() uses ddof=1
    if sd == 0:
        return np.nan
    return 100.0 + 10.0 * (avg - vaf_individual) / sd

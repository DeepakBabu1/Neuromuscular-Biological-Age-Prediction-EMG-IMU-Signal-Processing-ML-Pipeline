"""
Improved symbolic-regression model: fixed 3-feature set, heavier parsimony,
restricted operator set, per-fold standardization, linear-regression
competitor, and equation-stability analysis across folds.

DISCLOSURE (read before presenting this): the 3 features below were chosen
by ranking Spearman |rho| against age across all 29 participants (see
feature_correlations.csv from 07_feature_screening.py). That is a smaller,
but real, instance of the same feature-selection leakage that
08_nested_loocv.py was built to eliminate -- the difference is this fixes
the feature IDENTITY once (a common, disclosed compromise at small N),
rather than re-selecting from 33 candidates inside every fold (which is
what made the original 03_symbolic_regression.py leaky run look
artificially strong). Everything downstream of the feature choice --
imputation, scaling, model fitting -- is done correctly per fold, with the
test participant never touched until prediction.

Applies:
  1. Fixed features: peak_roll_walkingincline, stride_timing_cv_walking,
     rom_roll_ratio_stepforward_walk (highest whole-sample |rho|: .49/.49/.48)
  2. parsimony_coefficient = 0.05 (was 0.0005) -- suppresses GP bloat
  3. N_SEEDS=5, N_GENS=50, POPULATION=5000 per fold
  4. function_set restricted to add/sub/mul/div (no sqrt/abs/neg/inv)
  5. StandardScaler fit on the 28 training participants only, applied to both
     train and test
  6. LinearRegression competitor, same data/scaling, same fold
  7. Equation-stability analysis: constants abstracted to "C", grouped by
     structure, most frequent structure reported as "the equation"
"""

import os
import re
import time
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from gplearn.genetic import SymbolicRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA    = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS   = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_results_v2.csv"
NULL_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\null_mae_distribution_v2.csv"
os.makedirs(PLOTS, exist_ok=True)

FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk",
]

# Whole-sample Spearman rho for each feature (from feature_correlations.csv),
# used only for the biological-interpretation text at the end -- not used
# anywhere in the model itself.
FEATURE_RHO = {
    "peak_roll_walkingincline": -0.494,
    "stride_timing_cv_walking": +0.493,
    "rom_roll_ratio_stepforward_walk": +0.484,
}

N_SEEDS      = 3     # reduced from 5 -- fold 1 alone didn't finish in 7+ min at the original setting
N_GENS       = 25     # reduced from 50
POPULATION   = 1500   # reduced from 5000
PARSIMONY    = 0.05
FUNCTION_SET = ("add", "sub", "mul", "div")
RANDOM_STATE = 42

# Set >0 to run the permutation test (see the timing printed after the real
# run before choosing this -- do not blindly set to 500).
N_PERMS = 0

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df_model = df[["participant", "age"] + FEATURE_COLS].copy()
df_model = df_model[df_model["age"].notna()].reset_index(drop=True)
N = len(df_model)

print(f"Participants: {N}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")


def make_gp_model(seed):
    return SymbolicRegressor(
        population_size=POPULATION,
        generations=N_GENS,
        tournament_size=20,
        function_set=FUNCTION_SET,
        parsimony_coefficient=PARSIMONY,
        metric="mean absolute error",
        max_samples=1.0,
        n_jobs=1,
        verbose=0,
        random_state=seed,
    )


def run_loocv(df_in: pd.DataFrame, verbose: bool = False):
    n = len(df_in)
    gp_preds = np.full(n, np.nan)
    lin_preds = np.full(n, np.nan)
    fold_records = []

    for i in range(n):
        test_row = df_in.iloc[[i]]
        train_df = df_in.drop(index=df_in.index[i])

        # Impute with training median, THEN standardize on training stats only.
        train_medians = train_df[FEATURE_COLS].median()
        X_train_raw = train_df[FEATURE_COLS].fillna(train_medians).values.astype(float)
        X_test_raw = test_row[FEATURE_COLS].fillna(train_medians).values.astype(float)
        y_train = train_df["age"].values.astype(float)
        y_test = float(test_row["age"].values[0])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # gplearn multi-seed ensemble (median prediction, matching the
        # project's established LOOCV pattern elsewhere).
        fold_gp_preds, fold_gp_eqs = [], []
        for s in range(N_SEEDS):
            model = make_gp_model(seed=RANDOM_STATE + i * N_SEEDS + s)
            model.fit(X_train, y_train)
            fold_gp_preds.append(float(model.predict(X_test)[0]))
            fold_gp_eqs.append(str(model._program))
        gp_pred = float(np.median(fold_gp_preds))
        eq_idx = int(np.argmin(np.abs(np.array(fold_gp_preds) - gp_pred)))
        gp_eq = fold_gp_eqs[eq_idx]

        # Linear regression competitor, identical data/scaling.
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        lin_pred = float(lin.predict(X_test)[0])

        gp_preds[i] = gp_pred
        lin_preds[i] = lin_pred

        fold_records.append({
            "fold": i,
            "participant": test_row["participant"].values[0],
            "actual_age": y_test,
            "gp_predicted_age": gp_pred,
            "linear_predicted_age": lin_pred,
            "gp_abs_error": abs(gp_pred - y_test),
            "linear_abs_error": abs(lin_pred - y_test),
            "gp_equation": gp_eq,
        })

        if verbose:
            print(f"  [{i+1:2d}/{n}] participant={test_row['participant'].values[0]:<4} "
                  f"actual={y_test:.0f}  gp={gp_pred:.1f}  lin={lin_pred:.1f}  "
                  f"gp_err={abs(gp_pred-y_test):.1f}  lin_err={abs(lin_pred-y_test):.1f}")

    return gp_preds, lin_preds, fold_records


# ── Run the real LOOCV ────────────────────────────────────────────────────────
print("\nRunning fixed-3-feature LOOCV (gplearn ensemble + linear regression)...")
t0 = time.time()
gp_preds, lin_preds, fold_records = run_loocv(df_model, verbose=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold average).")


def readable(eq: str, feature_names) -> str:
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq


results_df = pd.DataFrame(fold_records)
results_df["gp_equation_readable"] = [readable(eq, FEATURE_COLS) for eq in results_df["gp_equation"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

# ── Performance ───────────────────────────────────────────────────────────────
actual = results_df["actual_age"].values
gp_mae = float(np.mean(np.abs(gp_preds - actual)))
lin_mae = float(np.mean(np.abs(lin_preds - actual)))
baseline_mae = float(np.mean(np.abs(actual - actual.mean())))

gp_r, gp_p = stats.pearsonr(actual, gp_preds)
lin_r, lin_p = stats.pearsonr(actual, lin_preds)
gp_rmse = float(np.sqrt(np.mean((gp_preds - actual) ** 2)))
lin_rmse = float(np.sqrt(np.mean((lin_preds - actual) ** 2)))

best_name = "gplearn" if gp_mae <= lin_mae else "linear regression"
if gp_mae <= lin_mae:
    best_mae, best_r, best_p, best_rmse, best_preds = gp_mae, gp_r, gp_p, gp_rmse, gp_preds
else:
    best_mae, best_r, best_p, best_rmse, best_preds = lin_mae, lin_r, lin_p, lin_rmse, lin_preds

# ── Equation stability across folds ──────────────────────────────────────────
def normalize_equation(eq: str) -> str:
    """Abstract numeric constants to 'C' so folds with the same structure but
    different fitted constants are grouped together."""
    return re.sub(r"-?\d+\.\d+|-?\d+", "C", eq)

templates = [normalize_equation(e) for e in results_df["gp_equation"]]
template_counts = Counter(templates)
most_common_template, freq = template_counts.most_common(1)[0]

matching_idx = [i for i, t in enumerate(templates) if t == most_common_template]
matching = results_df.iloc[matching_idx]
rep_row = matching.loc[matching["gp_abs_error"].idxmin()]
rep_equation_raw = rep_row["gp_equation"]
rep_equation_readable = readable(rep_equation_raw, FEATURE_COLS)

# ── Plot: predicted vs actual for the best model ─────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(actual, best_preds, alpha=0.75, s=55, zorder=3, c=actual, cmap="coolwarm")
lims = [min(actual.min(), best_preds.min()) - 3, max(actual.max(), best_preds.max()) + 3]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Age (years)")
ax.set_ylabel("Predicted Age (years)")
ax.set_title(f"Best model: {best_name}\nMAE={best_mae:.1f} yr  r={best_r:.3f}  n={N}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "improved_model_scatter.png"), dpi=150)
plt.close()
print(f"Saved -> {os.path.join(PLOTS, 'improved_model_scatter.png')}")

# ── Permutation test (optional -- see N_PERMS at top) ────────────────────────
perm_block = None
if N_PERMS > 0:
    print(f"\nRunning permutation test: {N_PERMS} shuffles...")
    rng = np.random.default_rng(RANDOM_STATE)
    null_maes = []
    t_perm0 = time.time()
    for p in range(N_PERMS):
        df_shuf = df_model.copy()
        df_shuf["age"] = rng.permutation(df_shuf["age"].values)
        gp_p_preds, lin_p_preds, _ = run_loocv(df_shuf, verbose=False)
        best_p_preds = gp_p_preds if best_name == "gplearn" else lin_p_preds
        mae_p = float(np.mean(np.abs(best_p_preds - df_shuf["age"].values)))
        null_maes.append(mae_p)
        print(f"  permutation {p+1}/{N_PERMS}: null MAE = {mae_p:.2f}  "
              f"(elapsed {time.time()-t_perm0:.0f}s)")
        pd.DataFrame({"permutation": range(1, len(null_maes)+1), "null_mae": null_maes}) \
            .to_csv(NULL_OUT, index=False)

    null_maes = np.array(null_maes)
    pctl_95 = float(np.percentile(null_maes, 95))
    perm_p = float(np.mean(null_maes <= best_mae))
    perm_block = (null_maes, pctl_95, perm_p)
else:
    print(f"\nPermutation test skipped (N_PERMS=0). Measured {elapsed/N:.2f}s/fold for the "
          f"real run -- a full repeat costs ~{elapsed:.0f}s, so N shuffles would take "
          f"~{elapsed/60:.1f}*N minutes. Decide N_PERMS with this number in hand.")

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
print(f"\n\n{'RESULTS SUMMARY':^60}")
print("=" * 60)
print(f"Features used: {FEATURE_COLS}")
print(f"Participants: N={N}")

print(f"\nMODEL COMPARISON:")
print(f"  gplearn MAE:           {gp_mae:.2f} years")
print(f"  Linear regression MAE: {lin_mae:.2f} years")
print(f"  Trivial baseline MAE:  {baseline_mae:.2f} years")

print(f"\nBEST MODEL: {best_name}")
print(f"  MAE:       {best_mae:.2f} years")
print(f"  Pearson r: {best_r:.3f} (p = {best_p:.4f})")
print(f"  RMSE:      {best_rmse:.2f} years")

print(f"\nTHE EQUATION (most stable across folds: {freq}/{N} folds share this structure):")
print(f"  Age ~ {rep_equation_readable}")

print(f"\nBIOLOGICAL INTERPRETATION:")
interp = {
    "peak_roll_walkingincline": (
        "Max foot-tilt angle (Roll) during walking-incline.",
        "NEGATIVE (rho=-0.49): lower peak roll predicts OLDER age.",
        "Plausible -- reduced ankle/foot range of motion during a harder "
        "(inclined) gait task is consistent with age-related joint "
        "stiffening and reduced ankle mobility reported in the gait "
        "literature."
    ),
    "stride_timing_cv_walking": (
        "Coefficient of variation of inter-footstrike timing during walking "
        "(gait rhythm variability).",
        "POSITIVE (rho=+0.49): higher stride-timing variability predicts "
        "OLDER age.",
        "Strongly plausible -- increased gait variability is one of the "
        "most consistently reported markers of neuromotor decline and "
        "fall risk in older adults."
    ),
    "rom_roll_ratio_stepforward_walk": (
        "Ratio of foot-roll range of motion in step-forward vs. walking.",
        "POSITIVE (rho=+0.48): a higher step-forward/walking ROM ratio "
        "predicts OLDER age.",
        "Plausible but more circumstantial -- could reflect older "
        "participants using relatively more foot rotation on the harder "
        "step-up task relative to an already-reduced walking baseline, "
        "i.e. a compensatory pattern rather than a direct decline marker. "
        "Interpret cautiously -- it is a ratio of two features, not a "
        "single physiological quantity."
    ),
}
for feat in FEATURE_COLS:
    what, direction, plausible = interp[feat]
    print(f"  {feat}:")
    print(f"    measures : {what}")
    print(f"    direction: {direction}")
    print(f"    plausible: {plausible}")

print(f"\nPERMUTATION TEST ({N_PERMS} shuffles):")
if perm_block is not None:
    null_maes, pctl_95, perm_p = perm_block
    print(f"  Null 95th percentile MAE: {pctl_95:.2f} years")
    print(f"  Observed MAE:             {best_mae:.2f} years")
    print(f"  Permutation p-value:      {perm_p:.4f}")
    print(f"  Conclusion: {'SIGNIFICANT' if perm_p < 0.05 else 'NOT significant'} at alpha=0.05")
else:
    print("  SKIPPED this run (N_PERMS=0) -- see timing note above.")

print(f"\nDISCLOSURE: the 3 features above were selected using whole-sample "
      f"correlation (all {N} participants), not re-derived per fold -- see "
      f"the module docstring. This is a smaller leak than the original "
      f"33-candidate-per-fold approach, but it is a leak, and should be "
      f"named as such when this is presented.")
print("=" * 60)

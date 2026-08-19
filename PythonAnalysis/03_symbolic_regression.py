"""
Step 3: Symbolic Regression with gplearn + LOOCV
- gplearn: pure-Python genetic programming (no Julia required)
- Feature set: 8 features screened from feature_table_full.csv (Spearman
  |rho| > 0.3 against age, then pruned to one representative per redundant
  pair with |Pearson r| > 0.8). 7 of 8 are IMU kinematic features; only
  vaf_stepforward is EMG-derived -- CCI and the other 3 VAF/DMCI-derived
  features did not pass screening.
- Target: age (continuous prediction)
- Validation: Leave-One-Out Cross-Validation (LOOCV). stride_timing_cv_walking
  and stride_timing_cv_walkingincline have substantial missingness (n=19/29
  and n=15/29) -- imputed with the training-fold median per LOOCV split
  (same leakage-free approach already used for all other missing values here).
- Evaluation: MAE, Pearson r, permutation test

NOTE: PySR (preferred by professor) requires Julia. If Julia is available on your
system and the Application Control policy allows DLL execution from user directories,
switch to 04_symbolic_regression_pysr.py which wraps the same pipeline with PySR.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

N_SEEDS      = 3          # seeds per fold (results averaged — reduces stochasticity)
N_GENS       = 30         # GP generations per seed
POPULATION   = 3000       # GP population size
N_PERMS      = 500        # permutation test iterations
RANDOM_STATE = 42

# ── Feature columns ───────────────────────────────────────────────────────────
# 8 features that passed Spearman |rho| > 0.3 against age in feature_correlations.csv,
# pruned to one representative per redundant pair (|Pearson r| > 0.8):
#   dropped rom_roll_walkingincline (kept peak_roll_walkingincline, stronger rho)
#   dropped rom_roll_walking        (kept peak_roll_walking, stronger rho)
#   dropped rom_pitch_walking       (kept rom_pitch_walkingincline, stronger rho)
FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk",
    "peak_roll_walking",
    "rom_pitch_walkingincline",
    "stride_timing_cv_walkingincline",
    "rom_pitch_stepforward",
    "vaf_stepforward",
]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)

df_model = df[["participant", "age"] + FEATURE_COLS].copy()
df_model = df_model[df_model["age"].notna()].copy()
# Keep rows with at least half the features present
min_features = len(FEATURE_COLS) // 2
df_model = df_model[df_model[FEATURE_COLS].notna().sum(axis=1) >= min_features].copy()
df_model = df_model.reset_index(drop=True)

print(f"Modelling dataset: n={len(df_model)} participants")
print(f"  Age range: {df_model['age'].min():.0f} – {df_model['age'].max():.0f} yrs")
print(f"  Features : {', '.join(FEATURE_COLS)}")

# Median for imputation (will be recomputed per fold to avoid leakage)
global_medians = df_model[FEATURE_COLS].median()

# ── GP model factory ──────────────────────────────────────────────────────────
def make_model(seed=0):
    return SymbolicRegressor(
        population_size=POPULATION,
        generations=N_GENS,
        tournament_size=20,
        stopping_criteria=0.0,
        const_range=(-10.0, 10.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "inv"),
        metric="mean absolute error",
        parsimony_coefficient=0.0005,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        warm_start=False,
        n_jobs=1,  # parallel is slower on small datasets
        verbose=0,
        random_state=seed,
    )

# ── LOOCV ─────────────────────────────────────────────────────────────────────
n          = len(df_model)
preds      = np.full(n, np.nan)
best_eqs   = []

print(f"\nRunning LOOCV ({n} folds, {N_SEEDS} seeds each)...")

for i in range(n):
    test_row  = df_model.iloc[[i]]
    train_set = df_model.drop(index=i).reset_index(drop=True)

    # Impute with training-set median (no leakage from test row)
    train_medians = train_set[FEATURE_COLS].median()
    X_train = train_set[FEATURE_COLS].fillna(train_medians).values.astype(float)
    X_test  = test_row[FEATURE_COLS].fillna(train_medians).values.astype(float)
    y_train = train_set["age"].values.astype(float)
    y_test  = float(test_row["age"].values[0])

    # Multi-seed ensemble
    fold_preds = []
    fold_eqs   = []
    for seed_offset in range(N_SEEDS):
        model = make_model(seed=RANDOM_STATE + i * N_SEEDS + seed_offset)
        model.fit(X_train, y_train)
        fold_preds.append(float(model.predict(X_test)[0]))
        fold_eqs.append(str(model._program))

    preds[i] = float(np.median(fold_preds))
    # Pick equation with prediction closest to median
    eq_idx = int(np.argmin(np.abs(np.array(fold_preds) - preds[i])))
    best_eqs.append(fold_eqs[eq_idx])

    err = abs(preds[i] - y_test)
    print(f"  [{i+1:2d}/{n}] {str(test_row['participant'].values[0]):22s}  "
          f"true={y_test:.0f}  pred={preds[i]:.1f}  |err|={err:.1f} yrs")

# ── Evaluation ────────────────────────────────────────────────────────────────
true_ages = df_model["age"].values.astype(float)
valid     = ~np.isnan(preds)
mae       = float(np.mean(np.abs(preds[valid] - true_ages[valid])))
r, p_r    = stats.pearsonr(true_ages[valid], preds[valid])
rmse      = float(np.sqrt(np.mean((preds[valid] - true_ages[valid])**2)))

print(f"\n{'='*55}")
print(f"LOOCV Results  (n={int(valid.sum())} participants)")
print(f"  MAE            : {mae:.2f} years")
print(f"  RMSE           : {rmse:.2f} years")
print(f"  Pearson r      : {r:.3f}  (p={p_r:.4f})")
print(f"  Baseline MAE   : {float(np.mean(np.abs(true_ages - np.mean(true_ages[valid])))):.2f} yrs (predict-mean baseline)")
print(f"{'='*55}")

# ── Permutation test ──────────────────────────────────────────────────────────
print(f"\nPermutation test ({N_PERMS} label shuffles)...")
rng       = np.random.default_rng(RANDOM_STATE)
perm_maes = []
for _ in range(N_PERMS):
    shuffled = rng.permutation(true_ages[valid])
    perm_maes.append(float(np.mean(np.abs(preds[valid] - shuffled))))
perm_p = float(np.mean(np.array(perm_maes) <= mae))
print(f"  Permutation p-value: {perm_p:.4f}")
print(f"  (fraction of {N_PERMS} shuffles with MAE <= observed {mae:.2f} yrs)")

# ── Save results ──────────────────────────────────────────────────────────────
def readable_equation(eq: str, feature_names) -> str:
    """Substitute X0, X1, ... with actual feature names, longest index first
    so e.g. X10 isn't corrupted by an earlier replace of X1."""
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq

results_df = df_model[["participant", "age"]].copy()
results_df["predicted_age"] = preds
results_df["abs_error"]     = np.abs(preds - true_ages)
results_df["loocv_equation"] = best_eqs
results_df["loocv_equation_readable"] = [readable_equation(eq, FEATURE_COLS) for eq in best_eqs]
results_df.to_csv(os.path.join(PLOTS, "loocv_results.csv"), index=False)

summary = {
    "n_participants": int(valid.sum()),
    "mae_years":      round(mae, 2),
    "rmse_years":     round(rmse, 2),
    "pearson_r":      round(r, 3),
    "pearson_p":      round(p_r, 4),
    "perm_p":         round(perm_p, 4),
}
pd.DataFrame([summary]).to_csv(os.path.join(PLOTS, "loocv_summary.csv"), index=False)
print(f"\nSummary: {summary}")

# ── Predicted vs actual ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(true_ages[valid], preds[valid], alpha=0.75, s=50, zorder=3,
           c=true_ages[valid], cmap="coolwarm")
lims = [min(true_ages.min(), preds[valid].min()) - 3,
        max(true_ages.max(), preds[valid].max()) + 3]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Age (years)", fontsize=12)
ax.set_ylabel("Predicted Age (years)", fontsize=12)
ax.set_title(f"Symbolic Regression – LOOCV\nMAE={mae:.1f} yr  r={r:.3f}  perm-p={perm_p:.3f}",
             fontsize=11)
ax.legend(fontsize=9)

# Error by participant
ax = axes[1]
err_sorted = results_df.sort_values("age")
colors_grp = ["#1f77b4" if a < 40 else "#d62728" for a in err_sorted["age"]]
ax.bar(range(len(err_sorted)), err_sorted["abs_error"], color=colors_grp)
ax.axhline(mae, color="black", linestyle="--", linewidth=1, label=f"Mean MAE={mae:.1f} yr")
ax.set_xticks(range(len(err_sorted)))
ax.set_xticklabels(err_sorted["participant"], rotation=90, fontsize=6)
ax.set_ylabel("Absolute Error (years)")
ax.set_title("Per-participant Absolute Error")
ax.legend(fontsize=9)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#1f77b4", label="Young (<40)"),
                   Patch(facecolor="#d62728", label="Older (>=40)")]
ax.legend(handles=legend_elements, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "loocv_results.png"), dpi=150, bbox_inches="tight")
plt.close()

# Residual plot
residuals = preds[valid] - true_ages[valid]
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(true_ages[valid], residuals, alpha=0.75, s=50, zorder=3)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.axhline(mae, color="red", linewidth=0.8, linestyle=":", label=f"+MAE={mae:.1f}")
ax.axhline(-mae, color="red", linewidth=0.8, linestyle=":", label=f"-MAE={mae:.1f}")
ax.set_xlabel("Actual Age (years)")
ax.set_ylabel("Residual (predicted - actual, years)")
ax.set_title("Residuals vs Age")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "loocv_residuals.png"), dpi=150)
plt.close()

print(f"\nAll outputs saved to: {PLOTS}")
print("\nSample equations from LOOCV folds (readable):")
for i, eq in enumerate(best_eqs[:8]):
    print(f"  Fold {i+1:2d}: {readable_equation(eq, FEATURE_COLS)}")

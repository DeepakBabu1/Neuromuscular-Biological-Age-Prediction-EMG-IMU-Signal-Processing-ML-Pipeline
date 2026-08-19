"""
Nested Leave-One-Out Cross-Validation with feature selection INSIDE the loop.

Why this replaces 03_symbolic_regression.py's approach: that script selected
features (Spearman |rho| > 0.3) once, using all 29 participants, BEFORE
running LOOCV. Since the test participant's age was part of that correlation
computation, information about the held-out participant leaked into feature
selection -- an optimistic bias. Here, feature selection is refit on only the
28 training participants at every fold, so the test participant is never
seen until the final prediction step.

Input: feature_table_full.csv
Target: age
Excluded from candidate features: participant, age, group, all dmci_* columns
  (dmci_* excluded because it's a deterministic affine transform of VAF built
  using the young-group labels -- leakage risk, not new information).

Outputs:
  loocv_predictions.csv       -- one row per fold (test participant)
  feature_selection_log.csv   -- one row per (fold, candidate feature)
  plots/nested_loocv_scatter.png
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gplearn.genetic import SymbolicRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

PRED_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_predictions.csv"
LOG_OUT  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_selection_log.csv"

RHO_THRESHOLD  = 0.3
TOP_N_FALLBACK = 3     # if fewer than 2 features pass the threshold, take the top-3 by |rho|
MIN_N_FOR_CORR = 5     # minimum non-NaN training pairs required to trust a correlation at all

GP_PARAMS = dict(
    population_size=1000,
    generations=20,
    tournament_size=20,
    function_set=("add", "sub", "mul", "div", "sqrt"),
    max_samples=0.9,
    parsimony_coefficient=0.01,
    metric="mean absolute error",  # not specified in the task, but matches the
                                    # project's established GP config elsewhere
                                    # rather than leaving it at gplearn's default
)

# Set to 0 to skip the permutation test entirely (it's the expensive part --
# see the printed time estimate after the main LOOCV finishes, and the
# conversation note on why 1000 full shuffles is not run by default).
N_PERMS = 100

NULL_MAE_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\null_mae_distribution.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df_raw.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df_raw.columns if c not in EXCLUDE and c not in dmci_cols]

df_raw = df_raw.dropna(subset=["age"]).reset_index(drop=True)
N = len(df_raw)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}")


# ── Inner feature selection (training fold only) ─────────────────────────────
def select_features(train_df: pd.DataFrame):
    """Spearman |rho| > 0.3 vs age, computed ONLY on train_df.
    Falls back to the top-3 by |rho| if fewer than 2 pass the threshold.
    Returns (selected_feature_list_sorted_by_abs_rho_desc, rho_lookup_dict).
    """
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            rows.append((col, np.nan))
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))

    rho_lookup = dict(rows)
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()

    passing = rho_df[rho_df["abs_rho"] > RHO_THRESHOLD].dropna(subset=["abs_rho"])
    if len(passing) < 2:
        passing = rho_df.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(TOP_N_FALLBACK)

    selected = passing.sort_values("abs_rho", ascending=False)["feature"].tolist()
    return selected, rho_lookup


def readable_equation(eq: str, feature_names) -> str:
    """Substitute X0, X1, ... with the feature names used to build that
    fold's X_train (order matches `selected`, since each fold reselects
    different features -- there is no single global mapping here)."""
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq


# ── One full nested-LOOCV pass (used for both the real run and each permutation) ─
def run_nested_loocv(df: pd.DataFrame, verbose_progress: bool = False):
    n = len(df)
    preds = np.full(n, np.nan)
    fold_records = []
    selection_log = []

    for i in range(n):
        test_row = df.iloc[[i]]
        train_df = df.drop(index=df.index[i])

        selected, rho_lookup = select_features(train_df)

        train_medians = train_df[selected].median()
        X_train = train_df[selected].fillna(train_medians).values.astype(float)
        y_train = train_df["age"].values.astype(float)
        X_test = test_row[selected].fillna(train_medians).values.astype(float)
        y_test = float(test_row["age"].values[0])

        model = SymbolicRegressor(
            population_size=GP_PARAMS["population_size"],
            generations=GP_PARAMS["generations"],
            tournament_size=GP_PARAMS["tournament_size"],
            function_set=GP_PARAMS["function_set"],
            max_samples=GP_PARAMS["max_samples"],
            parsimony_coefficient=GP_PARAMS["parsimony_coefficient"],
            metric=GP_PARAMS["metric"],
            random_state=i,   # "random_state = fold number", per spec
            n_jobs=1,
            verbose=0,
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(X_test)[0])
        preds[i] = pred

        train_pred = model.predict(X_train)
        train_mae = float(np.mean(np.abs(train_pred - y_train)))

        fold_records.append({
            "fold": i,
            "participant": test_row["participant"].values[0],
            "actual_age": y_test,
            "predicted_age": pred,
            "abs_error": abs(pred - y_test),
            "n_features_selected": len(selected),
            "selected_features": ";".join(selected),
            "equation": str(model._program),
            "equation_readable": readable_equation(str(model._program), selected),
            "train_mae": train_mae,
        })

        for feat in CANDIDATE_FEATURES:
            selection_log.append({
                "fold": i,
                "feature": feat,
                "rho": rho_lookup.get(feat, np.nan),
                "selected": feat in selected,
            })

        if verbose_progress:
            print(f"  [{i+1:2d}/{n}] participant={test_row['participant'].values[0]:<4} "
                  f"actual={y_test:.0f}  pred={pred:.1f}  |err|={abs(pred-y_test):.1f}  "
                  f"n_feat={len(selected)}  train_mae={train_mae:.2f}")

    return preds, fold_records, selection_log


# ── Run the real nested LOOCV ─────────────────────────────────────────────────
print("\nRunning nested LOOCV (feature selection refit inside every fold)...")
t0 = time.time()
preds, fold_records, selection_log = run_nested_loocv(df_raw, verbose_progress=True)
elapsed = time.time() - t0
per_fold = elapsed / N
print(f"\nDone in {elapsed:.1f}s ({per_fold:.2f}s/fold average).")

pred_df = pd.DataFrame(fold_records)
pred_df.to_csv(PRED_OUT, index=False)
print(f"Saved -> {PRED_OUT}")

sel_log_df = pd.DataFrame(selection_log)
sel_log_df.to_csv(LOG_OUT, index=False)
print(f"Saved -> {LOG_OUT}")

# ── Performance ───────────────────────────────────────────────────────────────
actual = pred_df["actual_age"].values
predicted = pred_df["predicted_age"].values
mae = float(np.mean(np.abs(predicted - actual)))
r, p_r = stats.pearsonr(actual, predicted)
rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

print(f"\n{'='*55}")
print("PERFORMANCE (nested LOOCV)")
print(f"  MAE       : {mae:.2f} years")
print(f"  RMSE      : {rmse:.2f} years")
print(f"  Pearson r : {r:.3f}  (p={p_r:.4f})")
print(f"{'='*55}")

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(actual, predicted, alpha=0.75, s=55, zorder=3, c=actual, cmap="coolwarm")
lims = [min(actual.min(), predicted.min()) - 3, max(actual.max(), predicted.max()) + 3]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Age (years)")
ax.set_ylabel("Predicted Age (years)")
ax.set_title(f"Nested LOOCV (feature selection inside the fold)\nMAE={mae:.1f} yr  r={r:.3f}  n={N}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "nested_loocv_scatter.png"), dpi=150)
plt.close()
print(f"Saved -> {os.path.join(PLOTS, 'nested_loocv_scatter.png')}")

# ── Feature stability ─────────────────────────────────────────────────────────
stability = (sel_log_df[sel_log_df["selected"]]
             .groupby("feature")["fold"].count()
             .reindex(CANDIDATE_FEATURES, fill_value=0)
             .sort_values(ascending=False))
stable_thresh = N / 2.0

print(f"\n{'='*55}")
print(f"FEATURE STABILITY (selected in how many of {N} folds)")
print(f"{'='*55}")
for feat, count in stability.items():
    tag = "  <-- STABLE (>50%)" if count > stable_thresh else ""
    print(f"  {feat:35s}  {count:2d}/{N}{tag}")

# ── Permutation test ──────────────────────────────────────────────────────────
if N_PERMS > 0:
    print(f"\nRunning permutation test: {N_PERMS} full nested-LOOCV repeats on shuffled age...")
    rng = np.random.default_rng(42)
    null_maes = []
    t_perm0 = time.time()
    for p in range(N_PERMS):
        df_shuf = df_raw.copy()
        df_shuf["age"] = rng.permutation(df_shuf["age"].values)
        preds_p, records_p, _ = run_nested_loocv(df_shuf, verbose_progress=False)
        mae_p = float(np.mean(np.abs(np.array([r["predicted_age"] for r in records_p]) -
                                      np.array([r["actual_age"] for r in records_p]))))
        null_maes.append(mae_p)
        print(f"  permutation {p+1}/{N_PERMS}: null MAE = {mae_p:.2f}  "
              f"(elapsed {time.time()-t_perm0:.0f}s)")

        # Checkpoint after every permutation so an overnight run survives an
        # interruption -- whatever has completed so far is always recoverable.
        pd.DataFrame({"permutation": range(1, len(null_maes) + 1), "null_mae": null_maes}) \
            .to_csv(NULL_MAE_OUT, index=False)

    perm_elapsed = time.time() - t_perm0

    null_maes = np.array(null_maes)
    pctl_95 = float(np.percentile(null_maes, 95))
    empirical_p = float(np.mean(null_maes <= mae))  # standard permutation p-value

    print(f"\n{'='*55}")
    print("PERMUTATION TEST")
    print(f"  Shuffles run           : {N_PERMS}  (took {perm_elapsed:.1f}s total)")
    print(f"  Null MAE distribution  : mean={null_maes.mean():.2f}  sd={null_maes.std():.2f}")
    print(f"  95th percentile of null MAE: {pctl_95:.2f} years")
    print(f"  Observed real MAE      : {mae:.2f} years")
    beats = mae < pctl_95
    print(f"  Real MAE {'BEATS' if beats else 'does NOT beat'} the 95th-percentile null threshold "
          f"({mae:.2f} {'<' if beats else '>='} {pctl_95:.2f})")
    print(f"  Empirical permutation p-value (fraction of null MAE <= observed): {empirical_p:.4f}")
    print(f"{'='*55}")

    print(f"Saved -> {NULL_MAE_OUT}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_maes, bins=20, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(mae, color="red", linewidth=2, linestyle="--",
               label=f"Real MAE = {mae:.2f} yr")
    ax.axvline(pctl_95, color="black", linewidth=1, linestyle=":",
               label=f"95th percentile = {pctl_95:.2f} yr")
    ax.set_xlabel("Null MAE (years) -- nested LOOCV on shuffled age labels")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test Null Distribution (N={N_PERMS})\n"
                 f"Empirical p = {empirical_p:.4f}  "
                 f"({'real MAE beats null' if beats else 'real MAE does NOT beat null'})")
    ax.legend()
    plt.tight_layout()
    perm_plot_path = os.path.join(PLOTS, "permutation_test_result.png")
    plt.savefig(perm_plot_path, dpi=150)
    plt.close()
    print(f"Saved -> {perm_plot_path}")
else:
    print(f"\nPermutation test skipped (N_PERMS=0). Measured {per_fold:.2f}s/fold for the real run --")
    print(f"a full nested-LOOCV repeat costs ~{elapsed:.0f}s, so N shuffles would take ~{elapsed/60:.1f}*N minutes.")
    print("Set N_PERMS at the top of this script once you've picked a feasible count.")

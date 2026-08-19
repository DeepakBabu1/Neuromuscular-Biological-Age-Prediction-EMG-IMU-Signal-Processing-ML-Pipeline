"""
session_summary_all_runs_v2.csv -- master table of every predictive-model
run completed this session, including everything done after
session_summary_all_runs.csv was last saved. Reads exclusively from already
-saved CSV outputs; no model is re-run or recomputed here except deriving
summary statistics (MAE/RMSE/r/p) directly from saved per-participant
predictions, which is reading+deriving, not re-running.
"""

import numpy as np
import pandas as pd
from scipy import stats

OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\session_summary_all_runs_v2.csv"

# The correct, properly-nested predict-mean baseline for N=29 (Task 1 finding).
# Used uniformly for every Beats_baseline comparison below.
BASELINE_MAE = 14.47


def p_from_r(r, n):
    """Analytic two-sided p-value from a saved Pearson r and n, used only
    where raw per-participant predictions were not saved (04, 13) so no
    other route to a p-value exists without re-fitting."""
    if r is None or abs(r) >= 1.0 or n <= 2:
        return None
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))


FULLY_RIGOROUS = "Per-fold nested (leakage-free)"
WHOLE_SAMPLE = "Whole-sample (fixed, leaky)"
NO_HOLDOUT = "No holdout (not cross-validated)"

rows = []

# ── Fully rigorous (per-fold nested) ─────────────────────────────────────────

# 05_baseline_comparison.py -- lasso, ridge, linear, baseline
b = pd.read_csv("baseline_comparison_v2_summary.csv").set_index("model")
n05 = len(pd.read_csv("baseline_comparison_v2.csv"))

r = b.loc["lasso"]
rows.append(dict(Run="Run 1: Lasso regression, nested top-3 features (nested)",
                  N_features="3 (varies/fold)", Feature_selection=FULLY_RIGOROUS,
                  Model="Lasso regression", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=r["Pearson_r"], p_value=r["p_value"],
                  Notes="Alpha=10.0 not used here -- Lasso alpha=1.0, per task spec."))

r = b.loc["ridge"]
rows.append(dict(Run="Run 2: Ridge regression, nested top-3 features (nested)",
                  N_features="3 (varies/fold)", Feature_selection=FULLY_RIGOROUS,
                  Model="Ridge regression", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=r["Pearson_r"], p_value=r["p_value"],
                  Notes="Alpha=10.0, nested top-3 feature reselection every fold."))

d8 = pd.read_csv("loocv_predictions.csv")
mae8 = d8["abs_error"].mean()
rmse8 = float(np.sqrt(((d8["predicted_age"] - d8["actual_age"]) ** 2).mean()))
r8, p8 = stats.pearsonr(d8["actual_age"], d8["predicted_age"])
rows.append(dict(Run="Run 3: Nested gplearn, 10-12 features per fold (nested)",
                  N_features=f"{int(d8['n_features_selected'].min())}-{int(d8['n_features_selected'].max())} (varies/fold)",
                  Feature_selection=FULLY_RIGOROUS, Model="gplearn", N=len(d8),
                  MAE=round(mae8, 4), RMSE=round(rmse8, 4), Pearson_r=round(r8, 4), p_value=round(p8, 4),
                  Notes="Threshold |rho|>0.3 selection re-derived every fold, top-3 fallback if <2 pass."))

r = b.loc["linear"]
rows.append(dict(Run="Run 4: Linear regression, nested top-3 features (nested)",
                  N_features="3 (varies/fold)", Feature_selection=FULLY_RIGOROUS,
                  Model="Linear regression", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=r["Pearson_r"], p_value=r["p_value"],
                  Notes="Nested top-3 feature reselection every fold."))

r = b.loc["baseline"]
rows.append(dict(Run="Run 5: Predict-mean baseline, properly nested (nested)",
                  N_features=0, Feature_selection=FULLY_RIGOROUS,
                  Model="Predict-mean baseline", N=n05, MAE=r["MAE"], RMSE=r["RMSE"],
                  Pearson_r=None, p_value=None,
                  Notes="Baseline r/p are a mathematical artifact of LOOCV mean construction "
                        "(pred_i = (sum-age_i)/28 is an exact negative linear function of age_i, "
                        "forcing r=-1.000) -- not a real relationship, left blank on purpose."))

d10 = pd.read_csv("loocv_reselected_features.csv")
mae10 = d10["abs_error"].mean()
rmse10 = float(np.sqrt(((d10["predicted_age"] - d10["actual_age"]) ** 2).mean()))
r10, p10 = stats.pearsonr(d10["actual_age"], d10["predicted_age"])
rows.append(dict(Run="Run 6: Nested linear regression, 10-12 features per fold (nested)",
                  N_features=f"{int(d10['n_features_selected'].min())}-{int(d10['n_features_selected'].max())} (varies/fold)",
                  Feature_selection=FULLY_RIGOROUS, Model="Linear regression", N=len(d10),
                  MAE=round(mae10, 4), RMSE=round(rmse10, 4), Pearson_r=round(r10, 4), p_value=round(p10, 4),
                  Notes="Unregularized OLS on ~11 features / 28 training points -- overfits badly, worst nested-threshold result."))

d12 = pd.read_csv("loocv_gplearn_top3_perfold_with_readable_eq.csv")
mae12 = d12["abs_error"].mean()
rmse12 = float(np.sqrt(((d12["predicted_age"] - d12["actual_age"]) ** 2).mean()))
r12, p12 = stats.pearsonr(d12["actual_age"], d12["predicted_age"])
rows.append(dict(Run="Run 7: Nested gplearn, exactly top-3 per fold (nested)",
                  N_features=3, Feature_selection=FULLY_RIGOROUS, Model="gplearn", N=len(d12),
                  MAE=round(mae12, 4), RMSE=round(rmse12, 4), Pearson_r=round(r12, 4), p_value=round(p12, 4),
                  Notes="Worst result of the whole session; r essentially zero. Features not standardized before fitting, unlike Run 8/10."))

# ── Partial leaky (whole-sample feature selection) ───────────────────────────

d9 = pd.read_csv("loocv_results_v2.csv")
lin_mae9 = d9["linear_abs_error"].mean()
lin_rmse9 = float(np.sqrt(((d9["linear_predicted_age"] - d9["actual_age"]) ** 2).mean()))
lin_r9, lin_p9 = stats.pearsonr(d9["actual_age"], d9["linear_predicted_age"])
rows.append(dict(Run="Run 8: Linear regression, fixed 3 features (leaky)",
                  N_features=3, Feature_selection=WHOLE_SAMPLE, Model="Linear regression", N=len(d9),
                  MAE=round(lin_mae9, 4), RMSE=round(lin_rmse9, 4), Pearson_r=round(lin_r9, 4), p_value=round(lin_p9, 4),
                  Notes="Best-looking result of the whole session, and the most leaky -- 3 features chosen from whole-sample correlation."))

s3 = pd.read_csv(r"plots\loocv_summary.csv").iloc[0]
rows.append(dict(Run="Run 9: gplearn, 8 fixed features (leaky)",
                  N_features=8, Feature_selection=WHOLE_SAMPLE, Model="gplearn", N=int(s3["n_participants"]),
                  MAE=s3["mae_years"], RMSE=s3["rmse_years"], Pearson_r=s3["pearson_r"], p_value=s3["pearson_p"],
                  Notes=f"500-shuffle permutation p={s3['perm_p']} also reported, but permutation test itself doesn't fix the leaky feature selection."))

gp_mae9 = d9["gp_abs_error"].mean()
gp_rmse9 = float(np.sqrt(((d9["gp_predicted_age"] - d9["actual_age"]) ** 2).mean()))
gp_r9, gp_p9 = stats.pearsonr(d9["actual_age"], d9["gp_predicted_age"])
rows.append(dict(Run="Run 10: gplearn, fixed 3 features (leaky)",
                  N_features=3, Feature_selection=WHOLE_SAMPLE, Model="gplearn", N=len(d9),
                  MAE=round(gp_mae9, 4), RMSE=round(gp_rmse9, 4), Pearson_r=round(gp_r9, 4), p_value=round(gp_p9, 4),
                  Notes="Same 3 fixed features as Run 8, gplearn instead of linear -- worse than its linear counterpart."))

# ── Not validated (no holdout) ────────────────────────────────────────────────

eq4 = pd.read_csv(r"plots\final_model_equation.csv").iloc[0]
rows.append(dict(Run="Run 11: gplearn, 452-node equation, full-dataset fit (not cross-validated)",
                  N_features=8, Feature_selection=NO_HOLDOUT, Model="gplearn", N=29,
                  MAE=round(eq4["train_mae"], 4), RMSE=None,
                  Pearson_r=round(eq4["train_r"], 4), p_value=round(eq4["train_p"], 4),
                  Notes="Not cross-validated -- trained and tested on the same 29 participants, for "
                        "equation discovery only. 452-node/depth-52 equation, severe overfitting. "
                        "RMSE was not saved for this run (only MAE/r/p were recorded)."))

d13 = pd.read_csv("final_model_constrained_seeds.csv")
valid13 = d13[d13["passes_ceiling"]]
best13 = valid13.loc[valid13["mae"].idxmin()]
p13 = p_from_r(best13["r"], 29)
rows.append(dict(Run="Run 12: gplearn, 3-node equation (heavy parsimony), full-dataset fit (not cross-validated)",
                  N_features=8, Feature_selection=NO_HOLDOUT, Model="gplearn", N=29,
                  MAE=round(best13["mae"], 4), RMSE=None,
                  Pearson_r=round(best13["r"], 4), p_value=round(p13, 4),
                  Notes=f"Not cross-validated -- trained and tested on the same 29 participants. "
                        f"3-node equation ('{best13['equation_readable']}') has no fitted coefficients; "
                        f"likely a numeric-scale coincidence (raw angle value in similar numeric range "
                        f"as age in years), not a discovered relationship. RMSE not saved for this run; "
                        f"p-value derived analytically from saved r (raw predictions weren't saved)."))

# ── Finalize ──────────────────────────────────────────────────────────────────
table = pd.DataFrame(rows)

def sigfig4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(f"{x:.4g}")

for col in ["MAE", "RMSE", "Pearson_r", "p_value"]:
    table[col] = table[col].apply(lambda v: sigfig4(v) if pd.notna(v) else None)

table["Beats_baseline"] = table["MAE"].apply(lambda m: bool(m < BASELINE_MAE) if pd.notna(m) else None)
table["Significant"] = table["p_value"].apply(lambda p: bool(p < 0.05) if pd.notna(p) else False)

table = table[["Run", "N_features", "Feature_selection", "Model", "N",
               "MAE", "RMSE", "Pearson_r", "p_value", "Beats_baseline", "Significant", "Notes"]]

table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}\n")

pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 250)
print(table.to_string(index=False))

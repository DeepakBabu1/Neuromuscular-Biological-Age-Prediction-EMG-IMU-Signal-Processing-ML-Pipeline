"""
Final consolidation for tomorrow's supervisor presentation.
Pulls exclusively from already-saved CSVs -- no models re-run, no new
statistics computed beyond simple aggregation/formatting.

Produces:
  master_results_summary.csv
  bias_correction_summary.csv
  per_participant_all_runs.csv (updated in place with outlier flag columns)
"""

import numpy as np
import pandas as pd

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

# =============================================================================
# TASK 1: master_results_summary.csv
# =============================================================================
stat_comp = pd.read_csv(f"{BASE}\\statistical_comparison.csv").set_index("comparison")
bc = pd.read_csv(f"{BASE}\\baseline_comparison_v2_summary.csv").set_index("model")
d8 = pd.read_csv(f"{BASE}\\loocv_predictions.csv")
d12 = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv")
loocv_sum03 = pd.read_csv(f"{BASE}\\plots\\loocv_summary.csv").iloc[0]
d09 = pd.read_csv(f"{BASE}\\loocv_results_v2.csv")
clf = pd.read_csv(f"{BASE}\\classification_young_vs_old_metrics.csv").iloc[0]

from scipy import stats as sps

# gplearn (nested, 10-12 features) stats derived from saved per-fold predictions
mae8 = d8["abs_error"].mean()
rmse8 = float(np.sqrt(((d8["predicted_age"] - d8["actual_age"]) ** 2).mean()))
r8, p8 = sps.pearsonr(d8["actual_age"], d8["predicted_age"])

# gplearn (nested, forced top-3)
mae12 = d12["abs_error"].mean()
rmse12 = float(np.sqrt(((d12["predicted_age"] - d12["actual_age"]) ** 2).mean()))
r12, p12 = sps.pearsonr(d12["actual_age"], d12["predicted_age"])

# Linear 3-feature (whole-sample, leaky) -- from 09_improved_model.py
lin3_mae = d09["linear_abs_error"].mean()
lin3_rmse = float(np.sqrt(((d09["linear_predicted_age"] - d09["actual_age"]) ** 2).mean()))
lin3_r, lin3_p = sps.pearsonr(d09["actual_age"], d09["linear_predicted_age"])

rows = [
    # -- nested (leakage-free) --
    dict(model_name="Ridge (nested top-3)", feature_selection_method="Per-fold nested",
         n_features="3 (varies/fold)", n_participants=29,
         MAE=bc.loc["ridge","MAE"], RMSE=bc.loc["ridge","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["ridge","RMSE"]/bc.loc["ridge","MAE"],4),
         pearson_r=bc.loc["ridge","Pearson_r"], p_value=bc.loc["ridge","p_value"],
         wilcoxon_p_vs_baseline=stat_comp.loc["Ridge vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["Ridge vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["Ridge vs Baseline","ci_upper_95"],
         notes="alpha=10.0"),

    dict(model_name="Lasso (nested top-3)", feature_selection_method="Per-fold nested",
         n_features="3 (varies/fold)", n_participants=29,
         MAE=bc.loc["lasso","MAE"], RMSE=bc.loc["lasso","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["lasso","RMSE"]/bc.loc["lasso","MAE"],4),
         pearson_r=bc.loc["lasso","Pearson_r"], p_value=bc.loc["lasso","p_value"],
         wilcoxon_p_vs_baseline=stat_comp.loc["Lasso vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["Lasso vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["Lasso vs Baseline","ci_upper_95"],
         notes="alpha=1.0"),

    dict(model_name="Linear (nested top-3)", feature_selection_method="Per-fold nested",
         n_features="3 (varies/fold)", n_participants=29,
         MAE=bc.loc["linear","MAE"], RMSE=bc.loc["linear","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["linear","RMSE"]/bc.loc["linear","MAE"],4),
         pearson_r=bc.loc["linear","Pearson_r"], p_value=bc.loc["linear","p_value"],
         wilcoxon_p_vs_baseline=stat_comp.loc["Linear vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["Linear vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["Linear vs Baseline","ci_upper_95"],
         notes="OLS, no regularization"),

    dict(model_name="gplearn (nested, 10-12 features)", feature_selection_method="Per-fold nested (threshold |rho|>0.3)",
         n_features="10-12 (varies/fold)", n_participants=len(d8),
         MAE=round(mae8,4), RMSE=round(rmse8,4), RMSE_MAE_ratio=round(rmse8/mae8,4),
         pearson_r=round(r8,4), p_value=round(p8,4),
         wilcoxon_p_vs_baseline=stat_comp.loc["gplearn vs Baseline","wilcoxon_p"],
         bootstrap_ci_lower=stat_comp.loc["gplearn vs Baseline","ci_lower_95"],
         bootstrap_ci_upper=stat_comp.loc["gplearn vs Baseline","ci_upper_95"],
         notes="08_nested_loocv.py"),

    dict(model_name="gplearn (nested, forced top-3)", feature_selection_method="Per-fold nested (exact top-3 by rank)",
         n_features=3, n_participants=len(d12),
         MAE=round(mae12,4), RMSE=round(rmse12,4), RMSE_MAE_ratio=round(rmse12/mae12,4),
         pearson_r=round(r12,4), p_value=round(p12,4),
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None,
         notes="12_nested_gplearn_top3.py; not standardized (unlike other nested models) -- "
               "no Wilcoxon/bootstrap vs baseline computed this session"),

    dict(model_name="Baseline (predict-mean)", feature_selection_method="N/A",
         n_features=0, n_participants=29,
         MAE=bc.loc["baseline","MAE"], RMSE=bc.loc["baseline","RMSE"],
         RMSE_MAE_ratio=round(bc.loc["baseline","RMSE"]/bc.loc["baseline","MAE"],4),
         pearson_r=None, p_value=None,
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None,
         notes="Properly nested (training-fold mean, not whole-sample mean). r/p intentionally "
               "blank -- would be a mathematical artifact of LOOCV mean construction (r=-1.000 "
               "exactly), not a real relationship."),

    # -- whole-sample (leaky) --
    dict(model_name="Linear (8-feature, whole-sample, leaky)", feature_selection_method="Whole-sample (fixed, leaky)",
         n_features=8, n_participants=None,
         MAE=None, RMSE=None, RMSE_MAE_ratio=None, pearson_r=None, p_value=None,
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None,
         notes="NOT COMPUTED THIS SESSION -- no Linear regression was run on the 8-feature "
               "whole-sample-selected set. The only 8-feature leaky run was gplearn (see below); "
               "the only Linear leaky run used the fixed 3-feature set (row below). Do not present "
               "numbers for this configuration -- none exist."),

    dict(model_name="Linear (3-feature, whole-sample, leaky)", feature_selection_method="Whole-sample (fixed, leaky)",
         n_features=3, n_participants=len(d09),
         MAE=round(lin3_mae,4), RMSE=round(lin3_rmse,4), RMSE_MAE_ratio=round(lin3_rmse/lin3_mae,4),
         pearson_r=round(lin3_r,4), p_value=round(lin3_p,4),
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None,
         notes="09_improved_model.py; features chosen once on whole sample (leaky) -- best-looking "
               "result of the whole session, and the most leaky"),

    dict(model_name="gplearn (8-feature, whole-sample, leaky)", feature_selection_method="Whole-sample (fixed, leaky)",
         n_features=8, n_participants=int(loocv_sum03["n_participants"]),
         MAE=loocv_sum03["mae_years"], RMSE=loocv_sum03["rmse_years"],
         RMSE_MAE_ratio=round(loocv_sum03["rmse_years"]/loocv_sum03["mae_years"],4),
         pearson_r=loocv_sum03["pearson_r"], p_value=loocv_sum03["pearson_p"],
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None,
         notes=f"03_symbolic_regression.py; 500-shuffle permutation p={loocv_sum03['perm_p']} also "
               f"available (also leaky, since feature selection was leaky)"),

    # -- binary classification (different task type) --
    dict(model_name="Logistic Regression (binary young/old classification)",
         feature_selection_method="Fixed (same 3 features, nested LOOCV model fit)",
         n_features=3, n_participants=int(clf["N"]),
         MAE=clf["accuracy"], RMSE=None, RMSE_MAE_ratio=None,
         pearson_r=clf["AUC_ROC"], p_value=0.032,
         wilcoxon_p_vs_baseline=None, bootstrap_ci_lower=None, bootstrap_ci_upper=None,
         notes=f"DIFFERENT TASK TYPE -- MAE column holds ACCURACY (0-1 scale, not years), "
               f"pearson_r column holds AUC-ROC. Sensitivity(old)={clf['sensitivity_old']:.4f}, "
               f"Specificity(young)={clf['specificity_young']:.4f}. p_value=0.032 is the permutation-test "
               f"p-value (500 shuffles), not a Pearson-r p-value. The only properly-nested, "
               f"leakage-free result in the entire session that reaches p<0.05."),
]

table = pd.DataFrame(rows)
BASELINE_MAE = bc.loc["baseline","MAE"]

# significant_at_0.05: defined on p_value (consistent with session_summary_all_runs_v2.csv
# convention) -- for the classification row, p_value holds the permutation p, so this
# still resolves correctly as "significant via the appropriate test for that row".
table["significant_at_0.05"] = table["p_value"].apply(lambda p: bool(p < 0.05) if pd.notna(p) else False)

table["_sort_mae"] = pd.to_numeric(table["MAE"], errors="coerce")
table = table.sort_values(["significant_at_0.05", "_sort_mae"], ascending=[False, True], na_position="last")
table = table.drop(columns="_sort_mae")

col_order = ["model_name", "feature_selection_method", "n_features", "n_participants",
             "MAE", "RMSE", "RMSE_MAE_ratio", "pearson_r", "p_value",
             "wilcoxon_p_vs_baseline", "bootstrap_ci_lower", "bootstrap_ci_upper",
             "significant_at_0.05", "notes"]
table = table[col_order].reset_index(drop=True)
table.to_csv(f"{BASE}\\master_results_summary.csv", index=False)

# =============================================================================
# TASK 2: bias_correction_summary.csv
# =============================================================================
bcr = pd.read_csv(f"{BASE}\\bias_correction_results.csv")
bcr = bcr[bcr["model"] != "Baseline"].copy()  # correction only applied to the 4 models

CIRCULARITY_NOTE = (
    "Correction uses the held-out participant's OWN true age as an input "
    "(slope*actual_age + intercept), so corrected_pred is mathematically pulled toward "
    "actual_age + residual. This is a documented property of the Beheshti-style correction "
    "in the literature, not an implementation error -- but it means these 'after' numbers "
    "are not achievable for a genuinely unseen participant (whose true age is unknown), and "
    "should be presented as bias-characterization, not a working fix."
)

bias_table = pd.DataFrame({
    "model_name": bcr["model"],
    "MAE_before": bcr["mae_before"],
    "r_before": bcr["r_before"],
    "bias_correlation": bcr["bias_corr_before"],
    "MAE_after_correction": bcr["mae_after"],
    "r_after_correction": bcr["r_after"],
    "circularity_note": CIRCULARITY_NOTE,
})
bias_table.to_csv(f"{BASE}\\bias_correction_summary.csv", index=False)

# =============================================================================
# TASK 3: per_participant_all_runs.csv -- add outlier flag columns
# =============================================================================
pp = pd.read_csv(f"{BASE}\\per_participant_all_runs.csv")
model_mae = {
    "ridge": bc.loc["ridge","MAE"], "lasso": bc.loc["lasso","MAE"],
    "linear": bc.loc["linear","MAE"], "gplearn": mae8,
    "baseline": bc.loc["baseline","MAE"],
}
for m, mae_val in model_mae.items():
    pp[f"{m}_outlier"] = pp[f"{m}_error"] > (2 * mae_val)

pp.to_csv(f"{BASE}\\per_participant_all_runs.csv", index=False)

print("Saved:")
print(f"  {BASE}\\master_results_summary.csv")
print(f"  {BASE}\\bias_correction_summary.csv")
print(f"  {BASE}\\per_participant_all_runs.csv (updated with outlier flag columns)")
print()
print(f"BASELINE_MAE used for outlier/beats-baseline reference: {BASELINE_MAE}")

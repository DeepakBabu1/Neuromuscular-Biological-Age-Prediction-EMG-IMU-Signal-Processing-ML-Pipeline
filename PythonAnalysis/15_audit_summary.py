"""
Task 4: master audit of every predictive-model script run this session.
Reads exclusively from already-saved CSV/log outputs -- no models re-run.

rigor_tier definitions:
  fully_rigorous          -- feature selection re-derived inside each LOOCV
                             fold using training data only, AND proper
                             train/test split for model fitting.
  partial_leaky           -- LOOCV train/test split is correct, but the
                             feature LIST was chosen once on the whole
                             sample before cross-validation began.
  not_validated           -- no cross-validation at all; fit on the full
                             dataset with no held-out test set.
  not_yet_run             -- planned but never executed (none remain after
                             Task 1 closed the 05_baseline_comparison.py gap).
"""

import numpy as np
import pandas as pd
from scipy import stats

OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\audit_summary.csv"


def p_from_r(r, n):
    """Analytic two-sided p-value for a Pearson r given n, when raw
    predictions aren't saved (only r itself was recorded)."""
    if abs(r) >= 1.0 or n <= 2:
        return 0.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))


rows = []

# ── 03_symbolic_regression.py ────────────────────────────────────────────────
s = pd.read_csv(r"plots\loocv_summary.csv").iloc[0]
rows.append({
    "script_name": "03_symbolic_regression.py",
    "rigor_tier": "partial_leaky",
    "feature_selection_method": "whole_sample_once",
    "n_features": 8, "N": int(s["n_participants"]),
    "MAE": s["mae_years"], "pearson_r": s["pearson_r"], "p_value": s["pearson_p"],
    "equation_or_note": "gplearn, 8 fixed features chosen by whole-sample Spearman screening; "
                         f"500-shuffle permutation p={s['perm_p']} (also leaky, since features were leaky)",
})

# ── 04_final_model.py ────────────────────────────────────────────────────────
eq4 = pd.read_csv(r"plots\final_model_equation.csv").iloc[0]
rows.append({
    "script_name": "04_final_model.py",
    "rigor_tier": "not_validated",
    "feature_selection_method": "whole_sample_no_holdout",
    "n_features": 8, "N": 29,
    "MAE": eq4["train_mae"], "pearson_r": eq4["train_r"], "p_value": None,
    "equation_or_note": f"NOT cross-validated -- fit on all 29, best of 5 seeds by training MAE. "
                        f"{eq4['n_nodes']}-node equation (depth {eq4['depth']}), severe overfitting risk. "
                        f"Eq: {eq4['equation_readable'][:120]}...",
})

# ── 05_baseline_comparison.py (Task 1, fresh this session) ──────────────────
b = pd.read_csv(r"baseline_comparison_v2_summary.csv")
per_fold_05 = pd.read_csv(r"baseline_comparison_v2.csv")
n05 = len(per_fold_05)
model_notes = {
    "baseline": "Predict-mean baseline, PROPERLY nested (training-mean per fold, not whole-sample "
                "mean). r=-1.000 is a mathematical artifact of leave-one-out mean construction "
                "(pred_i = (sum-age_i)/28, an exact negative linear function of age_i) -- not a "
                "real relationship; ignore significance for this row.",
    "linear": "LinearRegression, nested top-3 feature reselection every fold.",
    "ridge": "Ridge(alpha=10.0), nested top-3 feature reselection every fold.",
    "lasso": "Lasso(alpha=1.0), nested top-3 feature reselection every fold.",
}
for _, r in b.iterrows():
    rows.append({
        "script_name": f"05_baseline_comparison.py ({r['model']})",
        "rigor_tier": "fully_rigorous",
        "feature_selection_method": "n/a" if r["model"] == "baseline" else "per_fold",
        "n_features": 0 if r["model"] == "baseline" else "3 (varies/fold)",
        "N": n05, "MAE": r["MAE"], "pearson_r": r["Pearson_r"], "p_value": r["p_value"],
        "equation_or_note": model_notes[r["model"]],
    })

# ── 08_nested_loocv.py ───────────────────────────────────────────────────────
d8 = pd.read_csv(r"loocv_predictions.csv")
mae8 = d8["abs_error"].mean()
rmse8 = np.sqrt(((d8["predicted_age"] - d8["actual_age"]) ** 2).mean())
r8, p8 = stats.pearsonr(d8["actual_age"], d8["predicted_age"])
rows.append({
    "script_name": "08_nested_loocv.py",
    "rigor_tier": "fully_rigorous",
    "feature_selection_method": "per_fold",
    "n_features": f"{int(d8['n_features_selected'].min())}-{int(d8['n_features_selected'].max())} (varies/fold)",
    "N": len(d8), "MAE": round(mae8, 2), "pearson_r": round(r8, 3), "p_value": round(p8, 4),
    "equation_or_note": "gplearn, |rho|>0.3 threshold selection re-derived every fold (top-3 fallback "
                        "if <2 pass). First fully-rigorous nested run this session.",
})

# ── 09_improved_model.py (gplearn + linear, fixed 3 features) ───────────────
d9 = pd.read_csv(r"loocv_results_v2.csv")
gp_mae9 = d9["gp_abs_error"].mean()
gp_r9, gp_p9 = stats.pearsonr(d9["actual_age"], d9["gp_predicted_age"])
lin_mae9 = d9["linear_abs_error"].mean()
lin_r9, lin_p9 = stats.pearsonr(d9["actual_age"], d9["linear_predicted_age"])
rows.append({
    "script_name": "09_improved_model.py (gplearn)",
    "rigor_tier": "partial_leaky",
    "feature_selection_method": "whole_sample_once",
    "n_features": 3, "N": len(d9),
    "MAE": round(gp_mae9, 2), "pearson_r": round(gp_r9, 3), "p_value": round(gp_p9, 4),
    "equation_or_note": "gplearn, same fixed 3 features every fold (leaky), StandardScaler per fold.",
})
rows.append({
    "script_name": "09_improved_model.py (linear)",
    "rigor_tier": "partial_leaky",
    "feature_selection_method": "whole_sample_once",
    "n_features": 3, "N": len(d9),
    "MAE": round(lin_mae9, 2), "pearson_r": round(lin_r9, 3), "p_value": round(lin_p9, 4),
    "equation_or_note": "LinearRegression, same fixed 3 features every fold (leaky), StandardScaler per fold. "
                        "Best-looking result of the whole session -- and the most leaky.",
})

# ── 10_nested_linear_regression.py ───────────────────────────────────────────
d10 = pd.read_csv(r"loocv_reselected_features.csv")
mae10 = d10["abs_error"].mean()
r10, p10 = stats.pearsonr(d10["actual_age"], d10["predicted_age"])
rows.append({
    "script_name": "10_nested_linear_regression.py",
    "rigor_tier": "fully_rigorous",
    "feature_selection_method": "per_fold",
    "n_features": f"{int(d10['n_features_selected'].min())}-{int(d10['n_features_selected'].max())} (varies/fold)",
    "N": len(d10), "MAE": round(mae10, 2), "pearson_r": round(r10, 3), "p_value": round(p10, 4),
    "equation_or_note": "LinearRegression, same |rho|>0.3 nested selection as 08. Worst of the "
                        "threshold-based nested runs -- unregularized OLS on ~11 features / 28 "
                        "training points overfits badly.",
})

# ── 12_nested_gplearn_top3.py ────────────────────────────────────────────────
# original is locked (open in Excel) -- the readable-equation update went to
# the fallback file instead; read from there so the column is present.
d12 = pd.read_csv(r"loocv_gplearn_top3_perfold_with_readable_eq.csv")
mae12 = d12["abs_error"].mean()
r12, p12 = stats.pearsonr(d12["actual_age"], d12["predicted_age"])
rep = d12.loc[d12["abs_error"].idxmin(), "equation_readable"]
rows.append({
    "script_name": "12_nested_gplearn_top3.py",
    "rigor_tier": "fully_rigorous",
    "feature_selection_method": "per_fold",
    "n_features": 3, "N": len(d12), "MAE": round(mae12, 2), "pearson_r": round(r12, 3), "p_value": round(p12, 4),
    "equation_or_note": "gplearn, EXACTLY top-3 by rank every fold (no threshold gate). Worst result "
                        "of the entire session -- r essentially zero. Same 3 features as the fixed-3 "
                        "runs dominate selection (peak_roll_walkingincline, rom_roll_ratio_stepforward_walk, "
                        "stride_timing_cv_walking, 23/29 folds) yet still fails, likely because this "
                        "script does not standardize features before fitting (unlike 09).",
})

# ── 13_final_model_constrained.py ────────────────────────────────────────────
d13 = pd.read_csv(r"final_model_constrained_seeds.csv")
valid13 = d13[d13["passes_ceiling"]]
best13 = valid13.loc[valid13["mae"].idxmin()]
p13 = p_from_r(best13["r"], 29)
rows.append({
    "script_name": "13_final_model_constrained.py",
    "rigor_tier": "not_validated",
    "feature_selection_method": "whole_sample_no_holdout",
    "n_features": 8, "N": 29,
    "MAE": round(best13["mae"], 2), "pearson_r": round(best13["r"], 3), "p_value": round(p13, 4),
    "equation_or_note": f"NOT cross-validated -- fit on all 29. Heavy parsimony (0.5) collapsed to "
                        f"{int(best13['n_nodes'])}-node equation: '{best13['equation_readable']}'. "
                        f"No fitted coefficients on raw feature -- likely a numeric-scale coincidence "
                        f"(raw angle value in same numeric range as age in years), not a real relationship.",
})

# ── Assemble, sort, save ──────────────────────────────────────────────────────
table = pd.DataFrame(rows)
tier_order = {"fully_rigorous": 0, "partial_leaky": 1, "not_validated": 2, "not_yet_run": 3}
table["_sort"] = table["rigor_tier"].map(tier_order)
table = table.sort_values(["_sort", "MAE"]).drop(columns="_sort").reset_index(drop=True)

table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}\n")

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)
print(table.to_string(index=False))

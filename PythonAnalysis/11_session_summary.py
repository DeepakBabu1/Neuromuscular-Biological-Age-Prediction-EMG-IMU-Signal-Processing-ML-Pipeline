"""
Consolidated master comparison across every symbolic/linear regression LOOCV
run executed this session. Reads ONLY already-saved CSV/log outputs -- no
models are re-run here.

Data provenance for each row (see this file's printed notes for details on
two discrepancies found while assembling this table):
  - loocv_predictions.csv              -> 08_nested_loocv.py (nested gplearn)
  - plots/loocv_summary.csv            -> 03_symbolic_regression.py (8-feature, leaky)
  - loocv_results_v2.csv               -> 09_improved_model.py (fixed 3-feature, gplearn + linear)
  - loocv_reselected_features.csv      -> 10_nested_linear_regression.py (nested linear)
  - plots/model_comparison.csv         -> STALE (dated 2026-06-06, predates this
                                           session's feature-set changes; NOT used)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\session_summary_all_runs.csv"
OUT_PNG = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\session_summary_all_runs.png"

BASELINE_MAE = 13.97

rows = []

# ── Run 1/5: nested gplearn, per-fold reselection from 33 candidates ────────
df15 = pd.read_csv(r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_predictions.csv")
mae = df15["abs_error"].mean()
rmse = np.sqrt(((df15["predicted_age"] - df15["actual_age"]) ** 2).mean())
r, p = stats.pearsonr(df15["actual_age"], df15["predicted_age"])
rows.append({
    "Run": "Run 1/5: Nested gplearn (33->per-fold)",
    "N_features": "10-12 (varies/fold)",
    "Feature_selection": "Per-fold nested (leakage-free)",
    "Model": "gplearn",
    "N": len(df15), "MAE": round(mae, 2), "RMSE": round(rmse, 2),
    "Pearson_r": round(r, 3), "p_value": round(p, 4),
    "Beats_baseline": mae < BASELINE_MAE, "Significant": p < 0.05,
})

# ── Run 2: 8-feature pruned set, fixed/leaky selection, gplearn ─────────────
df2 = pd.read_csv(r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots\loocv_summary.csv")
rows.append({
    "Run": "Run 2: 8-feature gplearn (leaky)",
    "N_features": 8,
    "Feature_selection": "Whole-sample (fixed, leaky)",
    "Model": "gplearn",
    "N": int(df2["n_participants"].iloc[0]),
    "MAE": df2["mae_years"].iloc[0], "RMSE": df2["rmse_years"].iloc[0],
    "Pearson_r": df2["pearson_r"].iloc[0], "p_value": df2["pearson_p"].iloc[0],
    "Beats_baseline": df2["mae_years"].iloc[0] < BASELINE_MAE,
    "Significant": df2["pearson_p"].iloc[0] < 0.05,
})

# ── Run 3: baseline comparison (gplearn/linear/ridge/lasso) -- MISSING ──────
# plots/model_comparison.csv is dated 2026-06-06, predates every feature-set
# change made this session (old 12-feature set, baseline MAE=14.53 not 13.97,
# and a nonsensical Linear-all-features MAE of 1301.67). 05_baseline_comparison.py
# was updated to point at feature_table_full.csv + the pruned 8 features, but
# was never actually re-run afterward. Per instruction, this run is reported
# as MISSING rather than substituting the stale file.

# ── Run 4a/4b: fixed 3-feature set, gplearn and linear regression ───────────
df4 = pd.read_csv(r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_results_v2.csv")
gp_mae = df4["gp_abs_error"].mean()
gp_rmse = np.sqrt(((df4["gp_predicted_age"] - df4["actual_age"]) ** 2).mean())
gp_r, gp_p = stats.pearsonr(df4["actual_age"], df4["gp_predicted_age"])
rows.append({
    "Run": "Run 4a: Fixed 3-feature gplearn (leaky)",
    "N_features": 3,
    "Feature_selection": "Whole-sample (fixed, leaky)",
    "Model": "gplearn",
    "N": len(df4), "MAE": round(gp_mae, 2), "RMSE": round(gp_rmse, 2),
    "Pearson_r": round(gp_r, 3), "p_value": round(gp_p, 4),
    "Beats_baseline": gp_mae < BASELINE_MAE, "Significant": gp_p < 0.05,
})

lin_mae = df4["linear_abs_error"].mean()
lin_rmse = np.sqrt(((df4["linear_predicted_age"] - df4["actual_age"]) ** 2).mean())
lin_r, lin_p = stats.pearsonr(df4["actual_age"], df4["linear_predicted_age"])
rows.append({
    "Run": "Run 4b: Fixed 3-feature Linear (leaky)",
    "N_features": 3,
    "Feature_selection": "Whole-sample (fixed, leaky)",
    "Model": "Linear regression",
    "N": len(df4), "MAE": round(lin_mae, 2), "RMSE": round(lin_rmse, 2),
    "Pearson_r": round(lin_r, 3), "p_value": round(lin_p, 4),
    "Beats_baseline": lin_mae < BASELINE_MAE, "Significant": lin_p < 0.05,
})

# ── Run 6: nested linear regression, per-fold reselection ───────────────────
df6 = pd.read_csv(r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_reselected_features.csv")
mae6 = df6["abs_error"].mean()
rmse6 = np.sqrt(((df6["predicted_age"] - df6["actual_age"]) ** 2).mean())
r6, p6 = stats.pearsonr(df6["actual_age"], df6["predicted_age"])
rows.append({
    "Run": "Run 6: Nested Linear (33->per-fold)",
    "N_features": "10-12 (varies/fold)",
    "Feature_selection": "Per-fold nested (leakage-free)",
    "Model": "Linear regression",
    "N": len(df6), "MAE": round(mae6, 2), "RMSE": round(rmse6, 2),
    "Pearson_r": round(r6, 3), "p_value": round(p6, 4),
    "Beats_baseline": mae6 < BASELINE_MAE, "Significant": p6 < 0.05,
})

# ── Assemble, sort, save ──────────────────────────────────────────────────────
table = pd.DataFrame(rows)
table["_leaky_sort"] = table["Feature_selection"].str.contains("leaky").astype(int)
table = table.sort_values(["_leaky_sort", "MAE"]).drop(columns="_leaky_sort").reset_index(drop=True)
table.to_csv(OUT_CSV, index=False)

print("MASTER TABLE")
print(table.to_string(index=False))
print(f"\nSaved -> {OUT_CSV}")
print("\nNOTE: Run 1 and Run 5, as specified, both describe 'nested gplearn LOOCV' and "
      "resolve to the SAME saved file (loocv_predictions.csv, from 08_nested_loocv.py) -- "
      "there is no separate saved output for an '11-feature whole-sample-then-LOOCV' gplearn "
      "run distinct from this one, so they are reported once, not as two different numbers.")
print("NOTE: Run 3 (gplearn vs linear vs ridge vs lasso vs baseline, 05_baseline_comparison.py) "
      "is MISSING -- the only saved file (plots/model_comparison.csv) is dated 2026-06-06, "
      "predates this session's feature-set/pipeline changes entirely, and is not used.")

# ── Plot ──────────────────────────────────────────────────────────────────────
leaky_color = "#d62728"
nested_color = "#1f77b4"
colors = [nested_color if "leakage-free" in fs else leaky_color for fs in table["Feature_selection"]]

fig, ax = plt.subplots(figsize=(11, 6.5))
bars = ax.bar(range(len(table)), table["MAE"], color=colors, edgecolor="white")
ax.axhline(BASELINE_MAE, color="black", linestyle="--", linewidth=1.2,
           label=f"Trivial baseline (predict-mean) = {BASELINE_MAE} yr")
ax.set_xticks(range(len(table)))
ax.set_xticklabels(table["Run"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("LOOCV MAE (years)")
ax.set_title("Model Comparison Across All Runs -- Effect of Feature Selection Leakage")

for i, v in enumerate(table["MAE"]):
    ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=leaky_color, label="Leaky feature selection (whole-sample/fixed)"),
    Patch(facecolor=nested_color, label="Leakage-free (per-fold nested) feature selection"),
]
ax.legend(handles=legend_elements + [ax.get_lines()[0]], fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"Saved -> {OUT_PNG}")

# ── Written summary ───────────────────────────────────────────────────────────
leaky_rows = table[table["Feature_selection"].str.contains("leaky")]
nested_rows = table[table["Feature_selection"].str.contains("leakage-free")]

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{len(leaky_rows)} runs used leaky (whole-sample or fixed) feature selection; "
      f"{len(nested_rows)} runs used leakage-free per-fold nested feature selection. "
      f"(1 planned run -- Run 3, baseline model comparison -- is missing; see note above.)")
print(f"Leaky runs MAE range: {leaky_rows['MAE'].min():.2f} - {leaky_rows['MAE'].max():.2f} years, "
      f"{leaky_rows['Significant'].sum()}/{len(leaky_rows)} statistically significant (p<0.05).")
print(f"Leakage-free runs MAE range: {nested_rows['MAE'].min():.2f} - {nested_rows['MAE'].max():.2f} years, "
      f"{nested_rows['Significant'].sum()}/{len(nested_rows)} statistically significant (p<0.05).")
best_nested_mae = nested_rows["MAE"].min()
print(f"Conclusion: the best leakage-free result ({best_nested_mae:.2f} yr) is only marginally "
      f"below the trivial baseline ({BASELINE_MAE} yr) and is NOT statistically significant -- "
      f"no leakage-free run in this session achieves both a meaningful and significant "
      f"improvement over predicting the mean age.")

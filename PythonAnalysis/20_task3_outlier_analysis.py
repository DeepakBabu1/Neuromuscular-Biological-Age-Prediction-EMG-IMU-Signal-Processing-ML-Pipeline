"""
Task 3: RMSE/MAE ratio for all models, and gplearn-specific outlier flagging
(error > 2x gplearn's own MAE), using already-saved predictions.
"""
import pandas as pd
import numpy as np

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
GPLEARN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_predictions.csv"
OUT_RATIO = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\rmse_mae_ratio.csv"
OUT_OUTLIERS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\gplearn_outliers.csv"

df = pd.read_csv(IN_CSV)
actual = df["actual_age"].values.astype(float)

models = {
    "Ridge": df["ridge_pred"].values.astype(float),
    "Lasso": df["lasso_pred"].values.astype(float),
    "Linear": df["linear_pred"].values.astype(float),
    "gplearn": df["gplearn_pred"].values.astype(float),
    "Baseline": df["baseline_pred"].values.astype(float),
}

rows = []
for name, pred in models.items():
    err = np.abs(pred - actual)
    mae = float(err.mean())
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    rows.append({"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                 "RMSE_over_MAE": round(rmse / mae, 4)})

ratio_table = pd.DataFrame(rows).sort_values("RMSE_over_MAE", ascending=False)
ratio_table.to_csv(OUT_RATIO, index=False)
print(f"Saved -> {OUT_RATIO}\n")
print(ratio_table.to_string(index=False))

# ── gplearn outliers: error > 2x gplearn's own MAE ───────────────────────────
gplearn_mae = np.abs(models["gplearn"] - actual).mean()
threshold = 2 * gplearn_mae

g_detail = pd.read_csv(GPLEARN_CSV)  # has selected_features per fold
merged = df[["participant_id", "actual_age", "gplearn_pred", "gplearn_error", "group"]].merge(
    g_detail[["participant", "selected_features"]],
    left_on="participant_id", right_on="participant", how="left"
).drop(columns="participant")

outliers = merged[merged["gplearn_error"] > threshold].sort_values("gplearn_error", ascending=False)
outliers.to_csv(OUT_OUTLIERS, index=False)

print(f"\ngplearn MAE = {gplearn_mae:.4f}, outlier threshold (2x MAE) = {threshold:.4f}")
print(f"Saved -> {OUT_OUTLIERS}  ({len(outliers)} outlier participants)\n")
print(outliers.to_string(index=False))

"""
Task 2: age-regression bias quantification + Beheshti-style correction.

Bias correction detail: "each LOOCV fold's training set" is implemented as
the OTHER 28 participants' ALREADY-SAVED (actual_age, predicted_age) pairs
from the same nested LOOCV run being corrected -- this uses only saved
predictions, not a re-fit of the original Ridge/Lasso/Linear/gplearn models,
per the "do not re-run the original model fits" instruction. For each held-
out participant, predicted~actual is fit on the other 28 pairs, and that
fold-specific slope/intercept corrects the held-out participant's own
prediction. This keeps the correction itself leakage-free (participant i's
own point never contributes to its own correction).
"""
import numpy as np
import pandas as pd
from scipy import stats

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\bias_correction_results.csv"

df = pd.read_csv(IN_CSV)
actual = df["actual_age"].values.astype(float)
n = len(df)

models = {
    "Ridge": df["ridge_pred"].values.astype(float),
    "Lasso": df["lasso_pred"].values.astype(float),
    "Linear": df["linear_pred"].values.astype(float),
    "gplearn": df["gplearn_pred"].values.astype(float),
    "Baseline": df["baseline_pred"].values.astype(float),
}

rows = []
for name, pred in models.items():
    signed_error = pred - actual
    mae = float(np.mean(np.abs(signed_error)))
    rmse = float(np.sqrt(np.mean(signed_error ** 2)))
    r_before, p_before = stats.pearsonr(actual, pred)
    bias_r_before, bias_p_before = stats.pearsonr(actual, signed_error)

    row = {
        "model": name,
        "mae_before": round(mae, 4), "rmse_before": round(rmse, 4),
        "r_before": round(r_before, 4),
        "bias_corr_before": round(bias_r_before, 4), "bias_corr_p_before": round(bias_p_before, 4),
    }

    if name != "Baseline":
        corrected = np.empty(n)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            slope, intercept, _, _, _ = stats.linregress(actual[others], pred[others])
            corrected[i] = pred[i] - (slope * actual[i] + intercept - actual[i])

        signed_error_after = corrected - actual
        mae_after = float(np.mean(np.abs(signed_error_after)))
        rmse_after = float(np.sqrt(np.mean(signed_error_after ** 2)))
        r_after, _ = stats.pearsonr(actual, corrected)
        bias_r_after, bias_p_after = stats.pearsonr(actual, signed_error_after)

        row.update({
            "mae_after": round(mae_after, 4), "rmse_after": round(rmse_after, 4),
            "r_after": round(r_after, 4),
            "bias_corr_after": round(bias_r_after, 4), "bias_corr_p_after": round(bias_p_after, 4),
        })
    else:
        row.update({
            "mae_after": None, "rmse_after": None, "r_after": None,
            "bias_corr_after": None, "bias_corr_p_after": None,
        })
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}\n")
pd.set_option("display.width", 200)
print(table.to_string(index=False))

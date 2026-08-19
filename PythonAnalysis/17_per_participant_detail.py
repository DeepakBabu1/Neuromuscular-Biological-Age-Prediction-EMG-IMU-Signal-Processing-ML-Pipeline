"""
Per-participant prediction detail for 5 already-completed LOOCV runs.
Pulls actual saved predictions -- no models re-run.

Sources:
  Ridge/Lasso/Linear/Baseline (nested top-3)  -> baseline_comparison_v2.csv (05_baseline_comparison.py)
  gplearn (nested, 10-12 features per fold)   -> loocv_predictions.csv (08_nested_loocv.py)
  group (1=young, 2=older)                    -> feature_table_full.csv
"""

import pandas as pd

OUT_DIR = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

b = pd.read_csv(f"{OUT_DIR}\\baseline_comparison_v2.csv")
g = pd.read_csv(f"{OUT_DIR}\\loocv_predictions.csv")
groups = pd.read_csv(f"{OUT_DIR}\\feature_table_full.csv")[["participant", "group"]]
groups["group"] = groups["group"].astype("Int64")


def build_csv(df, pred_col, features_col, filename, features_present=True):
    out = pd.DataFrame({
        "participant_id": df["participant"],
        "actual_age": df["actual_age"],
        "predicted_age": df[pred_col],
    })
    out["absolute_error"] = (out["predicted_age"] - out["actual_age"]).abs()
    out = out.merge(groups, left_on="participant_id", right_on="participant", how="left").drop(columns="participant")
    if features_present:
        out["features_used_this_fold"] = df[features_col].str.replace(";", ", ", regex=False)
    else:
        out["features_used_this_fold"] = "(none -- predict-mean baseline uses no features)"

    out = out[["participant_id", "actual_age", "predicted_age", "absolute_error", "group", "features_used_this_fold"]]
    out = out.sort_values("absolute_error", ascending=False).reset_index(drop=True)

    mae = out["absolute_error"].mean()
    mean_row = pd.DataFrame([{
        "participant_id": "MEAN", "actual_age": None, "predicted_age": None,
        "absolute_error": mae, "group": None, "features_used_this_fold": None,
    }])
    out = pd.concat([out, mean_row], ignore_index=True)
    out.to_csv(f"{OUT_DIR}\\{filename}", index=False)
    print(f"Saved -> {filename}  (MAE={mae:.4f})")
    return mae


build_csv(b, "ridge_pred", "selected_features", "per_participant_ridge.csv")
build_csv(b, "lasso_pred", "selected_features", "per_participant_lasso.csv")
build_csv(b, "linear_pred", "selected_features", "per_participant_linear.csv")
build_csv(g, "predicted_age", "selected_features", "per_participant_gplearn.csv")
build_csv(b, "baseline_pred", None, "per_participant_baseline.csv", features_present=False)

# ── Combined file ─────────────────────────────────────────────────────────────
combined = b[["participant", "actual_age"]].copy()
combined = combined.merge(groups, on="participant", how="left")
combined["ridge_pred"] = b["ridge_pred"]
combined["ridge_error"] = (b["ridge_pred"] - b["actual_age"]).abs()
combined["lasso_pred"] = b["lasso_pred"]
combined["lasso_error"] = (b["lasso_pred"] - b["actual_age"]).abs()
combined["linear_pred"] = b["linear_pred"]
combined["linear_error"] = (b["linear_pred"] - b["actual_age"]).abs()
combined["baseline_pred"] = b["baseline_pred"]
combined["baseline_error"] = (b["baseline_pred"] - b["actual_age"]).abs()

g_indexed = g.set_index("participant")
combined["gplearn_pred"] = combined["participant"].map(g_indexed["predicted_age"])
combined["gplearn_error"] = (combined["gplearn_pred"] - combined["actual_age"]).abs()

combined = combined.rename(columns={"participant": "participant_id"})
combined = combined[["participant_id", "actual_age", "group",
                      "ridge_pred", "ridge_error",
                      "lasso_pred", "lasso_error",
                      "linear_pred", "linear_error",
                      "gplearn_pred", "gplearn_error",
                      "baseline_pred", "baseline_error"]]
combined = combined.sort_values("participant_id", ascending=True).reset_index(drop=True)
combined.to_csv(f"{OUT_DIR}\\per_participant_all_runs.csv", index=False)
print(f"\nSaved -> per_participant_all_runs.csv  ({len(combined)} rows)\n")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(combined.round(2).to_string(index=False))

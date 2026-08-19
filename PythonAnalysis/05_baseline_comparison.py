"""
Step 5 (rewritten): Baseline model comparison, fully nested LOOCV.

Rewritten from scratch this session because the original version was never
re-run after the pipeline corrections and feature-set changes -- the only
prior saved output (plots/model_comparison.csv) predates this session
entirely and used a different, since-corrected feature set.

All four models below use PROPERLY NESTED LOOCV:
  - Trivial baseline: per-fold prediction = mean age of the 28 TRAINING
    participants only (not the whole-sample mean, which would leak the
    held-out participant's own age into their own "prediction").
  - Linear / Ridge(alpha=10.0) / Lasso(alpha=1.0): feature selection is
    re-derived every fold from the 28 training participants only -- exactly
    top 3 features by |Spearman rho| (same mechanism as
    12_nested_gplearn_top3.py: ranked, not threshold-gated, so model
    complexity stays fixed at 3 features every fold even though which 3
    features they are can vary fold to fold).

Interpretation note: the task specified "the same 3-feature set as
08_nested_loocv.py" by name (peak_roll_walkingincline,
stride_timing_cv_walking, rom_roll_ratio_stepforward_walk), but also
required "nested feature selection" for Ridge/Lasso. 08_nested_loocv.py
itself actually selects 10-12 features per fold (threshold-based), not a
fixed 3 -- so the named triple more closely matches what
12_nested_gplearn_top3.py's per-fold TOP-3 selection converges to in most
folds. Given the task's opening line requires "feature selection re-derived
inside each fold" for everything in this comparison, all three models here
use genuine per-fold top-3 reselection (12's mechanism), not a hard-coded
fixed list -- so the specific 3 features can and do vary slightly by fold,
exactly as they did in 12_nested_gplearn_top3.py.
"""

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LinearRegression, Ridge, Lasso

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\baseline_comparison_v2.csv"

TOP_N = 3
MIN_N_FOR_CORR = 5

df = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["age"]).reset_index(drop=True)
N = len(df)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}), top-{TOP_N} reselected every fold\n")


def select_top3(train_df: pd.DataFrame):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()
    return rho_df.sort_values("abs_rho", ascending=False).head(TOP_N)["feature"].tolist()


models = {
    "baseline": None,
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=10.0),
    "lasso": Lasso(alpha=1.0),
}

preds = {name: np.full(N, np.nan) for name in models}
fold_selected_features = []

for i in range(N):
    test_row = df.iloc[[i]]
    train_df = df.drop(index=df.index[i])
    y_train = train_df["age"].values.astype(float)
    y_test = float(test_row["age"].values[0])

    # Baseline: training-only mean (properly nested -- excludes the held-out
    # participant's own age from the "prediction").
    preds["baseline"][i] = y_train.mean()

    # Nested top-3 feature selection, training data only.
    selected = select_top3(train_df)
    fold_selected_features.append(";".join(selected))

    train_medians = train_df[selected].median()
    X_train = train_df[selected].fillna(train_medians).values.astype(float)
    X_test = test_row[selected].fillna(train_medians).values.astype(float)

    for name, model in models.items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        preds[name][i] = float(model.predict(X_test)[0])

actual = df["age"].values.astype(float)

results = []
for name in models:
    p = preds[name]
    mae = float(np.mean(np.abs(p - actual)))
    rmse = float(np.sqrt(np.mean((p - actual) ** 2)))
    if np.std(p) > 1e-9:
        r, pval = stats.pearsonr(actual, p)
    else:
        r, pval = 0.0, 1.0
    results.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                     "Pearson_r": round(r, 3), "p_value": round(pval, 4)})

results_df = pd.DataFrame(results)
print("MODEL COMPARISON (all nested LOOCV, top-3 features re-selected every fold except baseline)")
print(results_df.to_string(index=False))

# Save per-participant predictions too, alongside the summary.
per_fold = pd.DataFrame({
    "participant": df["participant"].values,
    "actual_age": actual,
    "baseline_pred": preds["baseline"],
    "linear_pred": preds["linear"],
    "ridge_pred": preds["ridge"],
    "lasso_pred": preds["lasso"],
    "selected_features": fold_selected_features,
})
per_fold.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

summary_path = OUT_CSV.replace(".csv", "_summary.csv")
results_df.to_csv(summary_path, index=False)
print(f"Saved -> {summary_path}")

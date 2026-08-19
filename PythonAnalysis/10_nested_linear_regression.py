"""
Nested LOOCV with linear regression, feature selection refit INSIDE every
fold -- the leakage-free counterpart to 09_improved_model.py's fixed
3-feature linear regression result (MAE=12.28, r=0.451, p=0.014), which
selected its 3 features from whole-sample correlation (a disclosed leak).

For each of the 29 folds:
  1. Train = 28 participants, test = the held-out participant (fully excluded)
  2. Spearman rho vs age computed on the 28 training participants ONLY,
     over the same 33 candidate features used in 07/08 (all columns except
     participant/age/group/dmci_*)
  3. Keep features with |rho| > 0.3; if fewer than 2 pass, take the top 3
     by |rho| regardless of threshold
  4. Impute missing values with the training median
  5. StandardScaler fit on training data only, applied to both train and test
  6. Fit LinearRegression on the (scaled, imputed) training features
  7. Predict the held-out participant

Output: loocv_reselected_features.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_reselected_features.csv"

RHO_THRESHOLD  = 0.3
TOP_N_FALLBACK = 3
MIN_N_FOR_CORR = 5

# Previous fixed-3-feature result, for direct comparison at the end.
PREV_MAE, PREV_R, PREV_P = 12.28, 0.451, 0.0140

df = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]

df = df.dropna(subset=["age"]).reset_index(drop=True)
N = len(df)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}\n")


def select_features(train_df: pd.DataFrame):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < MIN_N_FOR_CORR:
            rows.append((col, np.nan))
            continue
        rho, _ = stats.spearmanr(valid["age"], valid[col])
        rows.append((col, rho))

    rho_df = pd.DataFrame(rows, columns=["feature", "rho"])
    rho_df["abs_rho"] = rho_df["rho"].abs()

    passing = rho_df[rho_df["abs_rho"] > RHO_THRESHOLD].dropna(subset=["abs_rho"])
    if len(passing) < 2:
        passing = rho_df.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(TOP_N_FALLBACK)

    return passing.sort_values("abs_rho", ascending=False)["feature"].tolist()


preds = np.full(N, np.nan)
fold_records = []
selection_counts = {f: 0 for f in CANDIDATE_FEATURES}

print("Running nested LOOCV (linear regression, feature selection refit inside every fold)...")
for i in range(N):
    test_row = df.iloc[[i]]
    train_df = df.drop(index=df.index[i])

    selected = select_features(train_df)
    for f in selected:
        selection_counts[f] += 1

    train_medians = train_df[selected].median()
    X_train_raw = train_df[selected].fillna(train_medians).values.astype(float)
    X_test_raw = test_row[selected].fillna(train_medians).values.astype(float)
    y_train = train_df["age"].values.astype(float)
    y_test = float(test_row["age"].values[0])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = float(model.predict(X_test)[0])
    preds[i] = pred

    fold_records.append({
        "fold": i,
        "participant": test_row["participant"].values[0],
        "actual_age": y_test,
        "predicted_age": pred,
        "abs_error": abs(pred - y_test),
        "n_features_selected": len(selected),
        "selected_features": ";".join(selected),
    })

    print(f"  [{i+1:2d}/{N}] participant={test_row['participant'].values[0]:<4} "
          f"actual={y_test:.0f}  pred={pred:.1f}  err={abs(pred-y_test):.1f}  "
          f"n_feat={len(selected)}  features={selected}")

results_df = pd.DataFrame(fold_records)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

# ── Performance ───────────────────────────────────────────────────────────────
actual = results_df["actual_age"].values
mae = float(np.mean(np.abs(preds - actual)))
r, p = stats.pearsonr(actual, preds)
rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
baseline_mae = float(np.mean(np.abs(actual - actual.mean())))

print(f"\n{'='*60}")
print("NESTED LINEAR REGRESSION -- LOOCV RESULTS")
print(f"{'='*60}")
print(f"  MAE            : {mae:.2f} years")
print(f"  RMSE           : {rmse:.2f} years")
print(f"  Pearson r      : {r:.3f}  (p={p:.4f})")
print(f"  Baseline MAE   : {baseline_mae:.2f} years (predict-mean)")

# ── Feature selection stability ───────────────────────────────────────────────
stability = pd.Series(selection_counts).sort_values(ascending=False)
stable_thresh = N / 2.0

print(f"\n{'='*60}")
print(f"FEATURE SELECTION STABILITY (selected in how many of {N} folds)")
print(f"{'='*60}")
for feat, count in stability.items():
    if count > 0:
        tag = "  <-- STABLE (>50%)" if count > stable_thresh else ""
        print(f"  {feat:35s}  {count:2d}/{N}{tag}")

# ── Direct comparison ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMPARISON: fixed 3-feature (leaky selection) vs. nested (nested selection)")
print(f"{'='*60}")
print(f"  Fixed 3-feature linear regression : MAE={PREV_MAE:.2f}  r={PREV_R:.3f}  p={PREV_P:.4f}")
print(f"  Nested (re-selected per fold)      : MAE={mae:.2f}  r={r:.3f}  p={p:.4f}")
delta = mae - PREV_MAE
print(f"  Delta MAE (nested - fixed)         : {delta:+.2f} years "
      f"({'nested is WORSE' if delta > 0 else 'nested is BETTER'})")
print(f"{'='*60}")

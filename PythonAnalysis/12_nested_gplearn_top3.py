"""
Nested gplearn LOOCV, exactly top-3 features per fold (by |Spearman rho|,
ranked, NOT threshold-gated) -- so model complexity (3 features) is held
constant across all 29 folds, unlike 08_nested_loocv.py (which used the
|rho|>0.3 threshold with a top-3 fallback only when fewer than 2 passed).

For each fold: select top 3 from the 28 training participants only, impute
with training median, fit gplearn (3-seed ensemble, median prediction),
predict the held-out participant.

Output: loocv_gplearn_top3_perfold.csv
"""

import os
import re
import time
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from gplearn.genetic import SymbolicRegressor

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\loocv_gplearn_top3_perfold.csv"

TOP_N = 3
MIN_N_FOR_CORR = 5
N_SEEDS = 3
RANDOM_STATE = 42

GP_PARAMS = dict(
    population_size=3000,
    generations=30,
    tournament_size=20,
    function_set=("add", "sub", "mul", "div"),
    parsimony_coefficient=0.05,
    const_range=(-10.0, 10.0),
    metric="mean absolute error",
)

# Comparison numbers supplied for this run, quoted exactly as given.
COMPARISON = {
    "Run 2 (nested, 8 features)":            dict(mae=13.72, r=0.160, p=0.41),
    "Run 3 (nested, fixed 3)":                dict(mae=14.26, r=None, p=None),
    "Run 5 (nested linear, 11-12 features)":  dict(mae=15.35, r=0.125, p=0.518),
    "Baseline (predict mean)":                dict(mae=13.97, r=None, p=None),
}

df = pd.read_csv(DATA)
EXCLUDE = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["age"]).reset_index(drop=True)
N = len(df)

print(f"Participants: {N}")
print(f"Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}\n")


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
    top3 = rho_df.sort_values("abs_rho", ascending=False).head(TOP_N)
    return top3["feature"].tolist(), dict(zip(rho_df["feature"], rho_df["rho"]))


def make_gp_model(seed):
    return SymbolicRegressor(
        population_size=GP_PARAMS["population_size"],
        generations=GP_PARAMS["generations"],
        tournament_size=GP_PARAMS["tournament_size"],
        function_set=GP_PARAMS["function_set"],
        parsimony_coefficient=GP_PARAMS["parsimony_coefficient"],
        const_range=GP_PARAMS["const_range"],
        metric=GP_PARAMS["metric"],
        n_jobs=1,
        verbose=0,
        random_state=seed,
    )


fold_records = []
selection_counts = {f: 0 for f in CANDIDATE_FEATURES}

print("Running nested gplearn LOOCV (exactly top-3 features per fold)...")
t0 = time.time()
for i in range(N):
    test_row = df.iloc[[i]]
    train_df = df.drop(index=df.index[i])

    selected, rho_lookup = select_top3(train_df)
    for f in selected:
        selection_counts[f] += 1

    train_medians = train_df[selected].median()
    X_train = train_df[selected].fillna(train_medians).values.astype(float)
    X_test = test_row[selected].fillna(train_medians).values.astype(float)
    y_train = train_df["age"].values.astype(float)
    y_test = float(test_row["age"].values[0])

    fold_preds, fold_eqs = [], []
    for s in range(N_SEEDS):
        model = make_gp_model(seed=RANDOM_STATE + i * N_SEEDS + s)
        model.fit(X_train, y_train)
        fold_preds.append(float(model.predict(X_test)[0]))
        fold_eqs.append(str(model._program))
    pred = float(np.median(fold_preds))
    eq_idx = int(np.argmin(np.abs(np.array(fold_preds) - pred)))
    eq = fold_eqs[eq_idx]

    def readable_equation(eq_str, feature_names):
        for idx in range(len(feature_names) - 1, -1, -1):
            eq_str = eq_str.replace(f"X{idx}", feature_names[idx])
        return eq_str

    fold_records.append({
        "fold": i,
        "participant": test_row["participant"].values[0],
        "actual_age": y_test,
        "predicted_age": pred,
        "abs_error": abs(pred - y_test),
        "selected_features": ";".join(selected),
        "feature_1": selected[0], "feature_2": selected[1], "feature_3": selected[2],
        "equation": eq,
        "equation_readable": readable_equation(eq, selected),
    })

    elapsed_so_far = time.time() - t0
    print(f"  [{i+1:2d}/{N}] participant={test_row['participant'].values[0]:<4} "
          f"actual={y_test:.0f}  pred={pred:.1f}  err={abs(pred-y_test):.1f}  "
          f"features={selected}  (elapsed {elapsed_so_far:.0f}s, "
          f"avg {elapsed_so_far/(i+1):.1f}s/fold)")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold average).")

results_df = pd.DataFrame(fold_records)
results_df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

# ── Performance ───────────────────────────────────────────────────────────────
actual = results_df["actual_age"].values
preds = results_df["predicted_age"].values
mae = float(np.mean(np.abs(preds - actual)))
rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
r, p = stats.pearsonr(actual, preds)
baseline_mae = float(np.mean(np.abs(actual - actual.mean())))

print(f"\n{'='*65}")
print("RESULTS: nested gplearn, exactly top-3 features per fold")
print(f"{'='*65}")
print(f"  N          : {N}")
print(f"  MAE        : {mae:.2f} years")
print(f"  RMSE       : {rmse:.2f} years")
print(f"  Pearson r  : {r:.3f}  (p={p:.4f})")

print(f"\nCOMPARISON:")
print(f"  {'This run (top-3 per fold)':40s}: MAE={mae:.2f}  r={r:.3f}  p={p:.4f}")
for name, vals in COMPARISON.items():
    r_str = f"{vals['r']:.3f}" if vals['r'] is not None else "n/a"
    p_str = f"{vals['p']:.4f}" if vals['p'] is not None else "n/a"
    print(f"  {name:40s}: MAE={vals['mae']:.2f}  r={r_str}  p={p_str}")

# ── Feature stability ─────────────────────────────────────────────────────────
stability = pd.Series(selection_counts).sort_values(ascending=False)
print(f"\n{'='*65}")
print(f"FEATURE STABILITY (how many of {N} folds each feature was in the top 3)")
print(f"{'='*65}")
for feat, count in stability.items():
    if count > 0:
        print(f"  {feat:35s}  {count:2d}/{N}")

top_feat, top_count = stability.index[0], stability.iloc[0]
same3_folds = (results_df[["feature_1", "feature_2", "feature_3"]]
               .apply(lambda row: frozenset(row), axis=1))
same3_counts = Counter(same3_folds)
most_common_set, most_common_freq = same3_counts.most_common(1)[0]
print(f"\nMost frequent single feature: '{top_feat}' in {top_count}/{N} folds.")
print(f"Most frequent EXACT top-3 SET: {sorted(most_common_set)} in {most_common_freq}/{N} folds.")
if most_common_freq > N / 2:
    print("-> The SAME 3 features were selected in a majority of folds (stable selection).")
else:
    print("-> The exact top-3 set VARIED across folds (not dominated by one fixed triple).")

# ── Equation stability ────────────────────────────────────────────────────────
def normalize_equation(eq: str) -> str:
    return re.sub(r"-?\d+\.\d+|-?\d+", "C", eq)

templates = [normalize_equation(e) for e in results_df["equation"]]
template_counts = Counter(templates)
most_common_template, freq = template_counts.most_common(1)[0]

print(f"\n{'='*65}")
print("EQUATION STABILITY")
print(f"{'='*65}")
print(f"Most common equation structure occurred in {freq}/{N} folds.")
matching_idx = [i for i, t in enumerate(templates) if t == most_common_template]
rep_row = results_df.iloc[matching_idx].loc[results_df.iloc[matching_idx]["abs_error"].idxmin()]
print(f"Representative equation (fold {int(rep_row['fold'])}, lowest error in that group):")
print(f"  Exact:    {rep_row['equation']}")
print(f"  Readable: {rep_row['equation_readable']}")
print(f"{'='*65}")

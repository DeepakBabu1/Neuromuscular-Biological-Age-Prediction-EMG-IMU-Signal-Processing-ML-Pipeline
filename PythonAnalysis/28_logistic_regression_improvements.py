"""
Systematic attempt to improve nested-LOOCV logistic regression classification
(young=1 vs old=2) beyond the original baseline (accuracy=0.621, AUC=0.678,
permutation p=0.032, using a FIXED 3-feature list).

IMPORTANT METHODOLOGICAL NOTE, discovered while building this script:
The ORIGINAL baseline (23_task6_classification.py) did NOT use per-fold
feature reselection -- it used a fixed list (peak_roll_walkingincline,
stride_timing_cv_walking, rom_roll_ratio_stepforward_walk), chosen once from
an earlier AGE-regression analysis on the whole sample, and never re-derived
against the actual classification target (group). This is a real, if
narrower, instance of the whole-sample-selection concern raised throughout
this session.

Since this task's instructions explicitly require "nested LOOCV and per-fold
feature reselection throughout... never select features on the whole sample"
for every improvement tested, a CORRECTED baseline is built first (nested
top-3 reselection, ranked by |Spearman rho| against the actual target,
group -- not age) and used as the true reference point for Improvements 1-5.
Both the original and corrected baselines are reported for transparency.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\logistic_regression_improvement_attempts.csv"

EXCLUDE = {"participant", "age", "group"}
df = pd.read_csv(DATA)
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
ORIGINAL_FIXED = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

print(f"N = {N}")


# ── Feature selection ─────────────────────────────────────────────────────────
def select_features(train_df, n=3, threshold=None, target_col="group"):
    rows = []
    for col in CANDIDATE_FEATURES:
        valid = train_df[[target_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 5:
            continue
        rho, _ = stats.spearmanr(valid[target_col], valid[col])
        rows.append((col, rho))
    rho_df = pd.DataFrame(rows, columns=["feature", "rho"]).dropna()
    rho_df["abs_rho"] = rho_df["rho"].abs()
    if threshold is None:
        return rho_df.sort_values("abs_rho", ascending=False).head(n)["feature"].tolist()
    passing = rho_df[rho_df["abs_rho"] > threshold]
    if len(passing) == 0:
        return rho_df.sort_values("abs_rho", ascending=False).head(1)["feature"].tolist()
    return passing.sort_values("abs_rho", ascending=False).head(n)["feature"].tolist()


def to_binary(group_array):
    return (np.asarray(group_array).astype(int) == 2).astype(int)


# ── Inner LOOCV over training fold (for hyperparameter/threshold selection) ──
def inner_loocv_proba(train_df, features, model_kwargs):
    idx_list = train_df.index.tolist()
    inner_actual, inner_proba = [], []
    for inner_test_idx in idx_list:
        inner_train = train_df.drop(index=inner_test_idx)
        inner_test = train_df.loc[[inner_test_idx]]
        med = inner_train[features].median()
        Xtr = inner_train[features].fillna(med).values.astype(float)
        ytr = to_binary(inner_train["group"].values)
        Xte = inner_test[features].fillna(med).values.astype(float)
        if len(np.unique(ytr)) < 2:
            inner_proba.append(0.5)
        else:
            clf = LogisticRegression(**model_kwargs)
            clf.fit(Xtr, ytr)
            classes = list(clf.classes_)
            old_idx = classes.index(1) if 1 in classes else 0
            inner_proba.append(clf.predict_proba(Xte)[0][old_idx])
        inner_actual.append(int(inner_test["group"].values[0] == 2))
    return np.array(inner_actual), np.array(inner_proba)


def select_best_C(train_df, features, C_grid, base_kwargs):
    best_C, best_acc = C_grid[0], -1
    for C in C_grid:
        kw = {**base_kwargs, "C": C}
        actual, proba = inner_loocv_proba(train_df, features, kw)
        acc = ((proba >= 0.5).astype(int) == actual).mean()
        if acc > best_acc:
            best_acc, best_C = acc, C
    return best_C


def select_best_threshold(train_df, features, model_kwargs, threshold_grid):
    actual, proba = inner_loocv_proba(train_df, features, model_kwargs)
    best_t, best_acc = 0.5, -1
    for t in threshold_grid:
        acc = ((proba >= t).astype(int) == actual).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


# ── Main nested LOOCV runner ──────────────────────────────────────────────────
def run_nested(group_labels, feature_kwargs, model_kwargs, decision_threshold=0.5,
               tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
               fixed_features=None, verbose=False):
    d = df.copy()
    d["group"] = group_labels
    pred_group = np.empty(N, dtype=int)
    pred_proba = np.empty(N)

    for i in range(N):
        train_df = d.drop(index=i)
        test_row = d.iloc[[i]]

        selected = fixed_features if fixed_features is not None else select_features(train_df, **feature_kwargs)

        med = train_df[selected].median()
        X_train = train_df[selected].fillna(med).values.astype(float)
        y_train = to_binary(train_df["group"].values)
        X_test = test_row[selected].fillna(med).values.astype(float)

        mk = dict(model_kwargs)
        if tune_C:
            mk["C"] = select_best_C(train_df, selected, C_grid, model_kwargs)

        clf = LogisticRegression(**mk)
        clf.fit(X_train, y_train)
        classes = list(clf.classes_)
        old_idx = classes.index(1) if 1 in classes else 0
        proba_old = clf.predict_proba(X_test)[0][old_idx]

        thresh = decision_threshold
        if tune_threshold:
            thresh = select_best_threshold(train_df, selected, mk, threshold_grid)

        pred_group[i] = 2 if proba_old >= thresh else 1
        pred_proba[i] = proba_old

        if verbose:
            print(f"  [{i+1:2d}/{N}] actual={int(test_row['group'].values[0])} pred={pred_group[i]} "
                  f"proba={proba_old:.3f} thresh={thresh:.2f} feat={selected}")

    return pred_group, pred_proba


def compute_metrics(pred_group, pred_proba, actual_group):
    accuracy = float((pred_group == actual_group).mean())
    cm = confusion_matrix(actual_group, pred_group, labels=[1, 2])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(actual_group, pred_proba)
    return accuracy, sensitivity, specificity, auc, cm


def permutation_test(feature_kwargs, model_kwargs, decision_threshold=0.5,
                      tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
                      fixed_features=None, n_perms=500, observed_auc=None):
    rng = np.random.default_rng(42)
    null_aucs = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = rng.permutation(df["group"].values)
        _, proba = run_nested(shuffled, feature_kwargs, model_kwargs, decision_threshold,
                               tune_C, C_grid, tune_threshold, threshold_grid, fixed_features)
        null_aucs.append(roc_auc_score((shuffled == 2).astype(int), proba))
    elapsed = time.time() - t0
    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return p_value, elapsed, null_aucs


actual_group = df["group"].values.astype(int)
results = []

# ── Row 1: ORIGINAL baseline (fixed features, as previously reported) ───────
print("\n" + "=" * 70)
print("ROW 1: ORIGINAL BASELINE (fixed features, as previously reported)")
print("=" * 70)
t0 = time.time()
pred1, proba1 = run_nested(df["group"].values, {}, {}, fixed_features=ORIGINAL_FIXED)
acc1, sens1, spec1, auc1, cm1 = compute_metrics(pred1, proba1, actual_group)
print(f"accuracy={acc1:.4f} sensitivity={sens1:.4f} specificity={spec1:.4f} AUC={auc1:.4f}  "
      f"(matches previously reported 0.621/0.678: {abs(acc1-0.6207)<0.001})")
results.append(dict(attempt="Row 1: Original baseline (fixed features)", accuracy=acc1, sensitivity=sens1,
                     specificity=spec1, AUC=auc1, permutation_p=0.032, n_features=3,
                     wilcoxon_p_vs_baseline=None, beats_baseline=None,
                     notes="Reproduces the previously reported result exactly; permutation p=0.032 from earlier run, not recomputed here."))

# ── Row 2: CORRECTED baseline (nested top-3, selected against GROUP, not age) ─
print("\n" + "=" * 70)
print("ROW 2: CORRECTED BASELINE (nested top-3 reselection vs GROUP, per fold)")
print("=" * 70)
BASE_FEATURE_KW = dict(n=3, threshold=None, target_col="group")
BASE_MODEL_KW = {}
t0 = time.time()
pred2, proba2 = run_nested(df["group"].values, BASE_FEATURE_KW, BASE_MODEL_KW, verbose=True)
elapsed2 = time.time() - t0
acc2, sens2, spec2, auc2, cm2 = compute_metrics(pred2, proba2, actual_group)
print(f"\naccuracy={acc2:.4f} sensitivity={sens2:.4f} specificity={spec2:.4f} AUC={auc2:.4f}  "
      f"(run took {elapsed2:.2f}s)")

correct1 = (pred1 == actual_group).astype(int)
correct2 = (pred2 == actual_group).astype(int)
diff_12 = correct2 - correct1
if np.any(diff_12 != 0):
    wstat, wp_2v1 = stats.wilcoxon(correct2, correct1)
else:
    wp_2v1 = 1.0
print(f"Wilcoxon (corrected baseline vs original baseline), per-participant correctness: p={wp_2v1:.4f}")

p2, elapsed_p2, null2 = permutation_test(BASE_FEATURE_KW, BASE_MODEL_KW, observed_auc=auc2, n_perms=500)
print(f"Permutation test (500 shuffles, {elapsed_p2:.1f}s): p={p2:.4f}")

results.append(dict(attempt="Row 2: Corrected baseline (nested top-3 vs group)", accuracy=acc2, sensitivity=sens2,
                     specificity=spec2, AUC=auc2, permutation_p=round(p2, 4), n_features=3,
                     wilcoxon_p_vs_baseline=round(wp_2v1, 4), beats_baseline=bool(acc2 > acc1 and wp_2v1 < 0.05),
                     notes="TRUE reference point for Improvements 1-5 below (per-fold reselection against the actual "
                           "classification target, group -- unlike Row 1, which reused fixed age-derived features)."))
print(f"\n>>> All Improvements 1-5 below are compared against ROW 2 (the corrected baseline), not Row 1. <<<")

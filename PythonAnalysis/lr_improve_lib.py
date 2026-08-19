"""
Reusable functions for the logistic regression improvement attempts.
No top-level execution here -- safe to import without side effects.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
EXCLUDE = {"participant", "age", "group"}

_df_raw = pd.read_csv(DATA)
_dmci_cols = [c for c in _df_raw.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in _df_raw.columns if c not in EXCLUDE and c not in _dmci_cols]
df = _df_raw.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
ORIGINAL_FIXED = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]
actual_group = df["group"].values.astype(int)


def select_features(train_df, n=3, threshold=None, target_col="age"):
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


def compute_metrics(pred_group, pred_proba, actual):
    accuracy = float((pred_group == actual).mean())
    cm = confusion_matrix(actual, pred_group, labels=[1, 2])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc = roc_auc_score(actual, pred_proba)
    return accuracy, sensitivity, specificity, auc, cm


def wilcoxon_vs(pred_a, pred_b, actual):
    ca = (pred_a == actual).astype(int)
    cb = (pred_b == actual).astype(int)
    if np.any(ca != cb):
        _, p = stats.wilcoxon(ca, cb)
        return float(p)
    return 1.0


def permutation_test(feature_kwargs=None, model_kwargs=None, decision_threshold=0.5,
                      tune_C=False, C_grid=None, tune_threshold=False, threshold_grid=None,
                      fixed_features=None, n_perms=500, observed_auc=None, seed=42):
    rng = np.random.default_rng(seed)
    null_aucs = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = rng.permutation(df["group"].values)
        _, proba = run_nested(shuffled, feature_kwargs or {}, model_kwargs or {}, decision_threshold,
                               tune_C, C_grid, tune_threshold, threshold_grid, fixed_features)
        null_aucs.append(roc_auc_score((shuffled == 2).astype(int), proba))
    elapsed = time.time() - t0
    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return p_value, elapsed, null_aucs

"""
Nested LOOCV XGBoost classifier for young(1) vs old(2) binary classification,
structured identically to 23_task6_classification.py (logistic regression)
for direct comparability.

NOTE ON "GRANULAR": the supervisor's request is titled "granular XGBoost
classifier," referencing the GXGBoost paper discussed earlier (pairwise
feature combination + multi-distance-metric granulation against reference
samples, feeding L parallel XGBoost classifiers). The concrete implementation
spec actually given here, however, does not include any of that granulation
machinery -- it specifies a single small, regularized XGBoost model on 3
nested-reselected raw features, matching the logistic regression script's
structure exactly. That is what is implemented below. If true feature
granulation (pairwise combinations + distance-to-reference-sample expansion)
is wanted, that is a separate, larger piece of work -- flagged for
confirmation rather than silently added or silently skipped.

Small-N safeguards per instructions: max_depth=2, n_estimators=50,
learning_rate=0.05, reg_alpha=1.0, reg_lambda=1.0.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\classification_xgboost_young_vs_old.csv"

EXCLUDE = {"participant", "age", "group"}

df = pd.read_csv(DATA)
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
CANDIDATE_FEATURES = [c for c in df.columns if c not in EXCLUDE and c not in dmci_cols]
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)

MIN_N_FOR_CORR = 5
TOP_N = 3

XGB_PARAMS = dict(
    max_depth=2,
    n_estimators=50,
    learning_rate=0.05,
    reg_alpha=1.0,
    reg_lambda=1.0,
    eval_metric="logloss",
)

print(f"N = {N} (nested top-3 feature reselection every fold, matching logistic regression script)")
print(f"XGBoost params: {XGB_PARAMS}")


def select_top3(train_df):
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


def nested_loocv_xgb(group_labels, verbose=False):
    """Runs the full nested LOOCV procedure for a given label vector (the
    real labels, or a shuffled version for the permutation test)."""
    d = df.copy()
    d["group"] = group_labels
    pred_group = np.empty(N, dtype=int)
    pred_proba_old = np.empty(N)
    fold_features = []

    for i in range(N):
        train_df = d.drop(index=i)
        test_row = d.iloc[[i]]

        selected = select_top3(train_df)
        fold_features.append(selected)

        train_medians = train_df[selected].median()
        X_train = train_df[selected].fillna(train_medians).values.astype(float)
        # XGBoost's sklearn wrapper requires 0-indexed class labels; remap
        # young(1)->0, old(2)->1 for fitting, then map predictions back.
        y_train = (train_df["group"].values.astype(int) == 2).astype(int)
        X_test = test_row[selected].fillna(train_medians).values.astype(float)

        clf = XGBClassifier(**XGB_PARAMS, random_state=i, n_jobs=1, verbosity=0)
        clf.fit(X_train, y_train)
        pred_binary = clf.predict(X_test)[0]  # 0=young, 1=old
        pred_group[i] = 2 if pred_binary == 1 else 1
        proba = clf.predict_proba(X_test)[0]
        classes = list(clf.classes_)
        old_idx = classes.index(1) if 1 in classes else 0  # class "1" here means old
        pred_proba_old[i] = proba[old_idx]

        if verbose:
            print(f"  [{i+1:2d}/{N}] participant={test_row['participant'].values[0]:<4} "
                  f"actual_group={test_row['group'].values[0]}  pred_group={pred_group[i]}  "
                  f"proba_old={pred_proba_old[i]:.3f}  features={selected}")

    return pred_group, pred_proba_old, fold_features


t0 = time.time()
pred_group, pred_proba_old, fold_features = nested_loocv_xgb(df["group"].values, verbose=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold average).")

actual_group = df["group"].values.astype(int)
accuracy = float((pred_group == actual_group).mean())
cm = confusion_matrix(actual_group, pred_group, labels=[1, 2])
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
auc = roc_auc_score(actual_group, pred_proba_old)

print(f"\n{'='*60}")
print("XGBOOST NESTED LOOCV RESULTS")
print(f"{'='*60}")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Sensitivity (recall for 'old'/group=2):   {sensitivity:.4f}")
print(f"Specificity (recall for 'young'/group=1): {specificity:.4f}")
print(f"AUC-ROC:     {auc:.4f}")
print(f"\nConfusion matrix (rows=actual, cols=predicted, order=[young(1), old(2)]):")
print(pd.DataFrame(cm, index=["actual_young(1)", "actual_old(2)"], columns=["pred_young(1)", "pred_old(2)"]))

print(f"\nComparison vs Logistic Regression (accuracy=0.6207, AUC=0.6779, perm_p=0.032):")
print(f"  XGBoost   accuracy={accuracy:.4f}  AUC={auc:.4f}")
print(f"  Logistic  accuracy=0.6207          AUC=0.6779")

# ── Overfitting check: fit on ALL 29 (diagnostic only, not a valid prediction) ─
selected_full = select_top3(df)
train_medians_full = df[selected_full].median()
X_full = df[selected_full].fillna(train_medians_full).values.astype(float)
y_full_binary = (df["group"].values.astype(int) == 2).astype(int)
clf_full = XGBClassifier(**XGB_PARAMS, random_state=0, n_jobs=1, verbosity=0)
clf_full.fit(X_full, y_full_binary)
train_preds = clf_full.predict(X_full)
train_accuracy = float((train_preds == y_full_binary).mean())
print(f"\nOVERFITTING CHECK (diagnostic only -- fit and evaluated on the same 29, not a valid prediction):")
print(f"  Training-set accuracy (fit on all 29, features selected on all 29): {train_accuracy:.4f}")
print(f"  Nested LOOCV accuracy (honest, out-of-sample): {accuracy:.4f}")
print(f"  Gap: {train_accuracy - accuracy:+.4f}  "
      f"({'LARGE GAP -- overfitting warning sign' if (train_accuracy - accuracy) > 0.15 else 'gap present but modest'})")

# ── Save per-participant predictions ─────────────────────────────────────────
out = pd.DataFrame({
    "participant_id": df["participant"],
    "actual_group": actual_group,
    "predicted_group": pred_group,
    "predicted_proba_old": np.round(pred_proba_old, 4),
    "correct": pred_group == actual_group,
})
summary_row = pd.DataFrame([{
    "participant_id": "SUMMARY", "actual_group": None, "predicted_group": None,
    "predicted_proba_old": None, "correct": None,
}])
out_full = pd.concat([out, summary_row], ignore_index=True)
out_full.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

# ── Permutation test (500 shuffles, matching logistic regression's test) ────
N_PERMS = 500
PERM_OUT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\xgboost_permutation_null.csv"

print(f"\nRunning permutation test ({N_PERMS} shuffles, measured {elapsed:.1f}s/run -> "
      f"est. {elapsed*N_PERMS/60:.1f} min total)...")
rng = np.random.default_rng(42)
null_aucs = []
t_perm0 = time.time()
for p in range(N_PERMS):
    shuffled = rng.permutation(df["group"].values)
    _, perm_proba, _ = nested_loocv_xgb(shuffled, verbose=False)
    null_aucs.append(roc_auc_score(shuffled.astype(int), perm_proba))
    if (p + 1) % 50 == 0:
        print(f"  {p+1}/{N_PERMS} done ({time.time()-t_perm0:.0f}s elapsed)")

perm_elapsed = time.time() - t_perm0
null_aucs = np.array(null_aucs)
perm_p = float(np.mean(null_aucs >= auc))

pd.DataFrame({"permutation": range(1, N_PERMS + 1), "null_auc": null_aucs}).to_csv(PERM_OUT, index=False)

print(f"\n{'='*60}")
print("XGBOOST PERMUTATION TEST")
print(f"{'='*60}")
print(f"Shuffles: {N_PERMS}  (took {perm_elapsed:.1f}s)")
print(f"Real AUC: {auc:.4f}")
print(f"Null AUC: mean={null_aucs.mean():.4f}  sd={null_aucs.std():.4f}  95th_pctile={np.percentile(null_aucs,95):.4f}")
print(f"Permutation p-value: {perm_p:.4f}  ({'SIGNIFICANT' if perm_p < 0.05 else 'NOT significant'})")
print(f"Saved -> {PERM_OUT}")

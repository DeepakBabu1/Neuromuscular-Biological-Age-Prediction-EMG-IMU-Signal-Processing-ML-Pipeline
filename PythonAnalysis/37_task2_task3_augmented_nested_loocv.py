"""
TASK 2 & 3 -- Nested LOOCV over the 29 REAL participants, with synthetic data
used ONLY to supplement each fold's 28-participant TRAINING set. Synthetic
data is regenerated FRESH inside each fold from that fold's 28 training
participants only -- the held-out participant never contributes to, and is
never drawn from, the synthetic pool. This is the central safeguard against
leakage requested in Task 2 and is enforced structurally: the generator
functions in augment_lib.py are only ever called on `train_df`, never on
data that includes the held-out row.

Evaluated models (matching existing baseline configurations exactly):
  - Logistic Regression, FIXED primary 3-feature set (matches Row 1)
  - Ridge (alpha=10.0), FIXED primary 3-feature set
  - Lasso (alpha=1.0), FIXED primary 3-feature set
NOTE: Ridge/Lasso normally use per-fold top-3 reselection from 33 candidates
(see feature_selection_by_model.csv). Synthetic data was generated only for
the primary 3-feature set (per the task spec), so that reselection is not
possible here -- the fixed primary set is used instead for ALL augmented
Ridge/Lasso runs. This is flagged explicitly in the comparison table's notes
column, since it is a real (if minor) methodological deviation from the
"Ridge (nested top-3)" / "Lasso (nested top-3)" baseline rows they are
compared against. In practice the primary set matches the top-3 selection
in 26-28 of 29 real folds anyway (per feature_selection_by_model.csv), so
the deviation is expected to be small.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

import augment_lib as A

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = A.FEATURES

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_group = df["group"].values.astype(int)
actual_age = df["age"].values.astype(float)

GEN_CONFIG = {
    "noise": dict(fn=A.generate_noise, kwargs=dict(n_per_real=3, seed=1)),
    "smote": dict(fn=A.generate_smote, kwargs=dict(target_total=round(100 * 28 / 29), seed=1)),
    "gmm":   dict(fn=A.generate_gmm,   kwargs=dict(target_total=round(100 * 28 / 29), n_components=1, seed=1)),
}


def build_training_set(train_real_df, method):
    """Real (median-imputed within this fold) + fresh synthetic, both tagged."""
    medians = train_real_df[FEATURES].median()
    real_part = train_real_df.copy()
    real_part[FEATURES] = real_part[FEATURES].fillna(medians)

    cfg = GEN_CONFIG[method]
    synth_part = cfg["fn"](train_real_df, **cfg["kwargs"])  # generator internally re-imputes for its own fitting

    combined = pd.concat([
        real_part[["participant", "age", "group"] + FEATURES],
        synth_part[["participant", "age", "group"] + FEATURES],
    ], ignore_index=True)
    return combined, medians, len(synth_part)


def run_augmented_loocv(method):
    lr_pred_group = np.empty(N, dtype=int)
    lr_pred_proba = np.empty(N)
    ridge_pred_age = np.empty(N)
    lasso_pred_age = np.empty(N)
    n_synth_list = []

    for i in range(N):
        train_real = df.drop(index=i)
        test_row = df.iloc[[i]]

        train_combined, medians, n_synth = build_training_set(train_real, method)
        n_synth_list.append(n_synth)

        X_train = train_combined[FEATURES].values.astype(float)
        X_test = test_row[FEATURES].fillna(medians).values.astype(float)

        # -- Logistic regression (classification), matching Row 1 config --
        y_train_bin = (train_combined["group"].values.astype(int) == 2).astype(int)
        if len(np.unique(y_train_bin)) < 2:
            lr_pred_proba[i] = 0.5
            lr_pred_group[i] = 1
        else:
            clf = LogisticRegression()
            clf.fit(X_train, y_train_bin)
            classes = list(clf.classes_)
            old_idx = classes.index(1) if 1 in classes else 0
            proba_old = clf.predict_proba(X_test)[0][old_idx]
            lr_pred_proba[i] = proba_old
            lr_pred_group[i] = 2 if proba_old >= 0.5 else 1

        # -- Ridge / Lasso (regression), matching master_results_summary alphas --
        y_train_age = train_combined["age"].values.astype(float)
        ridge = Ridge(alpha=10.0).fit(X_train, y_train_age)
        lasso = Lasso(alpha=1.0).fit(X_train, y_train_age)
        ridge_pred_age[i] = ridge.predict(X_test)[0]
        lasso_pred_age[i] = lasso.predict(X_test)[0]

    return dict(
        lr_pred_group=lr_pred_group, lr_pred_proba=lr_pred_proba,
        ridge_pred_age=ridge_pred_age, lasso_pred_age=lasso_pred_age,
        n_synth_per_fold=n_synth_list,
    )


def permutation_test_lr(method, observed_auc, n_perms=500, seed=42):
    rng = np.random.default_rng(seed)
    null_aucs = []
    t0 = time.time()
    for p in range(n_perms):
        shuffled = rng.permutation(df["group"].values)
        d2 = df.copy()
        d2["group"] = shuffled
        proba = np.empty(N)
        for i in range(N):
            train_real = d2.drop(index=i)
            test_row = d2.iloc[[i]]
            train_combined, medians, _ = build_training_set(train_real, method)
            X_train = train_combined[FEATURES].values.astype(float)
            X_test = test_row[FEATURES].fillna(medians).values.astype(float)
            y_train_bin = (train_combined["group"].values.astype(int) == 2).astype(int)
            if len(np.unique(y_train_bin)) < 2:
                proba[i] = 0.5
            else:
                clf = LogisticRegression()
                clf.fit(X_train, y_train_bin)
                classes = list(clf.classes_)
                old_idx = classes.index(1) if 1 in classes else 0
                proba[i] = clf.predict_proba(X_test)[0][old_idx]
        null_aucs.append(roc_auc_score((shuffled == 2).astype(int), proba))
    elapsed = time.time() - t0
    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    return p_value, elapsed, null_aucs


# ── Load existing baselines (no re-run, per Task 3 instructions) ────────────
baseline_class = pd.read_csv(f"{BASE}\\classification_young_vs_old.csv")
baseline_class = baseline_class[baseline_class["participant_id"] != "SUMMARY"].copy()
baseline_class["participant_id"] = baseline_class["participant_id"].astype(int)
baseline_class = baseline_class.set_index("participant_id")

baseline_ridge = pd.read_csv(f"{BASE}\\per_participant_ridge.csv")
baseline_ridge = baseline_ridge[baseline_ridge["participant_id"] != "MEAN"].copy()
baseline_ridge["participant_id"] = baseline_ridge["participant_id"].astype(int)
baseline_ridge = baseline_ridge.set_index("participant_id")

baseline_lasso = pd.read_csv(f"{BASE}\\per_participant_lasso.csv")
baseline_lasso = baseline_lasso[baseline_lasso["participant_id"] != "MEAN"].copy()
baseline_lasso["participant_id"] = baseline_lasso["participant_id"].astype(int)
baseline_lasso = baseline_lasso.set_index("participant_id")

participant_ids = df["participant"].astype(int).values

comparison_rows = []

# ── Baseline (no augmentation) rows, reused verbatim from saved files ──────
comparison_rows.append(dict(
    method="none", model="Logistic Regression", n_real=29, n_synthetic_per_fold=0,
    accuracy_or_MAE=0.6207, AUC_or_r=0.6779, permutation_p=0.032, wilcoxon_p_vs_no_augmentation=None,
    notes="Existing Row 1 result, reused verbatim (classification_young_vs_old.csv)."))
comparison_rows.append(dict(
    method="none", model="Ridge (nested top-3)", n_real=29, n_synthetic_per_fold=0,
    accuracy_or_MAE=13.58, AUC_or_r=0.194, permutation_p=None, wilcoxon_p_vs_no_augmentation=None,
    notes="Existing result, reused verbatim (master_results_summary.csv). Nested top-3 reselection, not fixed."))
comparison_rows.append(dict(
    method="none", model="Lasso (nested top-3)", n_real=29, n_synthetic_per_fold=0,
    accuracy_or_MAE=13.47, AUC_or_r=0.208, permutation_p=None, wilcoxon_p_vs_no_augmentation=None,
    notes="Existing result, reused verbatim (master_results_summary.csv). Nested top-3 reselection, not fixed."))

honesty_report = []

for method in ["noise", "smote", "gmm"]:
    print(f"\n{'='*70}\nMETHOD: {method}\n{'='*70}")
    t0 = time.time()
    res = run_augmented_loocv(method)
    elapsed = time.time() - t0
    avg_n_synth = np.mean(res["n_synth_per_fold"])
    print(f"Done in {elapsed:.1f}s. Avg synthetic participants per fold: {avg_n_synth:.1f}")

    # -- Logistic regression --
    lr_acc = float((res["lr_pred_group"] == actual_group).mean())
    lr_auc = roc_auc_score(actual_group, res["lr_pred_proba"])
    baseline_correct = (baseline_class.loc[participant_ids, "predicted_group"].values.astype(int)
                         == baseline_class.loc[participant_ids, "actual_group"].values.astype(int)).astype(int)
    aug_correct = (res["lr_pred_group"] == actual_group).astype(int)
    if np.any(baseline_correct != aug_correct):
        _, wp_lr = stats.wilcoxon(aug_correct, baseline_correct)
    else:
        wp_lr = 1.0
    print(f"LR: accuracy={lr_acc:.4f} AUC={lr_auc:.4f}  Wilcoxon vs baseline p={wp_lr:.4f}")

    acc_gain_pp = (lr_acc - 0.6207) * 100
    suspicious = abs(acc_gain_pp) > 15
    confirmed_lr = (lr_acc > 0.6207) and (wp_lr < 0.05)

    perm_p_lr = None
    if confirmed_lr:
        print("  -> Wilcoxon shows a significant, positive result: running full 500-shuffle permutation test...")
        perm_p_lr, perm_elapsed, _ = permutation_test_lr(method, lr_auc, n_perms=500)
        print(f"  Permutation p={perm_p_lr:.4f} (took {perm_elapsed:.1f}s)")
    else:
        print("  -> No positive, significant signal vs baseline -- permutation test skipped (matches established session practice).")

    comparison_rows.append(dict(
        method=method, model="Logistic Regression", n_real=29,
        n_synthetic_per_fold=round(avg_n_synth), accuracy_or_MAE=round(lr_acc, 4),
        AUC_or_r=round(lr_auc, 4), permutation_p=perm_p_lr, wilcoxon_p_vs_no_augmentation=round(wp_lr, 4),
        notes=f"Fixed primary 3-feature set (matches Row 1). {'SUSPICIOUS GAIN >15pp -- investigate' if suspicious else ''}"))

    honesty_report.append(dict(method=method, model="Logistic Regression", accuracy=lr_acc, auc=lr_auc,
                                wilcoxon_p=wp_lr, permutation_p=perm_p_lr, confirmed=confirmed_lr,
                                suspicious=suspicious, gain_pp=acc_gain_pp))

    # -- Ridge --
    ridge_mae = mean_absolute_error(actual_age, res["ridge_pred_age"])
    ridge_rmse = np.sqrt(mean_squared_error(actual_age, res["ridge_pred_age"]))
    ridge_r, ridge_p = stats.pearsonr(actual_age, res["ridge_pred_age"])
    baseline_ridge_err = baseline_ridge.loc[participant_ids, "absolute_error"].values
    aug_ridge_err = np.abs(res["ridge_pred_age"] - actual_age)
    _, wp_ridge = stats.wilcoxon(aug_ridge_err, baseline_ridge_err)
    print(f"Ridge: MAE={ridge_mae:.4f} r={ridge_r:.4f} p={ridge_p:.4f}  Wilcoxon vs baseline p={wp_ridge:.4f}")
    confirmed_ridge = (ridge_mae < 13.58) and (wp_ridge < 0.05)

    comparison_rows.append(dict(
        method=method, model="Ridge (fixed primary-3, augmented)", n_real=29,
        n_synthetic_per_fold=round(avg_n_synth), accuracy_or_MAE=round(ridge_mae, 4),
        AUC_or_r=round(ridge_r, 4), permutation_p=None, wilcoxon_p_vs_no_augmentation=round(wp_ridge, 4),
        notes="Fixed primary 3-feature set (synthetic data has no other candidates to reselect from) -- "
              "compared against the nested-top-3 baseline; feature-set difference noted."))
    honesty_report.append(dict(method=method, model="Ridge", accuracy=ridge_mae, auc=ridge_r,
                                wilcoxon_p=wp_ridge, permutation_p=None, confirmed=confirmed_ridge,
                                suspicious=False, gain_pp=None))

    # -- Lasso --
    lasso_mae = mean_absolute_error(actual_age, res["lasso_pred_age"])
    lasso_r, lasso_p = stats.pearsonr(actual_age, res["lasso_pred_age"])
    baseline_lasso_err = baseline_lasso.loc[participant_ids, "absolute_error"].values
    aug_lasso_err = np.abs(res["lasso_pred_age"] - actual_age)
    _, wp_lasso = stats.wilcoxon(aug_lasso_err, baseline_lasso_err)
    print(f"Lasso: MAE={lasso_mae:.4f} r={lasso_r:.4f} p={lasso_p:.4f}  Wilcoxon vs baseline p={wp_lasso:.4f}")
    confirmed_lasso = (lasso_mae < 13.47) and (wp_lasso < 0.05)

    comparison_rows.append(dict(
        method=method, model="Lasso (fixed primary-3, augmented)", n_real=29,
        n_synthetic_per_fold=round(avg_n_synth), accuracy_or_MAE=round(lasso_mae, 4),
        AUC_or_r=round(lasso_r, 4), permutation_p=None, wilcoxon_p_vs_no_augmentation=round(wp_lasso, 4),
        notes="Fixed primary 3-feature set (synthetic data has no other candidates to reselect from) -- "
              "compared against the nested-top-3 baseline; feature-set difference noted."))
    honesty_report.append(dict(method=method, model="Lasso", accuracy=lasso_mae, auc=lasso_r,
                                wilcoxon_p=wp_lasso, permutation_p=None, confirmed=confirmed_lasso,
                                suspicious=False, gain_pp=None))

comparison = pd.DataFrame(comparison_rows)
out_path = f"{BASE}\\augmentation_comparison.csv"
comparison.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
print(comparison.to_string(index=False))

# ── Final honesty summary ────────────────────────────────────────────────
print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
any_confirmed = [h for h in honesty_report if h["confirmed"]]
if any_confirmed:
    for h in any_confirmed:
        print(f"CONFIRMED improvement: {h['method']} / {h['model']}")
else:
    print("No augmentation method produced a confirmed improvement (metric better AND Wilcoxon p<0.05) "
          "over the real-data-only baseline, for any model.")

suspicious_any = [h for h in honesty_report if h["suspicious"]]
if suspicious_any:
    for h in suspicious_any:
        print(f"SUSPICIOUS GAIN FLAGGED: {h['method']} / {h['model']} -- gain={h['gain_pp']:+.1f}pp, investigate before trusting.")
else:
    print("No method showed a suspiciously large (>15pp) accuracy gain.")

print("\nSAFEGUARD CONFIRMATION: all reported performance above is evaluated exclusively on the 29 "
      "real held-out participants; synthetic data was used only inside each fold's training set and "
      "was regenerated fresh per fold from that fold's 28 real training participants only.")

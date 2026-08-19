"""
TASK 2B -- Bootstrap 95% CIs for the primary logistic regression classifier's
AUC and accuracy. Uses the ALREADY-SAVED nested LOOCV predictions in
classification_young_vs_old.csv -- no model is refit or retrained.

2000 resamples of participants (with replacement); AUC and accuracy
recomputed on each resample using the existing out-of-sample predictions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\classification_young_vs_old.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\primary_model_confidence_intervals.csv"

N_BOOT = 2000
SEED = 42

df = pd.read_csv(DATA)
df = df[df["participant_id"] != "SUMMARY"].copy()
df["actual_group"] = df["actual_group"].astype(int)
df["predicted_group"] = df["predicted_group"].astype(int)
df["predicted_proba_old"] = df["predicted_proba_old"].astype(float)
n = len(df)

actual = df["actual_group"].values
pred_group = df["predicted_group"].values
proba_old = df["predicted_proba_old"].values

point_auc = roc_auc_score(actual, proba_old)
point_acc = float((pred_group == actual).mean())

print(f"N participants (from saved predictions): {n}")
print(f"Point estimate AUC:      {point_auc:.4f}")
print(f"Point estimate accuracy: {point_acc:.4f}")

rng = np.random.default_rng(SEED)
idx = np.arange(n)
boot_aucs = []
boot_accs = []
skipped_single_class = 0

for b in range(N_BOOT):
    resample_idx = rng.choice(idx, size=n, replace=True)
    a = actual[resample_idx]
    p = proba_old[resample_idx]
    g = pred_group[resample_idx]

    boot_accs.append(float((g == a).mean()))

    if len(np.unique(a)) < 2:
        skipped_single_class += 1
        continue  # AUC undefined for a resample with only one class present
    boot_aucs.append(roc_auc_score(a, p))

boot_aucs = np.array(boot_aucs)
boot_accs = np.array(boot_accs)

auc_ci_lo, auc_ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
acc_ci_lo, acc_ci_hi = np.percentile(boot_accs, [2.5, 97.5])

print(f"\n{N_BOOT} bootstrap resamples ({skipped_single_class} skipped for AUC -- single-class resample, "
      f"{len(boot_aucs)} used for AUC CI, all {len(boot_accs)} used for accuracy CI)")
print(f"\nAUC       point={point_auc:.4f}  95% CI=[{auc_ci_lo:.4f}, {auc_ci_hi:.4f}]  width={auc_ci_hi-auc_ci_lo:.4f}")
print(f"Accuracy  point={point_acc:.4f}  95% CI=[{acc_ci_lo:.4f}, {acc_ci_hi:.4f}]  width={acc_ci_hi-acc_ci_lo:.4f}")

out = pd.DataFrame([
    {"metric": "AUC", "point_estimate": round(point_auc, 4), "ci_lower_2.5": round(auc_ci_lo, 4),
     "ci_upper_97.5": round(auc_ci_hi, 4), "ci_width": round(auc_ci_hi - auc_ci_lo, 4),
     "n_bootstrap": N_BOOT, "n_resamples_used": len(boot_aucs), "n_participants": n},
    {"metric": "Accuracy", "point_estimate": round(point_acc, 4), "ci_lower_2.5": round(acc_ci_lo, 4),
     "ci_upper_97.5": round(acc_ci_hi, 4), "ci_width": round(acc_ci_hi - acc_ci_lo, 4),
     "n_bootstrap": N_BOOT, "n_resamples_used": len(boot_accs), "n_participants": n},
])
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

# Does the CI include 0.5 (chance)?
print(f"\nAUC CI includes 0.5 (chance level)? {'YES' if auc_ci_lo <= 0.5 <= auc_ci_hi else 'NO'}")

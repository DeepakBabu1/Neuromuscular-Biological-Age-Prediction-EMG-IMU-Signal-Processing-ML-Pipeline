"""
Task 6: SEPARATE analysis -- binary classification (young=1 vs old=2)
instead of continuous age regression. Same 3 features, fixed (not
reselected), nested LOOCV logistic regression. This is a new model fit
(never done before this session), required because it's a genuinely
different task, not a re-run of the existing age-regression models.

Missing values in stride_timing_cv_walking are imputed with the TRAINING
fold's median (consistent with this project's established approach
elsewhere), so all 29 participants are retained -- unlike Task 5, which
had to drop to N=18 via listwise deletion.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\classification_young_vs_old.csv"

FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA).dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
print(f"N = {N} (fixed 3 features, missing values imputed with training-fold median)")

pred_group = np.empty(N, dtype=int)
pred_proba_old = np.empty(N)  # P(group=2)

for i in range(N):
    train_df = df.drop(index=i)
    test_row = df.iloc[[i]]

    train_medians = train_df[FEATURES].median()
    X_train = train_df[FEATURES].fillna(train_medians).values.astype(float)
    y_train = train_df["group"].values.astype(int)
    X_test = test_row[FEATURES].fillna(train_medians).values.astype(float)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred_group[i] = clf.predict(X_test)[0]
    proba = clf.predict_proba(X_test)[0]
    old_idx = list(clf.classes_).index(2)
    pred_proba_old[i] = proba[old_idx]

actual_group = df["group"].values.astype(int)

accuracy = float((pred_group == actual_group).mean())
cm = confusion_matrix(actual_group, pred_group, labels=[1, 2])
# cm rows = actual [1,2], cols = predicted [1,2]
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]  # "positive" class = old(2)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall for old
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # recall for young
auc = roc_auc_score(actual_group, pred_proba_old)

print(f"\nAccuracy:    {accuracy:.4f}")
print(f"Sensitivity (recall for 'old'/group=2):   {sensitivity:.4f}")
print(f"Specificity (recall for 'young'/group=1): {specificity:.4f}")
print(f"AUC-ROC:     {auc:.4f}")
print(f"\nConfusion matrix (rows=actual, cols=predicted, order=[young(1), old(2)]):")
print(pd.DataFrame(cm, index=["actual_young(1)", "actual_old(2)"], columns=["pred_young(1)", "pred_old(2)"]))

per_participant = pd.DataFrame({
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
out = pd.concat([per_participant, summary_row], ignore_index=True)
out.to_csv(OUT_CSV, index=False)

metrics = pd.DataFrame([{
    "N": N, "accuracy": round(accuracy, 4), "sensitivity_old": round(sensitivity, 4),
    "specificity_young": round(specificity, 4), "AUC_ROC": round(auc, 4),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
}])
metrics.to_csv(OUT_CSV.replace(".csv", "_metrics.csv"), index=False)

print(f"\nSaved -> {OUT_CSV}")
print(f"Saved -> {OUT_CSV.replace('.csv', '_metrics.csv')}")

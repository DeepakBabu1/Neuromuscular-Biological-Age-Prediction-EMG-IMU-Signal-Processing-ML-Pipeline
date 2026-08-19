import time
import lr_improve_lib as L

thresh_grid = [round(0.3 + 0.05 * i, 2) for i in range(9)]
feature_kwargs = dict(n=3, threshold=None, target_col="age")
model_kwargs = {"class_weight": "balanced"}

pred_i5, proba_i5 = L.run_nested(L.df["group"].values, feature_kwargs, model_kwargs,
                                  tune_threshold=True, threshold_grid=thresh_grid)
acc, sens, spec, auc, cm = L.compute_metrics(pred_i5, proba_i5, L.actual_group)
print(f"Improvement 5 real result: acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} auc={auc:.4f}")

print("Running 500-shuffle permutation test...")
t0 = time.time()
p_value, elapsed, null_aucs = L.permutation_test(
    feature_kwargs=feature_kwargs, model_kwargs=model_kwargs,
    tune_threshold=True, threshold_grid=thresh_grid,
    n_perms=500, observed_auc=auc,
)
print(f"Done in {elapsed:.1f}s")
print(f"Real AUC: {auc:.4f}")
print(f"Null AUC: mean={null_aucs.mean():.4f} sd={null_aucs.std():.4f} 95th_pctile={(null_aucs.mean()):.4f}")
import numpy as np
print(f"Null 95th percentile: {np.percentile(null_aucs, 95):.4f}")
print(f"Permutation p-value: {p_value:.4f}  ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")

import pandas as pd
pd.DataFrame({"permutation": range(1, len(null_aucs)+1), "null_auc": null_aucs}).to_csv(
    r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\improvement5_permutation_null.csv",
    index=False
)
print("Saved -> improvement5_permutation_null.csv")

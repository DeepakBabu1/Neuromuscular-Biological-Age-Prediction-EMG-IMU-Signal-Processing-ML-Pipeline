"""
Resume: RBF permutation test already completed and saved (p=0.100,
gpr_A_RBF_permutation_null.csv). This script runs ONLY the remaining
Matern config's 500-shuffle permutation test, identical methodology.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)

kernel_fn = lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2))
OBSERVED_R = 0.2152


def run_nested_gpr(kfn, age_labels):
    pred_age = np.empty(N)
    for i in range(N):
        train_df = df.drop(index=i)
        test_row = df.iloc[[i]]
        y_train = age_labels[train_df.index.values]
        medians = train_df[FEATURES].median()
        X_train_raw = train_df[FEATURES].fillna(medians).values.astype(float)
        X_test_raw = test_row[FEATURES].fillna(medians).values.astype(float)
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        gpr = GaussianProcessRegressor(kernel=kfn(), n_restarts_optimizer=10,
                                        normalize_y=True, random_state=0)
        gpr.fit(X_train, y_train)
        pred_age[i] = gpr.predict(X_test)[0]
    return pred_age


N_PERMS = 500
print(f"{'='*70}\nPERMUTATION TEST: B_Matern ({N_PERMS} shuffles) -- resumed run\n{'='*70}", flush=True)
rng = np.random.default_rng(42)
null_r = []
t0 = time.time()
for p in range(N_PERMS):
    shuffled_age = rng.permutation(actual_age)
    pred = run_nested_gpr(kernel_fn, shuffled_age)
    r, _ = stats.pearsonr(shuffled_age, pred)
    null_r.append(r)
    if (p + 1) % 50 == 0:
        print(f"  {p+1}/{N_PERMS} done ({time.time()-t0:.0f}s elapsed)", flush=True)
elapsed = time.time() - t0
null_r = np.array(null_r)
perm_p = float(np.mean(null_r >= OBSERVED_R))
print(f"Done in {elapsed:.1f}s. Observed r={OBSERVED_R:.4f}. "
      f"Null r: mean={null_r.mean():.4f} sd={null_r.std():.4f}. Permutation p={perm_p:.4f}")
pd.DataFrame({"permutation": range(1, N_PERMS + 1), "null_pearson_r": null_r}).to_csv(
    f"{BASE}\\gpr_B_Matern_permutation_null.csv", index=False)
print(f"Saved -> {BASE}\\gpr_B_Matern_permutation_null.csv")
print("MATERN PERMUTATION TEST COMPLETE")

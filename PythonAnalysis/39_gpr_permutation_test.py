"""
500-shuffle permutation test for both GPR kernel configs, explicitly
requested regardless of the real-run result. Age labels are shuffled across
participants and the full nested LOOCV GPR procedure (including per-fold
kernel hyperparameter refitting) is repeated on the shuffled labels.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats

import importlib.util
spec = importlib.util.spec_from_file_location("gpr_mod", r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\38_gpr_nested_loocv.py")
# Don't execute 38's top-level code (expensive) -- instead reimplement the minimal pieces needed.

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
DATA = f"{BASE}\\feature_table_full.csv"
FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
N = len(df)
actual_age = df["age"].values.astype(float)

CONFIGS = {
    "A_RBF": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2)),
    "B_Matern": lambda: ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2)),
}

OBSERVED_R = {"A_RBF": 0.1723, "B_Matern": 0.2152}


def run_nested_gpr(kernel_fn, age_labels):
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
        gpr = GaussianProcessRegressor(kernel=kernel_fn(), n_restarts_optimizer=10,
                                        normalize_y=True, random_state=0)
        gpr.fit(X_train, y_train)
        mean = gpr.predict(X_test)
        pred_age[i] = mean[0]
    return pred_age


N_PERMS = 500
for cfg_name, kernel_fn in CONFIGS.items():
    print(f"\n{'='*70}\nPERMUTATION TEST: {cfg_name} ({N_PERMS} shuffles)\n{'='*70}", flush=True)
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
    observed = OBSERVED_R[cfg_name]
    perm_p = float(np.mean(null_r >= observed))
    print(f"Done in {elapsed:.1f}s. Observed r={observed:.4f}. "
          f"Null r: mean={null_r.mean():.4f} sd={null_r.std():.4f}. Permutation p={perm_p:.4f}")
    pd.DataFrame({"permutation": range(1, N_PERMS + 1), "null_pearson_r": null_r}).to_csv(
        f"{BASE}\\gpr_{cfg_name}_permutation_null.csv", index=False)
    print(f"Saved -> {BASE}\\gpr_{cfg_name}_permutation_null.csv")

print("\nALL GPR PERMUTATION TESTS COMPLETE")

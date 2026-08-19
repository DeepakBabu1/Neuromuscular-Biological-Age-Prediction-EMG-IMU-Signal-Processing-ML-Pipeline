"""
TASK 1-3 -- Nested LOOCV Gaussian Process Regressor, two kernel configs,
matching the exact validation protocol used for every other model this
session: training-fold-only median imputation, training-fold-only feature
standardisation, kernel hyperparameters fit via marginal likelihood on the
training fold only (n_restarts_optimizer=10), primary 3-feature set.
"""
import time
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


def run_nested_gpr(kernel_fn, age_labels, verbose=False):
    pred_age = np.empty(N)
    pred_std = np.empty(N)
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
        mean, std = gpr.predict(X_test, return_std=True)
        pred_age[i] = mean[0]
        pred_std[i] = std[0]
        if verbose:
            print(f"  [{i+1:2d}/{N}] actual={age_labels[i]:.1f} pred={mean[0]:.2f} std={std[0]:.2f}")
    return pred_age, pred_std


def metrics(pred_age, actual):
    abs_err = np.abs(pred_age - actual)
    mae = abs_err.mean()
    rmse = np.sqrt(((pred_age - actual) ** 2).mean())
    r, p = stats.pearsonr(actual, pred_age)
    return mae, rmse, r, p, abs_err


def bootstrap_ci_mae_diff(abs_err_a, abs_err_b, n_boot=5000, seed=42):
    """MAE(a) - MAE(b) bootstrap CI, participant-level resampling with replacement."""
    rng = np.random.default_rng(seed)
    n = len(abs_err_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(abs_err_a[idx].mean() - abs_err_b[idx].mean())
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


# ── Load existing baselines for comparison ──────────────────────────────────
baseline_pp = pd.read_csv(f"{BASE}\\per_participant_baseline.csv")
baseline_pp = baseline_pp[baseline_pp["participant_id"] != "MEAN"].copy()
baseline_pp["participant_id"] = baseline_pp["participant_id"].astype(int)
baseline_pp = baseline_pp.set_index("participant_id")

ridge_pp = pd.read_csv(f"{BASE}\\per_participant_ridge.csv")
ridge_pp = ridge_pp[ridge_pp["participant_id"] != "MEAN"].copy()
ridge_pp["participant_id"] = ridge_pp["participant_id"].astype(int)
ridge_pp = ridge_pp.set_index("participant_id")

participant_ids = df["participant"].astype(int).values
baseline_err = baseline_pp.loc[participant_ids, "absolute_error"].values
ridge_err = ridge_pp.loc[participant_ids, "absolute_error"].values

results = {}
for cfg_name, kernel_fn in CONFIGS.items():
    print(f"\n{'='*70}\nCONFIG {cfg_name}\n{'='*70}")
    t0 = time.time()
    pred_age, pred_std = run_nested_gpr(kernel_fn, actual_age, verbose=True)
    elapsed = time.time() - t0
    mae, rmse, r, p, abs_err = metrics(pred_age, actual_age)
    avg_std = pred_std.mean()
    print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/fold).")
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f} p={p:.4f} avg_predictive_std={avg_std:.4f}")

    # Wilcoxon vs baseline & Ridge
    _, wp_base = stats.wilcoxon(abs_err, baseline_err)
    _, wp_ridge = stats.wilcoxon(abs_err, ridge_err)
    print(f"Wilcoxon vs baseline: p={wp_base:.4f}   Wilcoxon vs Ridge: p={wp_ridge:.4f}")

    ci_lo, ci_hi = bootstrap_ci_mae_diff(abs_err, baseline_err)
    print(f"Bootstrap 95% CI (MAE diff vs baseline, 5000 resamples): [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Calibration check
    calib_corr, calib_p = stats.pearsonr(pred_std, abs_err)
    lower = pred_age - 1.96 * pred_std
    upper = pred_age + 1.96 * pred_std
    within_95 = ((actual_age >= lower) & (actual_age <= upper)).mean()
    print(f"Calibration: corr(pred_std, abs_err)={calib_corr:.4f} (p={calib_p:.4f})   "
          f"fraction within 95% PI = {within_95:.4f}")

    results[cfg_name] = dict(pred_age=pred_age, pred_std=pred_std, mae=mae, rmse=rmse, r=r, p=p,
                              avg_std=avg_std, abs_err=abs_err, wp_base=wp_base, wp_ridge=wp_ridge,
                              ci_lo=ci_lo, ci_hi=ci_hi, calib_corr=calib_corr, calib_p=calib_p,
                              within_95=within_95, elapsed=elapsed)

# ── Save per-participant predictions ─────────────────────────────────────────
for cfg_name in CONFIGS:
    r = results[cfg_name]
    out = pd.DataFrame({
        "participant_id": df["participant"].astype(int),
        "actual_age": actual_age,
        "predicted_age": r["pred_age"],
        "predictive_std": r["pred_std"],
        "absolute_error": r["abs_err"],
        "within_95pct_interval": (actual_age >= r["pred_age"] - 1.96 * r["pred_std"]) &
                                  (actual_age <= r["pred_age"] + 1.96 * r["pred_std"]),
    })
    out_path = f"{BASE}\\per_participant_gpr_{cfg_name}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

# ── Update master_results_summary.csv ────────────────────────────────────────
master = pd.read_csv(f"{BASE}\\master_results_summary.csv")
new_rows = []
for cfg_name, label, kernel_desc in [
    ("A_RBF", "Gaussian Process Regressor (RBF kernel)", "ConstantKernel*RBF+WhiteKernel"),
    ("B_Matern", "Gaussian Process Regressor (Matern kernel, nu=1.5)", "ConstantKernel*Matern(nu=1.5)+WhiteKernel"),
]:
    r = results[cfg_name]
    new_rows.append({
        "model_name": label,
        "feature_selection_method": "Fixed primary 3-feature set (nested standardisation + imputation)",
        "n_features": 3, "n_participants": 29.0, "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4),
        "RMSE_MAE_ratio": round(r["rmse"] / r["mae"], 4), "pearson_r": round(r["r"], 4), "p_value": round(r["p"], 4),
        "wilcoxon_p_vs_baseline": round(r["wp_base"], 4), "bootstrap_ci_lower": round(r["ci_lo"], 4),
        "bootstrap_ci_upper": round(r["ci_hi"], 4), "significant_at_0.05": bool(r["wp_base"] < 0.05),
        "notes": (f"{kernel_desc}; n_restarts_optimizer=10; kernel hyperparameters fit via marginal "
                  f"likelihood on training fold only. avg_predictive_std={r['avg_std']:.4f}. "
                  f"Wilcoxon vs Ridge p={r['wp_ridge']:.4f}. Calibration: corr(std,abs_err)={r['calib_corr']:.4f}, "
                  f"fraction within 95% PI={r['within_95']:.4f}. Permutation test pending (see companion script)."),
    })

updated = pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True)
updated.to_csv(f"{BASE}\\master_results_summary.csv", index=False)
print(f"\nAppended 2 GPR rows -> master_results_summary.csv (now {len(updated)} rows)")

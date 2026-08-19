"""
Task 1: formal statistical comparison of the 4 models + baseline, using the
already-saved per-participant absolute errors in per_participant_all_runs.csv.
No models re-run -- paired tests only.
"""
import numpy as np
import pandas as pd
from scipy import stats

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\statistical_comparison.csv"

N_BOOT = 5000
RNG = np.random.default_rng(42)

df = pd.read_csv(IN_CSV)
errors = {
    "Ridge": df["ridge_error"].values,
    "Lasso": df["lasso_error"].values,
    "Linear": df["linear_error"].values,
    "gplearn": df["gplearn_error"].values,
    "Baseline": df["baseline_error"].values,
}
n = len(df)

def bootstrap_ci(a, b, n_boot=N_BOOT):
    diffs = np.empty(n_boot)
    idx_n = len(a)
    for i in range(n_boot):
        idx = RNG.integers(0, idx_n, idx_n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

comparisons = []
models = ["Ridge", "Lasso", "Linear", "gplearn"]
# vs baseline (4)
for m in models:
    comparisons.append((m, "Baseline"))
# every pair among the 4 (6)
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        comparisons.append((models[i], models[j]))

rows = []
for a_name, b_name in comparisons:
    a, b = errors[a_name], errors[b_name]
    stat, p = stats.wilcoxon(a, b)
    mae_diff = float(a.mean() - b.mean())
    ci_lo, ci_hi = bootstrap_ci(a, b)
    rows.append({
        "comparison": f"{a_name} vs {b_name}",
        "wilcoxon_stat": round(float(stat), 4),
        "wilcoxon_p": round(float(p), 4),
        "mae_difference": round(mae_diff, 4),
        "ci_lower_95": round(ci_lo, 4),
        "ci_upper_95": round(ci_hi, 4),
        "significant": bool(p < 0.05),
    })

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}  (n={n} participants, {N_BOOT} bootstrap resamples)\n")
pd.set_option("display.width", 160)
print(table.to_string(index=False))

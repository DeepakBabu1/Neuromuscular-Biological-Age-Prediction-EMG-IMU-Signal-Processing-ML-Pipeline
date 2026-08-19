"""
Task 4: Bland-Altman plots for Ridge, Lasso, Linear, gplearn.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\per_participant_all_runs.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"

df = pd.read_csv(IN_CSV)
actual = df["actual_age"].values.astype(float)
group = df["group"].values

model_cols = {"ridge": "Ridge", "lasso": "Lasso", "linear": "Linear", "gplearn": "gplearn"}

for col, label in model_cols.items():
    pred = df[f"{col}_pred"].values.astype(float)
    mean_vals = (actual + pred) / 2
    diff_vals = pred - actual
    mean_diff = diff_vals.mean()
    sd_diff = diff_vals.std(ddof=1)
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    young_mask = group == 1
    old_mask = group == 2
    ax.scatter(mean_vals[young_mask], diff_vals[young_mask], color="#1f77b4", label="Young (group=1)", s=55, alpha=0.8)
    ax.scatter(mean_vals[old_mask], diff_vals[old_mask], color="#d62728", label="Older (group=2)", s=55, alpha=0.8)

    ax.axhline(mean_diff, color="black", linestyle="-", linewidth=1.2, label=f"Mean diff = {mean_diff:.2f}")
    ax.axhline(upper, color="gray", linestyle="--", linewidth=1, label=f"+1.96 SD = {upper:.2f}")
    ax.axhline(lower, color="gray", linestyle="--", linewidth=1, label=f"-1.96 SD = {lower:.2f}")

    ax.set_xlabel("Mean of actual and predicted age (years)")
    ax.set_ylabel("Predicted - Actual age (years)")
    ax.set_title(f"Bland-Altman: {label} (nested LOOCV)")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_path = f"{PLOTS}\\bland_altman_{col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}  (mean_diff={mean_diff:.2f}, SD={sd_diff:.2f}, limits=[{lower:.2f}, {upper:.2f}])")

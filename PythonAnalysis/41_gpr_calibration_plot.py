"""
Figure 3.4 -- GPR predictive uncertainty vs. actual absolute error, both kernels.
Reads only the already-saved per-participant GPR prediction files; no recomputation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
OUT_PNG = f"{BASE}\\plots\\gpr_calibration.png"

configs = [
    ("A_RBF", "RBF kernel", f"{BASE}\\per_participant_gpr_A_RBF.csv"),
    ("B_Matern", "Matérn kernel (ν=1.5)", f"{BASE}\\per_participant_gpr_B_Matern.csv"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (key, label, path) in zip(axes, configs):
    df = pd.read_csv(path)
    std = df["predictive_std"].values
    err = df["absolute_error"].values
    r, p = stats.pearsonr(std, err)

    ax.scatter(std, err, color="#2a6f7f", alpha=0.8, s=45, zorder=3)
    slope, intercept = np.polyfit(std, err, 1)
    x_line = np.array([std.min(), std.max()])
    ax.plot(x_line, slope * x_line + intercept, "--", color="#c0392b", linewidth=2, zorder=2)

    ax.set_xlabel("GP predictive std. dev. (years)")
    ax.set_ylabel("Actual absolute error (years)")
    ax.set_title(f"{label}\nr = {r:.3f}, p = {p:.4f}")

fig.suptitle("Figure 3.4 — GPR calibration: predictive uncertainty vs. actual error\n"
             "(negative slope = model most confident where most wrong)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved -> {OUT_PNG}")

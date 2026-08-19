"""
Forest-plot style figure consolidating point estimates + 95% CIs already
computed and saved this session. No new computation -- pure visualization.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PNG = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\confidence_interval_forest_plot.png"

# label, point estimate, ci_lower, ci_upper, null value, panel (0=interaction coef, 1=proportion/AUC scale)
rows = [
    ("Interaction term\n(complete-case, N=18)", 0.5429, -0.0639, 1.1498, 0.0, 0),
    ("Interaction term\n(imputed, N=29)", 0.3163, -0.2145, 0.8470, 0.0, 0),
    ("Primary classifier AUC", 0.6779, 0.4545, 0.8824, 0.5, 1),
    ("Primary classifier accuracy", 0.6207, 0.4483, 0.7931, 0.5517, 1),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1]})

# -- Panel 0: interaction coefficient (own scale) --
ax = axes[0]
panel0 = [r for r in rows if r[5] == 0]
y_pos = np.arange(len(panel0))[::-1]
for i, (label, est, lo, hi, null, _) in zip(y_pos, panel0):
    crosses = lo <= null <= hi
    color = "#d62728" if crosses else "#2ca02c"
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, solid_capstyle="round")
    ax.plot(est, i, "o", color=color, markersize=8, zorder=3)
ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, label="Null (0)")
ax.set_yticks(y_pos)
ax.set_yticklabels([r[0] for r in panel0])
ax.set_xlabel("Coefficient estimate (age per unit feature)")
ax.set_title("Group×Feature Interaction Term\n(group:peak_roll_walkingincline)")
ax.legend(loc="lower right", fontsize=8)

# -- Panel 1: AUC / accuracy (0-1 scale, own nulls) --
ax = axes[1]
panel1 = [r for r in rows if r[5] == 1]
y_pos = np.arange(len(panel1))[::-1]
for i, (label, est, lo, hi, null, _) in zip(y_pos, panel1):
    crosses = lo <= null <= hi
    color = "#d62728" if crosses else "#2ca02c"
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, solid_capstyle="round")
    ax.plot(est, i, "o", color=color, markersize=8, zorder=3)
    ax.plot(null, i, "|", color="gray", markersize=18, markeredgewidth=2, zorder=2)
ax.set_yticks(y_pos)
ax.set_yticklabels([r[0] for r in panel1])
ax.set_xlabel("Value (0-1 scale)")
ax.set_xlim(0, 1)
ax.set_title("Primary Classifier\n(null = 0.5 for AUC, 0.5517 majority-class for accuracy)")

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color="#d62728", lw=2.5, label="95% CI crosses null"),
    Line2D([0], [0], color="#2ca02c", lw=2.5, label="95% CI excludes null"),
    Line2D([0], [0], marker="|", color="gray", lw=0, markersize=12, markeredgewidth=2, label="Null value"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

fig.suptitle("Confidence Intervals Across All Four Findings (N=29 study)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved -> {OUT_PNG}")

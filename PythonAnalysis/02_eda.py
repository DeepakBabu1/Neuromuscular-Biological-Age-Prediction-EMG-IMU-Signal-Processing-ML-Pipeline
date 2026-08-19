"""
Step 2: Exploratory Data Analysis
- Spearman correlations with age
- Distribution plots per group
- Identifies best features for modelling
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\participants_clean.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

df = pd.read_csv(DATA)

# ── Flag and exclude known artefact rows ──────────────────────────────────────
# Part11 & Part14: vaf_step_fwd ~ 99 (only 1 synergy, artefact of short recording)
artefact_mask = df["vaf_step_fwd"] > 95
print(f"Artefact rows (vaf_step_fwd > 95): {df.loc[artefact_mask, 'participant'].tolist()}")
# Set those step-forward values to NaN, keep participant for other exercises
df.loc[artefact_mask, ["vaf_step_fwd", "dmci_step_fwd", "ms_step_fwd"]] = np.nan

feature_cols = [
    "vaf_walk", "vaf_walk_inc", "vaf_step_fwd", "vaf_step_lat",
    "dmci_walk", "dmci_walk_inc", "dmci_step_fwd", "dmci_step_lat",
    "vaf_ratio_sfwd_walk", "dmci_ratio_sfwd_walk",
    "vaf_mean_all", "dmci_mean_all", "vaf_range", "dmci_range",
]

# ── 1. Spearman correlations with age ─────────────────────────────────────────
print("\nSpearman correlations with age (|rho| descending):")
corr_results = {}
for col in feature_cols:
    valid = df[["age", col]].dropna()
    if len(valid) < 5:
        continue
    r, p = stats.spearmanr(valid["age"], valid[col])
    corr_results[col] = (round(r, 3), round(p, 4))

corr_df = pd.DataFrame(corr_results, index=["rho", "p"]).T
corr_df = corr_df.reindex(corr_df["rho"].abs().sort_values(ascending=False).index)
print(corr_df.to_string())
corr_df.to_csv(os.path.join(PLOTS, "spearman_correlations.csv"))

# ── 2. Correlation heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = corr_df["rho"].values
bars = ax.barh(corr_df.index[::-1], corr_df["rho"].values[::-1],
               color=["#d62728" if v < 0 else "#1f77b4" for v in corr_df["rho"].values[::-1]])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman rho with Age")
ax.set_title("Feature–Age Correlations")
# Mark significance
for i, (idx, row) in enumerate(corr_df.iloc[::-1].iterrows()):
    star = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else ""))
    if star:
        ax.text(row["rho"] + (0.01 if row["rho"] >= 0 else -0.01), i, star,
                va="center", ha="left" if row["rho"] >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "feature_age_correlations.png"), dpi=150)
plt.close()

# ── 3. Scatter plots: top 6 features vs age ──────────────────────────────────
top6 = corr_df.index[:6].tolist()
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, col in zip(axes.flat, top6):
    valid = df[["age", col, "group"]].dropna()
    ya = valid[valid["group"] == 1]
    oa = valid[valid["group"] == 2]
    ax.scatter(ya["age"], ya[col], label="Young", alpha=0.7, color="#1f77b4")
    ax.scatter(oa["age"], oa[col], label="Older", alpha=0.7, color="#d62728")
    # Trend line
    m, b, *_ = stats.linregress(valid["age"], valid[col])
    xs = np.linspace(valid["age"].min(), valid["age"].max(), 100)
    ax.plot(xs, m * xs + b, "k--", linewidth=1)
    r, p = corr_results.get(col, (0, 1))
    ax.set_title(f"{col}\nrho={r}, p={p:.3f}")
    ax.set_xlabel("Age (yrs)")
    ax.legend(fontsize=7)
plt.suptitle("Top 6 Features vs Age", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "top6_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Violin plots: group comparison ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
main_feats = [
    "vaf_walk", "vaf_walk_inc", "vaf_step_fwd", "vaf_step_lat",
    "dmci_walk", "dmci_walk_inc", "dmci_step_fwd", "dmci_step_lat",
]
labels = {1: "Young", 2: "Older"}
for ax, col in zip(axes.flat, main_feats):
    for grp, color in [(1, "#1f77b4"), (2, "#d62728")]:
        vals = df.loc[df["group"] == grp, col].dropna()
        ax.violinplot([vals], positions=[grp], showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Young", "Older"])
    ax.set_title(col)
    # Mann-Whitney U test
    ya_v = df.loc[df["group"] == 1, col].dropna()
    oa_v = df.loc[df["group"] == 2, col].dropna()
    if len(ya_v) > 1 and len(oa_v) > 1:
        _, p = stats.mannwhitneyu(ya_v, oa_v, alternative="two-sided")
        ax.set_xlabel(f"MW p={p:.3f}")
plt.suptitle("Feature Distributions by Age Group", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "violin_group_comparison.png"), dpi=150)
plt.close()

# ── 5. Summary stats ─────────────────────────────────────────────────────────
print("\nGroup summary (mean +/- std):")
for col in main_feats:
    ya_v = df.loc[df["group"] == 1, col].dropna()
    oa_v = df.loc[df["group"] == 2, col].dropna()
    _, p = stats.mannwhitneyu(ya_v, oa_v, alternative="two-sided") if (len(ya_v)>1 and len(oa_v)>1) else (None, None)
    print(f"  {col:28s}  Young={ya_v.mean():.2f}+/-{ya_v.std():.2f}  "
          f"Older={oa_v.mean():.2f}+/-{oa_v.std():.2f}  MW-p={p:.3f}")

print(f"\nPlots saved to: {PLOTS}")

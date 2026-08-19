"""
Step 5: Feature screening.
Step 6: Final pipeline summary / recommended feature set report.

Correlates every candidate feature against age (Spearman + Pearson),
excluding DMCI (deterministic affine transform of VAF, constructed using
the young-group labels -- including it would be a data-leakage / circularity
risk) and the group column itself (label, not a feature).
"""

import pandas as pd
import numpy as np
from scipy import stats

IN_PATH  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_correlations.csv"

RHO_THRESHOLD = 0.3
REDUNDANCY_THRESHOLD = 0.8  # |Pearson r| between two passing features -> flag as non-independent

df = pd.read_csv(IN_PATH)

EXCLUDE_ALWAYS = {"participant", "age", "group"}
dmci_cols = [c for c in df.columns if c.startswith("dmci_")]
excluded_cols = EXCLUDE_ALWAYS.union(dmci_cols)

feature_cols = [c for c in df.columns if c not in excluded_cols]

vaf_cols = [c for c in feature_cols if c.startswith("vaf_") and "ratio" not in c]
cci_cols = [c for c in feature_cols if c.startswith("cci_") and "ratio" not in c]
kinematic_cols = [c for c in feature_cols if c not in vaf_cols + cci_cols
                   and "ratio" not in c]
ratio_cols = [c for c in feature_cols if "ratio" in c]

# ── Step 5: correlation screening ────────────────────────────────────────────
rows = []
for col in feature_cols:
    valid = df[["age", col]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    if n < 5:
        rows.append({"feature": col, "n": n, "spearman_rho": np.nan, "spearman_p": np.nan,
                      "pearson_r": np.nan, "pearson_p": np.nan, "passes_0.3": False})
        continue
    rho, p_s = stats.spearmanr(valid["age"], valid[col])
    r, p_p = stats.pearsonr(valid["age"], valid[col])
    rows.append({
        "feature": col, "n": n,
        "spearman_rho": round(rho, 4), "spearman_p": round(p_s, 4),
        "pearson_r": round(r, 4), "pearson_p": round(p_p, 4),
        "passes_0.3": abs(rho) > RHO_THRESHOLD,
    })

corr_df = pd.DataFrame(rows)
corr_df["abs_rho"] = corr_df["spearman_rho"].abs()
corr_df = corr_df.sort_values("abs_rho", ascending=False, na_position="last").drop(columns="abs_rho")
corr_df.to_csv(OUT_PATH, index=False)

print(f"Saved -> {OUT_PATH}\n")
print(corr_df.to_string(index=False))

passed = corr_df[corr_df["passes_0.3"]].copy()
failed = corr_df[~corr_df["passes_0.3"]].copy()

print(f"\n{'='*55}")
print("FEATURES THAT PASS |rho| > 0.3:")
for _, r in passed.iterrows():
    print(f"  {r['feature']:35s}  rho={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}  n={r['n']}")

print(f"\nFEATURES THAT FAIL |rho| > 0.3: {len(failed)} features")

if len(passed):
    complete_mask = df[passed["feature"].tolist()].notna().all(axis=1)
    n_complete = int(complete_mask.sum())
    print(f"\nParticipants with complete data across all {len(passed)} passing features: {n_complete} / {len(df)}")
else:
    n_complete = 0

# ── Redundancy check among passing features (for Step 6's "independent" note) ─
redundant_pairs = []
if len(passed) >= 2:
    pass_names = passed["feature"].tolist()
    sub = df[pass_names]
    rmat = sub.corr(method="pearson")
    for i in range(len(pass_names)):
        for j in range(i + 1, len(pass_names)):
            r_ij = rmat.iloc[i, j]
            if pd.notna(r_ij) and abs(r_ij) > REDUNDANCY_THRESHOLD:
                redundant_pairs.append((pass_names[i], pass_names[j], round(float(r_ij), 3)))

# ── Step 6: final summary block ──────────────────────────────────────────────
n_participants = len(df)
n_vaf = len(vaf_cols)
n_dmci = len(dmci_cols)
n_cci = len(cci_cols)
n_kinematic = len(kinematic_cols)
n_ratio = len(ratio_cols)
n_total_computed = n_vaf + n_dmci + n_cci + n_kinematic + n_ratio

print(f"\n\n{'PIPELINE SUMMARY':^55}")
print("=" * 55)
print(f"Participants in feature table: {n_participants}")
print(f"Features computed: {n_total_computed} total")
print(f"  - VAF features: {n_vaf}")
print(f"  - DMCI features: {n_dmci} (excluded from screening)")
print(f"  - CCI features: {n_cci}")
print(f"  - Kinematic features: {n_kinematic}")
print(f"  - Cross-task ratios: {n_ratio}")

print(f"\n{'SPEARMAN SCREENING RESULTS (|rho| > 0.3)':^55}")
print("=" * 55)
print("PASSED:")
for _, r in passed.iterrows():
    print(f"  {r['feature']:35s}  rho={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}")
print(f"FAILED: {len(failed)} features")

print("\nRECOMMENDED FEATURE SET FOR MODEL:")
if len(passed) == 0:
    print("  (none passed the |rho| > 0.3 threshold)")
else:
    flagged = set()
    for a, b, r_ij in redundant_pairs:
        flagged.add(a)
        flagged.add(b)
    for _, r in passed.iterrows():
        tag = " [REDUNDANT with another passing feature -- see below]" if r["feature"] in flagged else " [independent]"
        print(f"  {r['feature']}{tag}")
    if redundant_pairs:
        print("\n  Redundant pairs (|Pearson r| > {:.1f} between passing features):".format(REDUNDANCY_THRESHOLD))
        for a, b, r_ij in redundant_pairs:
            print(f"    {a}  <->  {b}   r={r_ij}")
    else:
        print("\n  No pairs of passing features exceed |r| > {:.1f} -- all appear independent.".format(REDUNDANCY_THRESHOLD))

print("\nDATA QUALITY NOTES:")
if len(passed):
    incomplete = df.loc[~complete_mask, "participant"].tolist() if "participant" in df.columns else []
    if incomplete:
        print(f"  {len(incomplete)} participant(s) have at least one NaN among the recommended features: {incomplete}")
    else:
        print("  No participants have missing data among the recommended feature set.")
else:
    print("  N/A -- no features passed screening.")

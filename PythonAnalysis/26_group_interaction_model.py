"""
Group interaction model: Model A (no interaction) vs Model B (group x feature
interactions), compared via likelihood-ratio test. Fit on the FULL cohort
(not LOOCV) -- this is a model-structure test, not a prediction task.

NOTE ON FILENAME: this script was requested as "06_group_interaction_model.py",
but 06_build_feature_table.py already exists as a load-bearing pipeline script
(it produces feature_table_full.csv, which most other scripts this session
depend on). To avoid overwriting it, this is saved as
26_group_interaction_model.py instead -- the next free number in sequence.

This supersedes 22_task5_interaction_model.py (same underlying model, but that
script did not save full coefficient tables or explicit dropped-participant
IDs -- this one does both, as requested).
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_COEF = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\group_interaction_model_results.csv"
OUT_TXT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\group_interaction_model_summary.txt"

FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]
NEEDED = ["participant", "age", "group"] + FEATURES

df = pd.read_csv(DATA)
all_participants = set(df["participant"])

df_model = df.dropna(subset=NEEDED).copy().reset_index(drop=True)
kept_participants = set(df_model["participant"])
dropped_participants = sorted(all_participants - kept_participants)

before_n = len(df)
after_n = len(df_model)

formula_a = "age ~ peak_roll_walkingincline + stride_timing_cv_walking + rom_roll_ratio_stepforward_walk + group"
formula_b = (formula_a +
             " + group:peak_roll_walkingincline + group:stride_timing_cv_walking + group:rom_roll_ratio_stepforward_walk")

model_a = smf.ols(formula_a, data=df_model).fit()
model_b = smf.ols(formula_b, data=df_model).fit()

llf_a, llf_b = model_a.llf, model_b.llf
df_diff = int(model_b.df_model - model_a.df_model)
lr_stat = 2 * (llf_b - llf_a)
p_value = float(stats.chi2.sf(lr_stat, df_diff))
interaction_significant = p_value < 0.05

# ── Coefficient tables ────────────────────────────────────────────────────────
def coef_table(model, model_label):
    t = pd.DataFrame({
        "model": model_label,
        "term": model.params.index,
        "estimate": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    return t

coef_a = coef_table(model_a, "Model A (no interaction)")
coef_b = coef_table(model_b, "Model B (with interaction)")
coef_all = pd.concat([coef_a, coef_b], ignore_index=True)
coef_all.to_csv(OUT_COEF, index=False)

# ── Console output ────────────────────────────────────────────────────────────
print(f"Participants in feature_table_full.csv: {before_n}")
print(f"Dropped due to missing data in {FEATURES}: {len(dropped_participants)}")
print(f"Dropped participant IDs: {dropped_participants}")
print(f"Final N used for both models: {after_n}")
print()
print(f"Model A (no interaction): R2={model_a.rsquared:.4f}, AIC={model_a.aic:.2f}, llf={llf_a:.4f}")
print(f"Model B (with interaction): R2={model_b.rsquared:.4f}, AIC={model_b.aic:.2f}, llf={llf_b:.4f}")
print()
print("MODEL A COEFFICIENTS")
print(coef_a[["term", "estimate", "std_error", "t_value", "p_value"]].to_string(index=False))
print()
print("MODEL B COEFFICIENTS")
print(coef_b[["term", "estimate", "std_error", "t_value", "p_value"]].to_string(index=False))
print()
print(f"Likelihood-ratio test: LR={lr_stat:.4f}, df={df_diff}, p={p_value:.4f}")
print(f"Saved -> {OUT_COEF}")

# ── Plain-language summary + save ────────────────────────────────────────────
conclusion = (
    "The interaction terms ARE statistically significant (p < 0.05): the relationship "
    "between the three features and age appears to differ between young and older "
    "participants, suggesting separate group-specific slopes may be warranted."
    if interaction_significant else
    "The interaction terms are NOT statistically significant (p >= 0.05): there is no "
    "evidence that the relationship between the three features and age differs between "
    "young and older participants. A single shared model is adequate; separate young/old "
    "models are not statistically justified by this test."
)

caveat = (
    f"CAVEAT: this test is fit on N={after_n}, not the full N={before_n} cohort -- "
    f"{len(dropped_participants)} participants ({dropped_participants}) were dropped via "
    f"listwise deletion, almost entirely driven by missingness in stride_timing_cv_walking "
    f"(footstep-detection failures in some trials). Model B has 8 parameters fit on "
    f"{after_n} observations ({after_n/8:.2f} observations per parameter) -- a thin ratio "
    f"that carries real overfitting risk and should temper confidence in this result "
    f"regardless of which way the p-value falls. Treat this as suggestive, not confirmatory."
)

summary_lines = [
    "GROUP INTERACTION MODEL -- SUMMARY",
    "=" * 60,
    f"Participants in feature_table_full.csv: {before_n}",
    f"Participants dropped (missing data): {len(dropped_participants)}",
    f"Dropped participant IDs: {dropped_participants}",
    f"Final N used: {after_n}",
    "",
    f"Model A (no interaction) formula: {formula_a}",
    f"Model B (with interaction) formula: {formula_b}",
    "",
    f"Model A: R2={model_a.rsquared:.4f}  AIC={model_a.aic:.2f}",
    f"Model B: R2={model_b.rsquared:.4f}  AIC={model_b.aic:.2f}",
    "",
    f"Likelihood-ratio test: LR statistic = {lr_stat:.4f}, df = {df_diff}, p-value = {p_value:.4f}",
    "",
    "PLAIN-LANGUAGE CONCLUSION:",
    conclusion,
    "",
    caveat,
]
with open(OUT_TXT, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Saved -> {OUT_TXT}")

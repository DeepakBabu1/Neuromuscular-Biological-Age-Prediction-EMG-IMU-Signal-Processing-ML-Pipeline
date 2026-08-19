"""
Task 5: nested linear models (age ~ features + group, with/without
group*feature interactions), fit on the full dataset (structure test, not
a LOOCV prediction task), compared via likelihood-ratio test.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"

FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]

df = pd.read_csv(DATA)
needed = ["age", "group"] + FEATURES
before_n = len(df)
df_model = df.dropna(subset=needed).copy()
after_n = len(df_model)
if after_n < before_n:
    print(f"NOTE: {before_n - after_n} participant(s) dropped due to missing values in "
          f"{FEATURES} (stride_timing_cv_walking has substantial missingness). "
          f"Fitting on N={after_n}, not the full 29.")

formula_a = "age ~ peak_roll_walkingincline + stride_timing_cv_walking + rom_roll_ratio_stepforward_walk + group"
formula_b = (formula_a +
             " + group:peak_roll_walkingincline + group:stride_timing_cv_walking + group:rom_roll_ratio_stepforward_walk")

model_a = smf.ols(formula_a, data=df_model).fit()
model_b = smf.ols(formula_b, data=df_model).fit()

llf_a, llf_b = model_a.llf, model_b.llf
df_diff = int(model_b.df_model - model_a.df_model)
lr_stat = 2 * (llf_b - llf_a)
p_value = float(stats.chi2.sf(lr_stat, df_diff))

print(f"N = {after_n}")
print(f"Model A (no interaction): R2={model_a.rsquared:.4f}, AIC={model_a.aic:.2f}, log-likelihood={llf_a:.4f}")
print(f"Model B (with interaction): R2={model_b.rsquared:.4f}, AIC={model_b.aic:.2f}, log-likelihood={llf_b:.4f}")
print()
print(f"Likelihood-ratio test: LR = {lr_stat:.4f}, df = {df_diff}, p = {p_value:.4f}")
print()
if p_value >= 0.05:
    print("p >= 0.05: the interaction terms are NOT statistically significant.")
    print("A single shared model is adequate; separate young/old models are NOT statistically justified.")
else:
    print("p < 0.05: the interaction terms ARE statistically significant.")
    print("Group-specific (young vs old) slopes differ significantly -- separate models may be justified.")

result = pd.DataFrame([{
    "N": after_n,
    "model_a_formula": formula_a,
    "model_b_formula": formula_b,
    "model_a_R2": round(model_a.rsquared, 4),
    "model_b_R2": round(model_b.rsquared, 4),
    "model_a_AIC": round(model_a.aic, 2),
    "model_b_AIC": round(model_b.aic, 2),
    "LR_statistic": round(lr_stat, 4),
    "df": df_diff,
    "p_value": round(p_value, 4),
    "interaction_significant": bool(p_value < 0.05),
}])
result.to_csv(r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\interaction_model_results.csv", index=False)
print(f"\nSaved -> interaction_model_results.csv")

"""
TASK 1 -- Missing-data sensitivity check for the group interaction model.
Sensitivity analysis ONLY on the EXISTING interaction model (26_group_interaction_model.py).
No new modelling variants are introduced.

A) Complete-case (N=18): re-fit Model B exactly as before, report full 95% CIs
   (not just p-values) for all terms.
B) Multiple imputation (target N=29): impute missing values in the 3 predictor
   columns via IterativeImputer (BayesianRidge, sample_posterior=True -- proper
   stochastic MICE-style draws, not a single deterministic fill), using
   age + group + the other 2 features as predictors for each missing column.
   20 imputed datasets, Model B refit on each, pooled via Rubin's rules.

NOTE ON SCOPE: 11 participants are dropped in the complete-case model. 10 are
missing stride_timing_cv_walking (as the supervisor flagged); 1 additional
participant is missing peak_roll_walkingincline. To reach the requested N=29
imputed dataset, both columns are imputed jointly (age and group have no
missingness and are used purely as predictors, never imputed).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\interaction_model_imputation_comparison.csv"

FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]
IMPUTE_COLS = ["age", "group"] + FEATURES  # age, group used as predictors only, never imputed (no missingness)
NEEDED = ["participant", "age", "group"] + FEATURES

formula_b = ("age ~ peak_roll_walkingincline + stride_timing_cv_walking + rom_roll_ratio_stepforward_walk + group"
             " + group:peak_roll_walkingincline + group:stride_timing_cv_walking + group:rom_roll_ratio_stepforward_walk")

INTERACTION_TERMS = ["group:peak_roll_walkingincline", "group:stride_timing_cv_walking", "group:rom_roll_ratio_stepforward_walk"]
ALL_TERMS = ["Intercept", "peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk",
             "group"] + INTERACTION_TERMS

df = pd.read_csv(DATA)
n_total = len(df)

# ── A) Complete-case (N=18) ──────────────────────────────────────────────────
df_cc = df.dropna(subset=NEEDED).copy().reset_index(drop=True)
n_cc = len(df_cc)
dropped_ids = sorted(set(df["participant"]) - set(df_cc["participant"]))

model_cc = smf.ols(formula_b, data=df_cc).fit()
ci_cc = model_cc.conf_int(alpha=0.05)
ci_cc.columns = ["ci_lower", "ci_upper"]

cc_table = pd.DataFrame({
    "term": model_cc.params.index,
    "estimate": model_cc.params.values,
    "std_error": model_cc.bse.values,
    "ci_lower": ci_cc["ci_lower"].values,
    "ci_upper": ci_cc["ci_upper"].values,
    "p_value": model_cc.pvalues.values,
})

print(f"N total in feature_table_full.csv: {n_total}")
print(f"Complete-case N: {n_cc}  (dropped {len(dropped_ids)}: {dropped_ids})")
print("\nA) COMPLETE-CASE MODEL B COEFFICIENTS (95% CI):")
print(cc_table.to_string(index=False))

# ── B) Multiple imputation (20 datasets, MICE-style via IterativeImputer) ───
N_IMPUTATIONS = 20
impute_source = df[["participant"] + IMPUTE_COLS].copy()

pooled_params = {term: [] for term in ALL_TERMS}
pooled_vars = {term: [] for term in ALL_TERMS}

for m in range(N_IMPUTATIONS):
    imputer = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                                random_state=m, max_iter=15)
    imputed_values = imputer.fit_transform(impute_source[IMPUTE_COLS])
    imputed_df = pd.DataFrame(imputed_values, columns=IMPUTE_COLS)
    imputed_df["participant"] = impute_source["participant"].values

    model_m = smf.ols(formula_b, data=imputed_df).fit()
    for term in ALL_TERMS:
        if term in model_m.params.index:
            pooled_params[term].append(model_m.params[term])
            pooled_vars[term].append(model_m.bse[term] ** 2)

# Rubin's rules pooling
def rubins_rule(term):
    q = np.array(pooled_params[term])
    u = np.array(pooled_vars[term])
    M = len(q)
    q_bar = q.mean()
    u_bar = u.mean()
    b = ((q - q_bar) ** 2).sum() / (M - 1)
    total_var = u_bar + b + b / M
    se_pooled = np.sqrt(total_var)
    # Rubin (1987) old degrees-of-freedom formula
    if b > 0:
        lam = (b + b / M) / total_var
        df_rubin = (M - 1) / (lam ** 2) if lam > 0 else np.inf
    else:
        df_rubin = np.inf
    df_rubin = max(df_rubin, 1)
    t_crit = stats.t.ppf(0.975, df_rubin)
    ci_lower = q_bar - t_crit * se_pooled
    ci_upper = q_bar + t_crit * se_pooled
    t_stat = q_bar / se_pooled
    p_val = 2 * stats.t.sf(abs(t_stat), df_rubin)
    return q_bar, se_pooled, ci_lower, ci_upper, df_rubin, p_val

mi_rows = []
for term in ALL_TERMS:
    est, se, lo, hi, dfr, p = rubins_rule(term)
    mi_rows.append(dict(term=term, estimate=est, std_error=se, ci_lower=lo, ci_upper=hi,
                         rubin_df=dfr, p_value=p))
mi_table = pd.DataFrame(mi_rows)

print(f"\nB) MULTIPLE IMPUTATION ({N_IMPUTATIONS} datasets, pooled via Rubin's rules, N=29):")
print(mi_table.to_string(index=False))

# ── Side-by-side comparison ──────────────────────────────────────────────────
comparison_rows = []
for term in ALL_TERMS:
    cc_row = cc_table[cc_table["term"] == term].iloc[0]
    mi_row = mi_table[mi_table["term"] == term].iloc[0]

    cc_sig = cc_row["p_value"] < 0.05
    mi_sig = mi_row["p_value"] < 0.05
    same_direction = np.sign(cc_row["estimate"]) == np.sign(mi_row["estimate"])
    ci_overlap = not (mi_row["ci_upper"] < cc_row["ci_lower"] or mi_row["ci_lower"] > cc_row["ci_upper"])

    if cc_sig == mi_sig and same_direction and ci_overlap:
        meaningfully_changed = "No -- same conclusion, overlapping CIs"
    elif cc_sig != mi_sig:
        meaningfully_changed = "Yes -- significance status flips"
    else:
        meaningfully_changed = "Partial -- same significance status, but magnitude/CI shifted"

    comparison_rows.append(dict(
        term=term,
        complete_case_N=n_cc,
        complete_case_estimate=round(cc_row["estimate"], 4),
        complete_case_se=round(cc_row["std_error"], 4),
        complete_case_ci_lower=round(cc_row["ci_lower"], 4),
        complete_case_ci_upper=round(cc_row["ci_upper"], 4),
        complete_case_p=round(cc_row["p_value"], 4),
        imputed_N=n_total,
        imputed_estimate=round(mi_row["estimate"], 4),
        imputed_se=round(mi_row["std_error"], 4),
        imputed_ci_lower=round(mi_row["ci_lower"], 4),
        imputed_ci_upper=round(mi_row["ci_upper"], 4),
        imputed_p=round(mi_row["p_value"], 4),
        imputed_rubin_df=round(mi_row["rubin_df"], 2),
        meaningfully_changed=meaningfully_changed,
    ))

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")

# ── Focused answer on the marginal term ─────────────────────────────────────
focus = comparison[comparison["term"] == "group:peak_roll_walkingincline"].iloc[0]
print(f"\n{'='*70}")
print("FOCUS: group:peak_roll_walkingincline (complete-case p=0.069)")
print(f"{'='*70}")
print(f"Complete-case (N={n_cc}): estimate={focus['complete_case_estimate']}, "
      f"95% CI=[{focus['complete_case_ci_lower']}, {focus['complete_case_ci_upper']}], p={focus['complete_case_p']}")
print(f"Imputed (N={n_total}):     estimate={focus['imputed_estimate']}, "
      f"95% CI=[{focus['imputed_ci_lower']}, {focus['imputed_ci_upper']}], p={focus['imputed_p']}")
print(f"Verdict: {focus['meaningfully_changed']}")

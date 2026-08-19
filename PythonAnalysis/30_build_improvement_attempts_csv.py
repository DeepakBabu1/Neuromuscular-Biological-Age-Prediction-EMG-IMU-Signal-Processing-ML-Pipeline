"""
Consolidates all logistic-regression improvement attempts (Row 1, Row 2/2b,
Improvements 1-5) into one CSV, reusing the exact numbers already computed
this session -- nothing here is recomputed.
"""
import pandas as pd

OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\logistic_regression_improvement_attempts.csv"

rows = [
    dict(attempt="Row 1: Original baseline (fixed 3 features, not re-derived per fold)",
         accuracy=0.6207, sensitivity=0.5385, specificity=0.6875, AUC=0.6779,
         n_features=3, wilcoxon_p_vs_row2b=0.0455, permutation_p=0.032,
         verdict="Reference result as originally reported. NOTE: feature list was fixed "
                 "(chosen once from an earlier age-regression analysis, never re-derived "
                 "against the classification target) -- not a fully nested/honest baseline."),

    dict(attempt="Row 2: Corrected baseline (nested top-3 vs GROUP, per fold)",
         accuracy=0.2414, sensitivity=0.3077, specificity=0.1875, AUC=0.1731,
         n_features=3, wilcoxon_p_vs_row2b=None, permutation_p=0.834,
         verdict="Far worse than Row 1 (Wilcoxon vs Row1 p=0.0023). Selecting features "
                 "against 'group' directly, per fold, performs worse than chance -- "
                 "an unexpected and important negative finding, not adopted as the working baseline."),

    dict(attempt="Row 2b: Corrected baseline (nested top-3 vs AGE, per fold) -- TRUE baseline for Improvements 1-5",
         accuracy=0.4828, sensitivity=0.5385, specificity=0.4375, AUC=0.4952,
         n_features=3, wilcoxon_p_vs_row2b=None, permutation_p=None,
         verdict="Worse than Row 1 (Wilcoxon vs Row1 p=0.0455) but the only methodologically "
                 "honest per-fold-reselected reference point (features re-derived from training "
                 "fold only, against age, matching how the original 3 features were originally chosen). "
                 "Used as the comparison baseline for all Improvements below. Permutation test not run "
                 "(not the primary result; superseded by testing whether any improvement beats it)."),

    dict(attempt="Improvement 1: Up to 5 features, threshold |rho|>0.3 (nested per fold)",
         accuracy=0.4828, sensitivity=0.4615, specificity=0.5000, AUC=0.3894,
         n_features="<=5, varies", wilcoxon_p_vs_row2b=0.1025, permutation_p=None,
         verdict="No improvement in accuracy over Row 2b; AUC worse. Not significant vs Row 2b. Not confirmed."),

    dict(attempt="Improvement 2: class_weight='balanced' (nested top-3 vs age)",
         accuracy=0.5172, sensitivity=0.6923, specificity=0.3750, AUC=0.5481,
         n_features=3, wilcoxon_p_vs_row2b=0.2568, permutation_p=None,
         verdict="Accuracy and AUC both nominally higher than Row 2b, but difference not "
                 "significant (Wilcoxon p=0.2568). Not confirmed."),

    dict(attempt="Improvement 3: C grid search via inner LOOCV (nested top-3 vs age)",
         accuracy=0.4483, sensitivity=0.4615, specificity=0.4375, AUC=0.4615,
         n_features=3, wilcoxon_p_vs_row2b=0.0588, permutation_p=None,
         verdict="No improvement over Row 2b. Permutation test skipped (231 min estimated cost, "
                 "no positive signal to justify it)."),

    dict(attempt="Improvement 4: Decision threshold tuned via inner LOOCV (nested top-3 vs age)",
         accuracy=0.5862, sensitivity=0.8462, specificity=0.3750, AUC=0.4952,
         n_features=3, wilcoxon_p_vs_row2b=0.1797, permutation_p=None,
         verdict="Highest raw accuracy among the per-fold-reselected configurations, but AUC "
                 "unchanged from Row 2b and not significant vs Row 2b (p=0.1797). Gain driven by "
                 "sensitivity/specificity imbalance (predicts 'old' far more often), not by genuine "
                 "improved discrimination. Not confirmed."),

    dict(attempt="Improvement 5: class_weight='balanced' + threshold tuning combined (nested top-3 vs age)",
         accuracy=0.6207, sensitivity=0.9231, specificity=0.3750, AUC=0.5481,
         n_features=3, wilcoxon_p_vs_row2b=0.1025, permutation_p="PENDING (500-shuffle test running, ~56 min)",
         verdict="Accuracy matches Row 1 numerically, and is statistically indistinguishable from Row 1 "
                 "(Wilcoxon vs Row1 p=1.0000) -- but NOT significantly better than Row 2b either (p=0.1025), "
                 "and AUC (0.5481) remains well below Row 1's AUC (0.6779). The gain comes almost entirely "
                 "from very high sensitivity (0.9231) and low specificity (0.3750), i.e. predicting 'old' "
                 "for most participants, not from better discrimination. Permutation test in progress to "
                 "confirm whether AUC=0.5481 exceeds the null distribution."),
]

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")
print(f"Rows: {len(df)}")
pd.set_option("display.max_colwidth", 30)
pd.set_option("display.width", 200)
print(df[["attempt", "accuracy", "AUC", "wilcoxon_p_vs_row2b"]].to_string(index=False))

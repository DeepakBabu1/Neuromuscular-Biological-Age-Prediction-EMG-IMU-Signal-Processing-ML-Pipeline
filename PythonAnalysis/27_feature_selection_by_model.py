"""
Documents exactly which features each model used and how they were chosen,
for every model run this session. Reads only from already-saved files.
"""
import pandas as pd
from collections import Counter

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
FIXED_3 = "peak_roll_walkingincline; stride_timing_cv_walking; rom_roll_ratio_stepforward_walk"
FIXED_8 = ("peak_roll_walkingincline; stride_timing_cv_walking; rom_roll_ratio_stepforward_walk; "
           "peak_roll_walking; rom_pitch_walkingincline; stride_timing_cv_walkingincline; "
           "rom_pitch_stepforward; vaf_stepforward")

def freq_string(series_of_lists, n_folds):
    counts = Counter()
    for feats in series_of_lists:
        for f in feats:
            counts[f] += 1
    ordered = sorted(counts.items(), key=lambda x: -x[1])
    return "; ".join(f"{f}: {c}/{n_folds}" for f, c in ordered)

rows = []

# -- Ridge / Lasso / Linear (nested top-3) -- all three share the SAME per-fold
# selection (05_baseline_comparison.py selects once per fold, reuses for all 3 models)
bc = pd.read_csv(f"{BASE}\\baseline_comparison_v2.csv")
bc_feats = bc["selected_features"].str.split(";")
n29 = len(bc)
freq_bc = freq_string(bc_feats, n29)
final_set_bc = "Varies by fold -- see frequency column (top-3 by |Spearman rho| vs age, re-ranked every fold)"

for model in ["Ridge (nested top-3)", "Lasso (nested top-3)", "Linear (nested top-3)"]:
    rows.append(dict(
        model_name=model,
        feature_selection_method="Per-fold nested",
        n_features_available_as_candidates=33,
        selection_criterion="Top 3 by |Spearman rho| vs age, ranked (not threshold-gated), recomputed on the 28 training participants each fold",
        features_selected_final=final_set_bc,
        selection_varies_by_fold=True,
        most_frequently_selected_features_with_counts=freq_bc,
    ))

# -- gplearn (nested, threshold-based, 10-12 features) --
d8 = pd.read_csv(f"{BASE}\\loocv_predictions.csv")
d8_feats = d8["selected_features"].str.split(";")
n8 = len(d8)
freq_8 = freq_string(d8_feats, n8)
rows.append(dict(
    model_name="gplearn (nested, threshold-based, 10-12 features)",
    feature_selection_method="Per-fold nested",
    n_features_available_as_candidates=33,
    selection_criterion="|Spearman rho| > 0.3 vs age on the 28 training participants; if fewer than 2 pass, top 3 by |rho| taken regardless",
    features_selected_final=f"Varies by fold, {int(d8['n_features_selected'].min())}-{int(d8['n_features_selected'].max())} features -- see frequency column",
    selection_varies_by_fold=True,
    most_frequently_selected_features_with_counts=freq_8,
))

# -- gplearn (nested, forced top-3) --
d12 = pd.read_csv(f"{BASE}\\loocv_gplearn_top3_perfold.csv")
d12_feats = d12[["feature_1", "feature_2", "feature_3"]].values.tolist()
n12 = len(d12)
freq_12 = freq_string(d12_feats, n12)
rows.append(dict(
    model_name="gplearn (nested, forced top-3)",
    feature_selection_method="Per-fold nested",
    n_features_available_as_candidates=33,
    selection_criterion="Top 3 by |Spearman rho| vs age, ranked (not threshold-gated), recomputed on the 28 training participants each fold",
    features_selected_final="Varies by fold -- see frequency column",
    selection_varies_by_fold=True,
    most_frequently_selected_features_with_counts=freq_12,
))

# -- Logistic Regression (classification) -- FIXED list, confirmed via code inspection --
rows.append(dict(
    model_name="Logistic Regression (young/old classification)",
    feature_selection_method="FIXED (not re-derived within this task)",
    n_features_available_as_candidates=33,
    selection_criterion=("Reused, unchanged, the top-3 whole-sample features originally identified for the "
                          "AGE-regression task (peak_roll_walkingincline, stride_timing_cv_walking, "
                          "rom_roll_ratio_stepforward_walk). NOT re-selected per fold, and NOT selected "
                          "against the classification target (group) at all -- selected against a different "
                          "variable (age) on the whole sample. This is a real, if narrower, instance of the "
                          "same whole-sample-selection concern raised elsewhere this session, even though "
                          "model FITTING (train/test split) was correctly nested."),
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A -- fixed every fold (29/29 for all 3)",
))

# -- XGBoost (classification) -- nested; not saved to CSV, but recoverable from run14.log --
import re
xgb_feats = []
with open(f"{BASE}\\run14.log") as f:
    for line in f:
        m = re.search(r"features=\[(.*?)\]", line)
        if m:
            feats = [x.strip().strip("'") for x in m.group(1).split(",")]
            xgb_feats.append(feats)
freq_xgb = freq_string(xgb_feats, len(xgb_feats)) if xgb_feats else "COULD NOT PARSE run14.log"
rows.append(dict(
    model_name="XGBoost (young/old classification)",
    feature_selection_method="Per-fold nested",
    n_features_available_as_candidates=33,
    selection_criterion="Top 3 by |Spearman rho| vs age, ranked (not threshold-gated), recomputed on the 28 training participants each fold",
    features_selected_final="Varies by fold -- see frequency column",
    selection_varies_by_fold=True,
    most_frequently_selected_features_with_counts=freq_xgb + " (recovered from run14.log console output, not a dedicated per-fold CSV)",
))

# -- Group interaction model --
rows.append(dict(
    model_name="Group interaction model (Model A & B)",
    feature_selection_method="FIXED (not selected at all -- pre-specified by hypothesis)",
    n_features_available_as_candidates=33,
    selection_criterion="Not a data-driven selection -- the same 3 features used throughout this session's "
                         "main analysis were specified directly as the model's terms, by design of the test itself.",
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A -- fit once on the full (listwise-deleted) cohort, not cross-validated",
))

# -- Leaky / whole-sample models --
rows.append(dict(
    model_name="Linear (3-feature, whole-sample, leaky)",
    feature_selection_method="Whole-sample (fixed, leaky)",
    n_features_available_as_candidates=33,
    selection_criterion="Top 3 by |Spearman rho| vs age, computed ONCE using all 29 participants before any LOOCV fold was run",
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A -- identical every fold by construction (29/29)",
))
rows.append(dict(
    model_name="gplearn (8-feature, whole-sample, leaky)",
    feature_selection_method="Whole-sample (fixed, leaky)",
    n_features_available_as_candidates=33,
    selection_criterion="|Spearman rho| > 0.3 vs age, computed ONCE using all 29 participants, then pruned to 8 by removing redundant pairs (|Pearson r| > 0.8)",
    features_selected_final=FIXED_8,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A -- identical every fold by construction (29/29)",
))
rows.append(dict(
    model_name="gplearn (3-feature, whole-sample, leaky)",
    feature_selection_method="Whole-sample (fixed, leaky)",
    n_features_available_as_candidates=33,
    selection_criterion="Same fixed 3-feature set as the Linear leaky row above, reused for the gplearn variant",
    features_selected_final=FIXED_3,
    selection_varies_by_fold=False,
    most_frequently_selected_features_with_counts="N/A -- identical every fold by construction (29/29)",
))

table = pd.DataFrame(rows)
out_path = f"{BASE}\\feature_selection_by_model.csv"
table.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
print(f"Rows: {len(table)}")
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.width", 220)
print(table[["model_name", "feature_selection_method", "selection_varies_by_fold"]].to_string(index=False))

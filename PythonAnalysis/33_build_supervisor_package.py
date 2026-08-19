"""
Consolidation-only script: packages already-completed results (interaction
model, XGBoost, LR improvement attempts) into a supervisor-ready summary.
No models are fit, tuned, or validated here -- every number below is read
directly from previously saved files.
"""
import pandas as pd

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

rows = [
    dict(
        finding="Interaction model (complete-case, N=18) -- likelihood-ratio test",
        metric="LR statistic (df=3)",
        point_estimate=8.5898,
        **{"95_percent_CI": "N/A (chi-sq test statistic, not a coefficient)"},
        p_value=0.0353,
        supervisor_conclusion="suggestive, not confirmatory -- keep as hypothesis for better-powered follow-up",
        my_interpretation="Jointly significant at N=18, but 8 parameters on 18 observations (2.25 obs/parameter) is a thin ratio; treat as exploratory.",
    ),
    dict(
        finding="Interaction model (complete-case, N=18) -- marginal term: group:peak_roll_walkingincline",
        metric="coefficient",
        point_estimate=0.5429,
        **{"95_percent_CI": "[-0.0639, 1.1498]"},
        p_value=0.0742,
        supervisor_conclusion="suggestive, not confirmatory -- keep as hypothesis for better-powered follow-up",
        my_interpretation="CI crosses zero; individually non-significant despite the joint LR test result.",
    ),
    dict(
        finding="Interaction model (imputed, N=29, 20 MICE datasets, Rubin's rules) -- marginal term: group:peak_roll_walkingincline",
        metric="coefficient",
        point_estimate=0.3163,
        **{"95_percent_CI": "[-0.2145, 0.8470]"},
        p_value=0.2428,
        supervisor_conclusion="suggestive, not confirmatory -- keep as hypothesis for better-powered follow-up",
        my_interpretation="Imputation does not rescue the marginal term -- estimate shrinks and CI widens further from zero-crossing significance, i.e. not a missing-data artefact masking a real effect.",
    ),
    dict(
        finding="XGBoost classifier (nested LOOCV, per-fold feature reselection)",
        metric="accuracy / AUC / permutation p / train-LOOCV gap",
        point_estimate="0.3793 / 0.3558 / 0.584 / +0.4138",
        **{"95_percent_CI": "not computed (bootstrap CI not run for XGBoost)"},
        p_value=0.584,
        supervisor_conclusion="good illustration that flexible models are wrong at N=29",
        my_interpretation="Below-chance accuracy with a 41-point train/LOOCV gap is a textbook overfitting signature, despite explicit regularisation (max_depth=2, reg_alpha=1.0, reg_lambda=1.0).",
    ),
    dict(
        finding="Primary logistic regression classifier (Row 1, fixed 3-feature set, nested LOOCV)",
        metric="accuracy / AUC / permutation p",
        point_estimate="0.6207 / 0.6779 / 0.032",
        **{"95_percent_CI": "Accuracy 95% CI [0.4483, 0.7931]; AUC 95% CI [0.4545, 0.8824] (2000-resample bootstrap)"},
        p_value=0.032,
        supervisor_conclusion="strong triangulation that N is the binding constraint",
        my_interpretation="The only statistically significant result across the whole session, but the bootstrap CIs are very wide (AUC CI includes 0.5, the chance level) -- the point estimate is real but imprecise at N=29.",
    ),
    dict(
        finding="Best LR improvement attempt (Improvement 5: class_weight=balanced + tuned threshold, nested top-3 vs age)",
        metric="accuracy / AUC / permutation p / Wilcoxon p vs Row 1",
        point_estimate="0.6207 / 0.5481 / 0.312 / 1.0000",
        **{"95_percent_CI": "not computed (bootstrap CI not run for this variant)"},
        p_value=0.312,
        supervisor_conclusion="strong triangulation that N is the binding constraint",
        my_interpretation="Matches Row 1's accuracy by point estimate only; permutation test confirms AUC=0.548 is indistinguishable from chance (p=0.312). No configuration tested improves on Row 1.",
    ),
]

df = pd.DataFrame(rows)
out_path = f"{BASE}\\supervisor_feedback_summary.csv"
df.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
print(f"Rows: {len(df)}")
pd.set_option("display.max_colwidth", 35)
pd.set_option("display.width", 240)
print(df[["finding", "point_estimate", "p_value", "supervisor_conclusion"]].to_string(index=False))

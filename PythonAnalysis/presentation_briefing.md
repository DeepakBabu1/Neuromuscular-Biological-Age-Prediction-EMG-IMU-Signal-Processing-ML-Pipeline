# Presentation Briefing — Interaction Model, XGBoost, LR Improvement Attempts

## Headline
N=29 is the binding constraint, confirmed from multiple independent angles, reported with confidence intervals throughout rather than binary significance calls.

## Finding 1: Group Interaction Model
- Complete-case (N=18): coefficient = 0.5429, 95% CI [-0.0639, 1.1498], p = 0.0742 (marginal term: group:peak_roll_walkingincline); joint likelihood-ratio test LR=8.5898, df=3, p=0.0353
- Imputed (N=29, 20 MICE datasets, Rubin's rules): coefficient = 0.3163, 95% CI [-0.2145, 0.8470], p = 0.2428
- Does imputation change the conclusion? **No** — the imputed estimate is smaller and its CI is wider relative to zero, not narrower. If missing data were suppressing a real effect, imputation would have pulled the estimate away from zero and tightened the CI; instead it moved further toward "no effect." This rules out missingness as the explanation for the marginal complete-case result.
- Verdict: suggestive only, not used to claim young/old need different models.

## Finding 2: XGBoost
- Accuracy = 0.3793, AUC = 0.3558, permutation p = 0.584, train-vs-LOOCV accuracy gap = +0.4138 (79.3% training vs 37.9% honest out-of-sample)
- Verdict: illustrates why flexible models fail at N=29 — even with explicit regularisation (max_depth=2, reg_alpha=1.0, reg_lambda=1.0), a 41-point train/test gap is a textbook overfitting signature, and out-of-sample performance falls below chance.

## Finding 3: LR Improvement Attempts
- Best attempt (Improvement 5: class_weight=balanced + tuned threshold): accuracy = 0.6207, AUC = 0.5481, permutation p = 0.312, Wilcoxon p vs Row 1 = 1.0000 (statistically indistinguishable from the original, not better than it)
- Verdict: triangulates N as the ceiling, not model choice — five independent tuning strategies (more features, class weighting, C tuning, threshold tuning, and their combination) were tried; none produced a confirmed improvement over the honest nested baseline, and the one that matched Row 1's accuracy failed its own permutation test.

## Finding 4: Primary Classifier Uncertainty
- AUC = 0.6779 with 95% CI [0.4545, 0.8824] (2000-resample bootstrap)
- Accuracy = 0.6207 with 95% CI [0.4483, 0.7931] (2000-resample bootstrap)
- What the CI width implies: both intervals are wide (AUC CI spans 0.43, accuracy CI spans 0.34), and the AUC CI **includes 0.5, the chance level**. This is the same message as Findings 1–3 from a different angle: the point estimate (0.678 AUC) is the best available estimate and the one statistically significant result in the whole session (permutation p=0.032), but at N=29 it should not be read as a precise number — the true value could plausibly range from near-chance to quite good.

## Bottom Line
Stop running variants — five improvement attempts, an alternative classifier, and a missing-data sensitivity check all point the same direction. Treat the original logistic regression (Row 1) as the primary result, report the interaction model and XGBoost as exploratory/robustness checks rather than standalone findings, and lead with the sample-size limitation: every analysis this session, regression and classification alike, is consistent with N=29 being the ceiling on what this dataset can resolve, not a failure of model choice or tuning.

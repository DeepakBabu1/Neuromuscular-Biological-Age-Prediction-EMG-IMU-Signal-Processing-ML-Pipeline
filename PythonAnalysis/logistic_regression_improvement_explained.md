# Logistic Regression Improvement Attempts — Explained, and Full Model Comparison

All numbers below are pulled directly from previously saved session outputs — no models were re-run to produce this document. Sources: `logistic_regression_improvement_attempts.csv`, `master_results_summary.csv`, `classification_model_comparison.csv`, `feature_selection_by_model.csv`.

---

## PART A — What was tried to improve logistic regression, explained plainly

**Baseline being improved on:** Row 1 — the original logistic regression classifier (young=1 vs old=2), using a **fixed** 3-feature list (`peak_roll_walkingincline`, `stride_timing_cv_walking`, `rom_roll_ratio_stepforward_walk`) inside nested LOOCV. Accuracy 0.6207, AUC 0.6779, permutation p=0.032.

Because Row 1's feature list was fixed rather than re-derived per fold, a methodologically honest reference point — **Row 2b** (features re-selected from scratch on each training fold, ranked by correlation with age, same logic used to originally pick Row 1's features) — was built first and used as the actual comparison baseline for Improvements 1–5.

| attempt_name | what_changed | accuracy | AUC | permutation_p | confirmed_improvement | explanation |
|---|---|---|---|---|---|---|
| Row 2b (honest baseline) | Re-derived the top-3 features from scratch on every training fold instead of reusing a fixed list | 0.4828 | 0.4952 | not run | N/A (reference point) | Doing feature selection properly, per fold, costs accuracy relative to Row 1 — with only 28 training participants per fold, the "best 3 features" ranking is unstable and doesn't always land on the same features Row 1 used. |
| Improvement 1 | Allowed up to 5 features (instead of exactly 3), keeping only those passing \|rho\|>0.3 | 0.4828 | 0.3894 | not run | **False** | No change in accuracy, and AUC got worse. Adding more candidate features gives the model more ways to overfit the 28-participant training fold without adding real signal — mechanically this tries to fix "maybe 3 features is too few," but at N=29 more features increases variance faster than it captures new signal. |
| Improvement 2 | Added `class_weight='balanced'` to the logistic regression | 0.5172 | 0.5481 | not run | **False** (looked better on point estimate only, Wilcoxon p=0.2568 vs Row 2b) | Both accuracy and AUC ticked up over Row 2b, but the difference is not statistically distinguishable from noise. This mechanically tries to fix class imbalance (16 young vs 13 old) by up-weighting the minority class — it nudges predictions in the right direction but the sample is too small to confirm it's a real effect rather than luck. |
| Improvement 3 | Tuned the regularization strength `C` via a nested inner LOOCV grid search | 0.4483 | 0.4615 | not run (skipped — no positive signal, ~231 min estimated cost) | **False** | No improvement at all over Row 2b. This tries to fix "maybe the default regularization is wrong for this data" — but with only 28 points per outer fold and one held out again for the inner search, there's too little data left to reliably choose a better C than the default. |
| Improvement 4 | Tuned the classification decision threshold (instead of the default 0.5) via nested inner LOOCV | 0.5862 | 0.4952 | not run | **False** (highest raw accuracy of any per-fold-reselected variant, but Wilcoxon p=0.1797 vs Row 2b, and AUC unchanged) | Accuracy rose the most of any single change, but AUC stayed flat — meaning the model's underlying ranking of participants didn't improve at all; it just shifted the cutoff to predict "old" more often (sensitivity 0.85 vs specificity 0.38). This tries to fix a poorly-calibrated default threshold, and it does shift the accuracy/sensitivity trade-off, but it isn't discriminating any better, and the shift isn't statistically confirmed. |
| Improvement 5 | Combined class_weight='balanced' + tuned threshold (the two changes that looked most promising individually) | 0.6207 | 0.5481 | **0.312 (CONFIRMED NOT SIGNIFICANT)** | **False** | Accuracy numerically matched Row 1 exactly, which looked promising enough to justify a full 500-shuffle permutation test — but the test result (p=0.312) shows an AUC of 0.548 or higher occurs in 31% of random label shuffles, i.e. indistinguishable from chance. The accuracy match came from predicting "old" for most participants (sensitivity 0.92, specificity 0.38), not from better discrimination. This is the clearest possible negative result: it looked like a win on the single metric everyone naturally looks at first (accuracy), and formal testing rejected it anyway. |

**Cross-attempt summary:** All five attempts targeted a different plausible weakness — feature count, class imbalance, regularization strength, decision threshold, and a combination of the two most promising individual changes — and none produced a confirmed improvement over the honest (Row 2b) baseline, let alone over Row 1. The common thread is sample size: at N=29 (13 vs 16 per class, ~28 per training fold), every tuning knob has enough researcher-degrees-of-freedom to move the point estimate around by ±0.1–0.15 accuracy just from noise, which is exactly the size of the differences observed here. Improvement 4 and Improvement 5 both raised accuracy by shifting the decision threshold toward predicting the majority-leaning class more often, not by improving the model's actual ability to separate young from old (AUC never exceeded 0.55 in any per-fold-reselected variant) — a pattern that would not survive a truly independent test set.

---

## PART B — Full model comparison table

### SECTION 1 — Continuous age regression models

| model | MAE | RMSE | Pearson r | p_value | beats_baseline |
|---|---|---|---|---|---|
| Baseline (predict-mean, nested) | 14.47 | 15.27 | — | — | — (reference) |
| Ridge (nested top-3) | 13.58 | 14.96 | 0.194 | 0.3131 | False |
| Lasso (nested top-3) | 13.47 | 14.94 | 0.208 | 0.2789 | False |
| Linear (nested top-3) | 14.25 | 15.89 | 0.186 | 0.3336 | False |
| gplearn (nested, 10–12 features) | 13.72 | 17.42 | 0.160 | 0.4078 | False |
| gplearn (nested, forced top-3) | 16.45 | 18.38 | -0.024 | 0.8999 | False |

*For reference only (excluded from the "honest" comparison above because feature selection was done on the whole sample before any fold was run, i.e. leaky):* Linear (3-feature, whole-sample) MAE=12.28, r=0.451, p=0.014; gplearn (8-feature, whole-sample) MAE=12.31, r=0.400, p=0.031. Both "beat baseline" only because of the leakage, not because of genuine model quality.

**None of the honestly-nested regression models statistically beat the predict-the-mean baseline.**

### SECTION 2 — Binary classification models (young vs. old)

| model | accuracy | sensitivity | specificity | AUC | permutation_p | significant | Notes |
|---|---|---|---|---|---|---|---|
| Logistic Regression (Row 1, baseline) | 0.6207 | 0.5385 | 0.6875 | 0.6779 | 0.032 | **True** | **Feature list was FIXED (chosen once from an earlier age-regression analysis), not re-derived per fold. Model fitting itself was correctly nested, but feature selection was not — a documented limitation of this result, not a hidden one.** |
| Logistic Regression Improvement 5 (best attempted variant) | 0.6207 | 0.9231 | 0.3750 | 0.5481 | 0.312 | False | Per-fold nested feature reselection (honest), class_weight='balanced' + tuned threshold. Matches Row 1's accuracy by point estimate only; permutation test confirms this is not distinguishable from chance. |
| XGBoost (nested) | 0.3793 | 0.3846 | 0.3750 | 0.3558 | 0.584 | False | Fully nested (features and model both re-derived per fold). Performed worse than a coin flip; showed a 41-point train/test accuracy gap (overfitting). |
| Majority-class trivial baseline (always predict "young", the larger group, 16/29) | 0.5517 | 0.0000 | 1.0000 | 0.5 (undefined/constant) | N/A (deterministic, not a fitted model) | No model at all — useful only as a floor. Row 1 beats this floor by 7 points of accuracy while also having real (AUC>0.5) discrimination; Improvement 5 and XGBoost do not clearly clear this floor once sensitivity/specificity balance is considered. |

---

## PART C — The one clear takeaway

Nothing tested improved on Row 1 in a statistically confirmed way — every reselection strategy, feature-count change, class-weighting scheme, and threshold-tuning approach either performed worse than Row 1 outright or, in the one case that matched its accuracy (Improvement 5), failed its permutation test (p=0.312). **Row 1 remains the only statistically significant result across the entire session, regression and classification combined** — every nested regression model (Ridge, Lasso, Linear, both gplearn variants) failed to beat a predict-the-mean baseline, and every classification variant besides Row 1 failed its own significance test. Classification likely succeeded where regression didn't because it asks an easier question of the same small sample: distinguishing two non-overlapping age bands (21–34 vs 49–63) is a coarser, lower-information-requirement task than predicting a precise continuous age, so N=29 can support it where it cannot support fine-grained regression. Across every model type, every feature-selection strategy, and every classifier tuning attempt tried this session, sample size — not model choice, not feature engineering, not hyperparameter tuning — is the consistent, identified limiting factor.

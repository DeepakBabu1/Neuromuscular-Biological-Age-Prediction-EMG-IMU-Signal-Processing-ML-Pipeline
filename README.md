# Neuromuscular Ageing: Predicting Biological Age from Wearable EMG/IMU Sensors

Reprocessing and extension of Burke et al. (2024)'s wearable sensor dataset (sEMG + IMU, N=29,
young vs. older adults) to predict continuous biological neuromuscular age via symbolic regression,
and to rigorously test whether the small sample size (N=29) is a solvable data problem or a hard
ceiling on what any model can extract.

## What this project does

1. **Reimplements the original MATLAB signal-processing pipeline in Python** (`emg_pipeline.py`,
   `run_pipeline.py`) — EMG filtering/rectification/envelope extraction, IMU-based stride
   segmentation, NMF-based muscle synergy extraction (VAF), Dynamic Motor Control Index (DMCI),
   Co-Contraction Index (CCI), and IMU kinematic features — while documenting every discrepancy
   found between the original code's stated intent and its actual numerical behaviour.
2. **Builds a 33-feature candidate table** (`06_build_feature_table.py`) spanning VAF, CCI, IMU
   kinematics, and cross-task ratios across four exercises (flat walking, incline walking,
   step-forward, step-lateral).
3. **Compares regression models** (baseline predict-mean, Linear, Ridge, Lasso, symbolic
   regression via gplearn, Gaussian Process Regression) for continuous age prediction, all under
   **nested leave-one-participant-out cross-validation** — feature selection, scaling, and
   imputation are re-derived from scratch on each fold's 28 training participants, never leaking
   information from the held-out participant.
4. **Reframes the problem as binary young/old classification** (logistic regression, XGBoost),
   given the cohort's natural age-distribution gap (21–34 vs. 49–63 years).
5. **Runs a battery of robustness checks**: permutation testing, bootstrap confidence intervals,
   Beheshti-style bias correction, Bland-Altman analysis, a group×feature interaction model with
   multiple-imputation sensitivity analysis, five systematic logistic-regression improvement
   attempts, three data-augmentation methods (with Bonferroni correction across comparisons), and
   a Gaussian Process calibration check (does the model's self-reported uncertainty track its
   actual error?).

## Headline finding

Across the entire modelling investigation, **exactly one result reaches statistical significance**
under honest, leakage-free evaluation: the primary logistic regression classifier for young/old
classification (accuracy 62.1%, AUC 0.678, permutation p=0.032) — and even that result's own
bootstrap confidence interval on AUC is wide enough to include chance-level performance. Every
continuous age-regression model, every alternative classifier, every improvement attempt, and
every data-augmentation method failed to beat its respective baseline in a statistically confirmed
way. The consistent, cross-validated conclusion is that **N=29 is the binding constraint** — not
model choice, feature engineering, or hyperparameter tuning.

## Repository structure

```
emg_pipeline.py, run_pipeline.py     Core signal-processing pipeline (MATLAB → Python)
01-17_*.py                           Feature engineering, screening, baseline/symbolic regression models
18-24_*.py                           Statistical robustness checks (bias correction, Bland-Altman,
                                      interaction model, classification, consolidation)
25-34_*.py                           Consolidated summaries, group interaction sensitivity analysis,
                                      logistic regression improvement attempts
36-42_*.py                           Data augmentation robustness check, Gaussian Process Regression,
                                      Figure 2.1 participant flow diagram
augment_lib.py, lr_improve_lib.py    Shared, importable analysis modules
*.csv, *.md, *.txt                   Results, per-participant predictions, and written summaries
plots/                                Final figures referenced in the thesis
pipeline_outputs/                    Final reprocessed pipeline output (pipeline_results_v7.csv)
```

## Methodology highlights

- **Nested cross-validation throughout**: every feature-selection, scaling, and imputation step is
  re-derived per fold from training data only — a central, repeatedly-verified theme of this
  project, since whole-sample feature selection was found to produce misleadingly optimistic
  results.
- **Statistical rigor**: every reported result is accompanied by a Wilcoxon signed-rank test,
  permutation test, and/or bootstrap confidence interval — never a bare point estimate.
- **Documented pipeline discrepancies**: e.g. the original MATLAB filter stage's code comment says
  "highpass filtering" but the actual `butter()` call produces a low-pass filter; this thesis
  replicates the pipeline's actual numerical behaviour and states the discrepancy explicitly
  rather than silently "fixing" it.

## Requirements

Python 3.11 (required for `gplearn` compatibility). Key dependencies: `numpy`, `pandas`, `scipy`,
`scikit-learn`, `gplearn`, `xgboost`, `statsmodels`, `matplotlib`.

## Data availability

Raw participant sEMG/IMU data are not included in this repository (participant privacy). Scripts
expect the source data directory structure documented in `emg_pipeline.py`.

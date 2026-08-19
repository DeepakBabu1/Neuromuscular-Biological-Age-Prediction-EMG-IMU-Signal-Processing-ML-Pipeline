import pandas as pd
pd.set_option('display.max_colwidth', 55)
pd.set_option('display.width', 250)

print('='*100)
print('1. MASTER RESULTS SUMMARY (master_results_summary.csv)')
print('='*100)
t = pd.read_csv('master_results_summary.csv')
print(t.to_string(index=False))

print()
print('='*100)
print('2. WHICH MODEL(S) ARE STATISTICALLY SIGNIFICANT AFTER PROPER NESTED VALIDATION')
print('='*100)
print('None of Ridge, Lasso, Linear, or gplearn (nested, either 10-12 features or forced')
print('top-3) reach p<0.05 -- the ONLY statistically significant, properly-nested result in')
print('the entire session is the binary young-vs-old logistic classification (p=0.032).')
print('Every regression result with p<0.05 (the 8-feature/3-feature gplearn/linear leaky rows)')
print('used feature selection that saw the whole sample, including test participants, before')
print('cross-validation -- i.e. the significant-looking regression results are the leaky ones.')

print()
print('='*100)
print('3. BIAS CORRECTION SUMMARY (bias_correction_summary.csv)')
print('='*100)
b = pd.read_csv('bias_correction_summary.csv')
print(b.drop(columns='circularity_note').to_string(index=False))
print()
print('Circularity note (applies to all 4 rows):')
print(' ', b['circularity_note'].iloc[0])

print()
print('='*100)
print('4. BINARY CLASSIFICATION RESULT')
print('='*100)
clf = pd.read_csv('classification_young_vs_old_metrics.csv').iloc[0]
print(f"Accuracy={clf['accuracy']:.4f}  AUC-ROC={clf['AUC_ROC']:.4f}  Permutation p-value=0.032 (500 shuffles, significant)")

print()
print('='*100)
print('5. OPEN ITEMS REQUIRING SUPERVISOR DECISION')
print('='*100)
items = [
    "Participant 27: steplateral and walking EMG files confirmed data-identical (SHA256 match "
    "on internal worksheet XML) -- exclude, or retain with documented caveat? Not yet decided.",

    "Filter configuration: Chapter 2 draft states the high-pass configuration was used per "
    "supervisory guidance, but emg_pipeline.py still implements low-pass (matches MATLAB's "
    "actual behaviour, not its comment). Every result in this session was computed with "
    "low-pass. Needs clarifying: was the switch to high-pass already decided and not yet "
    "implemented, or does the chapter text need correcting to describe what was actually run?",

    "Participant 11 walking-incline dropout count in Chapter 2 draft says '8 of 13' muscles; "
    "verified actual = 7 (8 belongs to the step-forward trial, not walking-incline).",

    "Stride-timing missingness counts in Chapter 2 draft are inverted: text implies 19/29 and "
    "15/29 participants are MISSING stride_timing_cv for flat/incline walking; verified these "
    "are actually the VALID (non-missing) counts -- true missing counts are 10/29 and 14/29.",

    "Participant 33 file correction (steplateral EMG substituted from participant 3, verified "
    "as a genuine error, and fixed) is undocumented in Chapter 2 Section 2.4.1 prose, though "
    "present in code comments.",

    "05_baseline_comparison.py does not apply StandardScaler despite Chapter 2 stating all "
    "non-trivial models were preceded by feature standardisation -- affects interpretation of "
    "the Ridge/Lasso alpha values reported.",

    "Group interaction model (Task 5) is fit on N=18, not 29, due to stride_timing_cv_walking "
    "missingness -- the p=0.035 result should be presented with this caveat, not as a clean finding.",
]
for i, it in enumerate(items, 1):
    print(f'{i}. {it}')

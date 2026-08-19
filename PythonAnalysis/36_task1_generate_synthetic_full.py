"""
TASK 1 -- Generate synthetic participants (full 29-real-participant sample,
for documentation/inspection purposes only). NOT used for model evaluation --
see 37_task2_task3_augmented_nested_loocv.py for the leakage-safe, per-fold
regeneration actually used to evaluate models.
"""
import pandas as pd
import augment_lib as A

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"

df = pd.read_csv(DATA)
df = df.dropna(subset=["group"]).reset_index(drop=True)
real_tagged = A.format_real(df)
print(f"Real participants: {len(df)}")

for method_name, gen_fn, kwargs in [
    ("noise", A.generate_noise, dict(n_per_real=3, seed=1)),
    ("smote", A.generate_smote, dict(target_total=100, seed=1)),
    ("gmm", A.generate_gmm, dict(target_total=100, seed=1)),
]:
    synth = gen_fn(df, **kwargs)
    combined = pd.concat([real_tagged, synth], ignore_index=True)
    out_path = f"{BASE}\\augmented_data_{method_name}.csv"
    combined.to_csv(out_path, index=False)
    print(f"{method_name}: {len(synth)} synthetic + {len(df)} real = {len(combined)} total -> {out_path}")

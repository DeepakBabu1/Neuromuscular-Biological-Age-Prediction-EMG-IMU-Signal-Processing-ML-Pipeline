"""
Refit of 04_final_model.py's "all 29 participants" equation-discovery model,
with heavy parsimony (0.5, up from 0.0005) and a hard 15-node ceiling.

IMPORTANT API NOTE: gplearn's SymbolicRegressor has no native "reject any
individual over N nodes outright during evolution" option -- the `metric`
callback (which parsimony_coefficient's penalty is layered on top of) only
receives (y, y_pred, sample_weight), not the program object, so a custom
fitness function cannot see node count either. The two available levers are:
  1. parsimony_coefficient (soft, but at 0.5 it's ~1000x the previous 0.0005,
     strong enough that oversized trees need near-zero error to be selected
     at all, which is astronomically unlikely)
  2. init_depth biased small, as a population-seeding nudge (does not
     strictly prevent later growth via crossover)
The HARD 15-node ceiling requested is therefore enforced here as a post-hoc
filter: each of the 5 seeds is fit and its result checked after the fact.
Any seed whose program exceeds 15 nodes is marked REJECTED and excluded
from "best equation" selection -- it is reported for transparency but never
presented as a usable result. This is the honest way to deliver "hard
rejection" given what gplearn's public API actually allows.
"""

import numpy as np
import pandas as pd
from scipy import stats
from gplearn.genetic import SymbolicRegressor

DATA = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
OUT_CSV = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\final_model_constrained_seeds.csv"

FEATURE_COLS = [
    "peak_roll_walkingincline",
    "stride_timing_cv_walking",
    "rom_roll_ratio_stepforward_walk",
    "peak_roll_walking",
    "rom_pitch_walkingincline",
    "stride_timing_cv_walkingincline",
    "rom_pitch_stepforward",
    "vaf_stepforward",
]

N_SEEDS = 5
POPULATION = 3000
GENERATIONS = 60
PARSIMONY = 0.5
MAX_NODES = 15

df = pd.read_csv(DATA)
df_model = df[["participant", "age"] + FEATURE_COLS].dropna(subset=["age"]).copy()
df_model = df_model[df_model[FEATURE_COLS].notna().sum(axis=1) >= len(FEATURE_COLS) // 2].copy()
df_model = df_model.reset_index(drop=True)

X = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median()).values.astype(float)
y = df_model["age"].values.astype(float)
baseline_mae = float(np.mean(np.abs(y - y.mean())))

print(f"n={len(df_model)} participants, baseline (predict-mean) MAE={baseline_mae:.2f} yr")
print(f"parsimony_coefficient={PARSIMONY}  hard ceiling={MAX_NODES} nodes (post-hoc filter, see docstring)\n")

results = []
for seed in range(N_SEEDS):
    model = SymbolicRegressor(
        population_size=POPULATION,
        generations=GENERATIONS,
        tournament_size=20,
        const_range=(-10.0, 10.0),
        init_depth=(2, 3),  # bias toward small initial trees, complements heavy parsimony
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "inv"),
        metric="mean absolute error",
        parsimony_coefficient=PARSIMONY,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        n_jobs=1,
        verbose=0,
        random_state=seed,
    )
    model.fit(X, y)
    preds = model.predict(X)
    mae = float(np.mean(np.abs(preds - y)))
    r, _ = stats.pearsonr(y, preds)
    n_nodes = int(model._program.length_)
    depth = int(model._program.depth_)
    eq = str(model._program)
    passes = n_nodes <= MAX_NODES

    def _readable(eq_str, feature_names):
        for idx in range(len(feature_names) - 1, -1, -1):
            eq_str = eq_str.replace(f"X{idx}", feature_names[idx])
        return eq_str

    eq_readable = _readable(eq, FEATURE_COLS)

    results.append({
        "seed": seed, "mae": mae, "r": r, "n_nodes": n_nodes,
        "depth": depth, "passes_ceiling": passes, "equation": eq,
        "equation_readable": eq_readable,
    })

    verdict = "OK  (<=15 nodes)" if passes else "REJECTED (>15 nodes -- hard ceiling)"
    print(f"Seed {seed}: MAE={mae:.2f} yr  r={r:.3f}  nodes={n_nodes}  depth={depth}  [{verdict}]")
    print(f"  eq (exact):    {eq}")
    print(f"  eq (readable): {eq_readable}\n")

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

valid = [res for res in results if res["passes_ceiling"]]

print(f"\n{'='*65}")
print(f"RESULT: {len(valid)}/{N_SEEDS} seeds produced an equation at or under {MAX_NODES} nodes.")
print(f"{'='*65}")

if valid:
    best = min(valid, key=lambda res: res["mae"])
    readable = best["equation"]
    for idx in range(len(FEATURE_COLS) - 1, -1, -1):
        readable = readable.replace(f"X{idx}", FEATURE_COLS[idx])

    print(f"\nBEST VALID EQUATION (seed {best['seed']}, confirmed {best['n_nodes']} <= {MAX_NODES} nodes):")
    print(f"  Training MAE: {best['mae']:.2f} yr   r: {best['r']:.3f}   nodes: {best['n_nodes']}")
    print(f"\n  Exact:    {best['equation']}")
    print(f"\n  Readable: {readable}")

    delta = baseline_mae - best["mae"]
    print(f"\nComparison to baseline: {best['mae']:.2f} yr vs {baseline_mae:.2f} yr baseline "
          f"(improvement: {delta:+.2f} yr)")
    if delta < 1.0:
        print("-> NOT a meaningful improvement over baseline. Even under a hard 15-node ceiling, "
              "no simple equation was found that beats predicting the mean age by a practically "
              "useful margin. This confirms no simple equation exists in this feature set either.")
    else:
        print("-> This is a meaningful improvement over baseline -- but note this is still an "
              "IN-SAMPLE training MAE with no cross-validation, so it should not be taken as "
              "evidence of a generalizable relationship without a LOOCV check.")
else:
    print("\nNO seed produced an equation at or under 15 nodes, even with parsimony_coefficient=0.5.")
    print("All 5 discovered equations exceeded the hard ceiling and are REJECTED -- there is no")
    print("valid 'best equation' to report from this run. This itself is informative: pushing")
    print("parsimony 1000x higher still didn't force gplearn into a truly simple solution here,")
    print("suggesting either no simple equation fits this data well, or the search needs a")
    print("different mechanism (e.g. lower generations/population, or a GP library with a true")
    print("hard depth/node cap enforced during evolution) to reliably land under 15 nodes.")

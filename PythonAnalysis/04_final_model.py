"""
Step 4: Final symbolic regression model on all data
Refits on the full dataset to discover the best interpretable formula.
Produces the Pareto front (complexity vs accuracy) and the final equation.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gplearn.genetic import SymbolicRegressor

DATA  = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\feature_table_full.csv"
PLOTS = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\plots"
os.makedirs(PLOTS, exist_ok=True)

# Same 8-feature screened + pruned set as 03_symbolic_regression.py (see that
# file's header comment for the screening/pruning rationale).
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

df = pd.read_csv(DATA)

df_model = df[["participant", "age"] + FEATURE_COLS].dropna(subset=["age"]).copy()
df_model = df_model[df_model[FEATURE_COLS].notna().sum(axis=1) >= len(FEATURE_COLS)//2].copy()
df_model = df_model.reset_index(drop=True)

X = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median()).values.astype(float)
y = df_model["age"].values.astype(float)

print(f"Full dataset: n={len(df_model)} participants")


def readable_equation(eq: str, feature_names) -> str:
    for idx in range(len(feature_names) - 1, -1, -1):
        eq = eq.replace(f"X{idx}", feature_names[idx])
    return eq

# Run multiple seeds and report best
N_SEEDS     = 5
N_GENS      = 60
POPULATION  = 3000

best_mae    = np.inf
best_model  = None
best_seed   = None

print(f"Running {N_SEEDS} seeds (gens={N_GENS}, pop={POPULATION})...")
for seed in range(N_SEEDS):
    model = SymbolicRegressor(
        population_size=POPULATION,
        generations=N_GENS,
        tournament_size=20,
        const_range=(-10.0, 10.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "abs", "neg", "inv"),
        metric="mean absolute error",
        parsimony_coefficient=0.0005,
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
    mae   = float(np.mean(np.abs(preds - y)))
    r, _  = stats.pearsonr(y, preds)
    print(f"  Seed {seed}: MAE={mae:.2f} yrs  r={r:.3f}  eq={readable_equation(str(model._program), FEATURE_COLS)[:90]}")
    if mae < best_mae:
        best_mae   = mae
        best_model = model
        best_seed  = seed

best_eq_str = str(best_model._program)
best_eq_readable = readable_equation(best_eq_str, FEATURE_COLS)

print(f"\nBest seed: {best_seed}  (MAE={best_mae:.2f} yrs)")
print(f"Best equation (exact):    {best_eq_str}")
print(f"Best equation (readable): {best_eq_readable}")
print(f"  Complexity (# nodes): {best_model._program.length_}")
print(f"  Depth               : {best_model._program.depth_}")

# Training metrics
preds_final = best_model.predict(X)
r_final, p_final = stats.pearsonr(y, preds_final)
print(f"  Train r             : {r_final:.3f}  (p={p_final:.4f})")
print(f"  Train MAE           : {best_mae:.2f} yrs  (note: not cross-validated)")

# Save equation summary
eq_df = pd.DataFrame({
    "seed":       [best_seed],
    "equation":   [best_eq_str],
    "equation_readable": [best_eq_readable],
    "n_nodes":    [best_model._program.length_],
    "depth":      [best_model._program.depth_],
    "train_mae":  [round(best_mae, 3)],
    "train_r":    [round(r_final, 3)],
    "train_p":    [round(p_final, 4)],
})
eq_df.to_csv(os.path.join(PLOTS, "final_model_equation.csv"), index=False)

# ── Plot: predicted vs actual (full dataset) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y, preds_final, alpha=0.75, s=55, zorder=3,
           c=y, cmap="coolwarm")
lims = [min(y.min(), preds_final.min()) - 3, max(y.max(), preds_final.max()) + 3]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Age (years)", fontsize=12)
ax.set_ylabel("Predicted Age (years)", fontsize=12)
ax.set_title(f"Final Symbolic Model (full dataset, n={len(df_model)})\n"
             f"Train MAE={best_mae:.1f} yr  r={r_final:.3f}\n"
             f"Eq: {str(best_model._program)[:60]}...", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "final_model_train.png"), dpi=150)
plt.close()

# ── Population fitness histogram ──────────────────────────────────────────────
# Extract fitness of best individual in each generation (from run_details_)
if hasattr(best_model, "run_details_"):
    rd = best_model.run_details_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rd["generation"], rd["best_fitness"], label="Best fitness (MAE)")
    ax.plot(rd["generation"], rd["average_fitness"], label="Population avg MAE", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("MAE (years)")
    ax.set_title("GP Convergence Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "gp_convergence.png"), dpi=150)
    plt.close()

print(f"\nDone. Outputs saved to: {PLOTS}")

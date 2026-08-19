"""
Reusable synthetic-data-augmentation functions. Pure importable module, no
top-level execution (avoids the exec/split bug from earlier this session).

CRITICAL SAFEGUARD, honored everywhere in this module: every function takes
an input real-participant dataframe and returns ONLY synthetic rows derived
from it. Callers are responsible for ensuring the input never includes a
held-out test participant -- these functions have no way to enforce that
themselves, so the caller-side discipline (see 37_task2_task3_...py) is the
actual safeguard.

FEATURES: augmentation operates ONLY on the primary 3-feature set (per the
task spec), not the full 33-candidate pool. This means Ridge/Lasso evaluated
on augmented data cannot use per-fold top-3 reselection from 33 candidates
(that data doesn't exist for synthetic rows) -- they use this fixed primary
set instead. This is flagged explicitly wherever results are reported, since
it is a real, if minor, deviation from the "Ridge (nested top-3)" baseline
methodology (in practice the primary set matches the top-3 selection in
26-28 of 29 real folds anyway, per feature_selection_by_model.csv).
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

FEATURES = ["peak_roll_walkingincline", "stride_timing_cv_walking", "rom_roll_ratio_stepforward_walk"]


def _prep_real(df_real):
    """Median-impute missing feature values using ONLY the given (training) real
    participants -- matches the training-fold-median convention used throughout
    this session. Returns a clean copy plus the medians used (for the caller to
    reuse on the held-out test participant)."""
    d = df_real.copy().reset_index(drop=True)
    medians = d[FEATURES].median()
    d[FEATURES] = d[FEATURES].fillna(medians)
    return d, medians


def generate_noise(df_real, n_per_real=3, noise_frac_range=(0.05, 0.10), seed=0):
    """METHOD A -- Gaussian noise injection. 3 synthetic copies per real
    participant (age/group unchanged), noise = 5-10% of each feature's SD
    (SD computed from the given real sample only)."""
    d, _ = _prep_real(df_real)
    rng = np.random.default_rng(seed)
    sds = d[FEATURES].std(ddof=1)

    rows = []
    for _, r in d.iterrows():
        for copy_i in range(n_per_real):
            frac = rng.uniform(*noise_frac_range, size=len(FEATURES))
            noise = rng.normal(0, 1, size=len(FEATURES)) * frac * sds.values
            new_feats = r[FEATURES].values.astype(float) + noise
            rows.append({
                "participant": f"{int(r['participant'])}_noise{copy_i}",
                "age": r["age"], "group": r["group"],
                **dict(zip(FEATURES, new_feats)),
            })
    out = pd.DataFrame(rows)
    out["is_synthetic"] = True
    out["synthetic_method"] = "noise"
    return out


def generate_smote(df_real, target_total=100, k_neighbors=5, seed=0):
    """METHOD B -- SMOTE-style interpolation, strictly within-group (never
    across the young/old age gap). Age is interpolated with the same ratio
    as the features, so synthetic ages stay within each group's real range."""
    d, _ = _prep_real(df_real)
    rng = np.random.default_rng(seed)
    n_real = len(d)
    n_synth_needed = max(target_total - n_real, 0)

    rows = []
    for group_val, gdf in d.groupby("group"):
        gdf = gdf.reset_index(drop=True)
        n_group = len(gdf)
        n_synth_group = round(n_synth_needed * n_group / n_real)
        if n_group < 2 or n_synth_group == 0:
            continue
        k = min(k_neighbors, n_group - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(gdf[FEATURES].values)
        _, neighbor_idx = nn.kneighbors(gdf[FEATURES].values)

        for i in range(n_synth_group):
            anchor_i = rng.integers(0, n_group)
            candidate_neighbors = neighbor_idx[anchor_i][1:]  # exclude self
            neighbor_i = rng.choice(candidate_neighbors)
            ratio = rng.uniform(0, 1)

            anchor = gdf.iloc[anchor_i]
            neighbor = gdf.iloc[neighbor_i]
            new_feats = anchor[FEATURES].values.astype(float) + ratio * (
                neighbor[FEATURES].values.astype(float) - anchor[FEATURES].values.astype(float))
            new_age = anchor["age"] + ratio * (neighbor["age"] - anchor["age"])

            rows.append({
                "participant": f"{int(anchor['participant'])}x{int(neighbor['participant'])}_smote{i}",
                "age": new_age, "group": group_val,
                **dict(zip(FEATURES, new_feats)),
            })
    out = pd.DataFrame(rows)
    out["is_synthetic"] = True
    out["synthetic_method"] = "smote"
    return out


def generate_gmm(df_real, target_total=100, n_components=1, seed=0):
    """METHOD C -- Gaussian Mixture Model sampling, fit separately per group
    on [age + 3 primary features] jointly (so sampled age stays correlated
    with sampled features rather than being assigned independently).

    SAFEGUARD: n_components=1 by default. With only ~13-16 real participants
    per group in a training fold, a multi-component GMM would itself be
    dangerously overfit (more free covariance parameters than data can
    support); a single multivariate Gaussian per group is the more
    defensible choice at this scale and is used deliberately, not as an
    oversight."""
    d, _ = _prep_real(df_real)
    rng_seed = seed
    n_real = len(d)
    n_synth_needed = max(target_total - n_real, 0)
    dims = ["age"] + FEATURES

    rows = []
    for group_val, gdf in d.groupby("group"):
        gdf = gdf.reset_index(drop=True)
        n_group = len(gdf)
        n_synth_group = round(n_synth_needed * n_group / n_real)
        if n_group < 2 or n_synth_group == 0:
            continue
        nc = min(n_components, max(1, n_group - 1))
        gmm = GaussianMixture(n_components=nc, covariance_type="full",
                               reg_covar=1e-3, random_state=rng_seed)
        gmm.fit(gdf[dims].values)
        sampled, _ = gmm.sample(n_synth_group)

        for i, s in enumerate(sampled):
            row = dict(zip(dims, s))
            rows.append({
                "participant": f"gmm_g{int(group_val)}_{i}",
                "age": row["age"], "group": group_val,
                **{f: row[f] for f in FEATURES},
            })
    out = pd.DataFrame(rows)
    out["is_synthetic"] = True
    out["synthetic_method"] = "gmm"
    return out


def format_real(df_real):
    """Tags real participants with is_synthetic=False, synthetic_method='none',
    for concatenation with synthetic output. Preserves original (possibly
    missing) values -- imputation happens only internally, inside the
    generator functions, when computing distances/noise/GMM fits."""
    out = df_real[["participant", "age", "group"] + FEATURES].copy()
    out["is_synthetic"] = False
    out["synthetic_method"] = "none"
    return out

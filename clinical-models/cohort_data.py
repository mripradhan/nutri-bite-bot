"""
Leakage-safe data handling for the MIMIC-IV cohort built by mimic_extract.py.

  make_split()     patient-level (subject_id) held-out test set + grouped CV folds
  india_weights()  raking weights matching Indian HTN/DM/CKD prevalence
  synthesize()     Gaussian-copula synthetic rows for under-represented comorbidity strata

Synthetic rows and weights must be computed from TRAINING rows only (call them
inside each CV fold / on the final training set); they are never used for
evaluation. Every reported metric comes from real held-out patients.

CLI (writes the fixed split + an aggregate-only split card):
    python cohort_data.py --tag v3.1
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

from mimic_extract import FEATURES, SMALL_CELL, TARGETS, ckd_epi_2021, suppress

DERIVED = Path(__file__).resolve().parent.parent / "data" / "derived"
PREVALENCE_PATH = Path(__file__).resolve().parent / "config" / "indian_prevalence.json"
FLAGS = ["has_htn", "has_dm", "has_ckd"]
# Continuous columns the copula models; eGFR is recomputed from creatinine/age/sex
# so synthetic rows stay internally consistent.
COPULA_COLS = ["age", "sex_male", "serum_sodium", "serum_potassium", "creatinine",
               "hba1c", "fbs", "sbp", "dbp", "bmi"] + TARGETS


BASELINE_CHEMISTRY = ["serum_sodium", "serum_potassium", "creatinine"]


def load_cohort(tag: str) -> pd.DataFrame:
    """Admissions eligible for modelling: at least one baseline chemistry result
    (Na, K or creatinine in the first 24 h); otherwise the row is mostly imputed values."""
    df = pd.read_parquet(DERIVED / f"cohort_{tag}.parquet")
    return df[df[BASELINE_CHEMISTRY].notna().any(axis=1)].reset_index(drop=True)


def n_excluded(tag: str) -> dict:
    full = pd.read_parquet(DERIVED / f"cohort_{tag}.parquet", columns=["subject_id"] + BASELINE_CHEMISTRY)
    drop = ~full[BASELINE_CHEMISTRY].notna().any(axis=1)
    return {"n_admissions_before": int(len(full)), "n_excluded_no_baseline_chemistry": int(drop.sum())}


def stratum(df: pd.DataFrame) -> pd.Series:
    return df["has_htn"].astype(str) + df["has_dm"].astype(str) + df["has_ckd"].astype(str)


def make_split(df: pd.DataFrame, n_test_folds: int = 5, n_cv_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Hold out ~1/n_test_folds of PATIENTS as the real test set, then assign grouped CV folds
    to the remaining patients. Stratified on comorbidity stratum; no patient spans splits."""
    strata, groups = stratum(df), df["subject_id"]
    outer = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    _, test_idx = next(outer.split(df, strata, groups))

    split = pd.DataFrame({"hadm_id": df["hadm_id"], "subject_id": df["subject_id"], "split": "train", "cv_fold": -1})
    split.iloc[test_idx, split.columns.get_loc("split")] = "test"

    train_pos = np.flatnonzero(split["split"].to_numpy() == "train")
    inner = StratifiedGroupKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(inner.split(train_pos, strata.iloc[train_pos], groups.iloc[train_pos])):
        split.iloc[train_pos[val_idx], split.columns.get_loc("cv_fold")] = fold

    assert not set(split.loc[split.split == "test", "subject_id"]) & set(split.loc[split.split == "train", "subject_id"])
    return split


def india_weights(train: pd.DataFrame, max_iter: int = 100, tol: float = 1e-8) -> pd.Series:
    """Iterative proportional fitting (raking) of the three comorbidity marginals to Indian
    population prevalence. Weights are normalised to mean 1."""
    target = {k: v["prevalence"] for k, v in json.loads(PREVALENCE_PATH.read_text()).items()}
    w = pd.Series(1.0, index=train.index)
    for _ in range(max_iter):
        prev = w.copy()
        for flag, p in target.items():
            pos = train[flag] == 1
            cur = w[pos].sum() / w.sum()
            w[pos] *= p / cur
            w[~pos] *= (1 - p) / (1 - cur)
        if (w - prev).abs().max() < tol:
            break
    return w / w.mean()


def _fit_copula(block: pd.DataFrame):
    """Empirical marginals + Gaussian dependence on normal scores (NaNs ignored pairwise)."""
    z = {}
    for c in COPULA_COLS:
        x = block[c].astype(float)
        ranks = x.rank(method="average")
        z[c] = pd.Series(stats.norm.ppf(ranks / (x.notna().sum() + 1)), index=x.index)
    corr = pd.DataFrame(z).corr(min_periods=5).fillna(0.0).to_numpy()
    np.fill_diagonal(corr, 1.0)
    # nearest positive semi-definite correlation matrix
    vals, vecs = np.linalg.eigh(corr)
    corr = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def synthesize(train: pd.DataFrame, target_per_stratum: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Gaussian-copula oversampling of comorbidity strata smaller than target_per_stratum
    (default: median stratum size). Each synthetic row copies the missingness pattern of a
    randomly drawn real row from the same stratum, so synthetic data is not 'cleaner'."""
    rng = np.random.default_rng(seed)
    st = stratum(train)
    sizes = st.value_counts()
    target_n = int(target_per_stratum or sizes.median())
    out = []
    for s, n_real in sizes.items():
        n_new = target_n - n_real
        # strata already within 10% of the target gain nothing from a handful of synthetic rows
        if n_new < 0.1 * target_n or n_real < SMALL_CELL:
            continue
        block = train[st == s]
        corr = _fit_copula(block)
        u = stats.norm.cdf(rng.multivariate_normal(np.zeros(len(COPULA_COLS)), corr, size=n_new))
        synth = pd.DataFrame(index=range(n_new))
        for j, c in enumerate(COPULA_COLS):
            observed = block[c].dropna().to_numpy(dtype=float)
            synth[c] = np.quantile(observed, u[:, j], method="inverted_cdf") if len(observed) else np.nan
        masks = block[COPULA_COLS].isna().to_numpy()[rng.integers(0, n_real, size=n_new)]
        synth = synth.mask(masks)
        for f, v in zip(FLAGS, s):
            synth[f] = int(v)
        out.append(synth)
    if not out:
        return train.iloc[0:0].assign(is_synthetic=1)
    synth = pd.concat(out, ignore_index=True)
    synth["egfr"] = ckd_epi_2021(synth["creatinine"], synth["age"], synth["sex_male"])
    for t in TARGETS:
        synth[t] = synth[t].round().clip(0, 2).astype("Int64")
    synth["sex_male"] = synth["sex_male"].round()
    synth["is_synthetic"] = 1
    synth["subject_id"] = -1
    synth["hadm_id"] = -np.arange(1, len(synth) + 1)
    return synth


def fidelity_summary(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Aggregate-only, stratum-matched comparison of real vs synthetic training rows:
    standardised mean difference per feature and mean |correlation difference|."""
    cols = [c for c in FEATURES if c not in FLAGS]
    real_st, synth_st = stratum(real), stratum(synth)
    out = {}
    for s in sorted(synth_st.unique()):
        r, g = real[real_st == s][cols], synth[synth_st == s][cols]
        smd = ((g.mean() - r.mean()) / r.std()).abs()
        rc, sc = r.corr().to_numpy(), g.corr().to_numpy()
        mask = ~np.isnan(rc) & ~np.isnan(sc) & ~np.eye(len(cols), dtype=bool)
        out[s] = {"max_abs_std_mean_diff": round(float(smd.max()), 3),
                  "mean_abs_corr_diff": round(float(np.abs(rc - sc)[mask].mean()), 3)}
    return out


def split_card(df: pd.DataFrame, split: pd.DataFrame, weights: pd.Series, synth: pd.DataFrame) -> dict:
    merged = df.merge(split[["hadm_id", "split", "cv_fold"]], on="hadm_id")
    card = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "splits": {}}
    for name in ("train", "test"):
        part = merged[merged.split == name]
        card["splits"][name] = {
            "n_admissions": int(len(part)),
            "n_patients": int(part.subject_id.nunique()),
            "strata": {k: suppress(int(v)) for k, v in stratum(part).value_counts().sort_index().items()},
            "targets": {t: {lbl: suppress(int((part[t] == i).sum())) for i, lbl in enumerate(["low", "moderate", "high"])}
                        for t in TARGETS},
        }
    train = merged[merged.split == "train"]
    card["cv_fold_sizes"] = {int(k): int(v) for k, v in train.cv_fold.value_counts().sort_index().items()}
    card["india_weights"] = {
        "weighted_prevalence": {f: round(float((train[f] * weights).sum() / weights.sum()), 4) for f in FLAGS},
        "unweighted_prevalence": {f: round(float(train[f].mean()), 4) for f in FLAGS},
        "effective_sample_size": round(float(weights.sum() ** 2 / (weights ** 2).sum()), 1),
        "weight_range": [round(float(weights.min()), 3), round(float(weights.max()), 3)],
    }
    card["synthetic"] = {
        "n_real_train": int(len(train)),
        "n_synthetic": int(len(synth)),
        "synthetic_share_pct": round(100 * len(synth) / (len(synth) + len(train)), 1),
        "per_stratum": {k: suppress(int(v)) for k, v in stratum(synth).value_counts().sort_index().items()} if len(synth) else {},
        "fidelity": fidelity_summary(train, synth) if len(synth) else None,
        "note": "Synthetic rows are for the training-set ablation only and are regenerated inside each CV fold; never evaluated.",
    }
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_cohort(args.tag)
    split = make_split(df, seed=args.seed)
    split.to_parquet(DERIVED / f"split_{args.tag}.parquet", index=False)

    train = df[df.hadm_id.isin(split.loc[split.split == "train", "hadm_id"])]
    weights = india_weights(train)
    synth = synthesize(train, seed=args.seed)

    card = split_card(df, split, weights, synth)
    card["eligibility"] = n_excluded(args.tag)
    (DERIVED / f"cohort_card_{args.tag}_split.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()

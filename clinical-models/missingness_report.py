"""
Missingness analysis (Reviewer 2, item 8). Answers four questions with numbers:

  1. How much of each feature is missing, per target, among the rows that carry that target's label?
  2. Is missingness related to the outcome (informative missingness)?  Outcome rates for
     measured vs. not-measured HbA1c / fasting glucose / BMI / blood pressure / chemistry.
  3. Can missingness ALONE predict the outcome?  AUROC of a model fed only the measured/not-measured
     indicators (0.5 = no information).
  4. Would giving the model the indicators help?  Same gradient-boosting model with vs. without
     indicator columns (patient-level test split, early stopping on CV fold 0, as elsewhere).

Fact stated for the paper: the deployed TabNet models receive NO missingness indicators; missing
values are median-imputed from training rows (train_model1.py, SimpleImputer(strategy="median")).
The indicator ablation here uses XGBoost, not TabNet, so it measures how much information the
indicators carry, not the effect on the deployed network.

Everything written is aggregate (rates, AUROCs, counts >= 10).  Outputs in
artifacts/models/reports/evaluation/: missingness_per_target.csv, missingness_outcome_rates.csv,
missingness_models.csv, missingness_summary.json

    python missingness_report.py [--targets ...] [--n-boot 200]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch  # noqa: F401  -- import before sklearn on Windows
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from mimic_extract import FEATURES, SMALL_CELL
from model1_artifacts import TARGETS
from train_model1 import load_frames

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "models" / "reports" / "evaluation"

# measured/not-measured groups (egfr is derived from creatinine; dbp is measured with sbp)
INDICATORS = {"na_measured": "serum_sodium", "k_measured": "serum_potassium", "creatinine_measured": "creatinine",
              "hba1c_measured": "hba1c", "fbs_measured": "fbs", "bp_measured": "sbp", "bmi_measured": "bmi"}
PARAMS = dict(learning_rate=0.05, max_depth=5, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
              n_estimators=600)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for ind, col in INDICATORS.items():
        out[ind] = out[col].notna().astype(int)
    return out


def per_target_missingness(df: pd.DataFrame, targets) -> pd.DataFrame:
    rows = []
    for t in targets:
        lab = df[df[t].notna()]
        for f in FEATURES:
            rows.append({"target": t, "feature": f, "n_labelled": int(len(lab)),
                         "pct_missing": round(100 * float(lab[f].isna().mean()), 2)})
    return pd.DataFrame(rows)


def outcome_rates(df: pd.DataFrame, targets) -> pd.DataFrame:
    rows = []
    for t in targets:
        lab = df[df[t].notna()]
        for ind in INDICATORS:
            for flag, name in ((1, "measured"), (0, "not_measured")):
                g = lab[lab[ind] == flag]
                if len(g) < SMALL_CELL:
                    continue
                rows.append({"target": t, "indicator": ind, "group": name, "n": int(len(g)),
                             "pct_moderate_or_high": round(100 * float((g[t] >= 1).mean()), 2),
                             "pct_high": round(100 * float((g[t] == 2).mean()), 2),
                             "mean_level": round(float(g[t].astype(float).mean()), 4)})
    return pd.DataFrame(rows)


def fit_eval(train, val, test, target, cols, impute: bool):
    tr, va, te = (d[d[target].notna()] for d in (train, val, test))
    if impute:
        imp = SimpleImputer(strategy="median").fit(tr[cols])
        X = [imp.transform(d[cols]) for d in (tr, va, te)]
    else:
        X = [d[cols].to_numpy(dtype=float) for d in (tr, va, te)]
    y = [d[target].astype(int).to_numpy() for d in (tr, va, te)]
    counts = np.bincount(y[0], minlength=3).astype(float)
    # inverse-frequency weights scaled to mean 1 (min_child_weight is a hessian sum, so scale matters)
    w = (1.0 / counts)[y[0]] * len(y[0]) / 3
    w_val = (1.0 / np.bincount(y[1], minlength=3).clip(1))[y[1]] * len(y[1]) / 3
    m = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist", eval_metric="mlogloss",
                      early_stopping_rounds=30, random_state=42, n_jobs=-1, **PARAMS)
    m.fit(X[0], y[0], sample_weight=w, eval_set=[(X[1], y[1])], sample_weight_eval_set=[w_val], verbose=False)
    proba = m.predict_proba(X[2])
    return roc_auc_score(y[2], proba, multi_class="ovr", average="macro", labels=[0, 1, 2]), te, y[2], proba


def boot_ci(y, proba, groups, n_boot, seed=42):
    """Patient-cluster bootstrap CI for the macro AUROC."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    vals = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
        keep = w > 0
        if len(np.unique(y[keep])) == 3:
            vals.append(roc_auc_score(y[keep], proba[keep], multi_class="ovr", average="macro",
                                      labels=[0, 1, 2], sample_weight=w[keep]))
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    df = add_indicators(load_frames(args.tag))
    train = df[(df.split == "train") & (df.cv_fold != 0)]
    val = df[(df.split == "train") & (df.cv_fold == 0)]
    test = df[df.split == "test"]
    feats = list(FEATURES)
    ind_cols = list(INDICATORS)

    miss = per_target_missingness(df, args.targets)
    rates = outcome_rates(df, args.targets)
    rows = []
    for t in args.targets:
        res = {}
        for name, cols, impute in (("indicators_only", ind_cols, False),
                                   ("features_median_imputed", feats, True),
                                   ("features_plus_indicators", feats + ind_cols, True)):
            auc, te, y, proba = fit_eval(train, val, test, t, cols, impute)
            lo, hi = boot_ci(y, proba, te["subject_id"].to_numpy(), args.n_boot)
            res[name] = auc
            rows.append({"target": t, "inputs": name, "auroc_macro": round(float(auc), 4), "ci_low": lo, "ci_high": hi,
                         "n_test": int(len(y))})
        print(f"{t:<24} AUROC indicators-only {res['indicators_only']:.3f} | features {res['features_median_imputed']:.3f} "
              f"| features+indicators {res['features_plus_indicators']:.3f}")
    models = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    miss.to_csv(OUT_DIR / "missingness_per_target.csv", index=False)
    rates.to_csv(OUT_DIR / "missingness_outcome_rates.csv", index=False)
    models.to_csv(OUT_DIR / "missingness_models.csv", index=False)
    (OUT_DIR / "missingness_summary.json").write_text(json.dumps({
        "deployed_tabnet_receives_indicators": False,
        "imputation": "median of training rows (SimpleImputer), then standard scaling",
        "ablation_model": "XGBoost (not TabNet); isolates information content of the indicators",
        "indicators": INDICATORS, "n_boot": args.n_boot,
        "per_target_missing_pct": miss.pivot(index="feature", columns="target", values="pct_missing").to_dict(),
        "models": models.to_dict("records")}, indent=2))
    print(f"wrote missingness_* files to {OUT_DIR}")


if __name__ == "__main__":
    main()

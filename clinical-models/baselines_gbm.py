"""
Conventional gradient-boosting baseline under the SAME protocol as TabNet (Reviewer 2, item 5).

Same cohort, same fixed patient-level split, same four observed-outcome targets, same
imputer + scaler, same class-balanced training + prior-correction calibration, same
metrics and patient-cluster bootstrap / repeated grouped CV code paths as the TabNet and
rules arms. Two input variants are run so the comparison is not hostage to preprocessing:

  xgb_matched  XGBoost on the identical inputs TabNet sees (median-imputed, scaled,
               per-target isotonic monotonic transform)
  xgb_raw      XGBoost on median-imputed features only (trees do not need the transform)

Modes
  --mode test   fit on training patients (CV folds != 0, early stopping on fold 0, exactly as the
                deployed TabNet), score the held-out test patients once, bootstrap 95% CIs,
                paired differences vs TabNet (deployed artifacts) and vs the guideline rules.
  --mode cv     5x5 repeated patient-grouped CV on TRAINING patients using repeated_cv.repeated_folds
                (same seeds/folds as the TabNet run), paired differences vs the TabNet rows already in
                reports/evaluation/repeated_cv_results.csv.

Hyperparameters are fixed defaults unless --tune is given, which picks from a small grid on CV fold 0
only (the same budget the TabNet tuning had) and writes gbm_params.json for later runs.

    python baselines_gbm.py --mode test --tune
    python baselines_gbm.py --mode cv --params gbm_params.json
    python baselines_gbm.py --mode test --targets carb_sensitivity --n-boot 50    # smoke test
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import torch  # noqa: F401  -- must be imported before sklearn (see repeated_cv.py)
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import cohort_data
from evaluate_model1 import ece, metrics
from mimic_extract import FEATURES
from model1_artifacts import LABELS, TARGETS, Model1Predictor, prior_correct
from repeated_cv import BASE_SEED, METRIC_KEYS, _ci, repeated_folds
from train_model1 import (LabelGenerator, MonotonicFeatureTransformer, build_monotonic_constraints,
                          load_frames)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "models" / "reports" / "evaluation"
VARIANTS = ("xgb_matched", "xgb_raw")

DEFAULT_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 10, "subsample": 0.8,
                  "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_estimators": 600}
TUNE_GRID = [{"max_depth": d, "learning_rate": lr, "min_child_weight": mcw}
             for d, lr, mcw in itertools.product((3, 5, 7), (0.05,), (10, 50))]
EARLY_STOP = 30


class GBMStratifier:
    """One XGBoost classifier per target; mirrors ClinicalRiskStratifier's preprocessing and calibration."""

    def __init__(self, variant: str, params: dict, seed: int = 42, targets=TARGETS, n_jobs: int = -1):
        assert variant in VARIANTS
        self.variant, self.params, self.seed, self.targets, self.n_jobs = variant, params, seed, tuple(targets), n_jobs
        self.models, self.mono, self.class_prior, self.best_iter = {}, {}, {}, {}

    def _base(self, df):
        return self.scaler.transform(self.imputer.transform(df[list(FEATURES)]))

    def _x(self, df, target):
        base = self._base(df)
        return self.mono[target].transform(base).astype(np.float32) if self.variant == "xgb_matched" else base.astype(np.float32)

    def fit(self, train_df, val_df):
        assert not set(train_df.subject_id) & set(val_df.subject_id), "patient overlap between train and validation"
        self.imputer = SimpleImputer(strategy="median").fit(train_df[list(FEATURES)])
        self.scaler = StandardScaler().fit(self.imputer.transform(train_df[list(FEATURES)]))
        constraints = build_monotonic_constraints(list(FEATURES))
        for target in self.targets:
            tr, va = train_df[train_df[target].notna()], val_df[val_df[target].notna()]
            y = tr[target].astype(int).to_numpy()
            y_val = va[target].astype(int).to_numpy()
            self.mono[target] = MonotonicFeatureTransformer(list(FEATURES), constraints).fit(self._base(tr), y)

            # inverse-frequency weights -> balanced training, same idea as TabNet's sampler; the
            # calibration factor is derived exactly as in ClinicalRiskStratifier.fit
            counts = np.bincount(y, minlength=3).astype(float)
            w = (1.0 / counts)[y] * len(y) / 3
            w_val = (1.0 / np.bincount(y_val, minlength=3).clip(1))[y_val]
            true_prior = counts / counts.sum()
            effective = np.bincount(y, weights=w, minlength=3)
            effective = effective / effective.sum()
            factor = np.divide(true_prior, effective, out=np.zeros(3), where=effective > 0)
            self.class_prior[target] = factor / factor.sum()

            model = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                                  eval_metric="mlogloss", early_stopping_rounds=EARLY_STOP,
                                  random_state=self.seed, n_jobs=self.n_jobs, **self.params)
            model.fit(self._x(tr, target), y, sample_weight=w,
                      eval_set=[(self._x(va, target), y_val)], sample_weight_eval_set=[w_val], verbose=False)
            self.models[target] = model
            self.best_iter[target] = int(model.best_iteration)
        return self

    def decision(self, df, target):
        return self.models[target].predict_proba(self._x(df, target))

    def evaluate(self, df):
        y_true, y_pred, y_proba = {}, {}, {}
        for target in self.targets:
            part = df[df[target].notna()]
            decision = self.decision(part, target)
            y_true[target] = part[target].astype(int).to_numpy()
            y_pred[target] = decision.argmax(axis=1)
            y_proba[target] = prior_correct(decision, self.class_prior[target])
        return y_true, y_pred, y_proba


# --------------------------------------------------------------------------------------
# tuning (CV fold 0 only; test set never opened)
# --------------------------------------------------------------------------------------
def tune(train_df, val_df, variant, targets):
    rows = []
    for g in TUNE_GRID:
        params = {**DEFAULT_PARAMS, **g}
        strat = GBMStratifier(variant, params, targets=targets).fit(train_df, val_df)
        y_true, y_pred, y_proba = strat.evaluate(val_df)
        f1 = float(np.mean([metrics(y_true[t], y_pred[t], y_proba[t])["f1_macro"] for t in targets]))
        rows.append({**g, "mean_f1_macro": f1})
        print(f"  tune {variant} {g} -> mean macro-F1 {f1:.4f}")
    best = max(rows, key=lambda r: r["mean_f1_macro"])
    return {**DEFAULT_PARAMS, **{k: best[k] for k in TUNE_GRID[0]}}, rows


# --------------------------------------------------------------------------------------
# test-set evaluation with patient-cluster bootstrap
# --------------------------------------------------------------------------------------
def arm_metrics(y, pred, proba, w=None):
    return metrics(y, pred, proba, w)


def cluster_bootstrap(y, arms, groups, n_boot, seed):
    """arms: name -> (pred, proba|None). CIs per arm plus paired differences gbm_* minus tabnet / rules."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    draws = {name: [] for name in arms}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        w = counts[codes].astype(float)
        keep = w > 0
        for name, (pred, proba) in arms.items():
            draws[name].append(arm_metrics(y[keep], pred[keep], None if proba is None else proba[keep], w[keep]))
    frames = {n: pd.DataFrame(d) for n, d in draws.items()}
    out = {n: {k: [round(float(f[k].quantile(0.025)), 4), round(float(f[k].quantile(0.975)), 4)] for k in f.columns}
           for n, f in frames.items()}
    for g in [n for n in arms if n.startswith("xgb")]:
        for ref in ("tabnet", "rules"):
            shared = [k for k in frames[g].columns if k in frames[ref].columns]
            diff = frames[g][shared] - frames[ref][shared]
            out[f"{g}_minus_{ref}"] = {k: [round(float(diff[k].quantile(0.025)), 4),
                                           round(float(diff[k].quantile(0.975)), 4)] for k in shared}
    return out


def run_test(args, params_by_variant):
    df = load_frames(args.tag)
    train_df = df[(df.split == "train") & (df.cv_fold != 0)]
    val_df = df[(df.split == "train") & (df.cv_fold == 0)]
    test = df[df.split == "test"].reset_index(drop=True)
    predictor = Model1Predictor(args.model_dir)
    rules = LabelGenerator.predict(test)
    X_test = test[predictor.feature_names].to_numpy(dtype=float)

    fitted = {}
    for v in VARIANTS:
        t0 = time.time()
        fitted[v] = GBMStratifier(v, params_by_variant[v], targets=args.targets).fit(train_df, val_df)
        print(f"fit {v} ({time.time() - t0:.0f}s), best iterations {fitted[v].best_iter}")

    summary = {"params": params_by_variant, "n_boot": args.n_boot, "targets": {}}
    rows = []
    for target in args.targets:
        mask = test[target].notna().to_numpy()
        y = test.loc[mask, target].astype(int).to_numpy()
        groups = test.loc[mask, "subject_id"].to_numpy()
        dec = predictor.predict_proba_matrix(X_test[mask], target)
        arms = {"tabnet": (dec.argmax(1), prior_correct(dec, predictor.class_prior[target])),
                "rules": (rules.loc[mask, target].to_numpy(), None)}
        for v in VARIANTS:
            part = test[mask]
            d = fitted[v].decision(part, target)
            arms[v] = (d.argmax(1), prior_correct(d, fitted[v].class_prior[target]))
        ci = cluster_bootstrap(y, arms, groups, args.n_boot, args.seed)
        res = {"n": int(len(y)), "point": {}, "ci95": ci, "ece_calibrated": {}}
        for name, (pred, proba) in arms.items():
            res["point"][name] = {k: round(float(x), 4) for k, x in arm_metrics(y, pred, proba).items()}
            if proba is not None:
                res["ece_calibrated"][name] = round(ece(y, proba), 4)
            for k, x in res["point"][name].items():
                rows.append({"target": target, "model": name, "metric": k, "value": x,
                             "ci_low": ci[name][k][0], "ci_high": ci[name][k][1]})
        summary["targets"][target] = res
        p = res["point"]
        print(f"{target:<24} macro-F1 tabnet {p['tabnet']['f1_macro']:.3f} | xgb_matched {p['xgb_matched']['f1_macro']:.3f} "
              f"| xgb_raw {p['xgb_raw']['f1_macro']:.3f} | rules {p['rules']['f1_macro']:.3f}   "
              f"AUROC tabnet {p['tabnet']['auroc_ovr_macro']:.3f} xgb_raw {p['xgb_raw']['auroc_ovr_macro']:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gbm_test_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(OUT_DIR / "gbm_test_metrics_ci.csv", index=False)
    print(f"wrote gbm_test_summary.json, gbm_test_metrics_ci.csv to {OUT_DIR}")


# --------------------------------------------------------------------------------------
# repeated grouped CV
# --------------------------------------------------------------------------------------
def run_cv(args, params_by_variant):
    df = load_frames(args.tag)
    train_df = df[df.split == "train"].reset_index(drop=True)
    rows, t0 = [], time.time()
    for r, fold, tr, va in repeated_folds(train_df, args.n_repeats, args.n_folds):
        for v in VARIANTS:
            strat = GBMStratifier(v, params_by_variant[v], targets=args.targets).fit(tr, va)
            y_true, y_pred, y_proba = strat.evaluate(va)
            for t in args.targets:
                rows.append({"repeat": r, "fold": fold, "target": t, "arm": v, "n": int(len(y_true[t])),
                             **metrics(y_true[t], y_pred[t], y_proba[t])})
        print(f"  repeat {r} fold {fold} done (elapsed {(time.time() - t0) / 60:.1f} min)")
    gbm = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gbm.to_csv(OUT_DIR / "gbm_repeated_cv_results.csv", index=False)

    ref_path = OUT_DIR / "repeated_cv_results.csv"
    ref = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()
    allrows = pd.concat([gbm, ref[ref["target"].isin(args.targets)]], ignore_index=True) if len(ref) else gbm
    wide = allrows.pivot_table(index=["target", "repeat", "fold"], columns="arm", values=METRIC_KEYS)
    summary = {}
    for target in args.targets:
        sub = wide.loc[target]
        summary[target] = {}
        for m in METRIC_KEYS:
            entry = {}
            for arm in (*VARIANTS, "tabnet", "rules"):
                if (m, arm) in sub.columns and sub[(m, arm)].notna().any():
                    entry[arm] = _ci(sub[(m, arm)].dropna().to_numpy())
            for v in VARIANTS:
                for refarm in ("tabnet", "rules"):
                    if (m, v) in sub.columns and (m, refarm) in sub.columns:
                        d = (sub[(m, v)] - sub[(m, refarm)]).dropna().to_numpy()
                        if len(d):
                            entry[f"{v}_minus_{refarm}"] = _ci(d)
            summary[target][m] = entry
    (OUT_DIR / "gbm_repeated_cv_summary.json").write_text(json.dumps(
        {"n_repeats": args.n_repeats, "n_folds": args.n_folds, "params": params_by_variant, "per_target": summary}, indent=2))
    for t, m in summary.items():
        f = m["f1_macro"]
        print(f"{t:<24} macro-F1 tabnet {f.get('tabnet', {}).get('mean', float('nan')):.3f} "
              f"xgb_matched {f['xgb_matched']['mean']:.3f} xgb_raw {f['xgb_raw']['mean']:.3f}")
    print(f"wrote gbm_repeated_cv_results.csv, gbm_repeated_cv_summary.json ({(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser(description="XGBoost baseline under the TabNet evaluation protocol")
    ap.add_argument("--mode", choices=["test", "cv"], required=True)
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="select params on CV fold 0 and write gbm_params.json")
    ap.add_argument("--params", type=Path, default=OUT_DIR / "gbm_params.json")
    args = ap.parse_args()

    params = {v: DEFAULT_PARAMS for v in VARIANTS}
    if args.tune:
        df = load_frames(args.tag)
        tr = df[(df.split == "train") & (df.cv_fold != 0)]
        va = df[(df.split == "train") & (df.cv_fold == 0)]
        params, grids = {}, {}
        for v in VARIANTS:
            params[v], grids[v] = tune(tr, va, v, args.targets)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        args.params.write_text(json.dumps({"params": params, "grid": grids}, indent=2))
    elif args.params.exists():
        params = json.loads(args.params.read_text())["params"]
        print(f"using tuned params from {args.params}")
    print("params:", json.dumps(params))

    (run_test if args.mode == "test" else run_cv)(args, params)


if __name__ == "__main__":
    main()

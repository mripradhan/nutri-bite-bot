"""
5x5 repeated grouped cross-validation of Model 1 (TabNet) vs the guideline-rule
baseline, on TRAINING patients only. The real held-out test set is never opened
here; this is the "external validation" stand-in from feedbackplan.md Sec 4,
reusing ClinicalRiskStratifier.fit/evaluate and evaluate_model1.metrics exactly
as the single-split test comparison in evaluate_model1.py does, so the numbers
are computed the same way.

For each of n_repeats independent random seeds, the training patients are
re-split into n_folds patient-level StratifiedGroupKFold folds (stratified by
comorbidity stratum). Each fold is held out in turn as the validation set for
that fit (used both for TabNet early stopping and as the fold's scored set,
matching how CV fold 0 is already used for --tune elsewhere in this pipeline).
No model artifacts are written; each of the n_repeats * n_folds fits is scored
and discarded.

Usage:
    python repeated_cv.py --tag v3.1                                  # full 5x5, ~4-6h
    python repeated_cv.py --tag v3.1 --n-repeats 1 --n-folds 2 \
        --targets carb_sensitivity                                    # smoke test
"""
import argparse
import json
import time
from pathlib import Path

import torch  # noqa: F401  -- must be imported before sklearn (see below) or torch's DLL init fails on Windows
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import StratifiedGroupKFold

import cohort_data
from evaluate_model1 import metrics
from train_model1 import ClinicalModelConfig, ClinicalRiskStratifier, LabelGenerator, load_frames

ROOT = Path(__file__).resolve().parent.parent
METRIC_KEYS = ["f1_macro", "balanced_accuracy", "cohen_kappa", "quadratic_kappa", "auroc_ovr_macro"]
BASE_SEED = 1000  # offset from cohort_data's split seed (42) so folds here are independent of the fixed test split


def run_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: ClinicalModelConfig,
             repeat: int, fold: int) -> list[dict]:
    strat = ClinicalRiskStratifier(cfg).fit(train_df, val_df, verbose=False)
    _, y_true, y_pred, y_proba = strat.evaluate(val_df)
    rules = LabelGenerator.predict(val_df)

    rows = []
    for target in cfg.targets:
        mask = val_df[target].notna().to_numpy()
        y = y_true[target]
        tab = metrics(y, y_pred[target], y_proba[target])
        rpred = rules.loc[mask, target].to_numpy()
        rule = metrics(y, rpred)
        for arm, m in (("tabnet", tab), ("rules", rule)):
            rows.append({"repeat": repeat, "fold": fold, "target": target, "arm": arm,
                         "n": int(mask.sum()), **m})
    return rows


def repeated_folds(train_df: pd.DataFrame, n_repeats: int, n_folds: int):
    strata = cohort_data.stratum(train_df)
    groups = train_df["subject_id"]
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=BASE_SEED + r)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, strata, groups)):
            tr, va = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            assert not set(tr["subject_id"]) & set(va["subject_id"]), "patient overlap between CV fold train/val"
            yield r, fold, tr, va


def summarize(df: pd.DataFrame) -> dict:
    """Mean, SD and a t-distribution 95% CI over the n_repeats*n_folds estimates,
    per target/metric/arm, plus the paired TabNet-rules difference per fold.
    Rules have no probabilities, so auroc_ovr_macro is tabnet-only (matches
    evaluate_model1.py's test-set comparison)."""
    out = {}
    wide = df.pivot_table(index=["target", "repeat", "fold"], columns="arm", values=METRIC_KEYS)
    for target in df["target"].unique():
        sub = wide.loc[target]
        out[target] = {}
        for metric in METRIC_KEYS:
            entry = {}
            for arm in ("tabnet", "rules"):
                if (metric, arm) not in sub.columns:
                    continue
                vals = sub[(metric, arm)].dropna().to_numpy()
                if len(vals):
                    entry[arm] = _ci(vals)
            if "tabnet" in entry and "rules" in entry:
                diff = (sub[(metric, "tabnet")] - sub[(metric, "rules")]).dropna().to_numpy()
                entry["diff"] = _ci(diff)
            if entry:
                out[target][metric] = entry
    return out


def _ci(vals: np.ndarray) -> dict:
    n = len(vals)
    mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(sp_stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
            "ci95": [round(mean - tcrit * se, 4), round(mean + tcrit * se, 4)]}


def main():
    ap = argparse.ArgumentParser(description="5x5 repeated grouped CV, TabNet vs guideline rules (training patients only)")
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models",
                    help="read the training hyperparameters from this deployed model's manifest.json, "
                         "so the CV can't silently drift from ClinicalModelConfig's dataclass defaults "
                         "(which do not match the deployed model's actual tuned config)")
    args = ap.parse_args()

    df = load_frames(args.tag)
    train_df = df[df.split == "train"].reset_index(drop=True)  # test set is never loaded here
    manifest = json.loads((args.model_dir.resolve() / "manifest.json").read_text())["config"]
    cfg = ClinicalModelConfig(tag=args.tag, device=args.device,
                              **{k: v for k, v in manifest.items() if k not in ("tag",)})
    if args.targets:
        cfg.targets = tuple(args.targets)
    print(f"Using deployed config from {args.model_dir.resolve()}/manifest.json: "
          f"batch_size={cfg.batch_size}, virtual_batch_size={cfg.virtual_batch_size}, lr={cfg.lr}")

    print(f"Repeated grouped CV: {args.n_repeats} repeats x {args.n_folds} folds x {len(cfg.targets)} targets "
          f"= {args.n_repeats * args.n_folds * len(cfg.targets)} fits, on {len(train_df):,} training admissions")

    rows = []
    t0 = time.time()
    for r, fold, tr, va in repeated_folds(train_df, args.n_repeats, args.n_folds):
        tf0 = time.time()
        rows.extend(run_fold(tr, va, cfg, r, fold))
        print(f"  repeat {r} fold {fold}: train {len(tr):,} val {len(va):,} ({time.time() - tf0:.0f}s, "
              f"elapsed {(time.time() - t0) / 60:.1f}min)")

    results = pd.DataFrame(rows)
    out_dir = ROOT / "artifacts" / "models" / "reports" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "repeated_cv_results.csv", index=False)

    summary = {"n_repeats": args.n_repeats, "n_folds": args.n_folds, "targets": list(cfg.targets),
              "n_fits": args.n_repeats * args.n_folds, "config": cfg.summary(),
              "per_target": summarize(results)}
    (out_dir / "repeated_cv_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRepeated CV mean [95% CI], TabNet vs rules macro-F1:")
    for target, m in summary["per_target"].items():
        f1 = m["f1_macro"]
        print(f"  {target:<24} TabNet {f1['tabnet']['mean']:.3f} {f1['tabnet']['ci95']} | "
              f"rules {f1['rules']['mean']:.3f} {f1['rules']['ci95']} | diff {f1['diff']['ci95']}")
    print(f"\n  Wrote {out_dir}/repeated_cv_results.csv, repeated_cv_summary.json "
          f"({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    main()

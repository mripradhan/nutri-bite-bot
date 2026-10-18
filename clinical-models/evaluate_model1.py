"""
Phase 4 evaluation of a saved Model 1 artifact set on the held-out real test patients.

  - TabNet vs guideline-rule baseline with patient-level (cluster) bootstrap 95% CIs,
    including paired TabNet - rules differences
  - the same metrics with the test set reweighted to ICMR-INDIAB prevalence
  - calibration (expected calibration error, reliability plots) of the balanced-training
    scores vs the deployed calibrated probabilities
  - comorbidity-subgroup and COVID-era (2020-2022) breakdowns
  - confusion matrices for TabNet and the rules

Everything written is aggregate (metrics, counts >= 10). Usage:
    python evaluate_model1.py --tag v3.1                                   # deployed model
    python evaluate_model1.py --tag v3.1 --model-dir ../artifacts/experiments/synthetic
"""
import argparse
import json
import warnings
from pathlib import Path

import torch  # noqa: F401  -- must be imported before sklearn (below) or torch's DLL init fails on Windows
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)

import cohort_data
from mimic_extract import SMALL_CELL
from model1_artifacts import LABELS, TARGETS, Model1Predictor, prior_correct
from train_model1 import LabelGenerator, load_frames

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
STRATUM_NAMES = {"000": "none", "100": "HTN only", "010": "DM only", "001": "CKD only", "110": "HTN+DM",
                 "101": "HTN+CKD", "011": "DM+CKD", "111": "HTN+DM+CKD"}


def metrics(y, pred, proba=None, w=None) -> dict:
    m = {
        "f1_macro": f1_score(y, pred, average="macro", sample_weight=w, labels=[0, 1, 2], zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred, sample_weight=w),
        "cohen_kappa": cohen_kappa_score(y, pred, sample_weight=w),
        "quadratic_kappa": cohen_kappa_score(y, pred, weights="quadratic", sample_weight=w),
    }
    if proba is not None:
        m["auroc_ovr_macro"] = (roc_auc_score(y, proba, multi_class="ovr", average="macro", sample_weight=w,
                                              labels=[0, 1, 2]) if len(np.unique(y)) == 3 else np.nan)
    return m


def cluster_bootstrap(y, tab_pred, tab_proba, rule_pred, groups, n_boot, seed, base_w=None) -> dict:
    """Resample PATIENTS with replacement; rows get weight = times their patient was drawn."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    n_pat = len(uniq)
    draws = {"tabnet": [], "rules": [], "diff": []}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)
        w = counts[codes].astype(float)
        if base_w is not None:
            w = w * base_w
        keep = w > 0
        t = metrics(y[keep], tab_pred[keep], tab_proba[keep], w[keep])
        r = metrics(y[keep], rule_pred[keep], None, w[keep])
        draws["tabnet"].append(t)
        draws["rules"].append(r)
        draws["diff"].append({k: t[k] - r[k] for k in r})
    out = {}
    for arm, rows in draws.items():
        df = pd.DataFrame(rows)
        out[arm] = {k: [round(float(df[k].quantile(0.025)), 4), round(float(df[k].quantile(0.975)), 4)]
                    for k in df.columns}
    return out


def ece(y, proba, n_bins=15) -> float:
    """Top-label expected calibration error."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(total)


def reliability_plot(y, raw, corrected, target, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, ax in enumerate(axes):
        for proba, name, style in ((raw, "balanced-training scores", "o-"), (corrected, "calibrated (deployed)", "s-")):
            p, obs = proba[:, k], (y == k).astype(float)
            edges = np.quantile(p, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
            xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= SMALL_CELL]
            ys = [obs[idx == b].mean() for b in range(10) if (idx == b).sum() >= SMALL_CELL]
            ax.plot(xs, ys, style, ms=4, label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=f"{LABELS[k]}", xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Reliability — {target} (held-out test)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = args.model_dir.resolve()
    out_dir = model_dir / "reports" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = Model1Predictor(model_dir)
    manifest = predictor.manifest

    df = load_frames(args.tag)
    test = df[df.split == "test"].reset_index(drop=True)
    rules = LabelGenerator.predict(test)
    india_w = cohort_data.india_weights(test).to_numpy()
    strata = cohort_data.stratum(test)
    X_test = test[predictor.feature_names].to_numpy(dtype=float)

    summary = {"model_dir": rel(model_dir), "config": manifest["config"],
               "git": manifest["git"], "n_boot": args.n_boot, "targets": {}}
    subgroup_rows, cm_rows = [], []

    for target in TARGETS:
        mask = test[target].notna().to_numpy()
        y = test.loc[mask, target].astype(int).to_numpy()
        decision = predictor.predict_proba_matrix(X_test[mask], target)          # balanced scores -> label
        proba = prior_correct(decision, predictor.class_prior[target])            # calibrated -> reported risk
        pred = decision.argmax(axis=1)
        rpred = rules.loc[mask, target].to_numpy()
        groups = test.loc[mask, "subject_id"].to_numpy()

        print(f"\n{target}: n={len(y):,} patients={len(np.unique(groups)):,}")
        res = {
            "n": int(len(y)), "n_patients": int(len(np.unique(groups))),
            "class_counts": {LABELS[k]: int((y == k).sum()) for k in range(3)},
            "point": {"tabnet": metrics(y, pred, proba), "rules": metrics(y, rpred)},
            "ci95": cluster_bootstrap(y, pred, proba, rpred, groups, args.n_boot, args.seed),
            "india_weighted": {
                "point": {"tabnet": metrics(y, pred, proba, india_w[mask]), "rules": metrics(y, rpred, None, india_w[mask])},
                "ci95": cluster_bootstrap(y, pred, proba, rpred, groups, args.n_boot // 2, args.seed, india_w[mask]),
            },
            "calibration": {
                "ece_balanced_scores": round(ece(y, decision), 4),
                "ece_calibrated": round(ece(y, proba), 4),
                "correction_factor": [round(float(p), 4) for p in predictor.class_prior[target]],
                "tabnet_calibrated_argmax": metrics(y, proba.argmax(axis=1), proba),
                "mean_severity_balanced_scores": round(float(np.mean(decision @ np.array([0, 1, 2.0]))), 4),
                "mean_severity_calibrated": round(float(np.mean(proba @ np.array([0, 1, 2.0]))), 4),
                "mean_observed_level": round(float(y.mean()), 4),
            },
        }
        summary["targets"][target] = res
        reliability_plot(y, decision, proba, target, out_dir / f"reliability_{target}.png")
        p, c = res["point"], res["ci95"]
        print(f"  macro-F1 TabNet {p['tabnet']['f1_macro']:.3f} {c['tabnet']['f1_macro']} | rules "
              f"{p['rules']['f1_macro']:.3f} {c['rules']['f1_macro']} | diff CI {c['diff']['f1_macro']}")
        print(f"  ECE balanced scores {res['calibration']['ece_balanced_scores']:.3f} -> calibrated {res['calibration']['ece_calibrated']:.3f}")

        for model_name, pr in (("tabnet", pred), ("rules", rpred)):
            cm = confusion_matrix(y, pr, labels=[0, 1, 2])
            for i in range(3):
                cm_rows.append({"target": target, "model": model_name, "true": LABELS[i],
                                **{f"pred_{LABELS[j]}": int(cm[i, j]) for j in range(3)}})

        sub = pd.DataFrame({"stratum": strata[mask].map(STRATUM_NAMES).to_numpy(),
                            "covid_era": test.loc[mask, "covid_era"].to_numpy()})
        for col, values in (("comorbidity", sub["stratum"]), ("era", np.where(sub["covid_era"] == 1, "2020-2022", "2008-2019"))):
            for g in pd.unique(values):
                gm = np.asarray(values == g)
                n = int(gm.sum())
                if n < SMALL_CELL:
                    continue
                t = metrics(y[gm], pred[gm], proba[gm])
                r = metrics(y[gm], rpred[gm])
                subgroup_rows.append({"target": target, "breakdown": col, "group": g, "n": n,
                                      "tabnet_f1_macro": round(t["f1_macro"], 4), "rules_f1_macro": round(r["f1_macro"], 4),
                                      "tabnet_balanced_accuracy": round(t["balanced_accuracy"], 4),
                                      "rules_balanced_accuracy": round(r["balanced_accuracy"], 4),
                                      "tabnet_auroc": round(t["auroc_ovr_macro"], 4)})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.DataFrame(subgroup_rows).to_csv(out_dir / "subgroups.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrices.csv", index=False)

    rows = []
    for t, r in summary["targets"].items():
        for arm in ("tabnet", "rules"):
            for k, v in r["point"][arm].items():
                rows.append({"target": t, "model": arm, "metric": k, "value": round(v, 4),
                             "ci_low": r["ci95"][arm][k][0], "ci_high": r["ci95"][arm][k][1],
                             "diff_vs_rules_ci": r["ci95"]["diff"].get(k) if arm == "tabnet" else None})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_ci.csv", index=False)
    print(f"\n  ✓ Wrote {rel(out_dir)}/ (summary.json, metrics_ci.csv, subgroups.csv, "
          f"confusion_matrices.csv, reliability_*.png)")


if __name__ == "__main__":
    main()

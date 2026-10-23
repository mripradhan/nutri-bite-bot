"""
Where does the learned model add value over the guideline rules? (Reviewer 3 item 2, Reviewer 2 item 4)

Held-out TEST set only, deployed TabNet vs the guideline rules vs the tuned XGBoost baseline. The model is
not changed. Four analyses, all with patient-cluster bootstrap 95% CIs:

  1. Strata where a learned model should earn its place
       near_threshold / not_near_threshold   any rule feature within +-10% of one of its guideline cut-offs
       rules_cannot_fire / complete_inputs   >=1 input the rule needs is missing (the rule fails closed)
       disagree / agree                      TabNet and the rules assign different tiers
       confidence terciles                   TabNet's own confidence (decision score of the assigned tier)
     per stratum: macro-F1, balanced accuracy, accuracy for each arm, and TabNet-minus-rules (and
     TabNet-minus-XGBoost) differences; for 'disagree', who is right when they differ.
  2. Decision curve: net benefit of flagging patients whose observed outcome is >= moderate (or high), for
     TabNet's calibrated probability vs the rules' flag vs treat-all / treat-none.
  3. Matched false-positive rate: sensitivity of TabNet when its threshold is set to the rules' false-positive rate.
  4. Missing inputs: macro-F1 / AUROC of TabNet, XGBoost and the rules when 1-3 lab inputs are removed at random
     (same pool and mechanism as robustness_test.py).

Macro-F1 uses labels [0,1,2] with zero_division=0, as everywhere else in this repo, so very small strata
that lack a class are deflated for ALL arms alike; compare arms within a stratum, not strata with each other.

    python disagreement_analysis.py [--n-boot 500]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import torch  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baselines_gbm import DEFAULT_PARAMS, GBMStratifier
from mimic_extract import SMALL_CELL
from model1_artifacts import TARGETS, Model1Predictor, prior_correct
from robustness_test import MISSING_POOL
from train_model1 import LabelGenerator, load_frames

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "models" / "reports" / "evaluation"

CUTOFFS = {  # target -> {feature: [guideline cut-offs the rule uses]}  (LabelGenerator.predict)
    "sodium_sensitivity": {"serum_sodium": [145, 150], "sbp": [160]},
    "potassium_sensitivity": {"serum_potassium": [4.5, 5.0], "egfr": [60, 30]},
    "protein_restriction": {"egfr": [60, 30], "creatinine": [2.0]},
    "carb_sensitivity": {"hba1c": [5.7, 6.5, 7.0], "fbs": [100, 126], "bmi": [25, 30]},
}
BAND = 0.10


# ---------------------------------------------------------------- fast weighted metrics (3 classes)
def conf3(y, p, w):
    return np.bincount(y * 3 + p, weights=w, minlength=9).reshape(3, 3)


def macro_f1(c):
    tp = np.diag(c)
    den = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(3), where=den > 0)))


def bal_acc(c):
    sup = c.sum(1)
    return float(np.mean(np.diag(c)[sup > 0] / sup[sup > 0])) if (sup > 0).any() else float("nan")


def accuracy(c):
    return float(np.trace(c) / c.sum()) if c.sum() > 0 else float("nan")


def arm_stats(y, p, w):
    c = conf3(y, p, w)
    return {"macro_f1": macro_f1(c), "balanced_accuracy": bal_acc(c), "accuracy": accuracy(c)}


# ---------------------------------------------------------------- strata
def strata_masks(target, df, tab_pred, rule_pred, conf):
    near = np.zeros(len(df), bool)
    anymiss = np.zeros(len(df), bool)
    for feat, cuts in CUTOFFS[target].items():
        x = df[feat].to_numpy(float)
        anymiss |= np.isnan(x)
        for c in cuts:
            near |= np.abs(x - c) <= BAND * c
    q1, q2 = np.quantile(conf, [1 / 3, 2 / 3])
    return {
        "all": np.ones(len(df), bool),
        "near_threshold": near, "not_near_threshold": ~near & ~anymiss,
        "rules_cannot_fire": anymiss, "complete_inputs": ~anymiss,
        "disagree": tab_pred != rule_pred, "agree": tab_pred == rule_pred,
        "conf_low": conf <= q1, "conf_mid": (conf > q1) & (conf <= q2), "conf_high": conf > q2,
    }


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, q * cw[-1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frames(args.tag)
    train = df[(df.split == "train") & (df.cv_fold != 0)]
    val = df[(df.split == "train") & (df.cv_fold == 0)]
    test = df[df.split == "test"].reset_index(drop=True)
    pred = Model1Predictor(ROOT / "artifacts" / "models")
    X = test[pred.feature_names].to_numpy(float)
    rules = LabelGenerator.predict(test)

    params_path = OUT / "gbm_params.json"
    params = json.loads(params_path.read_text())["params"]["xgb_raw"] if params_path.exists() else DEFAULT_PARAMS
    with contextlib.redirect_stdout(io.StringIO()):
        xgb = GBMStratifier("xgb_raw", params).fit(train, val)

    strata_rows, nb_rows, fpr_rows = [], [], []
    summary = {"n_boot": args.n_boot, "band": BAND, "targets": {}}
    rng = np.random.default_rng(args.seed)

    for target in TARGETS:
        m = test[target].notna().to_numpy()
        d = test[m].reset_index(drop=True)
        y = d[target].astype(int).to_numpy()
        rp = rules.loc[m, target].to_numpy()
        dec = pred.predict_proba_matrix(X[m], target)
        dec = dec / dec.sum(1, keepdims=True)
        tp_ = dec.argmax(1)
        cal = prior_correct(dec, pred.class_prior[target])
        conf = dec.max(1)
        xd = xgb.decision(d, target)
        xp = xd.argmax(1)
        xcal = prior_correct(xd, xgb.class_prior[target])
        arms = {"tabnet": tp_, "xgboost": xp, "rules": rp}
        masks = strata_masks(target, d, tp_, rp, conf)
        codes, uniq = pd.factorize(d["subject_id"].to_numpy())

        # ---- 1. strata: point estimates + cluster bootstrap on the differences
        boots = {s: {k: [] for k in ("tabnet_minus_rules", "xgboost_minus_rules", "tabnet_minus_xgboost")} for s in masks}
        for _ in range(args.n_boot):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
            for s, mk in masks.items():
                ww = w[mk]
                if ww.sum() == 0:
                    continue
                f = {a: macro_f1(conf3(y[mk], arms[a][mk], ww)) for a in arms}
                boots[s]["tabnet_minus_rules"].append(f["tabnet"] - f["rules"])
                boots[s]["xgboost_minus_rules"].append(f["xgboost"] - f["rules"])
                boots[s]["tabnet_minus_xgboost"].append(f["tabnet"] - f["xgboost"])
        for s, mk in masks.items():
            n = int(mk.sum())
            if n < SMALL_CELL:
                continue
            row = {"target": target, "stratum": s, "n": n, "pct_of_target": round(100 * n / len(y), 1),
                   "n_patients": int(len(np.unique(d.loc[mk, "subject_id"])))}
            ones = np.ones(n)
            for a, p in arms.items():
                for k, v in arm_stats(y[mk], p[mk], ones).items():
                    row[f"{a}_{k}"] = round(v, 4)
            for a, pr in (("tabnet", cal), ("xgboost", xcal)):
                row[f"{a}_auroc"] = (round(float(roc_auc_score(y[mk], pr[mk], multi_class="ovr", average="macro",
                                                              labels=[0, 1, 2])), 4)
                                     if len(np.unique(y[mk])) == 3 else np.nan)
            for k, vals in boots[s].items():
                if vals:
                    row[f"{k}_f1"] = round(float(np.mean(vals)), 4)
                    row[f"{k}_f1_ci_low"], row[f"{k}_f1_ci_high"] = (round(float(np.quantile(vals, q)), 4) for q in (0.025, 0.975))
            if s == "disagree":
                tc, rc = tp_[mk] == y[mk], rp[mk] == y[mk]
                row["disagree_tabnet_only_right_pct"] = round(100 * float((tc & ~rc).mean()), 1)
                row["disagree_rules_only_right_pct"] = round(100 * float((~tc & rc).mean()), 1)
                row["disagree_both_wrong_pct"] = round(100 * float((~tc & ~rc).mean()), 1)
            strata_rows.append(row)

        # ---- 2/3. decision curve and matched-FPR sensitivity
        for event_name, thr in (("moderate_or_high", 1), ("high", 2)):
            ev = (y >= thr).astype(int)
            score = cal[:, thr:].sum(1)
            flag = (rp >= thr).astype(int)
            n = len(ev)
            for pt in np.round(np.arange(0.05, 0.51, 0.05), 2):
                for name, f in (("tabnet", (score >= pt)), ("rules", flag.astype(bool))):
                    tp, fp = float((f & (ev == 1)).sum()), float((f & (ev == 0)).sum())
                    nb_rows.append({"target": target, "event": event_name, "threshold_prob": pt, "model": name,
                                    "net_benefit": round(tp / n - fp / n * pt / (1 - pt), 5)})
                prev = ev.mean()
                nb_rows.append({"target": target, "event": event_name, "threshold_prob": pt, "model": "treat_all",
                                "net_benefit": round(prev - (1 - prev) * pt / (1 - pt), 5)})
                nb_rows.append({"target": target, "event": event_name, "threshold_prob": pt, "model": "treat_none",
                                "net_benefit": 0.0})

            d_tpr, rules_tpr, rules_fpr, tab_tpr = [], [], [], []
            for b in range(args.n_boot + 1):
                w = (np.ones(n) if b == 0 else
                     np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float))
                neg, pos = ev == 0, ev == 1
                if w[neg].sum() == 0 or w[pos].sum() == 0:
                    continue
                r_fpr = float((w[neg] * flag[neg]).sum() / w[neg].sum())
                r_tpr = float((w[pos] * flag[pos]).sum() / w[pos].sum())
                if r_fpr <= 0 or r_fpr >= 1:
                    continue
                cut = weighted_quantile(score[neg], w[neg], 1 - r_fpr)
                t_tpr = float((w[pos] * (score[pos] > cut)).sum() / w[pos].sum())
                if b == 0:
                    point = (r_fpr, r_tpr, t_tpr)
                else:
                    d_tpr.append(t_tpr - r_tpr)
            fpr_rows.append({"target": target, "event": event_name, "rules_fpr": round(point[0], 4),
                             "rules_sensitivity": round(point[1], 4), "tabnet_sensitivity_at_same_fpr": round(point[2], 4),
                             "diff": round(point[2] - point[1], 4),
                             "diff_ci_low": round(float(np.quantile(d_tpr, 0.025)), 4),
                             "diff_ci_high": round(float(np.quantile(d_tpr, 0.975)), 4)})
        print(f"{target:<24} n={len(y):>6}  strata done")

    # ---- 4. missing inputs: TabNet vs XGBoost vs rules
    miss_rows = []
    for target in TARGETS:
        m = test[target].notna().to_numpy()
        base = test[m].reset_index(drop=True)
        y = base[target].astype(int).to_numpy()
        for k in (0, 1, 2, 3):
            draws = []
            for _ in range(1 if k == 0 else args.n_draws):
                nulled = base.copy()
                if k:
                    ranks = np.argsort(np.argsort(rng.random((len(nulled), len(MISSING_POOL))), axis=1), axis=1)
                    for j, col in enumerate(MISSING_POOL):
                        nulled.loc[ranks[:, j] < k, col] = np.nan
                dec = pred.predict_proba_matrix(nulled[pred.feature_names].to_numpy(float), target)
                cal = prior_correct(dec / dec.sum(1, keepdims=True), pred.class_prior[target])
                xd = xgb.decision(nulled, target)
                xcal = prior_correct(xd, xgb.class_prior[target])
                rp = LabelGenerator.predict(nulled)[target].to_numpy()
                ones = np.ones(len(y))
                draws.append({"tabnet_f1": macro_f1(conf3(y, dec.argmax(1), ones)),
                              "xgboost_f1": macro_f1(conf3(y, xd.argmax(1), ones)),
                              "rules_f1": macro_f1(conf3(y, rp, ones)),
                              "tabnet_auroc": roc_auc_score(y, cal, multi_class="ovr", average="macro", labels=[0, 1, 2]),
                              "xgboost_auroc": roc_auc_score(y, xcal, multi_class="ovr", average="macro", labels=[0, 1, 2])})
            miss_rows.append({"target": target, "k_missing": k, "n": len(y),
                              **{key: round(float(np.mean([r[key] for r in draws])), 4) for key in draws[0]}})
        print(f"{target:<24} missing-input sweep done")

    pd.DataFrame(strata_rows).to_csv(OUT / "disagreement_strata.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(OUT / "disagreement_net_benefit.csv", index=False)
    pd.DataFrame(fpr_rows).to_csv(OUT / "disagreement_matched_fpr.csv", index=False)
    pd.DataFrame(miss_rows).to_csv(OUT / "disagreement_missing_xgb.csv", index=False)
    (OUT / "disagreement_summary.json").write_text(json.dumps(
        {"n_boot": args.n_boot, "band": BAND, "xgb_params": params,
         "matched_fpr": fpr_rows, "note": "Macro-F1 uses labels [0,1,2] with zero_division=0; compare arms within a stratum."},
        indent=2))
    print("wrote disagreement_* files to", OUT)


if __name__ == "__main__":
    main()

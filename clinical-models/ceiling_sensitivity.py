"""
How much do the computed gram ceilings depend on the classifier? (Reviewer 3 item 1, Reviewer 2 item 4)

Part A  Deterministic: ceiling as a function of the severity score, for the use-case patient and ingredients,
        and the effect of a one-tier change in severity (0 -> 1, 1 -> 2). This is the engine's sensitivity.

Part B  Empirical: on held-out TEST admissions that have all four outcomes observed, compare the ceilings
        computed from
            tabnet     the deployed model's calibrated severity (probability-weighted class index)
            rules      the guideline-rule label used as severity (0/1/2)
            constant   severity 1.0 for every target (no personalisation: what the engine does with no model)
        with the ceilings computed from the OBSERVED outcome levels used as severity (the "oracle").
        The oracle is a proxy outcome, not a validated safe ceiling: a small gap to the oracle does not show
        the ceiling is clinically safe, and a large gap does not show it is unsafe. The numbers describe how
        much classifier error propagates into grams, and what the learned component adds over the rules.

Phosphorus severity is a fixed CKD / eGFR rule in the engine (train_model2._tabnet_predict) and is kept
identical in every scenario. Nothing here changes the deployed model; reports go to
artifacts/models/reports/evaluation/ceiling_sensitivity_*.

    python ceiling_sensitivity.py [--n-patients 1500] [--seed 42]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch  # noqa: F401  -- before sklearn on Windows
import numpy as np
import pandas as pd

from model1_artifacts import TARGETS
from train_model1 import LabelGenerator, load_frames

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "models" / "reports" / "evaluation"

USECASE = ["Green gram dal (Moong dal)", "Paneer (Cottage cheese)", "Egg, whole, boiled", "Rice, milled (white)",
           "Banana, ripe", "Apple", "Pineapple (Ananas)", "Bottle gourd (Lauki)"]
USECASE_PATIENT = {"age": 58, "sex_male": 1, "has_htn": 1, "has_dm": 0, "has_ckd": 1, "serum_sodium": 140,
                   "serum_potassium": 5.1, "creatinine": 1.9, "egfr": 45, "hba1c": 5.2, "fbs": 85, "sbp": 162,
                   "dbp": 96, "bmi": 26}
LABELS = ["low", "moderate", "high"]
WITHIN = 0.25          # "close" = within +-25% of the oracle ceiling


def phosphorus_entry(has_ckd: int, egfr: float) -> dict:
    """Mirror of the engine's rule-derived phosphorus severity (train_model2.Model1Integration._tabnet_predict)."""
    if has_ckd == 1 or egfr < 30:
        return {"label": "high", "severity_score": 1.8}
    if egfr < 60:
        return {"label": "moderate", "severity_score": 1.0}
    return {"label": "low", "severity_score": 0.2}


def risk_dict(severities: dict, has_ckd: int, egfr: float) -> dict:
    d = {t: {"label": LABELS[int(np.clip(round(severities[t]), 0, 2))], "severity_score": float(severities[t])}
         for t in TARGETS}
    d["phosphorus_sensitivity"] = phosphorus_entry(has_ckd, egfr)
    return d


def part_a(engine) -> dict:
    budget = engine.budget_calc.get_daily_budget(has_ckd=True, has_htn=True, has_dm=False)
    rows = []
    for s in np.round(np.arange(0, 2.0001, 0.1), 2):
        risk = risk_dict({t: s for t in TARGETS}, 1, 45)
        for ing in USECASE:
            rec = engine.recommender.recommend_one(ing, risk, budget)
            rows.append({"severity": float(s), "ingredient": ing, "max_grams": rec.max_grams,
                         "binding": rec.binding_constraint, "label": rec.label})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ceiling_sensitivity_curve.csv", index=False)

    tier = []
    for ing in USECASE:
        g = {s: float(df[(df.ingredient == ing) & (np.isclose(df.severity, s))].max_grams.iloc[0]) for s in (0.0, 1.0, 2.0)}
        tier.append({"ingredient": ing, "g_sev0": g[0.0], "g_sev1": g[1.0], "g_sev2": g[2.0],
                     "ratio_1_over_0": round(g[1.0] / g[0.0], 3) if g[0.0] else None,
                     "ratio_2_over_1": round(g[2.0] / g[1.0], 3) if g[1.0] else None,
                     "ratio_2_over_0": round(g[2.0] / g[0.0], 3) if g[0.0] else None})
    pd.DataFrame(tier).to_csv(OUT / "ceiling_sensitivity_tier_effect.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for ing in USECASE:
            d = df[df.ingredient == ing]
            ax.plot(d.severity, d.max_grams, label=ing.split(" (")[0], lw=1.6)
        ax.set(xlabel="severity score (probability-weighted outcome level, all four targets)",
               ylabel="computed gram ceiling (g)", title="Gram ceiling vs severity (use-case patient budget)")
        for x in (0, 1, 2):
            ax.axvline(x, color="grey", lw=0.5, ls=":")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / "ceiling_vs_severity.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print("figure skipped:", exc)
    return {"tier_effect": tier}


def part_b(engine, n_patients: int, seed: int) -> dict:
    df = load_frames("v3.1")
    test = df[df.split == "test"].reset_index(drop=True)
    full = test.dropna(subset=list(TARGETS)).reset_index(drop=True)
    n_full = len(full)
    sample = full.sample(n=min(n_patients, n_full), random_state=seed).reset_index(drop=True)
    pred = engine.model1.predictor
    X = sample[pred.feature_names].to_numpy(dtype=float)
    sev = {"tabnet": {}, "rules": {}, "constant": {}, "oracle": {}}
    rules = LabelGenerator.predict(sample)
    for t in TARGETS:
        proba = pred.predict_proba_matrix(X, t, calibrated=True)
        sev["tabnet"][t] = proba @ np.array([0.0, 1.0, 2.0])
        sev["rules"][t] = rules[t].to_numpy(dtype=float)
        sev["constant"][t] = np.full(len(sample), 1.0)
        sev["oracle"][t] = sample[t].to_numpy(dtype=float)

    ingredients = engine.ifct.df["ingredient"].tolist()
    budgets = {}
    out = {k: np.zeros((len(sample), len(ingredients))) for k in sev}
    for i, row in sample.iterrows():
        key = (int(row.has_ckd), int(row.has_htn), int(row.has_dm))
        if key not in budgets:
            budgets[key] = engine.budget_calc.get_daily_budget(has_ckd=bool(key[0]), has_htn=bool(key[1]), has_dm=bool(key[2]))
        for scen in sev:
            risk = risk_dict({t: sev[scen][t][i] for t in TARGETS}, int(row.has_ckd), float(row.egfr))
            for j, ing in enumerate(ingredients):
                out[scen][i, j] = engine.recommender.recommend_one(ing, risk, budgets[key]).max_grams

    oracle = out["oracle"]
    cap = engine.config.default_cap_g
    label_of = lambda g: np.where(g <= engine.config.avoid_threshold_g, 0, np.where(g <= engine.config.half_portion_g, 1, 2))
    summary = {"n_test_admissions_with_all_four_outcomes": int(n_full), "n_patients_sampled": int(len(sample)),
               "n_ingredients": len(ingredients), "practical_cap_g": cap, "within_tolerance": WITHIN,
               "note": "'all_pairs' includes pairs whose ceiling is the 300 g practical cap (trivially equal); "
                       "'nutrient_bound_pairs' keeps only pairs where the oracle ceiling is below the cap",
               "scenarios": {}}
    per_ing_rows = []

    def stats(g, mask):
        r = (g / np.where(oracle > 0, oracle, 1))[mask]
        lab_o, lab_s = label_of(oracle)[mask], label_of(g)[mask]
        return {"n_pairs": int(mask.sum()),
                "median_ratio_to_oracle": round(float(np.median(r)), 3),
                "mean_abs_log2_ratio": round(float(np.mean(np.abs(np.log2(np.clip(r, 1e-3, None))))), 3),
                f"pct_within_{int(WITHIN * 100)}pct": round(100 * float(np.mean(np.abs(r - 1) <= WITHIN)), 1),
                f"pct_over_permissive_gt_{int(WITHIN * 100)}pct": round(100 * float(np.mean(r > 1 + WITHIN)), 1),
                f"pct_over_restrictive_lt_{int(WITHIN * 100)}pct": round(100 * float(np.mean(r < 1 - WITHIN)), 1),
                "pct_same_portion_label": round(100 * float(np.mean(lab_o == lab_s)), 1),
                "pct_label_looser_than_oracle": round(100 * float(np.mean(lab_s > lab_o)), 1),
                "pct_allowed_where_oracle_avoid": round(100 * float(np.mean((lab_s == 2) & (lab_o == 0))), 2)}

    valid = oracle > 0                                     # ratio undefined when the oracle says 0 g
    bound = valid & (oracle < cap - 1e-6)
    for scen in ("tabnet", "rules", "constant"):
        g = out[scen]
        summary["scenarios"][scen] = {"all_pairs": stats(g, valid), "nutrient_bound_pairs": stats(g, bound)}
        ratio = np.where(valid, g / np.where(valid, oracle, 1), np.nan)
        for j, ing in enumerate(ingredients):
            v = valid[:, j]
            if v.sum() >= 10:
                rj = ratio[v, j]
                per_ing_rows.append({"scenario": scen, "ingredient": ing, "n": int(v.sum()),
                                     "median_ratio": round(float(np.median(rj)), 3),
                                     "pct_within": round(100 * float(np.mean(np.abs(rj - 1) <= WITHIN)), 1)})
    pd.DataFrame(per_ing_rows).to_csv(OUT / "ceiling_sensitivity_per_ingredient.csv", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from train_model2 import PortionControlModel
    with contextlib.redirect_stdout(io.StringIO()):
        engine = PortionControlModel()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"part_a": part_a(engine), "part_b": part_b(engine, args.n_patients, args.seed),
           "caveat": "The oracle uses observed outcome levels as severity. It is a proxy, not a validated safe ceiling."}
    (OUT / "ceiling_sensitivity_summary.json").write_text(json.dumps(res, indent=2))
    for t in res["part_a"]["tier_effect"]:
        print(f"{t['ingredient']:<30} g@sev0 {t['g_sev0']:6.1f}  g@sev1 {t['g_sev1']:6.1f}  g@sev2 {t['g_sev2']:6.1f}"
              f"   x{t['ratio_1_over_0']} then x{t['ratio_2_over_1']}")
    print(json.dumps(res["part_b"], indent=2))


if __name__ == "__main__":
    main()

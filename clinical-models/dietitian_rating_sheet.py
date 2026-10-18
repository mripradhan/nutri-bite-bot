"""
Blinded rating-sheet generator for the dietitian validation study (protocol: dietitian_validation_protocol.md).

Produces, for SYNTHETIC illustrative patient profiles only (never MIMIC records: the MIMIC data-use agreement
forbids sharing row-level data with raters), a sheet of proposed gram ceilings for blinded review:

  rating_sheet.csv   what the dietitians see: profile, ingredient, proposed ceiling (g), blank rating columns
  rating_key.csv     NOT shared with raters: which engine produced each ceiling (deployed TabNet engine, rules-only
                     engine, or both when they agree within 10%), and which rows are repeated for intra-rater checks

Raters are asked for: (a) a 4-level judgement of the ceiling, (b) the ceiling THEY would set (g), (c) comments.
Nothing here is a result; no rating has been collected.

    python dietitian_rating_sheet.py [--patients-per-stratum 6] [--ingredients 8] [--seed 7] [--out DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ceiling_sensitivity import risk_dict
from model1_artifacts import TARGETS
from _prototype.json_recipe_adherence_eval import STRATA, make_patient
from train_model1 import LabelGenerator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOL = 0.10
REPEAT_FRACTION = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients-per-stratum", type=int, default=6)
    ap.add_argument("--ingredients", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "dietitian_study")
    args = ap.parse_args()

    from train_model2 import PortionControlModel
    with contextlib.redirect_stdout(io.StringIO()):
        engine = PortionControlModel()
    rng = np.random.default_rng(args.seed)
    py = random.Random(args.seed)
    cat = engine.ifct.df.groupby("category").ingredient.apply(list).to_dict()
    cats = sorted(cat)

    items, n_id = [], 0
    for (h, d, c) in STRATA:
        for pi in range(args.patients_per_stratum):
            pt = make_patient(rng, h, d, c)
            # ingredient panel spread over food groups
            panel = [py.choice(cat[cats[(pi + j) % len(cats)]]) for j in range(args.ingredients)]
            panel = list(dict.fromkeys(panel))
            tab_risk = engine.model1.predict_risk_levels(pt)
            rule_lv = LabelGenerator.predict(pd.DataFrame([pt])).iloc[0]
            rule_risk = risk_dict({t: float(rule_lv[t]) for t in TARGETS}, pt["has_ckd"], pt["egfr"])
            budget = engine.budget_calc.get_daily_budget(has_ckd=bool(c), has_htn=bool(h), has_dm=bool(d))
            pid = f"P{len(items) // max(len(panel), 1) + 1:03d}"
            for ing in panel:
                g_tab = engine.recommender.recommend_one(ing, tab_risk, budget).max_grams
                g_rule = engine.recommender.recommend_one(ing, rule_risk, budget).max_grams
                same = abs(g_tab - g_rule) <= TOL * max(g_tab, g_rule, 1e-9)
                for engine_name, g in ((("both",), g_tab),) if same else ((("tabnet",), g_tab), (("rules",), g_rule)):
                    n_id += 1
                    items.append({"item_id": f"I{n_id:05d}", "patient_id": f"{len(items)}", "_pt": pt, "ingredient": ing,
                                  "ceiling_g": round(float(g), 0), "engine": engine_name[0], "stratum": f"htn{h}_dm{d}_ckd{c}",
                                  "patient_key": f"S{h}{d}{c}-{pi}"})

    # de-duplicated patient ids and a readable profile
    pkeys = {k: f"P{i + 1:03d}" for i, k in enumerate(dict.fromkeys(i["patient_key"] for i in items))}
    for it in items:
        it["patient_id"] = pkeys[it["patient_key"]]
    # repeated items for intra-rater reliability
    reps = py.sample(items, k=max(1, int(REPEAT_FRACTION * len(items))))
    for r in reps:
        n_id += 1
        items.append({**r, "item_id": f"I{n_id:05d}", "repeat_of": r["item_id"]})
    py.shuffle(items)

    def profile(pt):
        dx = [n for n, f in (("hypertension", pt["has_htn"]), ("type 2 diabetes", pt["has_dm"]), ("CKD", pt["has_ckd"])) if f]
        return (f"{pt['age']} y {'M' if pt['sex_male'] else 'F'}; {', '.join(dx) or 'none of HTN/DM/CKD'}; "
                f"Na {pt['serum_sodium']}, K {pt['serum_potassium']}, creatinine {pt['creatinine']}, eGFR {pt['egfr']}, "
                f"HbA1c {pt['hba1c']}, FBS {pt['fbs']}, BP {pt['sbp']:.0f}/{pt['dbp']:.0f}, BMI {pt['bmi']}")

    sheet = pd.DataFrame([{
        "item_id": it["item_id"], "patient_id": it["patient_id"], "patient_profile": profile(it["_pt"]),
        "ingredient": it["ingredient"], "proposed_max_grams_per_meal": it["ceiling_g"],
        "rating (1=too low, 2=appropriate, 3=too high, 4=unsafe)": "", "your_preferred_max_grams": "", "comment": ""}
        for it in items])
    key = pd.DataFrame([{"item_id": it["item_id"], "engine": it["engine"], "stratum": it["stratum"],
                         "repeat_of": it.get("repeat_of", "")} for it in items])
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out / "rating_sheet.csv", index=False)
    key.to_csv(args.out / "rating_key.csv", index=False)
    print(f"{len(sheet)} items ({len(reps)} repeats) for {len(pkeys)} synthetic patients -> {args.out}")
    print("engine split:", key.engine.value_counts().to_dict())


if __name__ == "__main__":
    main()

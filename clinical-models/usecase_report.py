"""
Regenerate the manuscript's clinical use case (Section 5) from the deployed system,
so every number in that section comes from code rather than hand transcription.

  Model 1: per-target risk tier, calibrated outcome probabilities, severity score,
           top TabNet attributions, and the guideline-rule baseline for the same patient
  Model 2: per-ingredient portion decisions from both implementations that exist in
           the repo (train_model2.PortionControlModel and the deployed app.py
           /api/recommend), with any disagreement flagged

Outputs (artifacts/models/reports/): usecase_report.json, usecase_report.md, usecase_table.tex
The patient is the manuscript's illustrative profile, not a MIMIC record.

    python usecase_report.py            # both engines
    python usecase_report.py --no-app   # skip app.py (needs flask/groq/supabase deps)
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

from model1_artifacts import TARGETS, Model1Predictor
from train_model1 import LabelGenerator

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "artifacts" / "models"
OUT_DIR = MODEL_DIR / "reports"

# Manuscript Section 5: 58-year-old male, CKD stage 3, uncontrolled HTN, no diabetes
PATIENT = {
    "age": 58, "sex_male": 1, "has_htn": 1, "has_dm": 0, "has_ckd": 1,
    "serum_sodium": 140, "serum_potassium": 5.1, "creatinine": 1.9, "egfr": 45,
    "hba1c": 5.2, "fbs": 85, "sbp": 162, "dbp": 96, "bmi": 26,
}
INGREDIENTS = [
    "Green gram dal (Moong dal)", "Paneer (Cottage cheese)", "Egg, whole, boiled", "Rice, milled (white)",
    "Banana, ripe", "Apple", "Pineapple (Ananas)", "Bottle gourd (Lauki)",
]


def model2_research(patient):
    from train_model2 import PortionControlModel
    with contextlib.redirect_stdout(io.StringIO()):
        model = PortionControlModel()
        res = model.get_recommendations(patient, INGREDIENTS, include_substitutes=False)
    return {r["ingredient"]: {"label": r["label"], "max_grams": round(float(r["max_grams"]), 1),
                              "binding_constraint": r.get("binding_constraint")}
            for r in res["recommendations"]}, res.get("clinical_warnings", [])


def model2_app(patient):
    sys.path.insert(0, str(ROOT))
    with contextlib.redirect_stdout(io.StringIO()):
        import app as deployed
    r = deployed.app.test_client().post("/api/recommend", json={"patient": patient, "ingredients": INGREDIENTS})
    if r.status_code != 200:
        raise RuntimeError(f"/api/recommend returned {r.status_code}: {r.get_data(as_text=True)[:300]}")
    body = r.get_json()
    return {x["ingredient"]: {"label": x["label"], "max_grams": round(float(x["max_grams"]), 1),
                              "binding_constraint": x.get("binding_constraint")}
            for x in body["recommendations"]}, body["daily_budget"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-app", action="store_true")
    args = ap.parse_args()

    predictor = Model1Predictor(MODEL_DIR)
    risk = predictor.predict(PATIENT)
    rules = LabelGenerator.predict(pd.DataFrame([PATIENT])).iloc[0]

    report = {
        "model_manifest": {k: predictor.manifest[k] for k in ("created_utc", "git", "config")},
        "patient": PATIENT,
        "model1": {t: {**risk[t], "guideline_rule_label": ["low", "moderate", "high"][int(rules[t])]} for t in TARGETS},
        "ingredients": INGREDIENTS,
    }
    research, warnings_ = model2_research(PATIENT)
    report["model2_research"] = research
    report["model2_research_warnings"] = warnings_
    if not args.no_app:
        deployed, budget = model2_app(PATIENT)
        report["model2_app"] = deployed
        report["daily_budget_app"] = budget
        report["engine_disagreements"] = [
            {"ingredient": i, "research": research.get(i), "app": deployed.get(i)} for i in INGREDIENTS
            if research.get(i, {}).get("label", "").lower() != deployed.get(i, {}).get("label", "").lower()
            or abs(research.get(i, {}).get("max_grams", 0) - deployed.get(i, {}).get("max_grams", 0)) > 0.5
        ]

    (OUT_DIR / "usecase_report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = ["# Clinical use case — regenerated from the deployed system", "",
             f"Model trained {report['model_manifest']['created_utc']}, commit {report['model_manifest']['git']['commit'][:7]}.", "",
             "## Model 1", "",
             "| Target | Tier (TabNet) | Guideline rules | P(low / moderate / high), calibrated | Severity | Top attributions |",
             "|---|---|---|---|---|---|"]
    for t in TARGETS:
        r = report["model1"][t]
        p = r["proba"]
        top = ", ".join(f"{f} ({w:.2f})" for f, w in list(r["feature_attribution"].items())[:3])
        lines.append(f"| {t} | {r['label'].upper()} | {r['guideline_rule_label']} | "
                     f"{p['low']:.2f} / {p['moderate']:.2f} / {p['high']:.2f} | {r['severity_score']:.2f} | {top} |")
    engine = report.get("model2_app", research)
    lines += ["", f"## Model 2 — portion decisions ({'deployed app.py' if 'model2_app' in report else 'train_model2'})", "",
              "| Ingredient | Decision | Max (g) | Binding constraint |", "|---|---|---|---|"]
    for i in INGREDIENTS:
        e = engine[i]
        lines.append(f"| {i} | {e['label']} | {e['max_grams']:.1f} | {e['binding_constraint']} |")
    if report.get("engine_disagreements"):
        lines += ["", "## ⚠ app.py and train_model2 disagree", ""]
        for d in report["engine_disagreements"]:
            lines.append(f"- {d['ingredient']}: train_model2 {d['research']} vs app {d['app']}")
    (OUT_DIR / "usecase_report.md").write_text("\n".join(lines) + "\n")

    tex = ["% generated by clinical-models/usecase_report.py — do not edit by hand",
           "\\begin{tabular}{p{3.0cm} l r p{1.8cm}}", "\\hline",
           "\\bfseries Ingredient & \\bfseries Decision & \\bfseries Max (g) & \\bfseries Binding Constraint \\\\", "\\hline"]
    for i in INGREDIENTS:
        e = engine[i]
        tex.append(f"{i} & {e['label']} & {e['max_grams']:.1f} & {str(e['binding_constraint']).capitalize()} \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (OUT_DIR / "usecase_table.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(lines))
    print(f"\n  ✓ Wrote usecase_report.json / .md / usecase_table.tex to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

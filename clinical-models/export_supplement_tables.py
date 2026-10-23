"""Emit the reproducibility tables for the manuscript supplement (Reviewer 1, items 1-3).

Reads the deployed artifacts only (no torch, no MIMIC data) and writes LaTeX + a JSON copy to
artifacts/models/reports/supplement/:

  tabnet_hyperparameters.tex   per-target TabNet configuration and training outcome
  label_definitions.tex        observed-outcome training labels (the targets)
  rule_baseline_thresholds.tex guideline-rule comparison baseline (LabelGenerator)
  monotonic_constraints.tex    constraint direction per feature per target

Run:  python export_supplement_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "artifacts" / "models"
OUT = MODELS / "reports" / "supplement"

TARGET_LABEL = {
    "sodium_sensitivity": "Follow-up BP risk",
    "potassium_sensitivity": "Hyperkalaemia risk",
    "protein_restriction": "AKI (renal) risk",
    "carb_sensitivity": "Hyperglycaemia risk",
}

# Outcome definitions mirror mimic_extract.add_derived (ordinal cut points are strict ">").
# References are the guideline the cut point follows; keep in sync with the manuscript bibliography.
LABEL_ROWS = [
    ("potassium_sensitivity", "max serum K after 24 h baseline, LOS $\\geq$ 48 h",
     "$\\leq$5.0", "5.1--5.5", "$>$5.5", "KDIGO hyperkalaemia conventions"),
    ("carb_sensitivity", "max plasma glucose after 24 h baseline, LOS $\\geq$ 48 h",
     "$\\leq$180", "181--250", "$>$250", "ADA Standards of Care 2024 (inpatient)"),
    ("sodium_sensitivity", "mean outpatient SBP 30--365 d post-discharge",
     "$\\leq$140", "141--160", "$>$160", "ACC/AHA 2017 BP guideline"),
    ("protein_restriction", "KDIGO AKI vs. baseline creatinine, LOS $\\geq$ 48 h",
     "no AKI", "stage 1 ($\\geq$0.3 mg/dL or $\\geq$1.5$\\times$)", "stage 2+ ($\\geq$2.0$\\times$)",
     "KDIGO AKI 2012"),
]

# Guideline-rule baseline; mirrors train_model1.LabelGenerator.predict (missing input => not met).
RULE_ROWS = [
    ("sodium_sensitivity", "Na $>$ 145 or HTN", "Na $>$ 150 or (HTN and SBP $>$ 160)"),
    ("potassium_sensitivity", "K $>$ 4.5 or eGFR $<$ 60", "K $>$ 5.0 or (CKD and eGFR $<$ 30)"),
    ("protein_restriction", "eGFR $<$ 60", "eGFR $<$ 30 or CKD or creatinine $>$ 2.0"),
    ("carb_sensitivity", "HbA1c $\\geq$ 5.7 or FBS $\\geq$ 100 or BMI $\\geq$ 25 or DM",
     "HbA1c $\\geq$ 7.0 or FBS $\\geq$ 126 or BMI $\\geq$ 30 or (DM and HbA1c $\\geq$ 6.5)"),
]


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def table(caption: str, label: str, header: list[str], rows: list[list[str]], colspec: str) -> str:
    lines = ["\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\hline", " & ".join(header) + " \\\\", "\\hline"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tabnet_table(manifest: dict) -> tuple[str, dict]:
    cfg = manifest["config"]
    shared = {"n_d / n_a": f"{cfg['n_d']} / {cfg['n_a']}", "decision steps": cfg["n_steps"],
              "relaxation $\\gamma$": cfg["gamma"], "mask": cfg["mask_type"],
              "weight decay": cfg["weight_decay"], "optimiser": "Adam",
              "initial lr": cfg["lr"], "lr schedule": "StepLR (step 50, factor 0.9)",
              "batch / virtual batch": f"{cfg['batch_size']} / {cfg['virtual_batch_size']}",
              "max epochs / patience": f"{cfg['max_epochs']} / {cfg['patience']}",
              "early-stop metric": "balanced accuracy, CV fold 0",
              "class balance": "inverse-frequency sampler" if cfg["class_balance"] else "none",
              "seed": cfg["seed"]}
    # lambda_sparse and other library defaults live in each target's model_params.json
    per_target = {}
    for t in manifest["targets"]:
        p = load_json(MODELS / "tabnet" / t / "model_params.json")["init_params"]
        tr = manifest["training"][t]
        per_target[t] = {"lambda_sparse": p["lambda_sparse"], "momentum": p["momentum"],
                         "n_independent": p["n_independent"], "n_shared": p["n_shared"],
                         "clip_value": p["clip_value"], "best_epoch": tr["best_epoch"],
                         "n_train": tr["n_train_real"]}
    rows = [[esc(k), esc(v)] for k, v in shared.items()]
    t1 = table("TabNet configuration shared by all four per-target classifiers.",
               "tab:tabnet-shared", ["Hyperparameter", "Value"], rows, "ll")
    keys = ["lambda_sparse", "momentum", "n_independent", "n_shared", "clip_value", "best_epoch", "n_train"]
    rows = [[esc(k)] + [esc(per_target[t][k]) for t in manifest["targets"]] for k in keys]
    t2 = table("Per-target library defaults and training outcome (from \\texttt{model\\_params.json} and the manifest).",
               "tab:tabnet-target", ["Item"] + [esc(TARGET_LABEL[t]) for t in manifest["targets"]], rows, "lcccc")

    grid_path = MODELS / "reports" / "tuning_results_fold0.csv"
    t3 = ""
    if grid_path.exists():
        by_setting: dict[tuple, list[float]] = {}
        with grid_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by_setting.setdefault((r["lr"], r["batch_size"], r["class_balance"]), []).append(float(r["f1_macro"]))
        rows = [[esc(a), esc(b), esc(c), f"{sum(v) / len(v):.4f}"]
                for (a, b, c), v in sorted(by_setting.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))]
        t3 = table("Tuning grid evaluated on CV fold 0 (mean macro-F1 over the four targets).",
                   "tab:tuning", ["lr", "batch size", "class balance", "mean macro-F1"], rows, "lccc")
    return t1 + t2 + t3, {"shared": shared, "per_target": per_target}


def monotonic_table(pre: dict) -> tuple[str, dict]:
    mono = pre["monotonic"]
    feats = pre["feature_names"]
    targets = list(mono)
    signs: dict[str, dict[str, int]] = {t: {} for t in targets}
    for t in targets:
        for f, curve in mono[t].items():
            signs[t][f] = 1 if curve["y"][-1] >= curve["y"][0] else -1
    rows = []
    for f in feats:
        cells = [{1: "$+$", -1: "$-$"}.get(signs[t].get(f), "--") for t in targets]
        rows.append([esc(f)] + cells)
    tex = table("Monotonic constraint direction fitted per feature and target "
                "(isotonic fit against the ordinal outcome; -- = unconstrained).",
                "tab:monotonic", ["Feature"] + [esc(TARGET_LABEL.get(t, t)) for t in targets], rows, "l" + "c" * len(targets))
    return tex, signs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_json(MODELS / "manifest.json")
    pre = load_json(MODELS / "preprocessing.json")

    tab_tex, tab_json = tabnet_table(manifest)
    mono_tex, mono_json = monotonic_table(pre)
    label_tex = table(
        "Observed-outcome definitions used as training targets (24 h baseline window; HbA1c look-back 90 d). "
        "These are outcome-risk proxies, not measured dietary responses.",
        "tab:labels", ["Target", "Outcome", "Low", "Moderate", "High", "Reference"],
        [[esc(TARGET_LABEL[t]), o, lo, mo, hi, esc(ref)] for t, o, lo, mo, hi, ref in LABEL_ROWS], "llcccl")
    rule_tex = table(
        "Guideline-rule comparison baseline (\\texttt{LabelGenerator}); not used to train any model. "
        "A missing input evaluates as ``threshold not met''.",
        "tab:rules", ["Target", "Moderate if", "High if"],
        [[esc(TARGET_LABEL[t]), m, h] for t, m, h in RULE_ROWS], "lll")

    files = {"tabnet_hyperparameters.tex": tab_tex, "label_definitions.tex": label_tex,
             "rule_baseline_thresholds.tex": rule_tex, "monotonic_constraints.tex": mono_tex}
    for name, body in files.items():
        (OUT / name).write_text(body, encoding="utf-8")
    (OUT / "supplement_tables.json").write_text(
        json.dumps({"tabnet": tab_json, "monotonic_signs": mono_json,
                    "git": manifest.get("git"), "model_created_utc": manifest.get("created_utc")}, indent=2),
        encoding="utf-8")
    print(f"wrote {len(files) + 1} files to {OUT}")
    if manifest.get("git", {}).get("clinical_models_dirty"):
        print("WARNING: deployed model was trained from a dirty working tree "
              f"(commit {manifest['git']['commit'][:7]}); retrain from a clean commit or disclose.")


if __name__ == "__main__":
    main()

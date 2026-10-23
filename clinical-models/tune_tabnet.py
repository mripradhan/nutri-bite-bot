"""
Time-boxed TabNet tuning on CV fold 0 ONLY (the held-out test set is never opened here).

One named configuration per process so several can run in parallel on a many-core CPU:

    python tune_tabnet.py --name s1_nomono --threads 5
    python tune_tabnet.py --name combo --set monotonic=false batch_size=1024 virtual_batch_size=128 lr=0.01

Each run fits the four per-target networks on training patients with CV fold != 0, early-stops on
fold 0 (balanced accuracy, exactly as the deployed model), scores fold 0, and writes
artifacts/experiments/tabnet_tuning/<name>.json. `summarize` ranks all runs by mean macro-F1
(ties: mean AUROC) next to the deployed configuration's fold-0 numbers from its manifest.

    python tune_tabnet.py --summarize
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "experiments" / "tabnet_tuning"

# Stage 1: one change at a time against the deployed configuration.
CONFIGS = {
    "s1_nomono": {"monotonic": False},
    "s1_b2048": {"batch_size": 2048, "virtual_batch_size": 256},
    "s1_b1024_lr01": {"batch_size": 1024, "virtual_batch_size": 128, "lr": 0.01},
    "s1_w32": {"n_d": 32, "n_a": 32},
    "s1_steps3": {"n_steps": 3},
    "s1_sparse1e-4": {"lambda_sparse": 1e-4},
    "s1_decay": {"step_size": 5, "scheduler_gamma": 0.8},
    "s1_long": {"patience": 25, "max_epochs": 150},
}


def parse_value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        return float(v)


def run(name: str, overrides: dict, threads: int | None):
    from train_model1 import ClinicalModelConfig, ClinicalRiskStratifier, load_frames
    if threads:
        torch.set_num_threads(threads)
    manifest = json.loads((ROOT / "artifacts" / "models" / "manifest.json").read_text())["config"]
    base = {k: v for k, v in manifest.items() if k != "tag"}
    cfg = ClinicalModelConfig(tag="v3.1", **{**base, **overrides})
    cfg.model_dir = OUT / "_unused_models"  # nothing is saved by this script

    df = load_frames(cfg.tag)
    train_df = df[(df.split == "train") & (df.cv_fold != 0)]
    val_df = df[(df.split == "train") & (df.cv_fold == 0)]
    t0 = time.time()
    strat = ClinicalRiskStratifier(cfg).fit(train_df, val_df, verbose=False)
    metrics, *_ = strat.evaluate(val_df)
    res = {"name": name, "overrides": overrides, "config": cfg.summary(), "minutes": round((time.time() - t0) / 60, 1),
           "targets": {t: {**m, **strat.training_info[t]} for t, m in metrics.items()},
           "mean_f1_macro": float(np.mean([m["f1_macro"] for m in metrics.values()])),
           "mean_auroc": float(np.mean([m["auroc_ovr_macro"] for m in metrics.values()]))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"{name}: mean fold-0 macro-F1 {res['mean_f1_macro']:.4f}  AUROC {res['mean_auroc']:.4f}  "
          f"({res['minutes']} min)  best epochs {[strat.training_info[t]['best_epoch'] for t in cfg.targets]}")


def summarize():
    manifest = json.loads((ROOT / "artifacts" / "models" / "manifest.json").read_text())
    val = manifest["metrics"]["validation_fold0"]
    rows = [("deployed", float(np.mean([v["f1_macro"] for v in val.values()])),
             float(np.mean([v["auroc_ovr_macro"] for v in val.values()])), {t: v["f1_macro"] for t, v in val.items()}, "-")]
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text())
        rows.append((r["name"], r["mean_f1_macro"], r["mean_auroc"],
                     {t: v["f1_macro"] for t, v in r["targets"].items()}, r["minutes"]))
    rows.sort(key=lambda x: (-x[1], -x[2]))
    print(f"{'run':<18}{'meanF1':>8}{'meanAUC':>9}  per-target macro-F1 (Na, K, protein, carb)   min")
    for n, f1, auc, per, mins in rows:
        print(f"{n:<18}{f1:8.4f}{auc:9.4f}  " + " ".join(f"{per.get(t, float('nan')):.3f}" for t in
              ("sodium_sensitivity", "potassium_sensitivity", "protein_restriction", "carb_sensitivity")) + f"   {mins}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides on top of the named config")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        ov = {**CONFIGS.get(a.name, {}), **{k: parse_value(v) for k, v in (s.split("=") for s in a.set)}}
        run(a.name, ov, a.threads)

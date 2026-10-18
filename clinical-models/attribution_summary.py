"""
Aggregate TabNet feature attributions over the held-out test patients (manuscript
attribution table). Per patient, TabNet's explain() mask is normalised to sum to 1;
masks are averaged per target, overall and within each predicted risk tier.

Output: artifacts/models/reports/evaluation/attribution_summary.{csv,json} (aggregates only)

    python attribution_summary.py --tag v3.1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model1_artifacts import LABELS, TARGETS, Model1Predictor, apply_monotonic
from train_model1 import load_frames

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    args = ap.parse_args()

    predictor = Model1Predictor(args.model_dir)
    feats = predictor.feature_names
    test = load_frames(args.tag).query("split == 'test'")
    out_dir = args.model_dir / "reports" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = [], {}
    for target in TARGETS:
        part = test[test[target].notna()]
        X_raw = part[feats].to_numpy(dtype=float)
        X = np.where(np.isnan(X_raw), predictor.medians, X_raw)
        X = apply_monotonic((X - predictor.mean) / predictor.scale, feats, predictor.monotonic[target]).astype(np.float32)
        model = predictor.models[target]
        masks, _ = model.explain(X)
        totals = masks.sum(axis=1, keepdims=True)
        masks = np.divide(masks, totals, out=np.zeros_like(masks), where=totals > 0)
        tier = predictor.predict_proba_matrix(X_raw, target).argmax(axis=1)

        groups = {"all": np.ones(len(X), bool), **{f"predicted_{LABELS[k]}": tier == k for k in range(3)}}
        summary[target] = {}
        for gname, gmask in groups.items():
            if gmask.sum() < 10:
                continue
            mean = masks[gmask].mean(axis=0)
            order = np.argsort(mean)[::-1]
            summary[target][gname] = {"n": int(gmask.sum()),
                                      "top5": {feats[i]: round(float(mean[i]), 4) for i in order[:5]}}
            for i in range(len(feats)):
                rows.append({"target": target, "group": gname, "n": int(gmask.sum()),
                             "feature": feats[i], "mean_attribution": round(float(mean[i]), 4)})
        top = summary[target]["all"]["top5"]
        print(f"{target:<24} " + ", ".join(f"{f} {w:.3f}" for f, w in top.items()))

    pd.DataFrame(rows).to_csv(out_dir / "attribution_summary.csv", index=False)
    (out_dir / "attribution_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  ✓ Wrote {out_dir.relative_to(ROOT)}/attribution_summary.csv/.json")


if __name__ == "__main__":
    main()

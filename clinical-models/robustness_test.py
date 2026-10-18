"""
TabNet vs guideline-rule robustness comparison on the real held-out test set,
using the already-deployed Model 1 artifacts. This is the comparison
feedbackplan.md Sec 2 asks for: a perturbed/noisy-label test showing whose
decision boundary is smoother near thresholds, and a missing-data test showing
the rules engine failing closed while TabNet still predicts.

Three experiments, all read-only against the test set (features are perturbed
in memory; nothing is refit and the deployed model is loaded once):

  (a) noise sweep     -- Gaussian noise on each lab feature at 0.5x/1x/2x an
                          assumed measurement-imprecision SD (documented below);
                          reports prediction flip rate and TabNet-rules
                          agreement rate per target per magnitude.
  (b) threshold curves -- for patients near a guideline threshold, sweep that
                          one feature continuously across the boundary and
                          record TabNet's calibrated probability curve vs the
                          rules' step function.
  (c) missing data     -- null 1-3 lab features per patient (NaN, so rules
                          "cannot fire" per LabelGenerator's documented
                          missing-input semantics) and compare TabNet's
                          (median-imputed) degradation to the rules'.

Usage:
    python robustness_test.py --tag v3.1
    python robustness_test.py --tag v3.1 --targets carb_sensitivity --n-draws 3   # smoke test
"""
import argparse
import json
from pathlib import Path

import torch  # noqa: F401  -- must be imported before sklearn (via mimic_extract/model1_artifacts) or torch's DLL init fails on Windows
import numpy as np
import pandas as pd

from mimic_extract import SMALL_CELL, ckd_epi_2021
from model1_artifacts import TARGETS, Model1Predictor
from train_model1 import LabelGenerator, load_frames

ROOT = Path(__file__).resolve().parent.parent

# Approximate combined analytical + within-subject biological SD per lab, used as the
# 1x noise magnitude below. These are documented, illustrative assumptions (broadly
# consistent with published CLIA total-allowable-error ranges) -- not a specific
# citation; state so if quoted in the manuscript.
NOISE_SD = {
    "serum_sodium": 2.0,      # mmol/L
    "serum_potassium": 0.15,  # mmol/L
    "hba1c": 0.2,             # percentage points
    "fbs": 8.0,                # mg/dL
    "sbp": 5.0,                # mmHg
    "dbp": 4.0,                # mmHg
    "bmi": 0.5,                 # kg/m^2
}
CREATININE_REL_SD = 0.08  # relative (8% CV); egfr is recomputed from noised creatinine, not noised directly

# Guideline thresholds a rule's condition can cross (LabelGenerator.predict), used for
# the threshold-boundary sweep. feature -> (threshold, target it gates).
THRESHOLDS = [
    ("hba1c", 5.7, "carb_sensitivity"), ("hba1c", 7.0, "carb_sensitivity"),
    ("fbs", 100.0, "carb_sensitivity"), ("fbs", 126.0, "carb_sensitivity"),
    ("serum_sodium", 145.0, "sodium_sensitivity"), ("serum_sodium", 150.0, "sodium_sensitivity"),
    ("serum_potassium", 4.5, "potassium_sensitivity"), ("serum_potassium", 5.0, "potassium_sensitivity"),
    ("egfr", 60.0, "protein_restriction"), ("egfr", 30.0, "protein_restriction"),
    ("creatinine", 2.0, "protein_restriction"),
]
MISSING_POOL = ["serum_sodium", "serum_potassium", "creatinine", "hba1c", "fbs", "sbp", "dbp", "bmi"]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def tabnet_pred(predictor: Model1Predictor, df: pd.DataFrame, target: str) -> np.ndarray:
    X = df[predictor.feature_names].to_numpy(dtype=float)
    return predictor.predict_proba_matrix(X, target).argmax(axis=1)


def tabnet_proba(predictor: Model1Predictor, df: pd.DataFrame, target: str) -> np.ndarray:
    X = df[predictor.feature_names].to_numpy(dtype=float)
    return predictor.predict_proba_matrix(X, target, calibrated=True)


def add_noise(df: pd.DataFrame, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    """Gaussian noise on labs; creatinine is noised multiplicatively and eGFR is recomputed
    from the noised creatinine so the two stay clinically consistent (mirrors cohort_data.synthesize)."""
    out = df.copy()
    for col, sd in NOISE_SD.items():
        out[col] = out[col] + rng.normal(0, sd * magnitude, size=len(out))
    out["creatinine"] = out["creatinine"] * (1 + rng.normal(0, CREATININE_REL_SD * magnitude, size=len(out)))
    out["creatinine"] = out["creatinine"].clip(lower=0.1)
    out["egfr"] = ckd_epi_2021(out["creatinine"], out["age"], out["sex_male"])
    return out


def noise_sweep(predictor: Model1Predictor, test: pd.DataFrame, targets: list[str],
                magnitudes: list[float], n_draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        mask = test[target].notna().to_numpy()
        base = test.loc[mask].reset_index(drop=True)
        base_tab = tabnet_pred(predictor, base, target)
        base_rule = LabelGenerator.predict(base)[target].to_numpy()
        for mag in magnitudes:
            flips, agree_draws = [], []
            for _ in range(n_draws):
                noised = add_noise(base, mag, rng)
                tab = tabnet_pred(predictor, noised, target)
                rule = LabelGenerator.predict(noised)[target].to_numpy()
                flips.append((tab != base_tab).mean())
                agree_draws.append((tab == rule).mean())
            base_agree = (base_tab == base_rule).mean()
            rows.append({"target": target, "noise_magnitude": mag, "n": int(mask.sum()),
                        "tabnet_flip_rate": round(float(np.mean(flips)), 4),
                        "tabnet_rules_agreement": round(float(np.mean(agree_draws)), 4),
                        "baseline_agreement_no_noise": round(float(base_agree), 4)})
    return pd.DataFrame(rows)


def threshold_stability(predictor: Model1Predictor, test: pd.DataFrame, band_frac: float,
                        n_points: int, max_patients: int, seed: int, out_dir: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    rows = []
    for feature, thr, target in THRESHOLDS:
        mask = test[target].notna().to_numpy()
        pool = test.loc[mask].reset_index(drop=True)
        band = band_frac * abs(thr)
        near = pool[(pool[feature] >= thr - band) & (pool[feature] <= thr + band)]
        if len(near) < SMALL_CELL:
            continue
        idx = rng.choice(near.index, size=min(max_patients, len(near)), replace=False)
        sample = near.loc[idx].reset_index(drop=True)

        sweep = np.linspace(thr - 3 * band, thr + 3 * band, n_points)
        tab_curve, rule_curve = [], []
        for v in sweep:
            probe = sample.copy()
            probe[feature] = v
            if feature == "creatinine":
                probe["egfr"] = ckd_epi_2021(probe["creatinine"], probe["age"], probe["sex_male"])
            proba = tabnet_proba(predictor, probe, target)
            tab_curve.append(float(proba[:, 1:].sum(axis=1).mean()))  # mean P(moderate or high)
            rule_curve.append(float((LabelGenerator.predict(probe)[target] > 0).mean()))
            rows.append({"feature": feature, "threshold": thr, "target": target, "swept_value": round(float(v), 4),
                        "n_patients": len(sample), "tabnet_mean_p_elevated": round(tab_curve[-1], 4),
                        "rules_frac_elevated": round(rule_curve[-1], 4)})

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(sweep, tab_curve, "o-", ms=3, label="TabNet P(moderate/high)")
        ax.plot(sweep, rule_curve, "s-", ms=3, label="rules (step function)")
        ax.axvline(thr, color="k", ls="--", lw=0.8, label=f"threshold={thr}")
        ax.set(xlabel=feature, ylabel="fraction / mean probability", title=f"{target}: {feature} @ {thr}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"threshold_{target}_{feature}_{thr}.png".replace(".", "p", 1), dpi=130)
        plt.close(fig)
    return pd.DataFrame(rows)


def missing_data_test(predictor: Model1Predictor, test: pd.DataFrame, targets: list[str],
                      ks: list[int], n_draws: int, seed: int) -> pd.DataFrame:
    from evaluate_model1 import metrics
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        mask = test[target].notna().to_numpy()
        base = test.loc[mask].reset_index(drop=True)
        y = base[target].astype(int).to_numpy()

        full_tab = tabnet_pred(predictor, base, target)
        full_tab_proba = tabnet_proba(predictor, base, target)
        full_rule = LabelGenerator.predict(base)[target].to_numpy()
        rows.append({"target": target, "k_missing": 0, "n": len(base),
                    **{f"tabnet_{k}": v for k, v in metrics(y, full_tab, full_tab_proba).items()},
                    **{f"rules_{k}": v for k, v in metrics(y, full_rule).items()},
                    "tabnet_rules_agreement": round(float((full_tab == full_rule).mean()), 4)})

        for k in ks:
            tab_m_draws, rule_m_draws, agree_draws, cant_fire = [], [], [], []
            for _ in range(n_draws):
                nulled = base.copy()
                # exactly k of len(MISSING_POOL) columns nulled per row, vectorized: rank random
                # scores per row and keep the k lowest-ranked columns (no per-row Python loop)
                ranks = np.argsort(np.argsort(rng.random((len(nulled), len(MISSING_POOL))), axis=1), axis=1)
                col_mask = ranks < k
                for j, col in enumerate(MISSING_POOL):
                    nulled.loc[col_mask[:, j], col] = np.nan
                tab = tabnet_pred(predictor, nulled, target)
                tab_p = tabnet_proba(predictor, nulled, target)
                rule = LabelGenerator.predict(nulled)[target].to_numpy()
                tab_m_draws.append(metrics(y, tab, tab_p))
                rule_m_draws.append(metrics(y, rule))
                agree_draws.append((tab == rule).mean())
                cant_fire.append((rule < full_rule).mean())  # rule dropped a tier it would have hit with full data
            tab_mean = {k2: round(float(np.mean([d[k2] for d in tab_m_draws])), 4) for k2 in tab_m_draws[0]}
            rule_mean = {k2: round(float(np.mean([d[k2] for d in rule_m_draws])), 4) for k2 in rule_m_draws[0]}
            rows.append({"target": target, "k_missing": k, "n": len(base),
                        **{f"tabnet_{k2}": v for k2, v in tab_mean.items()},
                        **{f"rules_{k2}": v for k2, v in rule_mean.items()},
                        "tabnet_rules_agreement": round(float(np.mean(agree_draws)), 4),
                        "rules_frac_fails_closed_lower": round(float(np.mean(cant_fire)), 4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="TabNet vs guideline-rule robustness comparison on the held-out test set")
    ap.add_argument("--tag", default="v3.1")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--magnitudes", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--missing-k", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--band-frac", type=float, default=0.02, help="threshold-sweep band as a fraction of the threshold value")
    ap.add_argument("--n-points", type=int, default=41)
    ap.add_argument("--max-patients-per-threshold", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = args.model_dir.resolve()
    out_dir = model_dir / "reports" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = Model1Predictor(model_dir)
    df = load_frames(args.tag)
    test = df[df.split == "test"].reset_index(drop=True)
    print(f"Loaded deployed model from {rel(model_dir)}; held-out test set {len(test):,} admissions")

    print("\n(a) noise sweep ...")
    noise_df = noise_sweep(predictor, test, args.targets, args.magnitudes, args.n_draws, args.seed)
    noise_df.to_csv(out_dir / "noise_sweep.csv", index=False)
    print(noise_df.to_string(index=False))

    print("\n(b) threshold-boundary stability ...")
    thr_df = threshold_stability(predictor, test, args.band_frac, args.n_points,
                                 args.max_patients_per_threshold, args.seed, out_dir)
    thr_df.to_csv(out_dir / "threshold_stability.csv", index=False)

    print("\n(c) missing-data test ...")
    miss_df = missing_data_test(predictor, test, args.targets, args.missing_k, args.n_draws, args.seed)
    miss_df.to_csv(out_dir / "missing_data.csv", index=False)
    print(miss_df[["target", "k_missing", "n", "tabnet_f1_macro", "rules_f1_macro",
                   "tabnet_rules_agreement", "rules_frac_fails_closed_lower"]].to_string(index=False))

    summary = {
        "model_dir": rel(model_dir), "n_test": int(len(test)), "targets": args.targets,
        "noise_magnitudes": args.magnitudes, "n_draws": args.n_draws, "missing_k": args.missing_k,
        "noise_sd_documented": {**NOISE_SD, "creatinine_relative": CREATININE_REL_SD},
        "noise_sweep": json.loads(noise_df.to_json(orient="records")),
        "missing_data": json.loads(miss_df.to_json(orient="records")),
    }
    (out_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  ✓ Wrote {rel(out_dir)}/ (noise_sweep.csv, threshold_stability.csv, missing_data.csv, "
          f"threshold_*.png, robustness_summary.json)")


if __name__ == "__main__":
    main()

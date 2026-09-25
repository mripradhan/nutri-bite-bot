"""
Quantitative evaluation of whether the bounded LLM respects the rules engine's gram
limits (Reviewer 1, point 7).

Drives the deployed /api/generate-recipe endpoint over a grid of patient profiles and
ingredient sets, then aggregates the per-ingredient adherence records the endpoint
returns: match rate, violation rate, and overage when violated.

The patient profiles are SYNTHETIC and illustrative, constructed here — not MIMIC-IV
rows. Recipe generation sends the profile to Groq, and PhysioNet's DUA forbids sending
credentialed data to third-party APIs.

Requires GROQ_API_KEY. Supabase is optional: the endpoint persists rows to
recipe_adherence when configured, and this script reports the row count as a check that
the migration was applied; the numbers themselves come from the API responses.

    python recipe_adherence_eval.py --repeats 2              # ~32 recipes
    python recipe_adherence_eval.py --repeats 1 --profiles htn_ckd htn_dm_ckd
"""
import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "models" / "reports" / "recipe_adherence"

# Synthetic profiles spanning the comorbidity strata the system targets. Values are
# clinically coherent for each condition set (not sampled from any patient record).
PROFILES = {
    "none":       dict(age=45, sex_male=1, has_htn=0, has_dm=0, has_ckd=0, serum_sodium=140, serum_potassium=4.2,
                       creatinine=0.9, egfr=95, hba1c=5.2, fbs=88, sbp=118, dbp=76, bmi=23),
    "htn":        dict(age=55, sex_male=0, has_htn=1, has_dm=0, has_ckd=0, serum_sodium=142, serum_potassium=4.1,
                       creatinine=1.0, egfr=88, hba1c=5.4, fbs=92, sbp=158, dbp=94, bmi=28),
    "dm":         dict(age=52, sex_male=1, has_htn=0, has_dm=1, has_ckd=0, serum_sodium=138, serum_potassium=4.3,
                       creatinine=0.95, egfr=92, hba1c=8.6, fbs=178, sbp=126, dbp=80, bmi=30),
    "ckd":        dict(age=61, sex_male=1, has_htn=0, has_dm=0, has_ckd=1, serum_sodium=137, serum_potassium=5.3,
                       creatinine=2.4, egfr=28, hba1c=5.5, fbs=95, sbp=132, dbp=82, bmi=25),
    "htn_dm":     dict(age=58, sex_male=0, has_htn=1, has_dm=1, has_ckd=0, serum_sodium=141, serum_potassium=4.4,
                       creatinine=1.1, egfr=78, hba1c=9.1, fbs=196, sbp=162, dbp=96, bmi=32),
    "htn_ckd":    dict(age=58, sex_male=1, has_htn=1, has_dm=0, has_ckd=1, serum_sodium=140, serum_potassium=5.1,
                       creatinine=1.9, egfr=45, hba1c=5.2, fbs=85, sbp=162, dbp=96, bmi=26),
    "dm_ckd":     dict(age=64, sex_male=0, has_htn=0, has_dm=1, has_ckd=1, serum_sodium=136, serum_potassium=5.5,
                       creatinine=2.8, egfr=22, hba1c=8.2, fbs=168, sbp=134, dbp=84, bmi=29),
    "htn_dm_ckd": dict(age=67, sex_male=1, has_htn=1, has_dm=1, has_ckd=1, serum_sodium=143, serum_potassium=5.6,
                       creatinine=3.1, egfr=19, hba1c=9.4, fbs=210, sbp=168, dbp=98, bmi=31),
}


def ingredient_sets(app_module, n_sets: int, size: int, seed: int) -> list:
    """Deterministic ingredient baskets drawn across IFCT categories."""
    rng = random.Random(seed)
    df = app_module._ifct_df
    by_cat = {c: sorted(g["ingredient"].tolist()) for c, g in df.groupby("category")}
    cats = sorted(by_cat)
    sets = []
    for _ in range(n_sets):
        picks, chosen = [], rng.sample(cats, min(size, len(cats)))
        for c in chosen:
            picks.append(rng.choice(by_cat[c]))
        sets.append(picks)
    return sets


def supabase_row_count() -> str:
    """Confirms the migration was applied and the endpoint is logging."""
    import requests
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return "not configured (numbers below come from API responses)"
    try:
        r = requests.get(f"{url}/rest/v1/recipe_adherence?select=id",
                         headers={"apikey": key, "Authorization": f"Bearer {key}",
                                  "Prefer": "count=exact", "Range": "0-0"}, timeout=15)
        if r.status_code >= 400:
            return f"unreachable (HTTP {r.status_code}) — is 20260923120000_recipe_adherence.sql applied?"
        return f"{r.headers.get('content-range', '?').split('/')[-1]} rows"
    except Exception as exc:
        return f"error: {exc}"


def summarise(records: pd.DataFrame, tolerance: float) -> dict:
    matched = records[records.matched]
    over = matched[matched.stated_grams > matched.max_grams]          # any overage at all
    viol = matched[matched.violated]                                   # beyond the tolerance
    return {
        "n_recipes": int(records.recipe_index.nunique()),
        "n_ingredient_checks": int(len(records)),
        "n_matched": int(len(matched)),
        "match_rate": round(len(matched) / len(records), 4) if len(records) else None,
        "tolerance_fraction": tolerance,
        "violation_rate_with_tolerance": round(len(viol) / len(matched), 4) if len(matched) else None,
        "violation_rate_zero_tolerance": round(len(over) / len(matched), 4) if len(matched) else None,
        "mean_overage_g_when_violated": round(float(viol.overage_grams.mean()), 2) if len(viol) else 0.0,
        "median_overage_g_when_violated": round(float(viol.overage_grams.median()), 2) if len(viol) else 0.0,
        "max_overage_g": round(float(matched.overage_grams.max()), 2) if len(matched) else None,
        "mean_overage_pct_when_violated": (round(float((viol.overage_grams / viol.max_grams * 100).mean()), 1)
                                           if len(viol) else 0.0),
        "recipes_with_any_violation": int(viol.recipe_index.nunique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2, help="ingredient sets per patient profile")
    ap.add_argument("--set-size", type=int, default=5, help="ingredients per recipe")
    ap.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=sorted(PROFILES))
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between recipes (Groq rate limits)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    with contextlib.redirect_stdout(io.StringIO()):
        import app as deployed
    if deployed.groq_client is None:
        sys.exit("GROQ_API_KEY is not set — recipe generation cannot run.")

    client = deployed.app.test_client()
    baskets = ingredient_sets(deployed, args.repeats, args.set_size, args.seed)

    rows, failures, idx = [], [], 0
    total = len(args.profiles) * len(baskets)
    for profile in args.profiles:
        for basket in baskets:
            idx += 1
            print(f"[{idx}/{total}] {profile}: {', '.join(basket)}")
            resp = client.post("/api/generate-recipe",
                               json={"patient": PROFILES[profile], "ingredients": basket})
            if resp.status_code != 200:
                failures.append({"recipe_index": idx, "profile": profile,
                                 "error": f"HTTP {resp.status_code}", "detail": resp.get_data(as_text=True)[:200]})
                continue
            body = resp.get_json()
            limits = {r["ingredient"]: r for r in body.get("portions_used", [])}
            adherence = body.get("adherence") or []
            if not adherence:
                failures.append({"recipe_index": idx, "profile": profile, "error": "no adherence records"})
                continue
            for rec in adherence:
                rows.append({
                    "recipe_index": idx, "profile": profile,
                    "has_htn": PROFILES[profile]["has_htn"], "has_dm": PROFILES[profile]["has_dm"],
                    "has_ckd": PROFILES[profile]["has_ckd"],
                    "binding_constraint": limits.get(rec["ingredient"], {}).get("binding_constraint"),
                    **rec,
                })
            time.sleep(args.sleep)

    if not rows:
        sys.exit(f"No adherence records collected. Failures: {json.dumps(failures[:3], indent=2)}")

    records = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUT_DIR / "records.csv", index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": deployed.RECIPE_LLM_MODEL,
        "patient_profiles": "synthetic (no MIMIC-IV rows sent to the LLM)",
        "profiles_used": args.profiles, "ingredients_per_recipe": args.set_size,
        "n_failures": len(failures), "failures": failures[:10],
        "supabase_recipe_adherence": supabase_row_count(),
        "overall": summarise(records, deployed.RECIPE_ADHERENCE_TOLERANCE),
        "by_profile": {p: summarise(g, deployed.RECIPE_ADHERENCE_TOLERANCE)
                       for p, g in records.groupby("profile")},
        "by_binding_constraint": {str(c): summarise(g, deployed.RECIPE_ADHERENCE_TOLERANCE)
                                  for c, g in records.groupby("binding_constraint")},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    o = summary["overall"]
    print(f"\n{'='*64}\nRECIPE ADHERENCE — {o['n_recipes']} recipes, {o['n_ingredient_checks']} ingredient checks")
    print(f"{'='*64}")
    print(f"  extraction match rate        {o['match_rate']:.1%} ({o['n_matched']}/{o['n_ingredient_checks']})")
    print(f"  violation rate (tol {o['tolerance_fraction']:.0%})     {o['violation_rate_with_tolerance']:.1%}")
    print(f"  violation rate (zero tol)    {o['violation_rate_zero_tolerance']:.1%}")
    print(f"  mean overage when violated   {o['mean_overage_g_when_violated']} g "
          f"({o['mean_overage_pct_when_violated']}% of the limit)")
    print(f"  max overage seen             {o['max_overage_g']} g")
    print(f"  recipes with ≥1 violation    {o['recipes_with_any_violation']}/{o['n_recipes']}")
    print(f"  failed generations           {len(failures)}")
    print(f"  supabase rows                {summary['supabase_recipe_adherence']}")
    print(f"\n  ✓ Wrote {OUT_DIR.relative_to(ROOT)}/records.csv and summary.json")


if __name__ == "__main__":
    main()

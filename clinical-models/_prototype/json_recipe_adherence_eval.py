"""
Recipe-generation adherence experiment (Reviewer 2 items 9-10).

Question 1  How often does the language model, left alone, break the computed gram ceilings or the exclusion
            list?  -> the FIRST attempt of every case (what an unchecked system would have delivered).
Question 2  What reaches the user once the deterministic verifier, regeneration and template fallback are in
            place?  -> delivered source (llm / template / none), attempts, and an independent cross-check.
Question 3  How many violations were observed in what was delivered, with an upper bound?
            -> "no violations observed in the evaluated sample (one-sided 95% upper bound x%)", NOT a guarantee.

Cases are generated from a fixed seed (recipe_eval_prompts.json, committed) so the run is reproducible:
24 SYNTHETIC illustrative patient profiles (all 8 HTN/DM/CKD strata x 3; lab values drawn from plausible ranges,
not from MIMIC records) x 10 ingredient sets, a cuisine/equipment/time variant, and exclusions on ~30% of cases.

The prompt is the one the API uses (recipe_prompts.py). The LLM is called over HTTPS (Groq's OpenAI-compatible
endpoint); needs GROQ_API_KEY. Delivered LLM recipes can optionally be re-checked by a SECOND, independent LLM
extraction call (--crosscheck): a model-mediated check, reported separately, never used to accept a recipe.

    python recipe_adherence_eval.py --build-prompts                  # writes recipe_eval_prompts.json
    python recipe_adherence_eval.py --run --crosscheck               # real run (GROQ_API_KEY required)
    python recipe_adherence_eval.py --run --mock --limit 20          # pipeline test only: output is MOCK_*, not a result
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats as sp_stats

from mimic_extract import ckd_epi_2021
from recipe_prompts import RECIPE_SYSTEM_PROMPT, build_recipe_prompt, clinical_context_lines
from json_recipe_verifier import DEFAULT_RESTRICTED, RecipeVerifier, generate_verified_recipe, render_markdown

HERE = Path(__file__).resolve().parent.parent          # clinical-models/ (this file lives in clinical-models/_mine/)
ROOT = HERE.parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # GROQ_API_KEY; never printed or logged
OUT = ROOT / "artifacts" / "models" / "reports" / "evaluation"
PROMPTS = HERE / "recipe_eval_prompts.json"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("RECIPE_LLM_MODEL", "llama-3.3-70b-versatile")   # override with --model / RECIPE_LLM_MODEL

STRATA = [(h, d, c) for h in (0, 1) for d in (0, 1) for c in (0, 1)]
CUISINES = ["North Indian", "South Indian", "Bengali", "Gujarati", "Maharashtrian", "Any"]
EQUIPMENT = ["Stovetop only", "Pressure cooker", "Stovetop and oven", "Microwave and kettle"]
TIMES = ["20 minutes", "30 minutes", "45 minutes", "Any"]
EXCLUSION_CHOICES = ["dairy", "gluten", "peanut", "tree_nuts", "egg", "fish", "shellfish", "soy", "legumes", "vegetarian"]


# ===================================================================================================== cases
def make_patient(rng: np.random.Generator, htn: int, dm: int, ckd: int) -> dict:
    age, sex = int(rng.integers(35, 81)), int(rng.integers(0, 2))
    cr = float(rng.uniform(1.4, 4.5) if ckd else rng.uniform(0.6, 1.2))
    egfr = float(ckd_epi_2021(pd.Series([cr]), pd.Series([age]), pd.Series([sex])).iloc[0])
    return {"age": age, "sex_male": sex, "has_htn": htn, "has_dm": dm, "has_ckd": ckd,
            "serum_sodium": round(float(rng.uniform(134, 145)), 1),
            "serum_potassium": round(float(rng.uniform(4.3, 5.8) if ckd else rng.uniform(3.8, 4.8)), 2),
            "creatinine": round(cr, 2), "egfr": round(egfr, 1),
            "hba1c": round(float(rng.uniform(6.5, 10) if dm else rng.uniform(4.9, 5.9)), 1),
            "fbs": round(float(rng.uniform(110, 220) if dm else rng.uniform(75, 108)), 0),
            "sbp": round(float(rng.uniform(140, 185) if htn else rng.uniform(105, 135)), 0),
            "dbp": round(float(rng.uniform(85, 105) if htn else rng.uniform(65, 85)), 0),
            "bmi": round(float(rng.uniform(20, 34)), 1)}


def build_prompts(seed: int = 2026, patients_per_stratum: int = 3, sets_per_patient: int = 10) -> list:
    rng = np.random.default_rng(seed)
    py = random.Random(seed)
    names = pd.read_csv(HERE / "ifct_database.csv")
    by_cat = {c: g.ingredient.tolist() for c, g in names.groupby("category")}
    staples = by_cat["Cereals"] + by_cat["Pulses"] + by_cat["Roots and Tubers"]
    veg = by_cat["Other Vegetables"] + by_cat["Green Leafy Vegetables"]
    all_names = names.ingredient.tolist()
    cases = []
    for (h, d, c) in STRATA:
        for pi in range(patients_per_stratum):
            patient = make_patient(rng, h, d, c)
            for si in range(sets_per_patient):
                ings = [py.choice(staples), py.choice(veg)] + py.sample(all_names, py.randint(2, 6))
                ings = list(dict.fromkeys(ings))
                excl = py.sample(EXCLUSION_CHOICES, py.randint(1, 2)) if py.random() < 0.30 else []
                cases.append({"id": f"s{h}{d}{c}-p{pi}-i{si}", "patient": patient, "ingredients": ings,
                              "cuisine": py.choice(CUISINES), "equipment": py.choice(EQUIPMENT),
                              "time_limit": py.choice(TIMES), "exclusions": excl})
    PROMPTS.write_text(json.dumps({"seed": seed, "note": "synthetic illustrative patients; not MIMIC records",
                                   "cases": cases}, indent=1), encoding="utf-8")
    return cases


# ===================================================================================================== LLM
def groq_chat(messages, key: str, json_mode: bool, temperature: float, max_tokens: int = 900, retries: int = 6) -> str:
    body = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if MODEL.startswith("openai/gpt-oss"):
        # reasoning models spend completion tokens on hidden reasoning; keep it short and leave room for the answer
        body["reasoning_effort"] = "low"
        body["max_tokens"] = max(max_tokens, 3000)
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        try:
            r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=(10, 45))
        except requests.exceptions.RequestException:
            time.sleep(min(2 ** attempt, 30))      # connection stalled / reset: retry
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code in (429, 500, 502, 503):
            wait = float(r.headers.get("retry-after", 2 ** attempt))
            time.sleep(min(wait, 60))
            continue
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Groq: retries exhausted")


def mock_llm(prompt: str, rng: random.Random) -> str:
    """PIPELINE TEST ONLY. A crude, deliberately imperfect stand-in for the language model."""
    items = re.findall(r"^- (.+?): MAXIMUM ([\d.]+) g", prompt, flags=re.M)
    chosen = rng.sample(items, k=min(len(items), rng.randint(2, 5))) if items else []
    ings = []
    for name, mx in chosen:
        f = rng.uniform(0.3, 0.95) if rng.random() > 0.35 else rng.uniform(1.1, 2.0)
        ings.append({"name": name, "grams": round(float(mx) * f, 1)})
    steps = ["Wash and chop everything.", "Cook in water until tender.", "Serve warm."]
    if rng.random() < 0.10:
        steps.append("Add salt to taste.")
    return json.dumps({"title": "Mock dish", "description": "", "ingredients": ings or [{"name": "water", "grams": 100}],
                       "steps": steps})


def crosscheck(recipe_md: str, offered: list, key: str, mock: bool) -> dict:
    """Independent LLM extraction of grams used per offered ingredient (reported separately; never an accept gate)."""
    names = [r["ingredient"] for r in offered]
    if mock:
        return {"extracted": {n: 0.0 for n in names}, "error": None}
    prompt = ("Extract the grams used of each ingredient in the recipe. Reply only with JSON "
              '{"ingredients":[{"name":str,"grams":number}]} using exactly these names, 0 if unused: '
              f"{names}\n\nRecipe:\n{recipe_md}")
    try:
        txt = groq_chat([{"role": "system", "content": "You extract structured quantities and answer only in JSON."},
                         {"role": "user", "content": prompt}], key, True, 0.0, 500)
        parsed = json.loads(txt)
        return {"extracted": {str(i.get("name")): i.get("grams") for i in parsed.get("ingredients", [])}, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"extracted": {}, "error": f"{type(exc).__name__}: {exc}"}


# ===================================================================================================== stats
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def upper_95_zero(n: int) -> float:
    """One-sided 95% upper bound on a rate when 0 events are seen in n trials (exact; ~3/n)."""
    return round(1 - 0.05 ** (1 / n), 4) if n > 0 else float("nan")


def clopper_upper(k: int, n: int) -> float:
    return round(float(sp_stats.beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0, 4)


# ===================================================================================================== run
def run(args):
    key = os.getenv("GROQ_API_KEY", "")
    if not args.mock and not key:
        raise SystemExit("GROQ_API_KEY is not set. Set it for a real run, or use --mock for a pipeline test "
                         "(mock output is not a result).")
    cases = json.loads(PROMPTS.read_text(encoding="utf-8"))["cases"]
    # the file is ordered by stratum; shuffle with a fixed seed so any partial run (daily API token limits!)
    # covers all strata. Completed cases are skipped by id, so this is safe to change between resumes.
    if not args.in_order:
        random.Random(2026).shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]
    from train_model2 import PortionControlModel
    with contextlib.redirect_stdout(io.StringIO()):
        engine = PortionControlModel()
    verifier = RecipeVerifier(engine.ifct.df["ingredient"].tolist())

    tag = "MOCK_" if args.mock else ""
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"{tag}recipe_adherence_cases.jsonl"
    done = set()
    if log_path.exists() and not args.fresh:
        done = {json.loads(line)["id"] for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    elif log_path.exists():
        log_path.unlink()
    if args.mock:
        print("*** MOCK MODE: a crude stand-in model is used. Output is for testing the pipeline only. ***")

    with log_path.open("a", encoding="utf-8") as log:
        for n_done, case in enumerate(cases, 1):
            if case["id"] in done:
                continue
            patient, excl_in = case["patient"], case["exclusions"]
            res = engine.get_recommendations(patient, case["ingredients"], include_substitutes=False,
                                             use_ledger=False, exclusions=excl_in)
            rows = res["recommendations"]
            excl = engine.allergen_filter.resolve(excl_in)
            offered = [r for r in rows if r["label"] in ("Allowed", "Half portion") and r["max_grams"] > 0
                       and not excl.is_excluded(r["ingredient"])]
            excluded_names = sorted(r["ingredient"] for r in rows if r["label"] == "Excluded")
            ctx = clinical_context_lines(bool(patient["has_ckd"]), bool(patient["has_htn"]), bool(patient["has_dm"]),
                                         res.get("clinical_warnings", []))
            rng = random.Random(case["id"])

            def llm_call(feedback):
                user = build_recipe_prompt(offered, ctx, case["equipment"], case["cuisine"], case["time_limit"],
                                           excluded_names, feedback)
                if args.mock:
                    return mock_llm(user, rng)
                time.sleep(args.delay)
                return groq_chat([{"role": "system", "content": RECIPE_SYSTEM_PROMPT},
                                  {"role": "user", "content": user}], key, True, 0.6)

            restricted = DEFAULT_RESTRICTED if (patient["has_htn"] or patient["has_ckd"]) else []
            out = generate_verified_recipe(llm_call, verifier, rows, excl, restricted, max_attempts=args.max_attempts)
            first = out.history[0]
            rec = {"id": case["id"], "strata": f"htn{patient['has_htn']}dm{patient['has_dm']}ckd{patient['has_ckd']}",
                   "n_requested": len(case["ingredients"]), "n_offered": len(offered), "exclusions": excl_in,
                   "n_excluded_items": len(excluded_names), "source": out.source, "attempts": out.attempts,
                   "first_ok": bool(first["ok"]), "first_codes": sorted({v["code"] for v in first["violations"]}),
                   "first_n_quantities": first.get("n_quantities", 0), "first_n_over": first.get("n_over_ceiling", 0),
                   "first_has_excluded_term": any(v["code"] == "excluded_term" for v in first["violations"]),
                   "first_has_restricted": any(v["code"] == "restricted_item" for v in first["violations"]),
                   "history_codes": [sorted({v["code"] for v in h["violations"]}) for h in out.history],
                   "first_details": [v["detail"][:140] for v in first["violations"]][:4],
                   "llm_errors": [v["detail"][:160] for h in out.history for v in h["violations"] if v["code"] == "llm_error"][:2]}
            if out.source == "llm" and out.recipe is not None:
                rec["delivered_n_quantities"] = len(out.per_ingredient)
                if args.crosscheck:
                    cc = crosscheck(render_markdown(out.recipe), offered, key, args.mock)
                    cap = {r["ingredient"]: r["max_grams"] for r in offered}
                    over = 0
                    for name, mx in cap.items():
                        g = cc["extracted"].get(name)
                        if g is not None and float(g) > mx + 1e-6:
                            over += 1
                    rec.update({"crosscheck_error": cc["error"], "crosscheck_over": over,
                                "crosscheck_quantities": len(cap) if not cc["error"] else 0})
            elif out.source == "template":
                rec["delivered_n_quantities"] = len(out.per_ingredient)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(cases)} cases")
    summarize(args, tag)


def summarize(args, tag: str):
    log_path = OUT / f"{tag}recipe_adherence_cases.jsonl"
    recs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(recs)
    n = len(df)
    first_fail = int((~df.first_ok).sum())
    q_first, q_over = int(df.first_n_quantities.sum()), int(df.first_n_over.sum())
    delivered = df[df.source != "none"]
    llm = df[df.source == "llm"]
    code_counts = pd.Series([c for codes in df.first_codes for c in codes]).value_counts().to_dict()
    summary = {
        "is_mock": bool(args.mock), "n_cases": n, "n_cases_with_exclusions": int((df.exclusions.str.len() > 0).sum()),
        "model": MODEL, "max_attempts": args.max_attempts,
        "first_attempt": {
            "fail_rate": round(first_fail / n, 4), "fail_ci95": wilson(first_fail, n), "n_fail": first_fail,
            "listed_quantities": q_first, "quantities_over_ceiling": q_over,
            "quantity_over_rate": round(q_over / q_first, 4) if q_first else None,
            "quantity_over_ci95": wilson(q_over, q_first) if q_first else None,
            "cases_with_excluded_term": int(df.first_has_excluded_term.sum()),
            "cases_with_restricted_item": int(df.first_has_restricted.sum()),
            "violation_code_counts": code_counts},
        "delivered": {
            "llm": int((df.source == "llm").sum()), "template": int((df.source == "template").sum()),
            "none": int((df.source == "none").sum()),
            "llm_first_try": int(((df.source == "llm") & (df.attempts == 1)).sum()),
            "mean_attempts": round(float(df.attempts.mean()), 3),
            "template_rate": round(float((df.source == "template").mean()), 4),
            "template_rate_ci95": wilson(int((df.source == "template").sum()), n)},
        "caveat": ("Delivered recipes pass the deterministic verifier by construction, so 'violations in delivered "
                   "recipes' measured with the same rules is 0 by design. The informative numbers are the first-attempt "
                   "rates above and the independent cross-check below. Zero observed violations is not a guarantee."),
    }
    # Failures caused only by spice/seasoning wording the verifier's approved list did not contain (a vocabulary gap
    # in the checker, not an unsafe output). Classified from the logged violation details (first 4 per case).
    spice = re.compile(r"coriander|chil+i|pepper|powder|cumin|turmeric|masala|ground|paprika|cinnamon|cardamom|clove|spice|seed", re.I)

    def spice_only(r):
        det = r.get("first_details") or []
        return (not r["first_ok"]) and set(r["first_codes"]) == {"unknown_ingredient"} and bool(det)             and all(spice.search(x) for x in det)
    n_spice = int(sum(spice_only(r) for r in recs))
    summary["first_attempt"]["fail_spice_wording_only"] = n_spice
    summary["first_attempt"]["fail_rate_excluding_spice_wording_only"] = round((first_fail - n_spice) / n, 4)
    summary["first_attempt"]["fail_excl_spice_ci95"] = wilson(first_fail - n_spice, n)
    by = {}
    for name, fn in (("hypertensive", lambda r: "htn1" in r["strata"]), ("non_hypertensive", lambda r: "htn0" in r["strata"]),
                     ("with_exclusions", lambda r: bool(r["exclusions"])), ("without_exclusions", lambda r: not r["exclusions"])):
        g = [r for r in recs if fn(r)]
        by[name] = {"n": len(g), "first_fail_rate": round(sum(not r["first_ok"] for r in g) / len(g), 3),
                    "template_rate": round(sum(r["source"] == "template" for r in g) / len(g), 3),
                    "quantities_over": int(sum(r["first_n_over"] for r in g)),
                    "quantities_listed": int(sum(r["first_n_quantities"] for r in g))}
    summary["subgroups"] = by
    summary["cases_by_stratum"] = {k: int(v) for k, v in df.strata.value_counts().sort_index().items()}
    if "crosscheck_quantities" in df and df.crosscheck_quantities.notna().any():
        qn = int(df.crosscheck_quantities.fillna(0).sum())
        qo = int(df.crosscheck_over.fillna(0).sum())
        summary["independent_crosscheck_on_delivered_llm_recipes"] = {
            "recipes_checked": int((df.crosscheck_quantities.fillna(0) > 0).sum()), "quantities_checked": qn,
            "quantities_over_ceiling": qo, "extraction_errors": int(df.crosscheck_error.notna().sum()),
            "upper95_violation_rate": clopper_upper(qo, qn) if qn else None,
            "note": "model-mediated second opinion; extraction errors make this an imperfect audit"}
    nq = int(delivered.delivered_n_quantities.fillna(0).sum()) if "delivered_n_quantities" in delivered else 0
    summary["delivered"]["quantities_in_delivered_recipes"] = nq
    summary["delivered"]["rule_of_three_style_upper95_if_zero_observed"] = upper_95_zero(nq)
    (OUT / f"{tag}recipe_adherence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-prompts", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--mock", action="store_true", help="PIPELINE TEST ONLY; output files are prefixed MOCK_")
    ap.add_argument("--crosscheck", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--in-order", action="store_true", help="do not shuffle (file order is by stratum)")
    ap.add_argument("--model", help="Groq model id (default: RECIPE_LLM_MODEL env or llama-3.3-70b-versatile)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between Groq calls (rate limits)")
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite an existing case log instead of resuming")
    args = ap.parse_args()
    if args.model:
        globals()["MODEL"] = args.model
    if args.build_prompts:
        cases = build_prompts()
        print(f"wrote {len(cases)} cases to {PROMPTS}")
    if args.run:
        run(args)
    elif args.summarize_only:
        summarize(args, "MOCK_" if args.mock else "")


if __name__ == "__main__":
    main()

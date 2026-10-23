"""
Large-scale evaluation of the DEPLOYED deterministic recipe verifier (recipe_verifier.py at the repo root).

Answers the open question in the paper: how often does the language model's first draft pass the deterministic
checks, what finally reaches the user (verified / verified after regeneration / template), and how many gram-limit
or exclusion violations were observed -- over 200+ cases rather than the earlier 32-recipe, model-checked run.

How it works: it drives the unmodified /api/generate-recipe route of app.py (Flask test client) over the 240
pre-generated synthetic cases in recipe_eval_prompts.json, and records every call that route makes to
recipe_verifier.verify_recipe, so the FIRST attempt (what an unchecked system would have delivered) is analysed
separately from the delivered recipe. Nothing in app.py or recipe_verifier.py is changed.

  * Cases are synthetic illustrative profiles (not MIMIC-IV rows): 24 profiles x 10 ingredient sets, ~30% with
    exclusions. The case file uses ten exclusion groups; "legumes" has no counterpart in allergens.py and is
    replaced by "sesame" (recorded in the summary). Vegetarian maps to "no_meat_fish".
  * Order is a seeded shuffle, so a partial run (daily Groq token limits!) still spans every stratum. Completed
    cases are skipped on resume.
  * If Groq reports the daily token limit (or asks for a very long wait) the run stops cleanly, keeps its
    progress and tells you to resume later; it never sleeps for hours.
  * Delivered recipes pass the verifier by construction, so "no violations in what was delivered" is not
    independent evidence. --audit re-reads delivered LLM recipes with a separate extraction call (model-mediated,
    reported separately, never used to accept anything).
  * Cost: the deployed route does not lower the model's reasoning effort, so expect roughly 2-4k tokens per call.
    A free-tier key (200k tokens/day) covers on the order of 60-90 cases per day.

    python recipe_verifier_eval_at_scale.py --run --audit            # real run (needs GROQ_API_KEY in .env)
    python recipe_verifier_eval_at_scale.py --run --limit 20         # pilot
    python recipe_verifier_eval_at_scale.py --run --mock --limit 40  # pipeline test only; output is MOCK_*
    python recipe_verifier_eval_at_scale.py --summarize
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import re
import sys
import time
import types
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "artifacts" / "models" / "reports" / "recipe_verifier_eval"
CASES_PATH = HERE / "recipe_eval_prompts.json"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROUP_MAP = {"dairy": "dairy", "gluten": "gluten", "peanut": "peanut", "tree_nuts": "tree_nuts", "egg": "egg",
             "fish": "fish", "shellfish": "shellfish", "soy": "soy", "vegetarian": "no_meat_fish", "legumes": "sesame"}


class QuotaExhausted(RuntimeError):
    """Groq daily token limit reached (or a wait too long to sit through)."""


# ----------------------------------------------------------------------------------------- model clients
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _completion(text: str):
    return _Obj(choices=[_Obj(message=_Obj(content=text))])


class GroqShim:
    """Same call shape as the groq SDK's chat.completions.create, over plain HTTPS (the SDK may not be installed)."""

    def __init__(self, key: str, max_wait: float = 90.0):
        self.key, self.max_wait = key, max_wait
        self.chat = _Obj(completions=_Obj(create=self._create))
        self.tokens_used = 0

    def _create(self, **kw):
        body = {k: kw[k] for k in ("model", "messages", "max_tokens", "temperature") if k in kw}
        for attempt in range(6):
            try:
                r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=(10, 90))
            except requests.exceptions.RequestException:
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 200:
                j = r.json()
                self.tokens_used += int(j.get("usage", {}).get("total_tokens", 0))
                return _completion(j["choices"][0]["message"].get("content") or "")
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after", 5))
                if "per day" in r.text or "TPD" in r.text or wait > self.max_wait:
                    raise QuotaExhausted(f"QuotaExhausted: Groq daily limit / wait {wait:.0f}s")
                time.sleep(wait + 1)
                continue
            if r.status_code in (500, 502, 503):
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:160]}")
        raise RuntimeError("Groq: retries exhausted")


class MockClient:
    """PIPELINE TEST ONLY: a crude, deliberately imperfect stand-in for the language model."""

    def __init__(self):
        self.chat = _Obj(completions=_Obj(create=self._create))
        self.tokens_used = 0

    def _create(self, **kw):
        user = kw["messages"][-1]["content"]
        rng = random.Random(hash(user) % (2 ** 31))
        items = re.findall(r"^- (.+?): MAXIMUM ([\d.]+)g", user, flags=re.M)
        chosen = rng.sample(items, k=min(len(items), rng.randint(2, 4))) if items else []
        lines = []
        for name, mx in chosen:
            f = rng.uniform(0.3, 0.95) if rng.random() > 0.3 else rng.uniform(1.1, 1.8)
            lines.append(f"- {name}: {float(mx) * f:.1f} g")
        if rng.random() < 0.15:
            lines.append("- Salt: 2 g")
        return _completion("# Mock dish\n\nA test dish.\n\n## Ingredients\n" + "\n".join(lines) +
                           "\n- Water: 200 g\n\n## Instructions\n1. Cook everything.\n2. Serve.\n")


# ----------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def upper95_zero(n: int) -> float:
    return round(1 - 0.05 ** (1 / n), 4) if n > 0 else float("nan")


# ----------------------------------------------------------------------------------------- run
def load_app():
    for name, attrs in {"groq": {"Groq": lambda **k: None}, "flask_cors": {"CORS": lambda *a, **k: None}}.items():
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
    sys.path.insert(0, str(ROOT))
    with contextlib.redirect_stdout(io.StringIO()):
        import app as deployed
    return deployed


def audit_recipe(client, model: str, recipe_md: str, usable: list) -> dict:
    names = [u["ingredient"] for u in usable]
    prompt = ("Extract the grams used of each ingredient in the recipe. Reply only with JSON "
              '{"ingredients":[{"name":str,"grams":number}]} using exactly these names, 0 if unused: '
              f"{names}\n\nRecipe:\n{recipe_md}")
    try:
        txt = client.chat.completions.create(model=model, max_tokens=700, temperature=0.0, messages=[
            {"role": "system", "content": "You extract structured quantities and answer only in JSON."},
            {"role": "user", "content": prompt}]).choices[0].message.content
        data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip()))
        got = {str(i.get("name")): i.get("grams") for i in data.get("ingredients", [])}
        over = sum(1 for u in usable if got.get(u["ingredient"]) is not None and float(got[u["ingredient"]]) > u["max_grams"] + 1e-6)
        return {"error": None, "checked": len(usable), "over": over}
    except QuotaExhausted:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "checked": 0, "over": 0}


def run(args):
    deployed = load_app()
    tag = "MOCK_" if args.mock else ""
    if args.mock:
        client = MockClient()
        print("*** MOCK MODE: crude stand-in model; output is for testing the pipeline only. ***")
    else:
        key = getattr(deployed, "GROQ_API_KEY", "")
        if not key:
            sys.exit("GROQ_API_KEY is not set (put it in .env). Use --mock to test the pipeline without a key.")
        client = GroqShim(key)
    deployed.groq_client = client
    model = getattr(deployed, "RECIPE_LLM_MODEL", "unknown")

    # record every verifier call the route makes (attempt 1, 2, ..., then the template check)
    calls = []
    real_verify = deployed.recipe_verifier.verify_recipe

    def recording_verify(text, allowed, **kw):
        res = real_verify(text, allowed, **kw)
        recs = res["records"]
        calls.append({"ok": res["ok"], "issues": [i["type"] for i in res["issues"]],
                      "n_quantities": sum(1 for r in recs if r["matched"]),
                      "n_over": sum(1 for r in recs if r["violated"])})
        return res
    deployed.recipe_verifier.verify_recipe = recording_verify

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    random.Random(2026).shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{tag}cases.jsonl"
    if args.fresh and log.exists():
        log.unlink()
    done = {json.loads(l)["id"] for l in log.read_text(encoding="utf-8").splitlines() if l.strip()} if log.exists() else set()
    http = deployed.app.test_client()
    stopped = False

    with log.open("a", encoding="utf-8") as fh:
        for n, case in enumerate(cases, 1):
            if case["id"] in done:
                continue
            calls.clear()
            groups = [GROUP_MAP[g] for g in case["exclusions"]]
            resp = http.post("/api/generate-recipe", json={
                "patient": case["patient"], "ingredients": case["ingredients"], "allergies": groups,
                "exclusion_terms": [], "equipment": case["equipment"], "cuisine": case["cuisine"],
                "time_limit": case["time_limit"]})
            body = resp.get_json() or {}
            if resp.status_code == 500 and "QuotaExhausted" in str(body.get("error", "")):
                print(f"\nStopped at case {n}: {body['error']}\nProgress is saved; resume later with the same command.")
                stopped = True
                break
            p = case["patient"]
            rec = {"id": case["id"], "strata": f"htn{p['has_htn']}dm{p['has_dm']}ckd{p['has_ckd']}",
                   "exclusions": case["exclusions"], "mapped_groups": groups, "http": resp.status_code}
            if resp.status_code == 200:
                v = body["verification"]
                first = calls[0] if calls else {"ok": None, "issues": [], "n_quantities": 0, "n_over": 0}
                rec.update({"status": v["status"], "attempts": v["attempts"], "first_ok": first["ok"],
                            "first_issues": first["issues"], "first_n_quantities": first["n_quantities"],
                            "first_n_over": first["n_over"],
                            "delivered_n_quantities": sum(1 for r in body["adherence"] if r["matched"]),
                            "delivered_n_over": sum(1 for r in body["adherence"] if r["violated"])})
                if args.audit and v["status"] != "template_fallback":
                    usable = [{"ingredient": r["ingredient"], "max_grams": r["max_grams"]} for r in body["portions_used"]
                              if r["label"] in ("Allowed", "Half Portion") and r["max_grams"] > 0]
                    try:
                        rec["audit"] = audit_recipe(client, model, body["recipe"], usable)
                    except QuotaExhausted as exc:
                        print(f"\nStopped during audit of case {n}: {exc}")
                        stopped = True
                        break
            else:
                rec.update({"status": "no_recipe", "error": str(body.get("error", ""))[:160]})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(cases)} cases")
            if not args.mock:
                time.sleep(args.sleep)
    summarize(tag, model, args.mock, stopped)


def summarize(tag: str, model: str, is_mock: bool, stopped: bool = False):
    log = OUT / f"{tag}cases.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r["http"] == 200]
    n, n_ok = len(rows), len(ok)
    first_fail = sum(1 for r in ok if not r["first_ok"])
    q_first, q_over = sum(r["first_n_quantities"] for r in ok), sum(r["first_n_over"] for r in ok)
    codes = {}
    for r in ok:
        for c in set(r["first_issues"]):
            codes[c] = codes.get(c, 0) + 1
    st = {k: sum(1 for r in ok if r["status"] == k) for k in ("verified", "verified_after_regeneration", "template_fallback")}
    nq = sum(r["delivered_n_quantities"] for r in ok)

    def grp(fn):
        g = [r for r in ok if fn(r)]
        return {"n": len(g), "first_fail_rate": round(sum(1 for r in g if not r["first_ok"]) / len(g), 3) if g else None,
                "template_rate": round(sum(1 for r in g if r["status"] == "template_fallback") / len(g), 3) if g else None}
    summary = {
        "is_mock": is_mock, "stopped_early_on_quota": stopped, "model": model,
        "verifier": "deployed deterministic parser (recipe_verifier.py, repo root)",
        "n_cases_attempted": n, "n_recipes_returned": n_ok, "n_no_recipe": n - n_ok,
        "cases_by_stratum": {k: sum(1 for r in rows if r["strata"] == k) for k in sorted({r["strata"] for r in rows})},
        "n_cases_with_exclusions": sum(1 for r in rows if r["exclusions"]),
        "legumes_replaced_by_sesame_in": sum(1 for r in rows if "legumes" in r["exclusions"]),
        "first_attempt": {"n_fail": first_fail, "fail_rate": round(first_fail / n_ok, 4) if n_ok else None,
                          "fail_ci95": wilson(first_fail, n_ok), "listed_quantities": q_first,
                          "quantities_over_ceiling": q_over,
                          "quantity_over_rate": round(q_over / q_first, 4) if q_first else None,
                          "quantity_over_ci95": wilson(q_over, q_first) if q_first else None,
                          "issue_type_case_counts": dict(sorted(codes.items(), key=lambda kv: -kv[1]))},
        "delivered": {**st, "mean_attempts": round(sum(r["attempts"] for r in ok) / n_ok, 3) if n_ok else None,
                      "template_rate": round(st["template_fallback"] / n_ok, 4) if n_ok else None,
                      "template_rate_ci95": wilson(st["template_fallback"], n_ok),
                      "quantities_in_delivered_recipes": nq,
                      "violations_in_delivered_recipes": sum(r["delivered_n_over"] for r in ok),
                      "upper95_if_zero_observed": upper95_zero(nq)},
        "subgroups": {"hypertensive": grp(lambda r: "htn1" in r["strata"]), "non_hypertensive": grp(lambda r: "htn0" in r["strata"]),
                      "with_exclusions": grp(lambda r: bool(r["exclusions"])), "without_exclusions": grp(lambda r: not r["exclusions"])},
        "caveat": ("Delivered recipes pass the verifier by construction, so zero violations in them measured with the same "
                   "rules is not independent evidence. The verifier checks quantities in the ingredient list only, not amounts "
                   "implied in the cooking steps. Zero observed violations is not a guarantee."),
    }
    au = [r["audit"] for r in ok if r.get("audit") and not r["audit"]["error"]]
    if au:
        qn, qo = sum(a["checked"] for a in au), sum(a["over"] for a in au)
        summary["independent_audit"] = {"recipes_audited": len(au), "quantities_checked": qn, "quantities_over_ceiling": qo,
                                        "extraction_errors": sum(1 for r in ok if r.get("audit") and r["audit"]["error"]),
                                        "upper95": round(float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.95, qo + 1, qn - qo)), 4) if qn else None,
                                        "note": "model-mediated second opinion; never used to accept a recipe"}
    (OUT / f"{tag}summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--mock", action="store_true", help="PIPELINE TEST ONLY; output files are prefixed MOCK_")
    ap.add_argument("--audit", action="store_true", help="independent extraction re-check of delivered LLM recipes")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between cases (Groq rate limits)")
    ap.add_argument("--fresh", action="store_true", help="discard an existing case log instead of resuming")
    args = ap.parse_args()
    if args.run:
        run(args)
    elif args.summarize:
        d = load_app()
        summarize("MOCK_" if args.mock else "", getattr(d, "RECIPE_LLM_MODEL", "unknown"), args.mock)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

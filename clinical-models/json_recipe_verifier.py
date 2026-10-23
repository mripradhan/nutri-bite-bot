"""
Deterministic recipe verification, regeneration and template fallback (Reviewer 2, items 9-10).

The LLM is untrusted. Nothing it writes reaches the user unless this module's rule-based checks pass;
no model call takes part in the decision. Pipeline (generate_verified_recipe):

    LLM draft  ->  verify_recipe()  --pass-->  delivered (source = "llm")
                        | fail
                        v
              regenerate with the violation list (up to max_attempts)
                        | all attempts fail
                        v
              build_template_recipe()  ->  verify_recipe()  ->  delivered (source = "template")
                                                    | cannot be built / fails
                                                    v
                                              no recipe (never an "unverified" recipe)

verify_recipe() checks, on a STRUCTURED draft (JSON: title, ingredients[{name, grams}], steps[]):

  parse_error                  not valid JSON / wrong shape / empty ingredients or steps
  invalid_quantity             a stated quantity is not a positive number
  non_allowed_ingredient       a listed ingredient is an IFCT item that is not offered (Avoid / excluded /
                               not selected)
  unknown_ingredient           a listed ingredient matches nothing offered, neutral, or known
  over_ceiling                 total grams of an ingredient exceed its computed ceiling (strict <=)
  quantity_in_steps_over_ceiling / unattributed_quantity
                               a weight/volume quantity inside the steps exceeds the ceiling of the
                               ingredient it refers to, or cannot be attributed to any ingredient
  non_allowed_ingredient_mentioned
                               the title / description / steps name an IFCT ingredient that is not offered
  neutral_over_cap             water-like / spice / oil items above their per-recipe cap
  restricted_item              sodium-rich items (salt, soy sauce, ...) when the caller restricts them
  excluded_term                any allergy / intolerance / diet term (allergen_filter.ExclusionSet)
                               appears anywhere in the draft

Limits (state in the paper): matching is lexical (names, aliases, word boundaries), so a novel misspelling
or an ingredient the vocabulary does not know ("jaggery") is rejected as unknown_ingredient rather than
accepted; volume units are converted with deliberately heavy (conservative) densities; the check does not
judge nutrition, taste or cooking feasibility.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from allergen_filter import ExclusionSet, term_pattern

EPS = 1e-6

# --- items that need no ceiling (cap = max grams per whole recipe; None = unlimited) ---------------------
NEUTRAL_ITEMS: Dict[str, Optional[float]] = {
    "water": None, "ice": None, "hot water": None,
    "turmeric": 10, "haldi": 10, "cumin": 10, "jeera": 10, "cumin seeds": 10, "coriander powder": 10, "dhania powder": 10,
    "garam masala": 10, "red chilli powder": 10, "chilli powder": 10, "black pepper": 10, "pepper": 10, "cinnamon": 10,
    "dalchini": 10, "cardamom": 10, "elaichi": 10, "cloves": 10, "bay leaf": 10, "bay leaves": 10, "tej patta": 10,
    "asafoetida": 2, "hing": 2, "mustard seeds": 10, "rai": 10, "fennel seeds": 10, "saunf": 10, "ajwain": 10,
    "saffron": 1, "kasuri methi": 10, "lemon juice": 30, "lemon": 30, "lime": 30,
    "oil": 20, "cooking oil": 20, "vegetable oil": 20, "olive oil": 20, "mustard oil": 20, "sunflower oil": 20,
    "rice bran oil": 20,
}
# --- sodium-rich items; enforced only when the caller passes them (hypertension / CKD) -----------------
DEFAULT_RESTRICTED = ["salt", "table salt", "sea salt", "rock salt", "pink salt", "kala namak", "sendha namak",
                      "salt substitute", "lite salt", "soy sauce", "fish sauce", "worcestershire sauce", "ketchup",
                      "stock cube", "bouillon", "msg", "ajinomoto", "baking soda", "baking powder", "pickle",
                      "achar", "papad"]
NEGATIONS = {"no", "without", "avoid", "omit", "skip", "not", "never", "dont", "don", "free"}

# conservative (heavy) grams per unit: over-estimating makes the check stricter
UNIT_GRAMS = {"g": 1, "gm": 1, "gms": 1, "gram": 1, "grams": 1, "kg": 1000, "ml": 1, "l": 1000, "litre": 1000,
              "litres": 1000, "liter": 1000, "liters": 1000, "tbsp": 18, "tablespoon": 18, "tablespoons": 18,
              "tsp": 6, "teaspoon": 6, "teaspoons": 6, "cup": 240, "cups": 240, "oz": 28.35, "pinch": 1,
              "pinches": 1, "handful": 40, "handfuls": 40}
_QTY = re.compile(r"(?P<num>\d+(?:\.\d+)?(?:\s*/\s*\d+)?|[¼½¾])\s*(?:[-–]\s*(?P<hi>\d+(?:\.\d+)?)\s*)?"
                  r"(?P<unit>kg|gms?|grams?|g|ml|litres?|liters?|l|tbsp|tablespoons?|tsp|teaspoons?|cups?|oz|"
                  r"pinch(?:es)?|handfuls?)\b", re.I)
_VULGAR = {"¼": 0.25, "½": 0.5, "¾": 0.75}

_STOP_ALIASES = {"white", "whole", "raw", "green", "dry", "fresh", "ripe", "black", "red", "pale brown", "tender",
                 "pod", "leaves", "seed", "milled"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def _words(s: str) -> List[str]:
    return _norm(s).split()


def _qty_value(m: "re.Match") -> float:
    """Numeric part of a quantity match; for a range ('2-3 tbsp') the upper end, the stricter reading."""
    lo = _num(m.group("num"))
    return max(lo, float(m.group("hi"))) if m.group("hi") else lo


def _num(tok: str) -> float:
    tok = tok.strip()
    if tok in _VULGAR:
        return _VULGAR[tok]
    if "/" in tok:
        a, b = (float(x) for x in tok.split("/"))
        return a / b if b else math.inf
    return float(tok)


# =======================================================================================================
# vocabulary
# =======================================================================================================
def ingredient_keys(name: str) -> Set[str]:
    """Lexical keys by which a recipe may refer to an IFCT ingredient."""
    base = name.split("(")[0]
    parens = re.findall(r"\(([^)]*)\)", name)
    segs = [s.strip() for s in base.split(",") if s.strip()]
    keys = {_norm(base)}
    if segs:
        keys.add(_norm(segs[0]))
        if len(segs) == 2:
            keys.add(_norm(f"{segs[1]} {segs[0]}"))            # "Rice, brown" -> "brown rice"
    for p in parens:
        for alias in re.split(r"[/]| or ", p):
            a = _norm(alias)
            if not a:
                continue
            if a in _STOP_ALIASES or len(a) <= 3:
                if segs:
                    keys.add(_norm(f"{a} {segs[0]}"))          # "Rice, milled (white)" -> "white rice"
            else:
                keys.add(a)
    keys.discard("")
    return keys


@dataclass
class Vocabulary:
    key_to_ingredients: Dict[str, Set[str]]
    patterns: Dict[str, re.Pattern]
    ordered_keys: List[str]            # longest first

    @classmethod
    def build(cls, ifct_names: Iterable[str]) -> "Vocabulary":
        k2i: Dict[str, Set[str]] = {}
        for n in ifct_names:
            for k in ingredient_keys(n):
                k2i.setdefault(k, set()).add(n)
        return cls(k2i, {k: term_pattern(k) for k in k2i}, sorted(k2i, key=lambda k: (-len(k), k)))


# =======================================================================================================
# result types
# =======================================================================================================
@dataclass
class VerificationResult:
    ok: bool
    violations: List[Dict[str, str]] = field(default_factory=list)
    recipe: Optional[Dict[str, Any]] = None          # normalised draft (only when it parsed)
    per_ingredient: List[Dict[str, Any]] = field(default_factory=list)   # allowed items: ceiling vs stated

    def codes(self) -> List[str]:
        return sorted({v["code"] for v in self.violations})


def _v(code: str, detail: str) -> Dict[str, str]:
    return {"code": code, "detail": detail}


# =======================================================================================================
# parsing
# =======================================================================================================
def parse_draft(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Accept a dict or a JSON string (optionally fenced). Returns (draft, error)."""
    if isinstance(raw, dict):
        data = raw
    else:
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        try:
            data = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            return None, f"not valid JSON ({type(exc).__name__})"
    if not isinstance(data, dict):
        return None, "top-level JSON value is not an object"
    ing, steps = data.get("ingredients"), data.get("steps")
    if not isinstance(ing, list) or not ing:
        return None, "'ingredients' must be a non-empty list of {name, grams}"
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) for s in steps):
        return None, "'steps' must be a non-empty list of strings"
    for i in ing:
        if not isinstance(i, dict) or "name" not in i or "grams" not in i:
            return None, "every ingredient must be an object with 'name' and 'grams'"
    return {"title": str(data.get("title", "")), "description": str(data.get("description", "")),
            "ingredients": ing, "steps": steps}, None


# =======================================================================================================
# the verifier
# =======================================================================================================
# densities for NEUTRAL items (spices, oil, water): realistic, because these are not nutrient-ceilinged
NEUTRAL_UNIT_GRAMS = {**UNIT_GRAMS, "tbsp": 15, "tablespoon": 15, "tablespoons": 15, "tsp": 5, "teaspoon": 5,
                      "teaspoons": 5}


class RecipeVerifier:
    def __init__(self, ifct_names: Sequence[str], vocab: Optional[Vocabulary] = None,
                 restricted_terms: Sequence[str] = tuple(DEFAULT_RESTRICTED)):
        self.ifct_names = list(ifct_names)
        self.vocab = vocab or Vocabulary.build(self.ifct_names)
        # one lexicon of everything the verifier recognises: key -> (kind, payload)
        lex: Dict[str, Tuple[str, Any]] = {}
        for k, owners in self.vocab.key_to_ingredients.items():
            lex[k] = ("ifct", owners)
        for k in NEUTRAL_ITEMS:                       # neutral wins over an IFCT alias of the same words
            lex[_norm(k)] = ("neutral", k)
        for k in restricted_terms:
            lex[_norm(k)] = ("restricted", k)
        self._lex = lex
        self._pat = {k: term_pattern(k) for k in lex}
        self._order = sorted(lex, key=lambda k: (-len(k), k))

    # ---- longest-match-wins lexical spans ---------------------------------------------------------
    def spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Non-overlapping (start, end, key) matches in the normalised text; longer keys claim text first,
        so 'spring onion' is not also read as 'onion' and 'bell pepper' is not read as the spice 'pepper'."""
        t = _norm(text)
        taken: List[Tuple[int, int]] = []
        out: List[Tuple[int, int, str]] = []
        for key in self._order:
            for m in self._pat[key].finditer(t):
                a, b = m.start(), m.end()
                if any(a < e and b > s0 for s0, e in taken):
                    continue
                taken.append((a, b))
                out.append((a, b, key))
        return sorted(out)

    def _classify(self, text: str, offered: Dict[str, float]) -> List[Tuple[str, str]]:
        """[(kind, name)] per recognised item, kind in offered | neutral | restricted | other."""
        res = []
        for _, _, key in self.spans(text):
            kind, payload = self._lex[key]
            if kind == "ifct":
                cands = [o for o in payload if o in offered]
                res.append(("offered", min(cands, key=lambda i: offered[i])) if cands else ("other", sorted(payload)[0]))
            else:
                res.append((kind, payload))
        return res

    # ---- main entry -----------------------------------------------------------------------------
    def verify(self, raw: Any, allowed: Dict[str, float], exclusions: Optional[ExclusionSet] = None,
               restricted_terms: Optional[Sequence[str]] = None) -> VerificationResult:
        draft, err = parse_draft(raw)
        if draft is None:
            return VerificationResult(False, [_v("parse_error", err or "unparseable")])
        vio: List[Dict[str, str]] = []
        offered = {k: float(v) for k, v in allowed.items()}
        restricted = list(restricted_terms or [])
        totals: Dict[str, float] = {k: 0.0 for k in offered}
        neutral_totals: Dict[str, float] = {}

        # 1. ingredient list ------------------------------------------------------------------------
        for item in draft["ingredients"]:
            name = str(item["name"])
            try:
                grams = float(re.sub(r"(?i)\s*g(rams?)?\s*$", "", str(item["grams"])).strip())
            except ValueError:
                vio.append(_v("invalid_quantity", f"{name}: grams={item['grams']!r}"))
                continue
            if not math.isfinite(grams) or grams <= 0:
                vio.append(_v("invalid_quantity", f"{name}: grams={item['grams']!r}"))
                continue
            kinds = self._classify(name, offered)
            main = kinds[0] if kinds else None
            n_norm = _norm(name)
            exact = next((a for a in offered if _norm(a) == n_norm), None)
            if exact is not None:
                totals[exact] += grams
            elif main and main[0] == "offered":
                totals[main[1]] += grams
            elif main and main[0] == "neutral":
                neutral_totals[main[1]] = neutral_totals.get(main[1], 0.0) + grams
            elif main and main[0] == "restricted":
                pass                                             # reported by the restricted-item scan below
            elif main and main[0] == "other":
                vio.append(_v("non_allowed_ingredient", f"{name} (= {main[1]}) is not offered for this patient"))
            else:
                vio.append(_v("unknown_ingredient", f"{name}: matches no offered, neutral or known item"))

        per_ing = []
        for name, ceiling in offered.items():
            stated = totals[name]
            if stated > ceiling + EPS:
                vio.append(_v("over_ceiling", f"{name}: {stated:g} g listed > ceiling {ceiling:g} g"))
            if stated > 0:
                per_ing.append({"ingredient": name, "max_grams": ceiling, "stated_grams": round(stated, 2),
                                "overage_grams": round(stated - ceiling, 2), "violated": stated > ceiling + EPS})
        for k, g in neutral_totals.items():
            cap = NEUTRAL_ITEMS[k]
            if cap is not None and g > cap + EPS:
                vio.append(_v("neutral_over_cap", f"{k}: {g:g} g > cap {cap:g} g"))

        # 2. free text: title, description, ingredient names, steps -----------------------------------
        prose = " . ".join([draft["title"], draft["description"]] + draft["steps"])
        fulltext = " . ".join([prose] + [str(i["name"]) for i in draft["ingredients"]])

        if exclusions:
            for term, grp in exclusions.hits_in_text(fulltext):
                vio.append(_v("excluded_term", f"'{term}' ({grp}) appears in the recipe"))

        norm_full = _norm(fulltext)
        for t in restricted:
            for m in term_pattern(t).finditer(norm_full):
                if not any(w in NEGATIONS for w in norm_full[:m.start()].split()[-3:]):
                    vio.append(_v("restricted_item", f"'{t}' is restricted for this patient"))
                    break

        for kind, nm in self._classify(prose, offered):
            if kind == "other":
                vio.append(_v("non_allowed_ingredient_mentioned", f"{nm} is mentioned but not offered"))

        # 3. quantities written inside the steps -----------------------------------------------------
        for step in draft["steps"]:
            for m in _QTY.finditer(step):
                unit = m.group("unit").lower()
                window = " ".join(re.split(r"[.;:!?\n]", step[m.end():])[0].split()[:8])
                first = self._classify(window, offered)
                first = first[0] if first else None
                if first is None:
                    vio.append(_v("unattributed_quantity", f"'{m.group(0)}' in step: {step.strip()[:80]!r}"))
                elif first[0] == "offered":
                    grams = _qty_value(m) * UNIT_GRAMS[unit]
                    if grams > offered[first[1]] + EPS:
                        vio.append(_v("quantity_in_steps_over_ceiling",
                                      f"{first[1]}: '{m.group(0)}' ~ {grams:g} g > ceiling {offered[first[1]]:g} g"))
                elif first[0] == "neutral":
                    grams = _qty_value(m) * NEUTRAL_UNIT_GRAMS[unit]
                    cap = NEUTRAL_ITEMS[first[1]]
                    if cap is not None and grams > cap + EPS:
                        vio.append(_v("neutral_over_cap", f"{first[1]}: '{m.group(0)}' ~ {grams:g} g > cap {cap:g} g"))
                # 'other' / 'restricted' are reported by the scans above
        seen, uniq = set(), []
        for x in vio:
            key = (x["code"], x["detail"])
            if key not in seen:
                seen.add(key)
                uniq.append(x)
        return VerificationResult(not uniq, uniq, draft, per_ing)


# =======================================================================================================
# template fallback
# =======================================================================================================
def build_template_recipe(allowed_rows: Sequence[Dict[str, Any]], max_items: int = 6,
                          fraction: float = 0.5, min_grams: float = 10.0) -> Optional[Dict[str, Any]]:
    """Deterministic one-pot dish from the offered ingredients, each at `fraction` of its ceiling
    (rounded DOWN to a whole gram, never above the ceiling). None if nothing usable."""
    usable = [r for r in allowed_rows
              if r.get("label") in ("Allowed", "Half Portion", "Half portion") and float(r["max_grams"]) >= min_grams]
    usable.sort(key=lambda r: -float(r["max_grams"]))
    usable = usable[:max_items]
    if not usable:
        return None
    ings = [{"name": r["ingredient"], "grams": float(math.floor(float(r["max_grams"]) * fraction))} for r in usable]
    ings = [i for i in ings if i["grams"] >= 1]
    if not ings:
        return None
    names = ", ".join(i["name"] for i in ings)
    return {
        "title": "Simple one-pot dish",
        "description": "A plain, quickly prepared dish using only the listed ingredients at conservative portions.",
        "ingredients": ings + [{"name": "water", "grams": 250}],
        "steps": [
            f"Wash, peel and chop the ingredients: {names}.",
            "Weigh each ingredient exactly as listed.",
            "Heat a pot, add the water and bring it to a gentle boil.",
            "Add the chopped ingredients, cover and simmer until everything is tender.",
            "Season only with turmeric and cumin; add no salt.",
            "Serve warm.",
        ],
    }


def render_markdown(recipe: Dict[str, Any]) -> str:
    lines = [f"# {recipe.get('title') or 'Recipe'}", ""]
    if recipe.get("description"):
        lines += [recipe["description"], ""]
    lines += ["## Ingredients"] + [f"- {i['name']}: {i['grams']:g} g" for i in recipe["ingredients"]]
    lines += ["", "## Steps"] + [f"{n}. {s}" for n, s in enumerate(recipe["steps"], 1)]
    return "\n".join(lines)


# =======================================================================================================
# generate -> verify -> regenerate -> template
# =======================================================================================================
@dataclass
class GeneratedRecipe:
    recipe: Optional[Dict[str, Any]]
    source: str                      # "llm" | "template" | "none"
    attempts: int
    history: List[Dict[str, Any]]
    per_ingredient: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def markdown(self) -> Optional[str]:
        return render_markdown(self.recipe) if self.recipe else None


def generate_verified_recipe(llm_call: Callable[[Optional[str]], str], verifier: RecipeVerifier,
                             allowed_rows: Sequence[Dict[str, Any]], exclusions: Optional[ExclusionSet] = None,
                             restricted_terms: Optional[Sequence[str]] = None, max_attempts: int = 3) -> GeneratedRecipe:
    """`llm_call(feedback)` returns the raw draft text; `feedback` is None on the first attempt, else the
    previous attempt's violation list as text. `allowed_rows` are portion-engine rows with
    ingredient / max_grams / label; only Allowed and Half-portion rows with max_grams > 0 are offered."""
    offered = {r["ingredient"]: float(r["max_grams"]) for r in allowed_rows
               if r.get("label") in ("Allowed", "Half Portion", "Half portion") and float(r["max_grams"]) > 0
               and not (exclusions and exclusions.is_excluded(r["ingredient"]))}
    history: List[Dict[str, Any]] = []
    feedback: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm_call(feedback)
            res = verifier.verify(raw, offered, exclusions, restricted_terms)
        except Exception as exc:  # noqa: BLE001  (network error, rate limit, ...)
            res = VerificationResult(False, [_v("llm_error", f"{type(exc).__name__}: {exc}")])
        history.append({"attempt": attempt, "ok": res.ok, "violations": res.violations,
                        "n_quantities": len(res.per_ingredient),
                        "n_over_ceiling": sum(1 for q in res.per_ingredient if q["violated"])})
        if res.ok:
            return GeneratedRecipe(res.recipe, "llm", attempt, history, res.per_ingredient)
        feedback = "; ".join(f"{v['code']}: {v['detail']}" for v in res.violations[:8])

    tmpl = build_template_recipe([r for r in allowed_rows if r["ingredient"] in offered])
    if tmpl is not None:
        res = verifier.verify(tmpl, offered, exclusions, restricted_terms)
        history.append({"attempt": "template", "ok": res.ok, "violations": res.violations})
        if res.ok:
            return GeneratedRecipe(res.recipe, "template", max_attempts, history, res.per_ingredient)
    return GeneratedRecipe(None, "none", max_attempts, history)

"""
Deterministic post-generation verification of LLM-written recipes.

No language model is involved here. The recipe prompt requires a fixed ingredient-line
format, and this module parses that format with regular expressions and checks it against
(i) the per-ingredient gram ceilings, (ii) the allergy / preference exclusions, and
(iii) a closed list of ingredients. Any line that cannot be parsed, any quantity not
stated in grams, any unlisted ingredient, any quantity over a ceiling and any excluded
term anywhere in the text is a failure. The caller then regenerates, and finally falls
back to a template recipe built directly from the ceilings, so an unverifiable recipe is
never returned as if it were compliant.

Required ingredient-line format (one per line, under an "Ingredients" heading):

    - <exact ingredient name from the allowed list>: <number> g

Limits of this check (stated plainly in the manuscript): it verifies quantities stated in
the ingredient list, not amounts implied only in the cooking steps; name matching is
exact (after normalisation) or by an unambiguous short alias; and the allergen scan is
keyword-based (see allergens.py).
"""
import re
from typing import Dict, List, Optional

import allergens

# Ingredients the prompt allows in addition to the allowed list. They carry no gram
# ceiling. Salt is additionally rejected for patients with HTN or CKD.
EXTRAS = {"water", "pepper", "black pepper", "oil", "cooking oil", "salt"}
SALT_NAMES = {"salt"}

_GRAM_UNITS = {"g": 1.0, "gm": 1.0, "gram": 1.0, "grams": 1.0, "kg": 1000.0}
_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(?P<name>.+?)\s*:\s*(?P<rest>.*?)\s*$")
_QTY_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)?\b")
_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*|\*\*)?(?P<h>[A-Za-z][A-Za-z ]*?)\**:?\s*$")
_ING_HEADINGS = {"ingredients", "ingredient list"}
_END_HEADINGS = {"instructions", "method", "directions", "steps", "preparation",
                 "cooking instructions", "procedure", "notes", "tips"}


def _norm(name: str) -> str:
    name = re.sub(r"[*_`]", "", str(name))
    return " ".join(name.lower().split())


def _short_aliases(allowed_names: List[str]) -> Dict[str, str]:
    """Unambiguous short names, e.g. 'paneer' -> 'Paneer (Cottage cheese)'."""
    counts: Dict[str, List[str]] = {}
    for n in allowed_names:
        short = _norm(re.split(r"[(,]", n)[0])
        counts.setdefault(short, []).append(n)
    return {s: ns[0] for s, ns in counts.items() if len(ns) == 1}


def _ingredient_block(text: str) -> Optional[List[str]]:
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        m = _HEADING_RE.match(ln)
        if m and _norm(m.group("h")) in _ING_HEADINGS:
            start = i + 1
            break
    if start is None:
        return None
    block = []
    for ln in lines[start:]:
        m = _HEADING_RE.match(ln)
        if m and _norm(m.group("h")) in _END_HEADINGS:
            break
        if ln.strip().startswith("#"):
            break
        block.append(ln)
    return block


def verify_recipe(text: str, allowed: List[dict], *, groups=(), terms=(),
                  restrict_salt: bool = False) -> dict:
    """
    allowed: [{"ingredient": str, "max_grams": float}, ...]  (max_grams 0 = must not be used)
    Returns {"ok": bool, "issues": [{type, ...}], "records": [adherence record per allowed
    ingredient, schema-compatible with the recipe_adherence table]}.
    """
    issues: List[dict] = []
    allowed_by_norm = {_norm(a["ingredient"]): a for a in allowed}
    aliases = _short_aliases([a["ingredient"] for a in allowed])
    stated: Dict[str, float] = {}

    block = _ingredient_block(text)
    if block is None:
        issues.append({"type": "no_ingredient_section"})
        block = []

    parsed_lines = 0
    for ln in block:
        if not ln.strip():
            continue
        m = _LINE_RE.match(ln)
        if not m:
            issues.append({"type": "unparsed_line", "line": ln.strip()[:120]})
            continue
        parsed_lines += 1
        name = _norm(m.group("name"))
        rest = m.group("rest").lstrip("*_` ")

        a = allowed_by_norm.get(name)
        if a is None and name in aliases:
            a = allowed_by_norm[_norm(aliases[name])]

        if a is None:
            if name in EXTRAS:
                if name in SALT_NAMES and restrict_salt:
                    issues.append({"type": "salt_not_allowed", "line": ln.strip()[:120]})
                continue
            issues.append({"type": "unknown_ingredient", "name": m.group("name").strip()})
            continue

        q = _QTY_RE.match(rest)
        if not q:
            issues.append({"type": "no_quantity", "ingredient": a["ingredient"]})
            continue
        unit = (q.group("unit") or "").lower()
        if unit not in _GRAM_UNITS:
            issues.append({"type": "unit_not_grams", "ingredient": a["ingredient"],
                           "unit": unit or "(none)"})
            continue
        grams = float(q.group("num")) * _GRAM_UNITS[unit]
        stated[a["ingredient"]] = stated.get(a["ingredient"], 0.0) + grams

    if block and parsed_lines == 0:
        issues.append({"type": "no_parseable_ingredient_lines"})

    records = []
    for a in allowed:
        name, cap = a["ingredient"], float(a["max_grams"])
        got = stated.get(name)
        matched = got is not None
        violated = bool(matched and round(got, 2) > round(cap, 2))
        if violated:
            issues.append({"type": "over_ceiling", "ingredient": name,
                           "stated_grams": got, "max_grams": cap})
        records.append({
            "ingredient": name,
            "max_grams": cap,
            "stated_grams": got,
            "matched": matched,
            "violated": violated,
            "overage_grams": round(got - cap, 2) if matched else None,
        })

    for hit in allergens.find_in_text(text, groups, terms):
        issues.append({"type": "excluded_term", **hit})

    return {"ok": not issues, "issues": issues, "records": records}


def describe_issues(issues: List[dict], limit: int = 8) -> str:
    """Plain-English list of verification failures, fed back to the model on a retry."""
    out = []
    for it in issues[:limit]:
        t = it["type"]
        if t == "over_ceiling":
            out.append(f'{it["ingredient"]}: you used {it["stated_grams"]:g} g, above the '
                       f'{it["max_grams"]:g} g maximum')
        elif t == "unknown_ingredient":
            out.append(f'"{it["name"]}" is not an allowed ingredient')
        elif t == "unit_not_grams":
            out.append(f'{it["ingredient"]}: quantity must be written in grams (g), not {it["unit"]}')
        elif t == "no_quantity":
            out.append(f'{it["ingredient"]}: no quantity given')
        elif t == "unparsed_line":
            out.append(f'ingredient line not in "- <name>: <number> g" format: {it["line"]}')
        elif t == "excluded_term":
            out.append(f'the recipe mentions "{it["term"]}", which the patient must avoid')
        elif t == "salt_not_allowed":
            out.append("do not add salt")
        elif t == "no_ingredient_section":
            out.append('there is no "Ingredients" section')
        elif t == "no_parseable_ingredient_lines":
            out.append("no ingredient lines could be read")
    if len(issues) > limit:
        out.append(f"... and {len(issues) - limit} more")
    return "; ".join(out)


def build_template_recipe(allowed: List[dict], *, fraction: float = 0.5, cap_g: float = 150.0,
                          min_g: float = 5.0) -> Optional[str]:
    """
    Fallback recipe assembled WITHOUT a language model, directly from the ceilings: each
    usable ingredient at `fraction` of its ceiling (rounded down, capped). Returns None if
    no ingredient has a usable ceiling.
    """
    lines = []
    for a in allowed:
        cap = float(a["max_grams"])
        if cap <= min_g:
            continue
        grams = min(cap_g, int(cap * fraction * 10) / 10)
        if grams >= min_g:
            lines.append(f'- {a["ingredient"]}: {grams:g} g')
    if not lines:
        return None
    return (
        "# Simple one-pot dish (template)\n\n"
        "Basic template assembled directly from your computed gram ceilings, without the "
        "language model, because no generated recipe passed verification.\n\n"
        "## Ingredients\n" + "\n".join(lines) + "\n- Water: as needed\n\n"
        "## Instructions\n"
        "1. Wash and chop all ingredients.\n"
        "2. Add the ingredients to a pot with water and bring to a gentle boil.\n"
        "3. Simmer until everything is tender, adding water if needed.\n"
        "4. Serve warm as a single meal.\n"
    )

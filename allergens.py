"""
Deterministic allergy / intolerance / preference exclusions.

Applied as a hard filter BEFORE portion computation, substitution and recipe generation,
and re-checked against the generated recipe text by recipe_verifier.py. Matching is
group-level and keyword-based: it is conservative (it can over-exclude, e.g. coconut milk
under "dairy") but it is NOT an exhaustive allergen database. Patients can add free-text
terms for anything the groups do not cover, and outputs still need review by a clinician
or dietitian who knows the patient.
"""
import re
from typing import Dict, Iterable, List, Optional

# group id -> label, IFCT categories excluded wholesale, and keywords (matched on word
# boundaries, case-insensitive, against ingredient names and generated recipe text)
ALLERGEN_GROUPS: Dict[str, dict] = {
    "dairy": {
        "label": "Dairy / milk",
        "categories": ["Milk and Dairy"],
        "keywords": ["milk", "curd", "dahi", "yogurt", "yoghurt", "paneer", "ghee", "khoya",
                     "mawa", "butter", "buttermilk", "cheese", "cream", "lassi", "whey",
                     "casein", "kefir", "raita"],
    },
    "egg": {
        "label": "Egg",
        "categories": [],
        "keywords": ["egg", "eggs", "omelette", "omelet", "mayonnaise"],
    },
    "fish": {
        "label": "Fish",
        "categories": [],
        "keywords": ["fish", "sardine", "mathi", "pomfret", "rohu", "salmon", "tuna",
                     "anchovy", "anchovies", "mackerel"],
    },
    "shellfish": {
        "label": "Shellfish / prawn",
        "categories": [],
        "keywords": ["prawn", "prawns", "jhinga", "shrimp", "crab", "lobster", "shellfish",
                     "oyster", "mussel", "clam", "squid"],
    },
    "peanut": {
        "label": "Peanut",
        "categories": [],
        "keywords": ["peanut", "peanuts", "groundnut", "groundnuts", "moongphali"],
    },
    "tree_nuts": {
        "label": "Tree nuts",
        "categories": [],
        "keywords": ["almond", "almonds", "badam", "cashew", "cashewnut", "kaju", "walnut",
                     "walnuts", "akhrot", "pistachio", "pistachios", "hazelnut", "pecan",
                     "macadamia"],
    },
    "sesame": {
        "label": "Sesame",
        "categories": [],
        "keywords": ["sesame", "til", "tahini", "gingelly"],
    },
    "gluten": {
        "label": "Gluten (wheat, barley, rye)",
        "categories": [],
        "keywords": ["wheat", "atta", "maida", "semolina", "suji", "sooji", "rava", "barley",
                     "rye", "gluten", "seitan", "bread", "chapati", "roti", "naan", "paratha",
                     "pasta", "noodles", "couscous", "bulgur"],
    },
    "soy": {
        "label": "Soy",
        "categories": [],
        "keywords": ["soy", "soya", "soybean", "soybeans", "tofu", "tempeh", "edamame"],
    },
    "no_meat_fish": {
        "label": "Vegetarian (no meat, poultry or fish)",
        "categories": ["Meat and Poultry", "Fish and Seafood"],
        "keywords": ["chicken", "mutton", "goat", "lamb", "beef", "pork", "bacon", "meat",
                     "fish", "prawn", "shrimp", "gelatin", "gelatine"],
    },
}


def group_options() -> List[dict]:
    """[{id, label}] for the UI selector."""
    return [{"id": gid, "label": g["label"]} for gid, g in ALLERGEN_GROUPS.items()]


def _word_re(term: str) -> "re.Pattern":
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower().strip()) + r"(?![a-z0-9])")


def normalize_groups(groups: Optional[Iterable[str]]) -> List[str]:
    """Keep only known group ids (unknown ids are ignored, never silently widened)."""
    return [g for g in (groups or []) if g in ALLERGEN_GROUPS]


def normalize_terms(terms: Optional[Iterable[str]]) -> List[str]:
    """Clean user-supplied free-text exclusion terms (lower-cased, de-duplicated, <=40 chars)."""
    out = []
    for t in terms or []:
        t = str(t).lower().strip()
        if 2 <= len(t) <= 40 and t not in out:
            out.append(t)
    return out


def exclusion_reason(name: str, category: str, groups: Iterable[str],
                     terms: Iterable[str] = ()) -> Optional[str]:
    """Why this ingredient is excluded for the patient, or None if it is not."""
    low = name.lower()
    for gid in groups:
        g = ALLERGEN_GROUPS[gid]
        if category in g["categories"]:
            return g["label"]
        if any(_word_re(k).search(low) for k in g["keywords"]):
            return g["label"]
    for t in terms:
        if _word_re(t).search(low):
            return f'excluded term "{t}"'
    return None


def find_in_text(text: str, groups: Iterable[str], terms: Iterable[str] = ()) -> List[dict]:
    """All excluded keywords/terms found in a block of text: [{group, term}]."""
    low = text.lower()
    hits = []
    for gid in groups:
        for k in ALLERGEN_GROUPS[gid]["keywords"]:
            if _word_re(k).search(low):
                hits.append({"group": gid, "term": k})
    for t in terms:
        if _word_re(t).search(low):
            hits.append({"group": "custom", "term": t})
    return hits

"""
Deterministic allergy / intolerance / diet exclusions (Reviewer 2, item 11).

Exclusions are HARD constraints applied before anything else in the pipeline:

  1. portion computation   an excluded ingredient gets max_grams = 0 and label "Excluded"
  2. substitution          an excluded ingredient is never proposed as a substitute
  3. recipe generation     excluded ingredients are never offered to the LLM, and every field of the
                           generated recipe is scanned for excluded terms (recipe_verifier.py)

No model and no LLM is involved: resolution is a table lookup (allergen_map.json) plus
exact / substring / difflib matching against the IFCT ingredient names.

Limits (state these in the paper): the map is curated, not exhaustive; it covers the 101 IFCT
ingredients and a list of common derivative terms; it does not model cross-reactivity or trace
contamination; free-text terms that match nothing are still scanned as literal words but cannot
be mapped to an IFCT ingredient (reported in `unresolved`).
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

MAP_PATH = Path(__file__).resolve().parent / "allergen_map.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def term_pattern(term: str) -> re.Pattern:
    """Whole-word(s) match, tolerating a plural 's' / 'es' on the last word."""
    words = [re.escape(w) for w in _norm(term).split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"(?:s|es)?\b")


@dataclass
class ExclusionSet:
    groups: Set[str] = field(default_factory=set)
    excluded_ingredients: Set[str] = field(default_factory=set)      # exact IFCT names
    terms: Dict[str, str] = field(default_factory=dict)              # term -> group (or "free-text")
    unresolved: List[str] = field(default_factory=list)              # free text that matched no IFCT item
    _patterns: Dict[str, re.Pattern] = field(default_factory=dict, repr=False)

    def __bool__(self) -> bool:
        return bool(self.excluded_ingredients or self.terms)

    def _pattern(self, term: str) -> re.Pattern:
        if term not in self._patterns:
            self._patterns[term] = term_pattern(term)
        return self._patterns[term]

    def hits_in_text(self, text: str) -> List[tuple]:
        """[(term, group)] for every excluded term appearing as a whole word in `text`."""
        t = _norm(text)
        return [(term, grp) for term, grp in self.terms.items() if self._pattern(term).search(t)]

    def is_excluded(self, ingredient: str) -> bool:
        return ingredient in self.excluded_ingredients or bool(self.hits_in_text(ingredient))

    def reason(self, ingredient: str) -> str:
        groups = sorted({g for g, _ in ((grp, 1) for _, grp in self.hits_in_text(ingredient))})
        if ingredient in self.excluded_ingredients and not groups:
            groups = sorted(self.groups) or ["exclusion list"]
        return ", ".join(groups) if groups else "exclusion list"

    def to_dict(self) -> dict:
        return {"groups": sorted(self.groups), "excluded_ingredients": sorted(self.excluded_ingredients),
                "unresolved": list(self.unresolved)}


class AllergenFilter:
    def __init__(self, ifct_names: Iterable[str], map_path: Path = MAP_PATH):
        spec = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.groups: Dict[str, dict] = spec["groups"]
        self.aliases: Dict[str, List[str]] = {
            _norm(k): ([v] if isinstance(v, str) else list(v)) for k, v in spec.get("group_aliases", {}).items()}
        self.ifct_names = list(ifct_names)
        self._norm_to_name = {_norm(n): n for n in self.ifct_names}
        unknown = {i for g in self.groups.values() for i in g["ingredients"]} - set(self.ifct_names)
        if unknown:
            raise ValueError(f"allergen_map.json lists ingredients not in the IFCT database: {sorted(unknown)}")

    def _match_ifct(self, text: str) -> List[str]:
        """Ingredient names a free-text exclusion refers to (substring either way, then difflib)."""
        q = _norm(text)
        if not q:
            return []
        if q in self._norm_to_name:
            return [self._norm_to_name[q]]
        pat = term_pattern(q)
        hits = [n for k, n in self._norm_to_name.items() if pat.search(k) or (len(k) > 3 and k in q)]
        if hits:
            return hits
        close = difflib.get_close_matches(q, list(self._norm_to_name), n=3, cutoff=0.8)
        return [self._norm_to_name[c] for c in close]

    def resolve(self, exclusions: Optional[Sequence[str]]) -> ExclusionSet:
        out = ExclusionSet()
        for raw in exclusions or []:
            key = _norm(raw)
            if not key:
                continue
            group_keys: List[str] = []
            if key in self.groups:
                group_keys = [key]
            elif key.replace(" ", "_") in self.groups:
                group_keys = [key.replace(" ", "_")]
            elif key in self.aliases:
                group_keys = self.aliases[key]
            for g in group_keys:
                spec = self.groups[g]
                out.groups.add(g)
                out.excluded_ingredients.update(spec["ingredients"])
                for term in spec["terms"]:
                    out.terms.setdefault(term, g)
            if group_keys:
                continue
            # free text: map to IFCT ingredients when possible; always scan the literal words
            matched = self._match_ifct(key)
            out.terms.setdefault(key, "free-text")
            if matched:
                out.excluded_ingredients.update(matched)
            else:
                out.unresolved.append(str(raw))
        return out

    def group_labels(self) -> Dict[str, str]:
        return {g: spec["label"] for g, spec in self.groups.items()}

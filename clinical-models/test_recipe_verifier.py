"""
Tests for the deterministic recipe verifier, regeneration loop and template fallback.
No LLM, no network, no TabNet: pure rule checks over the IFCT names.

    python -m pytest test_recipe_verifier.py -q          (from clinical-models/)
"""
import json
import math

import pandas as pd
import pytest
from pathlib import Path

from allergen_filter import AllergenFilter
from json_recipe_verifier import (DEFAULT_RESTRICTED, RecipeVerifier, build_template_recipe, generate_verified_recipe,
                             render_markdown)

NAMES = pd.read_csv(Path(__file__).resolve().parent / "ifct_database.csv")["ingredient"].tolist()
RICE, SPINACH, SPRING, DAL = "Rice, milled (white)", "Spinach (Palak)", "Onion stalk (Spring onion)", "Green gram dal (Moong dal)"
ALLOWED = {RICE: 80.0, SPINACH: 40.0, SPRING: 30.0, DAL: 60.0}


@pytest.fixture(scope="module")
def ver():
    return RecipeVerifier(NAMES)


@pytest.fixture(scope="module")
def filt():
    return AllergenFilter(NAMES)


def draft(ings=None, steps=None, title="Dal rice", description=""):
    return {"title": title, "description": description,
            "ingredients": ings if ings is not None else [{"name": RICE, "grams": 60}, {"name": DAL, "grams": 50},
                                                         {"name": SPINACH, "grams": 30}],
            "steps": steps if steps is not None else ["Wash the rice and dal.", "Cook everything in water until soft.",
                                                      "Add the spinach at the end and serve."]}


def codes(res):
    return set(res.codes())


# ---------------------------------------------------------------------------------------------------- pass
def test_clean_recipe_passes_and_reports_per_ingredient(ver):
    res = ver.verify(draft(), ALLOWED)
    assert res.ok and res.violations == []
    assert {p["ingredient"] for p in res.per_ingredient} == {RICE, DAL, SPINACH}
    assert all(not p["violated"] for p in res.per_ingredient)


def test_ceiling_boundary_is_strict_less_or_equal(ver):
    assert ver.verify(draft([{"name": RICE, "grams": 80}]), ALLOWED).ok
    assert "over_ceiling" in codes(ver.verify(draft([{"name": RICE, "grams": 80.01}]), ALLOWED))


def test_duplicate_entries_are_summed(ver):
    res = ver.verify(draft([{"name": RICE, "grams": 50}, {"name": "white rice", "grams": 40}]), ALLOWED)
    assert "over_ceiling" in codes(res)


def test_name_variants_resolve_to_the_offered_ingredient(ver):
    for nm in ["rice", "White Rice", "milled rice", "Rice, milled (white)"]:
        assert ver.verify(draft([{"name": nm, "grams": 90}]), ALLOWED).codes() == ["over_ceiling"], nm
    res = ver.verify(draft([{"name": "moong dal", "grams": 20}, {"name": "spring onion", "grams": 10}]), ALLOWED)
    assert res.ok, res.violations


def test_grams_given_as_string_with_unit_is_accepted(ver):
    assert ver.verify(draft([{"name": RICE, "grams": "50 g"}]), ALLOWED).ok
    assert ver.verify(draft([{"name": RICE, "grams": "50g"}]), ALLOWED).ok


def test_fenced_json_string_is_accepted(ver):
    assert ver.verify("```json\n" + json.dumps(draft()) + "\n```", ALLOWED).ok


# ---------------------------------------------------------------------------------------------------- reject
@pytest.mark.parametrize("raw", ["not json at all", "[1, 2]", "{}", '{"ingredients": [], "steps": ["x"]}',
                                 '{"ingredients": [{"name": "rice", "grams": 5}], "steps": []}',
                                 '{"ingredients": ["rice"], "steps": ["x"]}',
                                 '{"ingredients": [{"name": "rice"}], "steps": ["x"]}',
                                 '{"ingredients": [{"name": "rice", "grams": 5}], "steps": [1, 2]}'])
def test_malformed_drafts_are_rejected(ver, raw):
    res = ver.verify(raw, ALLOWED)
    assert not res.ok and codes(res) == {"parse_error"}


@pytest.mark.parametrize("grams", ["abc", 0, -5, "nan", "inf", None, ""])
def test_invalid_quantities_are_rejected(ver, grams):
    res = ver.verify(draft([{"name": RICE, "grams": grams}]), ALLOWED)
    assert not res.ok and "invalid_quantity" in codes(res)


def test_listing_a_non_offered_ifct_ingredient_is_rejected(ver):
    res = ver.verify(draft([{"name": RICE, "grams": 30}, {"name": "Paneer (Cottage cheese)", "grams": 20}]), ALLOWED)
    assert "non_allowed_ingredient" in codes(res)


def test_unknown_ingredient_is_rejected_not_accepted(ver):
    res = ver.verify(draft([{"name": RICE, "grams": 30}, {"name": "jaggery", "grams": 20}]), ALLOWED)
    assert "unknown_ingredient" in codes(res)


def test_mentioning_a_non_offered_ingredient_in_the_steps_is_rejected(ver):
    res = ver.verify(draft(steps=["Cook the rice.", "Garnish with crumbled paneer."]), ALLOWED)
    assert "non_allowed_ingredient_mentioned" in codes(res)


def test_longest_match_wins_spring_onion_is_not_onion_and_bell_pepper_is_not_spice(ver):
    ok = ver.verify(draft(steps=["Chop the spring onion finely.", "Cook the rice."]), ALLOWED)
    assert ok.ok, ok.violations
    bell = ver.verify(draft(steps=["Add chopped bell pepper.", "Cook."]), ALLOWED)
    assert "non_allowed_ingredient_mentioned" in codes(bell)       # capsicum, not the neutral spice 'pepper'
    spice = ver.verify(draft(steps=["Finish with black pepper.", "Cook."]), ALLOWED)
    assert spice.ok, spice.violations


# ---------------------------------------------------------------------------------------------------- steps
@pytest.mark.parametrize("step,bad", [
    ("Add 50 g rice to the pot.", False),
    ("Add 200 g rice to the pot.", True),
    ("Add 1 cup rice.", True),               # 240 g by the conservative density
    ("Add 1/2 cup rice.", True),             # 120 g
    ("Add ½ cup rice.", True),
    ("Add 0.2 kg rice.", True),
    ("Add 120g rice.", True),
    ("Add 20 g spinach.", False),
    ("Add 2-3 tbsp spinach.", True),         # range -> upper end, 54 g > 40 g
    ("Add 1 tbsp oil and 2 tsp cumin.", False),
    ("Add 200 ml water.", False),
    ("Add 3 tbsp oil.", True),               # 45 g > 20 g cap
    ("Add 6 tsp turmeric.", True),           # 30 g > 10 g cap
    ("Stir in 30 g of mystery paste.", True),   # unattributed
    ("Cook for 20 minutes at 180 degrees.", False),
    ("Add 3 large rice cakes.", False),
])
def test_quantities_written_in_the_steps(ver, step, bad):
    res = ver.verify(draft(steps=[step, "Serve."]), ALLOWED)
    assert (not res.ok) == bad, (step, res.violations)


# ---------------------------------------------------------------------------------------------------- sodium
def test_restricted_items_are_rejected_only_when_requested(ver):
    d = draft(steps=["Cook the rice.", "Add salt to taste."])
    assert ver.verify(d, ALLOWED).ok                                       # not restricted for this patient
    res = ver.verify(d, ALLOWED, restricted_terms=DEFAULT_RESTRICTED)
    assert "restricted_item" in codes(res)


@pytest.mark.parametrize("phrase", ["Do not add salt.", "Cook without salt.", "Add no salt.", "Avoid soy sauce."])
def test_negated_restricted_mentions_are_allowed(ver, phrase):
    assert ver.verify(draft(steps=["Cook the rice.", phrase]), ALLOWED, restricted_terms=DEFAULT_RESTRICTED).ok


def test_salt_as_a_listed_ingredient_is_flagged_as_restricted(ver):
    res = ver.verify(draft([{"name": RICE, "grams": 30}, {"name": "salt", "grams": 3}]), ALLOWED,
                     restricted_terms=DEFAULT_RESTRICTED)
    assert "restricted_item" in codes(res) and "unknown_ingredient" not in codes(res)


# ---------------------------------------------------------------------------------------------------- allergens
def test_excluded_terms_are_caught_in_every_field(ver, filt):
    ex = filt.resolve(["dairy"])
    for d in [draft(steps=["Cook the rice.", "Finish with a spoon of ghee."]),
              draft(title="Cream of rice"),
              draft(description="Served with raita."),
              draft(steps=["Simmer in coconut milk."])]:
        assert "excluded_term" in codes(ver.verify(d, ALLOWED, ex)), d
    assert ver.verify(draft(), ALLOWED, ex).ok


def test_excluded_ingredient_in_the_list_is_caught(ver, filt):
    ex = filt.resolve(["peanut"])
    res = ver.verify(draft([{"name": RICE, "grams": 30}, {"name": "peanut butter", "grams": 10}]), ALLOWED, ex)
    assert "excluded_term" in codes(res) and not res.ok


# ---------------------------------------------------------------------------------------------------- template
def rows(**kw):
    base = [{"ingredient": RICE, "max_grams": 80.0, "label": "Allowed"},
            {"ingredient": DAL, "max_grams": 60.0, "label": "Half Portion"},
            {"ingredient": SPINACH, "max_grams": 4.0, "label": "Avoid"}]
    return base


def test_template_respects_ceilings_and_verifies(ver, filt):
    t = build_template_recipe(rows())
    assert t is not None
    names = {i["name"] for i in t["ingredients"]}
    assert SPINACH not in names                                            # Avoid items are never used
    for i in t["ingredients"]:
        if i["name"] in ALLOWED:
            assert i["grams"] <= ALLOWED[i["name"]] and i["grams"] == math.floor(i["grams"])
    assert ver.verify(t, {RICE: 80.0, DAL: 60.0}, restricted_terms=DEFAULT_RESTRICTED).ok
    assert "Simple one-pot dish" in render_markdown(t)


def test_template_is_none_when_nothing_is_usable():
    assert build_template_recipe([{"ingredient": RICE, "max_grams": 5.0, "label": "Avoid"}]) is None
    assert build_template_recipe([]) is None


# ---------------------------------------------------------------------------------------------------- loop
def good():
    """A clean draft that only uses ingredients rows() offers (spinach is an 'Avoid' row there)."""
    return draft([{"name": RICE, "grams": 60}, {"name": DAL, "grams": 50}],
                 ["Wash the rice and dal.", "Cook both in water until soft.", "Serve warm."])


def run(ver, llm, rows_=None, **kw):
    return generate_verified_recipe(llm, ver, rows_ or rows(), **kw)


def test_loop_returns_first_verified_llm_draft(ver):
    out = run(ver, lambda fb: json.dumps(good()))
    assert out.source == "llm" and out.attempts == 1 and out.recipe


def test_loop_regenerates_with_feedback_then_succeeds(ver):
    seen = []
    bad = draft([{"name": RICE, "grams": 500}])

    def llm(fb):
        seen.append(fb)
        return json.dumps(bad if len(seen) < 3 else good())
    out = run(ver, llm)
    assert out.source == "llm" and out.attempts == 3
    assert seen[0] is None and "over_ceiling" in seen[1]                  # violation list is fed back
    assert [h["ok"] for h in out.history] == [False, False, True]


def test_loop_falls_back_to_a_verified_template(ver):
    out = run(ver, lambda fb: json.dumps(draft([{"name": RICE, "grams": 500}])))
    assert out.source == "template" and out.recipe and out.history[-1]["attempt"] == "template"
    assert ver.verify(out.recipe, {RICE: 80.0, DAL: 60.0}).ok


def test_llm_errors_count_as_failed_attempts_and_fall_back(ver):
    def boom(fb):
        raise TimeoutError("groq timeout")
    out = run(ver, boom)
    assert out.source == "template" and out.history[0]["violations"][0]["code"] == "llm_error"


def test_garbage_output_never_reaches_the_user(ver):
    out = run(ver, lambda fb: "Sure! Here is a lovely recipe with 500 g of paneer and plenty of salt.")
    assert out.source == "template"                                       # not the garbage


def test_no_usable_ingredient_means_no_recipe_not_an_unverified_one(ver):
    out = run(ver, lambda fb: json.dumps(good()), [{"ingredient": RICE, "max_grams": 3.0, "label": "Avoid"}])
    assert out.recipe is None and out.source == "none"


def test_excluded_ingredient_is_not_offered_even_if_the_row_says_allowed(ver, filt):
    ex = filt.resolve(["gluten"])
    r = rows() + [{"ingredient": "Wheat flour, atta", "max_grams": 90.0, "label": "Allowed"}]
    out = run(ver, lambda fb: json.dumps(draft([{"name": "Wheat flour, atta", "grams": 30}])), r, exclusions=ex)
    assert out.source == "template"
    assert "wheat" not in render_markdown(out.recipe).lower()

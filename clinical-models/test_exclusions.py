"""
Tests for deterministic allergy / intolerance / diet exclusions (allergen_filter.py + the engine).

    python -m pytest test_exclusions.py -q          (from clinical-models/)

The engine is built once per module (it loads the TabNet artifacts, ~seconds).
"""
import contextlib
import io
import json
from pathlib import Path

import pytest

from allergen_filter import AllergenFilter, term_pattern

ROOT = Path(__file__).resolve().parent.parent
PATIENT = {"age": 58, "sex_male": 1, "has_htn": 1, "has_dm": 0, "has_ckd": 1, "serum_sodium": 140,
           "serum_potassium": 5.1, "creatinine": 1.9, "egfr": 45, "hba1c": 5.2, "fbs": 85, "sbp": 162,
           "dbp": 96, "bmi": 26}
USECASE = ["Green gram dal (Moong dal)", "Paneer (Cottage cheese)", "Egg, whole, boiled", "Rice, milled (white)",
           "Banana, ripe", "Apple", "Pineapple (Ananas)", "Bottle gourd (Lauki)"]


@pytest.fixture(scope="module")
def engine():
    from train_model2 import PortionControlModel
    with contextlib.redirect_stdout(io.StringIO()):
        return PortionControlModel()


@pytest.fixture(scope="module")
def filt(engine):
    return AllergenFilter(engine.ifct.df["ingredient"].tolist())


def test_map_lists_only_real_ifct_ingredients(filt):
    # AllergenFilter raises on construction if any listed ingredient is not in the database
    assert len(filt.groups) >= 10


def test_no_exclusions_reproduces_committed_use_case(engine):
    """The deployed behaviour is unchanged when no exclusions are given."""
    committed = json.loads((ROOT / "artifacts/models/reports/usecase_report.json").read_text())
    res = engine.get_recommendations(PATIENT, USECASE, include_substitutes=False, use_ledger=False)
    got = {r["ingredient"]: r["max_grams"] for r in res["recommendations"]}
    want = {k: v["max_grams"] for k, v in committed["model2_research"].items()}
    assert set(got) == set(want) == set(USECASE)
    assert got == pytest.approx(want, abs=0.11)
    assert res["summary"]["excluded"] == []


def test_dairy_group_is_excluded_everywhere(engine):
    res = engine.get_recommendations(PATIENT, USECASE, use_ledger=False, exclusions=["dairy"])
    by = {r["ingredient"]: r for r in res["recommendations"]}
    assert by["Paneer (Cottage cheese)"]["label"] == "Excluded"
    assert by["Paneer (Cottage cheese)"]["max_grams"] == 0
    assert "Paneer (Cottage cheese)" in res["summary"]["excluded"]
    assert [r["ingredient"] for r in res["recommendations"]] == USECASE      # order preserved
    # non-excluded items are still computed
    assert by["Rice, milled (white)"]["max_grams"] > 0


@pytest.mark.parametrize("group", ["dairy", "lactose", "gluten", "peanut", "tree_nuts", "coconut", "sesame", "soy",
                                   "egg", "fish", "shellfish", "legumes", "allium", "vegetarian", "vegan"])
def test_excluded_group_never_gets_grams_or_appears_as_substitute(engine, filt, group):
    everything = engine.ifct.df["ingredient"].tolist()
    res = engine.get_recommendations(PATIENT, everything, use_ledger=False, include_substitutes=True,
                                     exclusions=[group])
    excl = filt.resolve([group])
    assert excl.excluded_ingredients, group
    for r in res["recommendations"]:
        if r["ingredient"] in excl.excluded_ingredients or excl.is_excluded(r["ingredient"]):
            assert r["label"] == "Excluded" and r["max_grams"] == 0, (group, r["ingredient"])
        for sub in r["substitutes"]:
            assert not excl.is_excluded(sub["ingredient"]), (group, r["ingredient"], sub["ingredient"])


def test_alias_and_free_text_resolution(filt):
    assert filt.resolve(["Wheat allergy"]).groups == {"gluten"}
    assert filt.resolve(["nuts"]).groups == {"tree_nuts", "peanut"}
    assert filt.resolve(["lactose intolerant"]).groups == {"lactose"}
    e = filt.resolve(["Peanut Butter"])                     # not an IFCT item: kept as scan term, reported
    assert e.unresolved == ["Peanut Butter"] and e.hits_in_text("2 tbsp peanut butter")
    e = filt.resolve(["bhindi"])                            # free text that is a real IFCT item (okra)
    assert e.excluded_ingredients == set() or "Ladies finger (Bhindi/Okra)" in e.excluded_ingredients
    e = filt.resolve(["ladies finger"])
    assert "Ladies finger (Bhindi/Okra)" in e.excluded_ingredients


def test_dairy_aliases_are_caught_in_text(filt):
    e = filt.resolve(["dairy"])
    for text in ["fry in ghee", "add a dollop of curd", "garnish with khoya", "stir in some malai",
                 "serve with Raita", "top with grated Cheese"]:
        assert e.hits_in_text(text), text


def test_word_boundaries_avoid_false_positives(filt):
    assert not filt.resolve(["sesame"]).hits_in_text("cook until tender")           # 'til' vs 'until'
    assert not filt.resolve(["egg"]).hits_in_text("slice the eggplant")             # 'egg' vs 'eggplant'
    assert not filt.resolve(["tree_nuts"]).hits_in_text("add coconut and nutmeg")   # 'nut' inside words
    assert filt.resolve(["tree_nuts"]).hits_in_text("add 10 g cashews")             # plural handled
    assert term_pattern("peanut").search("peanuts are excluded")


def test_conservative_overblocking_is_documented_behaviour(filt):
    # 'coconut milk' contains the dairy term 'milk'; blocking it is deliberate (fail-safe)
    assert filt.resolve(["dairy"]).hits_in_text("simmer in coconut milk")


def test_everything_excluded_returns_empty_but_valid(engine):
    names = engine.ifct.df["ingredient"].tolist()
    res = engine.get_recommendations(PATIENT, names[:3], use_ledger=False, exclusions=names[:3])
    assert all(r["label"] == "Excluded" for r in res["recommendations"])

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import allergens
import recipe_verifier as rv

ALLOWED = [
    {"ingredient": "Rice, milled (white)", "max_grams": 116.3},
    {"ingredient": "Green gram dal (Moong dal)", "max_grams": 34.2},
    {"ingredient": "Paneer (Cottage cheese)", "max_grams": 24.9},
    {"ingredient": "Banana, ripe", "max_grams": 0.0},
]

GOOD = """# Dal rice
Simple dish.
## Ingredients
- Rice, milled (white): 100 g
- Green gram dal (Moong dal): 30 g
- Water: 400 ml
## Instructions
1. Cook.
"""


def test_good_recipe_passes():
    r = rv.verify_recipe(GOOD, ALLOWED)
    assert r["ok"], r["issues"]
    assert [x["stated_grams"] for x in r["records"]][:2] == [100.0, 30.0]


def test_over_ceiling_fails_even_by_a_gram():
    r = rv.verify_recipe(GOOD.replace("30 g", "35 g"), ALLOWED)
    assert not r["ok"] and any(i["type"] == "over_ceiling" for i in r["issues"])


def test_forbidden_ingredient_with_zero_ceiling_fails():
    t = GOOD.replace("- Water", "- Banana, ripe: 5 g\n- Water")
    r = rv.verify_recipe(t, ALLOWED)
    assert any(i["type"] == "over_ceiling" and i["ingredient"] == "Banana, ripe" for i in r["issues"])


def test_non_gram_unit_and_unknown_ingredient_fail():
    r = rv.verify_recipe(GOOD.replace("100 g", "1 cup"), ALLOWED)
    assert any(i["type"] == "unit_not_grams" for i in r["issues"])
    r = rv.verify_recipe(GOOD.replace("- Water", "- Potato (Aloo): 20 g\n- Water"), ALLOWED)
    assert any(i["type"] == "unknown_ingredient" for i in r["issues"])


def test_short_alias_and_bold_markup_accepted():
    t = GOOD.replace("Green gram dal (Moong dal)", "**Green gram dal (Moong dal)**")
    assert rv.verify_recipe(t, ALLOWED)["ok"]
    t = GOOD.replace("- Water", "- Paneer: 20 g\n- Water")
    assert rv.verify_recipe(t, ALLOWED)["ok"]


def test_missing_section_and_unparsed_line_fail():
    assert not rv.verify_recipe("Just cook rice.", ALLOWED)["ok"]
    t = GOOD.replace("- Water: 400 ml", "Some free text line")
    assert any(i["type"] == "unparsed_line" for i in rv.verify_recipe(t, ALLOWED)["issues"])


def test_salt_rejected_when_restricted():
    t = GOOD.replace("- Water", "- Salt: 2 g\n- Water")
    assert rv.verify_recipe(t, ALLOWED)["ok"]
    assert not rv.verify_recipe(t, ALLOWED, restrict_salt=True)["ok"]


def test_allergen_term_anywhere_in_text_fails():
    t = GOOD.replace("1. Cook.", "1. Cook and finish with ghee.")
    r = rv.verify_recipe(t, ALLOWED, groups=["dairy"])
    assert any(i["type"] == "excluded_term" and i["term"] == "ghee" for i in r["issues"])
    assert rv.verify_recipe(GOOD, ALLOWED, groups=["dairy"])["ok"]
    assert not rv.verify_recipe(GOOD, ALLOWED, terms=["moong"])["ok"]  # custom term


def test_template_recipe_passes_its_own_verifier():
    t = rv.build_template_recipe(ALLOWED)
    assert t and rv.verify_recipe(t, ALLOWED, restrict_salt=True)["ok"]
    assert "Banana" not in t  # zero-ceiling ingredient not used
    assert rv.build_template_recipe([{"ingredient": "X", "max_grams": 3.0}]) is None


def test_exclusion_groups_on_ifct_names():
    cases = {
        ("Paneer (Cottage cheese)", "Milk and Dairy"): "dairy",
        ("Ghee", "Milk and Dairy"): "dairy",
        ("Groundnut (Moongphali/Peanut)", "Nuts and Oilseeds"): "peanut",
        ("Cashewnut (Kaju)", "Nuts and Oilseeds"): "tree_nuts",
        ("Egg, whole, boiled", "Meat and Poultry"): "egg",
        ("Prawn (Jhinga)", "Fish and Seafood"): "shellfish",
        ("Rohu fish", "Fish and Seafood"): "fish",
        ("Wheat flour, atta", "Cereals"): "gluten",
        ("Soybean", "Pulses"): "soy",
        ("Sesame seeds (Til)", "Nuts and Oilseeds"): "sesame",
        ("Chicken, breast", "Meat and Poultry"): "no_meat_fish",
    }
    for (name, cat), gid in cases.items():
        assert allergens.exclusion_reason(name, cat, [gid]), (name, gid)
    # no false positives on look-alikes
    assert not allergens.exclusion_reason("Brinjal (Eggplant)", "Other Vegetables", ["egg"])
    assert not allergens.exclusion_reason("Rice, milled (white)", "Cereals", list(allergens.ALLERGEN_GROUPS))
    assert not allergens.exclusion_reason("Coconut, fresh (Nariyal)", "Nuts and Oilseeds", ["tree_nuts"])
    assert allergens.exclusion_reason("Banana, ripe", "Fruits", [], ["banana"])
    assert allergens.normalize_groups(["dairy", "bogus"]) == ["dairy"]

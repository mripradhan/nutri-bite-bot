"""
Test Model2 Portion Control System
===================================
Comprehensive test cases with user-friendly output
"""

import sys
from train_model2 import PortionControlModel
from train_model1 import MonotonicFeatureTransformer  # noqa: F401 — needed for joblib deserialization
from output_formatter import OutputFormatter, print_recommendations


def test_ckd_patient():
    """Test CKD patient - should restrict potassium and protein"""
    print("\n" + "="*70)
    print("  TEST 1: CKD PATIENT (Stage 4)")
    print("="*70)
    
    model = PortionControlModel()
    
    patient = {
        "age": 62,
        "sex_male": 1,
        "has_htn": 0,
        "has_dm": 0,
        "has_ckd": 1,
        "serum_sodium": 138,
        "serum_potassium": 5.8,
        "creatinine": 3.5,
        "egfr": 22,
        "hba1c": 5.5,
        "fbs": 95,
        "sbp": 135,
        "dbp": 82,
        "bmi": 26
    }
    
    # High-potassium foods - should be restricted
    ingredients = [
        "Banana, ripe",
        "Spinach (Palak)",
        "Potato (Aloo)",
        "Tomato, ripe",
        "Orange",
        "Rice, milled (white)",
        "Apple",
    ]
    
    results = model.get_recommendations(patient, ingredients)
    print_recommendations(results)
    
    print("\n💡 Expected: Banana, Spinach, Potato should be restricted (high potassium)")
    
    return results


def test_dm_patient():
    """Test Diabetes patient - should restrict carbs"""
    print("\n" + "="*70)
    print("  TEST 2: DIABETES PATIENT (Uncontrolled)")
    print("="*70)
    
    model = PortionControlModel()
    
    patient = {
        "age": 48,
        "sex_male": 1,
        "has_htn": 0,
        "has_dm": 1,
        "has_ckd": 0,
        "serum_sodium": 140,
        "serum_potassium": 4.0,
        "creatinine": 1.0,
        "egfr": 92,
        "hba1c": 9.2,
        "fbs": 210,
        "sbp": 125,
        "dbp": 80,
        "bmi": 32
    }
    
    # High-carb foods - should be restricted for DM
    ingredients = [
        "Rice, milled (white)",
        "Potato (Aloo)",
        "Banana, ripe",
        "Mango, ripe",
        "Dates, dry",
        "Chicken, breast",
        "Spinach (Palak)",
    ]
    
    results = model.get_recommendations(patient, ingredients)
    print_recommendations(results)
    
    print("\n💡 Expected: High-carb foods (Rice, Dates, Mango) should be portion controlled")
    
    return results


def test_multi_condition():
    """Test patient with CKD + HTN + DM - most restrictive"""
    print("\n" + "="*70)
    print("  TEST 3: CKD + HTN + DM PATIENT (Most Restrictive)")
    print("="*70)
    
    model = PortionControlModel()
    
    patient = {
        "age": 65,
        "sex_male": 1,
        "has_htn": 1,
        "has_dm": 1,
        "has_ckd": 1,
        "serum_sodium": 145,
        "serum_potassium": 5.9,
        "creatinine": 4.0,
        "egfr": 18,
        "hba1c": 8.8,
        "fbs": 190,
        "sbp": 170,
        "dbp": 100,
        "bmi": 34
    }
    
    ingredients = [
        "Banana, ripe",
        "Spinach (Palak)",
        "Potato (Aloo)",
        "Rice, milled (white)",
        "Dates, dry",
        "Soybean",
        "Prawn (Jhinga)",
        "Cucumber",
        "Apple",
    ]
    
    results = model.get_recommendations(patient, ingredients)
    print_recommendations(results)
    
    print("\n💡 Expected: Most items should be restricted due to multiple conditions")
    
    return results


def test_healthy_patient():
    """Test healthy patient - minimal restrictions"""
    print("\n" + "="*70)
    print("  TEST 4: HEALTHY PATIENT (No Chronic Conditions)")
    print("="*70)
    
    model = PortionControlModel()
    
    patient = {
        "age": 35,
        "sex_male": 0,
        "has_htn": 0,
        "has_dm": 0,
        "has_ckd": 0,
        "serum_sodium": 140,
        "serum_potassium": 4.0,
        "creatinine": 0.9,
        "egfr": 110,
        "hba1c": 5.2,
        "fbs": 88,
        "sbp": 118,
        "dbp": 75,
        "bmi": 22
    }
    
    ingredients = [
        "Banana, ripe",
        "Spinach (Palak)",
        "Rice, milled (white)",
        "Chicken, breast",
        "Curd (Dahi/Yogurt)",
    ]
    
    results = model.get_recommendations(patient, ingredients)
    print_recommendations(results)
    
    print("\n💡 Expected: Most items should be allowed with generous portions")
    
    return results


def test_ingredient_search():
    """Test ingredient fuzzy search"""
    print("\n" + "="*70)
    print("  TEST 5: INGREDIENT SEARCH")
    print("="*70)
    
    model = PortionControlModel()
    
    print("\n  Testing ingredient search with common names & typos:")
    print("  " + "-"*50)
    
    # Test with common names and variations
    test_queries = [
        ("palak", "📗 Local name"),
        ("aloo", "📗 Local name"),
        ("dahi", "📗 Local name"),
        ("toor dal", "📗 Dal variety"),
        ("moong", "📗 Partial match"),
        ("bananna", "📗 Typo"),
        ("xyz food", "📕 Unknown"),
    ]
    
    for query, note in test_queries:
        results = model.ifct.search_ingredient(query)
        if results:
            print(f"  {note}: '{query}' → ✅ {results[0]}")
        else:
            print(f"  {note}: '{query}' → ❌ No matches")
    
    return True


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  MODEL2 PORTION CONTROL SYSTEM - TEST SUITE")
    print("█"*70)
    
    try:
        test_ckd_patient()
        test_dm_patient()
        test_multi_condition()
        test_healthy_patient()
        test_ingredient_search()
        
        print("\n" + "█"*70)
        print("  ✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("█"*70 + "\n")
    except Exception as e:
        print(f"\n  ❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

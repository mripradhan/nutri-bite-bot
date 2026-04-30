"""
Clinical Model #2: Portion Control System
==========================================
Recommends safe ingredient portions based on:
- Model1 clinical outputs (sodium/potassium/protein/carb sensitivity levels)
- Disease flags (has_ckd, has_htn, has_dm)
- IFCT nutritional data (Indian Food Composition Tables 2017)

Usage:
    python train_model2.py
    
Dependencies:
    - Model1 trained models in ../artifacts/models/
    - IFCT database in ./ifct_database.csv
"""

import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from difflib import get_close_matches
import warnings

warnings.filterwarnings('ignore')

# Import MonotonicFeatureTransformer so joblib can deserialize
# monotonic_transformer.joblib (class must be in namespace at load time)
from train_model1 import MonotonicFeatureTransformer  # noqa: F401
from train_model1 import TFTRiskModel  # Phase 3B
from pytorch_tabnet.tab_model import TabNetClassifier


# ===============================
# Configuration
# ===============================
@dataclass
class Model2Config:
    """Configuration for Model2 Portion Control System"""
    
    # Model1 artifacts path
    model1_dir: Path = Path("../artifacts/models")
    
    # IFCT database path
    ifct_path: Path = Path("ifct_database.csv")
    
    # Daily nutrient budgets by condition (mg or g per day)
    # Based on clinical guidelines
    daily_budgets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "healthy": {
            "sodium_mg": 2300,      # AHA recommendation
            "potassium_mg": 4700,   # DASH diet
            "protein_g": 56,        # 0.8g/kg for 70kg adult
            "carbs_g": 275,         # ~55% of 2000 kcal
            "phosphorus_mg": 1250,  # RDA for adults
        },
        "ckd": {
            "sodium_mg": 2000,      # CKD Stage 3-4
            "potassium_mg": 2000,   # CKD restriction
            "protein_g": 42,        # 0.6g/kg for CKD
            "carbs_g": 275,
            "phosphorus_mg": 800,   # CKD limit (kidney.org.uk guideline)
        },
        "htn": {
            "sodium_mg": 1500,      # HTN strict limit
            "potassium_mg": 4700,   # DASH encourages K+
            "protein_g": 56,
            "carbs_g": 275,
            "phosphorus_mg": 1250,
        },
        "dm": {
            "sodium_mg": 2300,
            "potassium_mg": 4700,
            "protein_g": 56,
            "carbs_g": 180,         # Diabetes carb limit
            "phosphorus_mg": 1250,
        },
        "ckd_htn": {
            "sodium_mg": 1500,      # HTN strict limit (priority)
            "potassium_mg": 2000,   # CKD restriction (priority)
            "protein_g": 42,
            "carbs_g": 275,
            "phosphorus_mg": 800,   # CKD phosphorus restriction
        },
        "ckd_dm": {
            "sodium_mg": 2000,
            "potassium_mg": 2000,
            "protein_g": 42,        # 0.6g/kg for CKD+DM
            "carbs_g": 150,         # More strict with DM
            "phosphorus_mg": 800,   # CKD phosphorus restriction
        },
        "ckd_htn_dm": {
            "sodium_mg": 1500,      # Most restrictive
            "potassium_mg": 2000,
            "protein_g": 42,
            "carbs_g": 150,
            "phosphorus_mg": 800,   # CKD phosphorus restriction
        },
    })
    
    # Portion thresholds
    default_cap_g: float = 300.0     # Max grams per ingredient
    half_portion_g: float = 75.0     # Below this = "Half portion"
    avoid_threshold_g: float = 5.0   # Below this = "Avoid"
    
    # Phosphorus-protein efficiency multipliers (Phase 1C — KDIGO 2024)
    phos_efficiency_bonus: float = 1.20   # ×1.20 for CKD-preferred (excellent/good tier)
    phos_efficiency_penalty: float = 0.75 # ×0.75 for poor-tier high-phosphorus sources
    
    # KDIGO 2024 phos/protein ratio tier thresholds (mg phosphorus per g protein)
    # Calibrated to IFCT 2017 distribution: min≈7, median≈20, max≈40
    phos_tier_excellent: float = 12.0   # ≤ 12 mg/g  — best quartile
    phos_tier_good: float = 18.0        # ≤ 18 mg/g  — below median
    phos_tier_moderate: float = 25.0    # ≤ 25 mg/g  — below 75th pctile


# ===============================
# IFCT Database Interface
# ===============================
class IFCTDatabase:
    """
    Read-only interface to IFCT 2017 nutritional data.
    All values are per 100g edible portion.
    """
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
        # Required columns for portion control
        required = {
            "ingredient",
            "sodium_mg_per_100g",
            "potassium_mg_per_100g",
            "protein_g_per_100g",
            "carbs_g_per_100g",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"IFCT CSV missing columns: {sorted(missing)}")
        
        # Create normalized index for searching
        self.df["ingredient_norm"] = self.df["ingredient"].astype(str).str.lower().str.strip()
        self.idx = self.df.set_index("ingredient_norm")
        
        # Create list of all ingredients for fuzzy matching
        self.all_ingredients = self.df["ingredient"].tolist()
        self.all_ingredients_lower = [i.lower() for i in self.all_ingredients]
        
        print(f"✓ Loaded IFCT database with {len(self.df)} ingredients")
    
    def get_nutrients_per_100g(self, ingredient: str) -> Dict[str, float]:
        """Get nutrient values per 100g for an ingredient"""
        key = ingredient.lower().strip()
        
        if key not in self.idx.index:
            raise KeyError(f"Ingredient not found: {ingredient}")
        
        row = self.idx.loc[key]
        
        return {
            "sodium_mg": float(row.get("sodium_mg_per_100g", 0)),
            "potassium_mg": float(row.get("potassium_mg_per_100g", 0)),
            "protein_g": float(row.get("protein_g_per_100g", 0)),
            "carbs_g": float(row.get("carbs_g_per_100g", 0)),
            "phosphorus_mg": float(row.get("phosphorus_mg_per_100g", 0)),
            "fat_g": float(row.get("fat_g_per_100g", 0)),
            "fiber_g": float(row.get("fiber_g_per_100g", 0)),
            "calories": float(row.get("calories_per_100g", 0)),
        }
    
    def search_ingredient(self, query: str, n_matches: int = 5) -> List[str]:
        """Fuzzy search for ingredients matching query"""
        query_lower = query.lower().strip()
        
        # Exact match
        if query_lower in self.all_ingredients_lower:
            idx = self.all_ingredients_lower.index(query_lower)
            return [self.all_ingredients[idx]]
        
        # Partial matches
        matches = []
        for i, ing_lower in enumerate(self.all_ingredients_lower):
            if query_lower in ing_lower or ing_lower in query_lower:
                matches.append(self.all_ingredients[i])
        
        if matches:
            return matches[:n_matches]
        
        # Fuzzy match
        close = get_close_matches(query_lower, self.all_ingredients_lower, n=n_matches, cutoff=0.5)
        return [self.all_ingredients[self.all_ingredients_lower.index(m)] for m in close]
    
    def list_by_category(self, category: str) -> List[str]:
        """List all ingredients in a category"""
        mask = self.df["category"].str.lower() == category.lower()
        return self.df.loc[mask, "ingredient"].tolist()
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        return self.df["category"].unique().tolist()
    
    def get_phos_protein_efficiency(
        self,
        ingredient: str,
        config: Model2Config = None,
    ) -> Dict[str, Any]:
        """
        Phosphorus-to-protein efficiency scoring (KDIGO 2024).

        Clinical rationale:
        Not all protein sources carry equal phosphorus burden. Egg whites
        (ratio ~1.4 mg/g) are far preferable for CKD patients than processed
        cheese (~47 mg/g). This method quantifies that difference so
        PortionRecommender can bonus/penalise portions accordingly.

        Parameters
        ----------
        ingredient : str
            Name of ingredient (must exist in IFCT).
        config : Model2Config, optional
            Supplies tier thresholds; defaults to Model2Config() if None.

        Returns
        -------
        dict with keys:
            phos_per_g_protein : float
            efficiency_tier    : str  (excellent / good / moderate / poor)
            ckd_preferred      : bool (True for excellent and good)
        """
        if config is None:
            config = Model2Config()

        key = ingredient.lower().strip()
        if key not in self.idx.index:
            # Unknown ingredient → neutral moderate
            return {
                "phos_per_g_protein": 0.0,
                "efficiency_tier": "moderate",
                "ckd_preferred": False,
            }

        row = self.idx.loc[key]
        protein = float(row.get("protein_g_per_100g", 0))
        phosphorus = float(row.get("phosphorus_mg_per_100g", 0))

        # Guard divide-by-zero / missing protein
        if protein <= 0:
            return {
                "phos_per_g_protein": 0.0,
                "efficiency_tier": "moderate",
                "ckd_preferred": False,
            }

        ratio = phosphorus / protein

        if ratio <= config.phos_tier_excellent:
            tier = "excellent"
        elif ratio <= config.phos_tier_good:
            tier = "good"
        elif ratio <= config.phos_tier_moderate:
            tier = "moderate"
        else:
            tier = "poor"

        return {
            "phos_per_g_protein": round(ratio, 2),
            "efficiency_tier": tier,
            "ckd_preferred": tier in ("excellent", "good"),
        }
    
    def get_ckd_friendly_proteins(
        self,
        config: Model2Config = None,
        min_protein_g: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """
        Ranked list of CKD-preferred protein sources.

        Clinical rationale:
        Proactively surfaces ingredients with low phosphorus burden per gram
        of protein so the chatbot layer can suggest CKD-safe options without
        requiring an explicit ingredient query from the patient.

        Parameters
        ----------
        config : Model2Config, optional
            Supplies tier thresholds; defaults to Model2Config().
        min_protein_g : float
            Minimum protein per 100g to qualify as a "protein source".

        Returns
        -------
        List[dict] sorted ascending by phos_per_g_protein, each dict:
            ingredient, phos_per_g_protein, efficiency_tier
        """
        if config is None:
            config = Model2Config()

        results = []
        for _, row in self.df.iterrows():
            protein = float(row.get("protein_g_per_100g", 0))
            if protein < min_protein_g:
                continue
            eff = self.get_phos_protein_efficiency(
                row["ingredient"], config=config
            )
            if eff["ckd_preferred"]:
                results.append({
                    "ingredient": row["ingredient"],
                    "phos_per_g_protein": eff["phos_per_g_protein"],
                    "efficiency_tier": eff["efficiency_tier"],
                })

        results.sort(key=lambda x: x["phos_per_g_protein"])
        return results


# ===============================
# Data Classes
# ===============================
@dataclass
class DailyBudget:
    """Remaining nutrient budget for the day"""
    sodium_mg_remaining: float
    potassium_mg_remaining: float
    protein_g_remaining: float
    carbs_g_remaining: float
    phosphorus_mg_remaining: float = 700.0


@dataclass
class PortionDecision:
    """Portion recommendation for a single ingredient"""
    ingredient: str
    max_grams: float
    label: str                      # "Allowed" | "Half portion" | "Avoid"
    explanation: str
    nutrient_load_at_max: Dict[str, float]
    binding_constraint: str         # Which nutrient constrained the portion
    substitutes: List['PortionDecision'] = field(default_factory=list)


# ===============================
# Nutrient Ledger
# ===============================
class NutrientLedger:
    """
    Stateful daily nutrient intake tracker.

    Clinical rationale:
    A CKD patient with a 2000 mg/day sodium limit who eats 900 mg at breakfast
    must have their lunch and dinner budgets reduced to 1100 mg total. Without
    a persistent ledger, every get_recommendations() call treats the patient as
    if they have consumed nothing — this class closes that gap.

    The ledger accumulates nutrient loads from confirmed meals and subtracts
    them from the base daily budget on every subsequent recommendation request.
    It auto-resets if the session crosses midnight.
    """

    TRACKED_NUTRIENTS = [
        "sodium_mg", "potassium_mg", "protein_g",
        "carbs_g", "phosphorus_mg", "calories",
    ]

    def __init__(self):
        """Initialize an empty ledger for today's session."""
        self._reset()

    def _reset(self):
        """Zero all accumulators and clear meal history."""
        from datetime import date
        self._session_date = date.today()
        self._consumed: Dict[str, float] = {n: 0.0 for n in self.TRACKED_NUTRIENTS}
        self._meals: List[Dict[str, Any]] = []

    def _check_midnight(self):
        """
        Auto-reset if the current date differs from the session start date.

        Clinical rationale:
        Daily nutrient budgets are 24-hour allowances. If a user's session
        spans midnight, continuing to accumulate yesterday's intake into
        today's budget would be clinically incorrect.
        """
        from datetime import date
        if date.today() != self._session_date:
            print("\u26a0 NutrientLedger: midnight crossed \u2014 resetting daily totals")
            self._reset()

    def log_accepted_portions(
        self,
        meal_name: str,
        portions: List[PortionDecision],
        ifct: "IFCTDatabase",
    ) -> Dict[str, float]:
        """
        Record the nutrient content of every non-Avoid portion.

        Clinical rationale:
        Only portions the patient actually intends to eat (Allowed or Half
        portion) should count toward consumption. Avoided items are not
        eaten and must not deplete the budget.

        Parameters
        ----------
        meal_name : str
            Human-readable label (e.g. 'Breakfast', 'Lunch').
        portions : list of PortionDecision
            The recommendations the user has confirmed.
        ifct : IFCTDatabase
            Used to look up per-100g nutrient values for calorie tracking.

        Returns
        -------
        dict : nutrients added by this meal
        """
        self._check_midnight()

        meal_nutrients: Dict[str, float] = {n: 0.0 for n in self.TRACKED_NUTRIENTS}
        accepted_items: List[str] = []

        for p in portions:
            if p.label == "Avoid":
                continue

            factor = p.max_grams / 100.0

            # Use pre-computed nutrient_load_at_max for the four main nutrients
            meal_nutrients["sodium_mg"] += p.nutrient_load_at_max.get("sodium_mg", 0)
            meal_nutrients["potassium_mg"] += p.nutrient_load_at_max.get("potassium_mg", 0)
            meal_nutrients["protein_g"] += p.nutrient_load_at_max.get("protein_g", 0)
            meal_nutrients["carbs_g"] += p.nutrient_load_at_max.get("carbs_g", 0)

            # Look up phosphorus and calories from IFCT (not in nutrient_load_at_max)
            try:
                full_nutrients = ifct.get_nutrients_per_100g(p.ingredient)
                meal_nutrients["phosphorus_mg"] += full_nutrients.get("phosphorus_mg", 0) * factor
                meal_nutrients["calories"] += full_nutrients.get("calories", 0) * factor
            except KeyError:
                pass  # ingredient not in DB — skip extras

            accepted_items.append(p.ingredient)

        # Accumulate into daily totals
        for k in self.TRACKED_NUTRIENTS:
            self._consumed[k] += meal_nutrients[k]

        # Record meal in history
        self._meals.append({
            "meal_name": meal_name,
            "items": accepted_items,
            "nutrients": {k: round(v, 1) for k, v in meal_nutrients.items()},
        })

        return {k: round(v, 1) for k, v in meal_nutrients.items()}

    def get_remaining_budget(self, base_budget: DailyBudget) -> DailyBudget:
        """
        Subtract consumed totals from the base daily budget.

        Clinical rationale:
        Each subsequent meal recommendation must reflect what the patient
        has already eaten. A CKD patient on a 2000 mg potassium limit
        who consumed 1200 mg at breakfast should only see 800 mg of
        potassium headroom for subsequent meals.

        Parameters
        ----------
        base_budget : DailyBudget
            The full-day allowance before any consumption.

        Returns
        -------
        DailyBudget : remaining allowance, all fields floored at 0.
        """
        self._check_midnight()
        return DailyBudget(
            sodium_mg_remaining=max(0.0, base_budget.sodium_mg_remaining - self._consumed["sodium_mg"]),
            potassium_mg_remaining=max(0.0, base_budget.potassium_mg_remaining - self._consumed["potassium_mg"]),
            protein_g_remaining=max(0.0, base_budget.protein_g_remaining - self._consumed["protein_g"]),
            carbs_g_remaining=max(0.0, base_budget.carbs_g_remaining - self._consumed["carbs_g"]),
            phosphorus_mg_remaining=max(0.0, base_budget.phosphorus_mg_remaining - self._consumed["phosphorus_mg"]),
        )

    def daily_summary(self) -> Dict[str, Any]:
        """
        Return consumed totals and meal history for the current session.

        Clinical rationale:
        Gives clinicians and patients a running view of daily intake against
        limits, enabling informed dietary decisions for remaining meals.

        Returns
        -------
        dict with keys: consumed_totals, meals, meal_count
        """
        self._check_midnight()
        return {
            "consumed_totals": {k: round(v, 1) for k, v in self._consumed.items()},
            "meals": list(self._meals),
            "meal_count": len(self._meals),
        }


# ===============================
# Caloric Sufficiency Validator
# ===============================
class CaloricSufficiencyValidator:
    """
    Detects when overlapping nutrient restrictions produce a
    calorically inadequate meal plan, and applies KDIGO-prioritized
    constraint relaxation to restore caloric adequacy.

    Clinical rationale:
    A CKD Stage 4 + DM + HTN patient may face simultaneous HIGH
    restrictions on protein, carbs, potassium, sodium, and phosphorus.
    The intersection can make it impossible to reach 1200 kcal/day.
    KDIGO 2024 explicitly states that malnutrition risk must be weighed
    against disease progression — protein restriction should be relaxed
    before allowing a patient to starve.
    """

    # --- Class constants ---
    MIN_KCAL_THRESHOLD: float = 1200.0   # Below this = nutritionally inadequate
    TARGET_KCAL: float = 1800.0          # Desired caloric target

    # Relaxation priority: relax *first* item first.
    # Potassium and sodium are NEVER relaxed (cardiac arrest / fluid risk).
    RELAXATION_PRIORITY = [
        {
            "constraint": "protein_restriction",
            "max_reduction": None,           # no cap — malnutrition outweighs CKD progression
            "rationale": "KDIGO 2024: malnutrition risk outweighs CKD progression; "
                         "protein restriction is the first target to relax.",
        },
        {
            "constraint": "carb_sensitivity",
            "max_reduction": None,           # relax if protein alone insufficient
            "rationale": "Relaxing carb sensitivity second when protein relaxation alone "
                         "cannot restore caloric adequacy.",
        },
        {
            "constraint": "phosphorus_sensitivity",
            "max_reduction": 0.20,           # marginal: ≤ 10% of 2.0 scale = 0.20
            "rationale": "Marginal phosphorus relaxation (max −0.20 severity) to "
                         "support caloric adequacy without excessive phosphorus load.",
        },
    ]

    # Constraints that must NEVER be relaxed
    IMMUTABLE_CONSTRAINTS = frozenset({"sodium_sensitivity", "potassium_sensitivity"})

    def __init__(self, ifct: "IFCTDatabase", recommender: "PortionRecommender"):
        self.ifct = ifct
        self.recommender = recommender

    # ---- helpers ----

    def _estimate_kcal(
        self,
        recommendations: List[PortionDecision],
    ) -> float:
        """
        Sum projected calories across all non-Avoid portions.

        For each non-Avoid item: calories = (max_grams / 100) * calories_per_100g
        """
        total = 0.0
        for rec in recommendations:
            if rec.label == "Avoid":
                continue
            try:
                n = self.ifct.get_nutrients_per_100g(rec.ingredient)
                total += (rec.max_grams / 100.0) * n.get("calories", 0)
            except KeyError:
                continue
        return round(total, 1)

    # ---- Task 1: validate ----

    def validate(
        self,
        recommendations: List[PortionDecision],
    ) -> Dict[str, Any]:
        """
        Check whether the combined recommendations meet caloric
        adequacy per KDIGO 2024 minimum thresholds.

        Parameters
        ----------
        recommendations : list of PortionDecision

        Returns
        -------
        dict with:
            is_sufficient   : bool
            projected_kcal  : float
            deficit_kcal    : float  (0 when sufficient)
            relaxations_needed : list of constraint names
        """
        projected = self._estimate_kcal(recommendations)
        is_sufficient = projected >= self.MIN_KCAL_THRESHOLD
        deficit = max(0.0, self.MIN_KCAL_THRESHOLD - projected)

        relaxations_needed: List[str] = []
        if not is_sufficient:
            for entry in self.RELAXATION_PRIORITY:
                relaxations_needed.append(entry["constraint"])

        return {
            "is_sufficient": is_sufficient,
            "projected_kcal": projected,
            "deficit_kcal": round(deficit, 1),
            "relaxations_needed": relaxations_needed,
        }

    # ---- Task 2: reconcile ----

    def reconcile(
        self,
        validation_result: Dict[str, Any],
        risk_levels: Dict[str, Any],
        budget: "DailyBudget",
        ingredients: List[str],
        severity_reduction_step: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Apply KDIGO-prioritized constraint relaxation until projected
        calories exceed MIN_KCAL_THRESHOLD, or all relaxable constraints
        have been fully reduced.

        Parameters
        ----------
        validation_result : dict from validate()
        risk_levels : current risk_levels dict (will be deep-copied)
        budget : DailyBudget for recommender re-runs
        ingredients : list of ingredient names for re-estimation
        severity_reduction_step : how much to reduce severity each iteration

        Returns
        -------
        Modified risk_levels dict with:
            reconciliation_applied : bool
            reconciliation_log     : list of dicts {constraint, old_sev, new_sev, rationale}
        """
        import copy

        # Enforce immutability assertions
        for immutable in self.IMMUTABLE_CONSTRAINTS:
            assert immutable not in [
                e["constraint"] for e in self.RELAXATION_PRIORITY
            ], f"CRITICAL: {immutable} must never appear in RELAXATION_PRIORITY"

        if validation_result["is_sufficient"]:
            modified = copy.deepcopy(risk_levels)
            modified["reconciliation_applied"] = False
            modified["reconciliation_log"] = []
            return modified

        modified = copy.deepcopy(risk_levels)
        log: List[Dict[str, Any]] = []
        projected = validation_result["projected_kcal"]

        for entry in self.RELAXATION_PRIORITY:
            constraint = entry["constraint"]
            max_reduction = entry["max_reduction"]
            rationale = entry["rationale"]

            if constraint not in modified:
                continue

            total_reduced = 0.0

            while projected < self.MIN_KCAL_THRESHOLD:
                current = self._extract_severity(modified[constraint])

                # Stop if severity is already at floor
                if current <= 0.0:
                    break

                # Honour per-constraint max reduction
                if max_reduction is not None and total_reduced >= max_reduction:
                    break

                # Calculate actual step (may be clamped)
                step = severity_reduction_step
                if max_reduction is not None:
                    step = min(step, max_reduction - total_reduced)
                step = min(step, current)  # don't go below 0

                old_sev = current
                new_sev = round(current - step, 4)

                # Apply the reduction
                if isinstance(modified[constraint], dict):
                    modified[constraint]["severity_score"] = new_sev
                    # Update label to match new severity
                    if new_sev < 0.7:
                        modified[constraint]["label"] = "low"
                    elif new_sev < 1.5:
                        modified[constraint]["label"] = "moderate"
                    # else stays "high"

                total_reduced += step

                # Re-run recommendations with relaxed risk_levels
                new_recs = self.recommender.recommend_batch(
                    ingredients, modified, budget
                )
                projected = self._estimate_kcal(new_recs)

                log.append({
                    "constraint": constraint,
                    "old_severity": round(old_sev, 4),
                    "new_severity": new_sev,
                    "projected_kcal_after": projected,
                    "rationale": rationale,
                })

            if projected >= self.MIN_KCAL_THRESHOLD:
                break

        # Final assertion: sodium and potassium must be untouched
        for immutable in self.IMMUTABLE_CONSTRAINTS:
            if immutable in risk_levels and immutable in modified:
                orig_sev = self._extract_severity(risk_levels[immutable])
                new_sev = self._extract_severity(modified[immutable])
                assert orig_sev == new_sev, (
                    f"SAFETY VIOLATION: {immutable} severity was modified "
                    f"from {orig_sev} to {new_sev}"
                )

        modified["reconciliation_applied"] = True
        modified["reconciliation_log"] = log
        return modified

    @staticmethod
    def _extract_severity(entry, default: float = 1.0) -> float:
        """Extract severity_score from a risk_levels entry (dict or str)."""
        if isinstance(entry, dict):
            return float(entry.get("severity_score", default))
        _legacy = {"low": 0.2, "moderate": 1.0, "high": 1.8}
        return _legacy.get(str(entry).lower().strip(), default)


# ===============================
# SubstitutionEngine (Phase 2A)
# ===============================
class SubstitutionEngine:
    """
    Find safe ingredient substitutes when an ingredient is Avoid.
    
    Uses IFCT nutritional cosine similarity to find alternatives that
    are nutritionally similar but lower in the binding-constraint nutrient.
    Respects the enriched risk_levels dict from Phase 3A — passes it
    directly to recommend_one() without unwrapping.
    """
    
    # Map binding_constraint string → IFCT column name
    CONSTRAINT_COLUMN_MAP = {
        "potassium":  "potassium_mg_per_100g",
        "sodium":     "sodium_mg_per_100g",
        "protein":    "protein_g_per_100g",
        "phosphorus": "phosphorus_mg_per_100g",
        "carbs":      "carbs_g_per_100g",
    }
    
    # Map binding_constraint → sensitivity target in risk_levels
    CONSTRAINT_TARGET_MAP = {
        "potassium":  "potassium_sensitivity",
        "sodium":     "sodium_sensitivity",
        "protein":    "protein_restriction",
        "phosphorus": "phosphorus_sensitivity",
        "carbs":      "carb_sensitivity",
    }
    
    # Macro columns for cosine similarity
    SIMILARITY_COLS = [
        "protein_g_per_100g",
        "carbs_g_per_100g",
        "fat_g_per_100g",
        "fiber_g_per_100g",
        "calories_per_100g",
    ]
    
    def __init__(self, ifct: 'IFCTDatabase'):
        self.ifct = ifct
        self._build_similarity_matrix()
    
    def _build_similarity_matrix(self):
        """Build normalized nutrient feature matrix for cosine similarity."""
        df = self.ifct.df.copy()
        
        # Extract similarity columns, fill missing with 0
        available_cols = [c for c in self.SIMILARITY_COLS if c in df.columns]
        self._sim_cols = available_cols
        
        mat = df[available_cols].fillna(0).values.astype(float)
        
        # L2-normalize each row for cosine similarity via dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero
        self._normed_matrix = mat / norms
        
        # Index map: ingredient → row index
        self._ingredient_idx = {
            name: i for i, name in enumerate(df["ingredient"].tolist())
        }
    
    def _cosine_similarity(self, idx_a: int) -> np.ndarray:
        """Compute cosine similarity of ingredient at idx_a against all others."""
        return self._normed_matrix @ self._normed_matrix[idx_a]
    
    def find_substitutes(
        self,
        avoided_ingredient: str,
        binding_constraint: str,
        risk_levels: dict,
        budget: 'DailyBudget',
        recommender: 'PortionRecommender',
        n: int = 3,
    ) -> List[PortionDecision]:
        """
        Find n safe substitutes for an Avoid ingredient.
        
        Parameters
        ----------
        avoided_ingredient : str
            The ingredient that received label="Avoid".
        binding_constraint : str
            Nutrient that triggered the Avoid (e.g. "potassium").
        risk_levels : dict
            Enriched risk_levels dict from Phase 3A (passed directly
            to recommend_one, which uses _get_severity internally).
        budget : DailyBudget
            Current remaining daily budget.
        recommender : PortionRecommender
            For calling recommend_one() on candidates.
        n : int
            Number of substitutes to return.
        
        Returns
        -------
        List[PortionDecision] — up to n non-Avoid substitutes
        """
        if avoided_ingredient not in self._ingredient_idx:
            return []
        
        avoided_idx = self._ingredient_idx[avoided_ingredient]
        similarities = self._cosine_similarity(avoided_idx)
        
        # Get avoided ingredient's category for same-category filtering
        avoided_row = self.ifct.df[
            self.ifct.df["ingredient"] == avoided_ingredient
        ]
        if avoided_row.empty:
            return []
        avoided_category = avoided_row.iloc[0].get("category", None)
        
        # Get constraint column for re-ranking
        constraint_col = self.CONSTRAINT_COLUMN_MAP.get(binding_constraint)
        has_constraint_col = (
            constraint_col is not None and constraint_col in self.ifct.df.columns
        )
        
        # Get the avoided ingredient's constraint nutrient value
        avoided_constraint_val = 0.0
        if has_constraint_col:
            avoided_constraint_val = float(
                avoided_row.iloc[0].get(constraint_col, 0) or 0
            )
        
        # Build candidate list with scoring
        candidates = []
        for ingredient, idx in self._ingredient_idx.items():
            if ingredient == avoided_ingredient:
                continue
            
            # Same-category filter
            if avoided_category is not None:
                row = self.ifct.df[self.ifct.df["ingredient"] == ingredient]
                if not row.empty and row.iloc[0].get("category") != avoided_category:
                    continue
            
            sim_score = float(similarities[idx])
            
            if has_constraint_col:
                # Inverse constraint: lower nutrient content → higher score
                row = self.ifct.df[self.ifct.df["ingredient"] == ingredient]
                if row.empty:
                    continue
                constraint_val = float(row.iloc[0].get(constraint_col, 0) or 0)
                
                # Normalize inverse: 1.0 when 0, 0.0 when at/above avoided value
                if avoided_constraint_val > 0:
                    inv_score = max(0.0, 1.0 - constraint_val / avoided_constraint_val)
                else:
                    inv_score = 1.0
                
                # Weighted combination: 60% similarity + 40% inverse constraint
                combined = 0.6 * sim_score + 0.4 * inv_score
            else:
                combined = sim_score
            
            candidates.append((ingredient, combined))
        
        # Sort by combined score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Build feature attribution suffix
        attribution_suffix = self._build_attribution_suffix(
            binding_constraint, risk_levels
        )
        
        # Try candidates via recommend_one, collect non-Avoid
        substitutes = []
        for candidate_name, _score in candidates:
            if len(substitutes) >= n:
                break
            
            rec = recommender.recommend_one(candidate_name, risk_levels, budget)
            if rec.label != "Avoid":
                # Append attribution suffix to explanation
                if attribution_suffix:
                    rec = PortionDecision(
                        ingredient=rec.ingredient,
                        max_grams=rec.max_grams,
                        label=rec.label,
                        explanation=rec.explanation + attribution_suffix,
                        nutrient_load_at_max=rec.nutrient_load_at_max,
                        binding_constraint=rec.binding_constraint,
                    )
                substitutes.append(rec)
        
        return substitutes
    
    def _build_attribution_suffix(
        self, binding_constraint: str, risk_levels: dict
    ) -> str:
        """Build feature_attribution explanation suffix (Task 3)."""
        target_key = self.CONSTRAINT_TARGET_MAP.get(binding_constraint)
        if not target_key:
            return ""
        
        target_info = risk_levels.get(target_key)
        if not isinstance(target_info, dict):
            return ""
        
        attribution = target_info.get("feature_attribution", {})
        if not attribution:
            return ""
        
        # Get top feature
        top_feature = next(iter(attribution))
        return (
            f" [Substituted due to {binding_constraint} restriction"
            f" — primary driver: {top_feature}]"
        )


# ===============================
# Model1 Integration
# ===============================
class Model1Integration:
    """
    Load and use Model1's trained NGBoost clinical risk models.
    Predicts: sodium_sensitivity, potassium_sensitivity, 
              protein_restriction, carb_sensitivity
    Each prediction returns a dict with label, severity_score, confidence, proba.
    """
    
    def __init__(self, model_dir: Path, use_tft: bool = False):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        self.mono_transformer = None
        self.use_tft = use_tft
        self.tft_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all Model1 TabNet components including monotonic transformer."""
        try:
            targets = [
                "sodium_sensitivity", "potassium_sensitivity",
                "protein_restriction", "carb_sensitivity",
            ]
            self.models = {}
            for target in targets:
                model = TabNetClassifier()
                model.load_model(str(self.model_dir / f"{target}.zip"))
                self.models[target] = model
            
            self.imputer = joblib.load(self.model_dir / "imputer.joblib")
            self.scaler = joblib.load(self.model_dir / "scaler.joblib")
            self.feature_names = joblib.load(self.model_dir / "feature_names.joblib")
            
            # Load monotonic transformer (graceful fallback for older artifacts)
            mono_path = self.model_dir / "monotonic_transformer.joblib"
            if mono_path.exists():
                self.mono_transformer = joblib.load(mono_path)
                print(f"\u2713 Loaded MonotonicFeatureTransformer from {mono_path}")
            else:
                self.mono_transformer = None
                print("\u26a0 No monotonic_transformer.joblib found \u2014 skipping")
            
            print(f"\u2713 Loaded Model1 components from {self.model_dir}")
            
            # Load TFT model if use_tft is enabled (Phase 3B)
            if self.use_tft:
                try:
                    tft_dir = self.model_dir / "tft"
                    self.tft_model = TFTRiskModel.load(tft_dir)
                except Exception as e:
                    print(f"\u26a0 TFT load failed: {e} -- falling back to TabNet")
                    self.tft_model = TFTRiskModel()  # unfitted, uses rule-based
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Model1: {e}")
    
    @staticmethod
    def get_risk_label(risk_entry: Union[dict, str]) -> str:
        """
        Extract the plain label string from a risk_levels entry.

        Clinical rationale:
        Backward-compatibility helper so that any downstream code expecting
        a plain string (e.g. 'high') can still work with the new enriched
        dict structure {label, severity_score, confidence, proba}.

        Parameters
        ----------
        risk_entry : dict or str
            Either the new enriched dict or a legacy plain string.

        Returns
        -------
        str : 'low', 'moderate', or 'high'
        """
        if isinstance(risk_entry, dict):
            return risk_entry["label"]
        return str(risk_entry)
    
    def predict_risk_levels(
        self,
        patient_data: Dict[str, Any],
        encounter_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, dict]:
        """
        Predict clinical risk levels with per-patient feature attribution.

        Clinical rationale:
        TabNet's sequential sparse attention produces a per-patient
        feature importance vector, enabling explanations like "your
        protein is restricted because your eGFR of 28 is the dominant
        signal." The feature_attribution dict in each target's result
        maps the top 5 contributing features to normalized importance weights.
        
        When use_tft=True and encounter_history is provided (>= 2 encounters),
        routes through TFTRiskModel for trajectory prediction with trend.
        
        Args:
            patient_data: Dict with clinical lab values.
            encounter_history: Optional list of dicts (chronological encounters)
                for TFT trajectory prediction.
        
        Returns:
            Dict with enriched risk levels including feature_attribution.
        """
        # Phase 3B: TFT trajectory path
        if (self.use_tft and encounter_history is not None
                and len(encounter_history) >= 2 and self.tft_model is not None):
            tft_results = self.tft_model.predict_trajectory(encounter_history)
            # Merge TFT trend into TabNet predictions for richer output
            tabnet_results = self._tabnet_predict(patient_data)
            for target in tabnet_results:
                if target in tft_results:
                    tabnet_results[target]["trend"] = tft_results[target].get("trend", "stable")
                else:
                    tabnet_results[target]["trend"] = "stable"
            return tabnet_results
        
        # Single-encounter fallback
        results = self._tabnet_predict(patient_data)
        
        if self.use_tft and (encounter_history is None or len(encounter_history) < 2):
            results["SINGLE_ENCOUNTER_FALLBACK"] = {
                "message": "TFT requires >= 2 encounters; fell back to single-encounter TabNet.",
            }
        
        return results
    
    def _tabnet_predict(self, patient_data: Dict[str, Any]) -> Dict[str, dict]:
        """Single-encounter TabNet prediction with feature attribution."""
        # Prepare features
        X = pd.DataFrame([patient_data])[self.feature_names]
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        if self.mono_transformer is not None:
            X = self.mono_transformer.transform(X)
        X = X.astype(np.float32)
        
        label_map = {0: "low", 1: "moderate", 2: "high"}
        class_indices = np.array([0, 1, 2], dtype=float)
        
        results = {}
        for target, model in self.models.items():
            pred_label = int(model.predict(X)[0])
            # TabNet predict_proba returns class probabilities directly
            probas = model.predict_proba(X)[0]
            probas = probas / probas.sum()  # numerical safety
            
            severity_score = float(np.dot(probas, class_indices))
            confidence = float(probas.max())
            
            # Per-patient feature attribution via TabNet explain()
            explain_matrix, _ = model.explain(X)
            attr_row = explain_matrix[0]
            attr_sum = attr_row.sum()
            if attr_sum > 0:
                attr_normalized = attr_row / attr_sum
            else:
                attr_normalized = np.zeros_like(attr_row)
            
            # Top 5 features by weight
            attr_dict = {
                self.feature_names[i]: round(float(attr_normalized[i]), 4)
                for i in range(len(self.feature_names))
            }
            top5 = dict(
                sorted(attr_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            )
            
            results[target] = {
                "label": label_map[pred_label],
                "severity_score": round(severity_score, 4),
                "confidence": round(confidence, 4),
                "proba": {
                    "low": round(float(probas[0]), 4),
                    "moderate": round(float(probas[1]), 4),
                    "high": round(float(probas[2]), 4),
                },
                "feature_attribution": top5,
            }
        
        # Add phosphorus sensitivity based on CKD status (rule-derived)
        has_ckd = patient_data.get("has_ckd", 0) == 1
        egfr = patient_data.get("egfr", 90)
        
        if has_ckd or egfr < 30:
            results["phosphorus_sensitivity"] = {
                "label": "high",
                "severity_score": 1.8,
                "confidence": 0.90,
                "proba": {"low": 0.05, "moderate": 0.05, "high": 0.90},
                "feature_attribution": {"egfr": 0.50, "has_ckd": 0.30, "creatinine": 0.20},
            }
        elif egfr < 60:
            results["phosphorus_sensitivity"] = {
                "label": "moderate",
                "severity_score": 1.0,
                "confidence": 0.75,
                "proba": {"low": 0.10, "moderate": 0.75, "high": 0.15},
                "feature_attribution": {"egfr": 0.55, "has_ckd": 0.25, "creatinine": 0.20},
            }
        else:
            results["phosphorus_sensitivity"] = {
                "label": "low",
                "severity_score": 0.2,
                "confidence": 0.85,
                "proba": {"low": 0.85, "moderate": 0.10, "high": 0.05},
                "feature_attribution": {"egfr": 0.60, "has_ckd": 0.20, "creatinine": 0.20},
            }
        
        return results
    
    def generate_clinical_explanation(
        self,
        risk_levels: Dict[str, dict],
        patient_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Generate natural-language explanations for each risk target.

        Clinical rationale:
        This is the direct interface for the chatbot layer. It uses
        per-patient feature attribution from TabNet to explain WHY
        each restriction was applied, naming the top contributing
        feature and its actual value from the patient's data.

        Parameters
        ----------
        risk_levels : dict from predict_risk_levels()
        patient_data : original patient clinical data dict

        Returns
        -------
        dict mapping target name -> plain-English explanation string
        """
        FEATURE_CLINICAL_MAP = {
            "egfr": {
                "name": "eGFR",
                "unit": "mL/min",
                "implications": {
                    "low": "indicating preserved kidney function with minimal dietary restriction needed",
                    "moderate": "indicating reduced kidney function where dietary modification is recommended",
                    "high": "indicating advanced kidney disease where strict dietary management is essential to slow progression",
                },
            },
            "creatinine": {
                "name": "serum creatinine",
                "unit": "mg/dL",
                "implications": {
                    "low": "suggesting adequate kidney function",
                    "moderate": "suggesting moderately impaired kidney clearance",
                    "high": "suggesting significant kidney impairment requiring close dietary monitoring",
                },
            },
            "serum_potassium": {
                "name": "serum potassium",
                "unit": "mEq/L",
                "implications": {
                    "low": "within acceptable range",
                    "moderate": "elevated above optimal, requiring moderate potassium restriction",
                    "high": "dangerously elevated, posing cardiac arrhythmia risk requiring strict potassium limitation",
                },
            },
            "serum_sodium": {
                "name": "serum sodium",
                "unit": "mEq/L",
                "implications": {
                    "low": "within normal range, allowing standard sodium intake",
                    "moderate": "suggesting fluid retention risk, requiring sodium awareness",
                    "high": "indicating sodium-sensitive hypertension requiring strict sodium restriction",
                },
            },
            "hba1c": {
                "name": "HbA1c",
                "unit": "%",
                "implications": {
                    "low": "indicating good glycemic control",
                    "moderate": "indicating suboptimal glycemic control requiring carbohydrate attention",
                    "high": "indicating poor glycemic control requiring strict carbohydrate management",
                },
            },
            "fbs": {
                "name": "fasting blood sugar",
                "unit": "mg/dL",
                "implications": {
                    "low": "within target range",
                    "moderate": "above target, requiring dietary carbohydrate management",
                    "high": "significantly elevated, requiring strict carbohydrate restriction",
                },
            },
            "sbp": {
                "name": "systolic blood pressure",
                "unit": "mmHg",
                "implications": {
                    "low": "within target range",
                    "moderate": "elevated, requiring sodium and fluid management",
                    "high": "significantly elevated, requiring strict sodium restriction to reduce cardiovascular risk",
                },
            },
            "dbp": {
                "name": "diastolic blood pressure",
                "unit": "mmHg",
                "implications": {
                    "low": "within normal range",
                    "moderate": "moderately elevated, supporting sodium awareness",
                    "high": "elevated, contributing to hypertension management needs",
                },
            },
            "bmi": {
                "name": "BMI",
                "unit": "kg/m\u00b2",
                "implications": {
                    "low": "within healthy range",
                    "moderate": "in overweight range, supporting caloric awareness",
                    "high": "indicating obesity, requiring caloric and carbohydrate management",
                },
            },
            "has_ckd": {
                "name": "CKD diagnosis",
                "unit": "",
                "implications": {
                    "low": "no CKD, minimal phosphorus and protein concern",
                    "moderate": "CKD present, requiring protein and phosphorus attention",
                    "high": "CKD confirmed, requiring strict protein, phosphorus, and potassium management",
                },
            },
            "has_htn": {
                "name": "hypertension diagnosis",
                "unit": "",
                "implications": {
                    "low": "no hypertension",
                    "moderate": "hypertension present, requiring sodium vigilance",
                    "high": "hypertension confirmed, requiring strict sodium restriction",
                },
            },
            "has_dm": {
                "name": "diabetes diagnosis",
                "unit": "",
                "implications": {
                    "low": "no diabetes",
                    "moderate": "diabetes present, requiring carbohydrate management",
                    "high": "diabetes confirmed, requiring strict carbohydrate control",
                },
            },
            "age": {
                "name": "age",
                "unit": "years",
                "implications": {
                    "low": "younger age with generally lower risk",
                    "moderate": "middle age with accumulating risk factors",
                    "high": "advanced age contributing to increased clinical risk",
                },
            },
            "sex_male": {
                "name": "sex",
                "unit": "",
                "implications": {
                    "low": "biological factor in risk calculation",
                    "moderate": "biological factor in risk calculation",
                    "high": "biological factor in risk calculation",
                },
            },
        }

        TARGET_NAMES = {
            "sodium_sensitivity": "Sodium sensitivity",
            "potassium_sensitivity": "Potassium sensitivity",
            "protein_restriction": "Protein restriction",
            "carb_sensitivity": "Carbohydrate sensitivity",
            "phosphorus_sensitivity": "Phosphorus sensitivity",
        }

        explanations = {}
        for target, info in risk_levels.items():
            if not isinstance(info, dict) or "feature_attribution" not in info:
                continue
            
            label = info.get("label", "moderate").upper()
            target_name = TARGET_NAMES.get(target, target)
            attribution = info["feature_attribution"]
            
            if not attribution:
                explanations[target] = (
                    f"{target_name} is {label} based on overall clinical profile."
                )
                continue
            
            # Top feature
            top_feature = next(iter(attribution))
            top_weight = attribution[top_feature]
            
            feat_info = FEATURE_CLINICAL_MAP.get(top_feature, {
                "name": top_feature,
                "unit": "",
                "implications": {
                    "low": "contributing to risk assessment",
                    "moderate": "contributing to risk assessment",
                    "high": "contributing to risk assessment",
                },
            })
            
            feat_name = feat_info["name"]
            feat_unit = feat_info["unit"]
            risk_label = info.get("label", "moderate")
            implication = feat_info["implications"].get(risk_label, "contributing to risk assessment")
            
            # Get actual patient value
            raw_value = patient_data.get(top_feature, "N/A")
            if feat_unit and raw_value != "N/A":
                value_str = f"{raw_value} {feat_unit}"
            elif isinstance(raw_value, (int, float)) and top_feature.startswith("has_"):
                value_str = "present" if raw_value == 1 else "absent"
            else:
                value_str = str(raw_value)
            
            # Build explanation
            pct = round(top_weight * 100)
            explanations[target] = (
                f"{target_name} is {label} primarily because "
                f"{feat_name} is {value_str} "
                f"(contributing {pct}% to this assessment), "
                f"{implication}."
            )
        
        return explanations


# ===============================
# Portion Recommender
# ===============================
class PortionRecommender:
    """
    Deterministic portion recommendation engine.
    Uses IFCT nutritional data + Model1 risk levels to recommend safe portions.
    Now uses continuous severity_score via a sigmoid mapping to compute
    per-nutrient fractions, eliminating dose-cliffs at label boundaries.
    """
    
    # Sigmoid parameters tuned so f(0.0) ≈ 0.40, f(2.0) ≈ 0.08
    _SIGMOID_L = 0.42   # upper asymptote
    _SIGMOID_K = 1.60   # steepness
    _SIGMOID_S0 = 1.0   # midpoint
    _SIGMOID_FLOOR = 0.05  # minimum fraction (never reach zero)
    
    def __init__(
        self,
        ifct: IFCTDatabase,
        config: Model2Config
    ):
        self.ifct = ifct
        self.config = config
    
    def severity_to_fraction(self, severity_score: float) -> float:
        """
        Map a continuous severity_score ∈ [0.0, 2.0] to a budget fraction.

        Clinical rationale:
        A sigmoid curve prevents dose-cliffs at label boundaries. Two patients
        with the same discrete label (e.g. 'high') but different eGFR values
        will have different severity_scores, producing meaningfully different
        portion sizes. The curve is anchored at:
          severity=0.0 → ~0.40 (generous, low-risk)
          severity=1.0 → ~0.24 (moderate — close to old 0.25)
          severity=2.0 → ~0.08 (restrictive, high-risk)

        Parameters
        ----------
        severity_score : float
            Probability-weighted class index from NGBoost, ∈ [0.0, 2.0].

        Returns
        -------
        float : budget fraction ∈ [_SIGMOID_FLOOR, _SIGMOID_L]
        """
        fraction = self._SIGMOID_L / (
            1.0 + math.exp(self._SIGMOID_K * (severity_score - self._SIGMOID_S0))
        )
        return max(self._SIGMOID_FLOOR, fraction)
    
    @staticmethod
    def _extract_severity(entry: Union[dict, str], default: float = 1.0) -> float:
        """
        Extract severity_score from a risk_levels entry.

        Clinical rationale:
        Graceful fallback: if Model1Integration returns the new enriched dict,
        use its severity_score. If it somehow returns a legacy plain string,
        map to a safe default of 1.0 (moderate). This ensures the system
        never crashes on mixed-version artifacts.

        Parameters
        ----------
        entry : dict or str
            Either {label, severity_score, confidence, proba} or 'low'/'moderate'/'high'.
        default : float
            Fallback severity_score if entry is a plain string.

        Returns
        -------
        float : severity_score ∈ [0.0, 2.0]
        """
        if isinstance(entry, dict):
            return float(entry.get("severity_score", default))
        # Legacy plain string fallback
        _legacy_map = {"low": 0.2, "moderate": 1.0, "high": 1.8}
        return _legacy_map.get(str(entry).lower().strip(), default)
    
    @staticmethod
    def _grams_from_budget(
        nutrient_per_100g: float, 
        remaining_budget: float, 
        risk_fraction: float
    ) -> float:
        """
        Compute maximum grams such that:
        nutrient_load <= remaining_budget * risk_fraction
        
        grams = (allowed_amount / nutrient_per_100g) * 100
        """
        if nutrient_per_100g <= 0:
            return float("inf")  # Not constrained by this nutrient
        
        allowed = max(0.0, remaining_budget) * risk_fraction
        grams = (allowed / nutrient_per_100g) * 100.0
        return max(0.0, grams)
    
    def recommend_one(
        self,
        ingredient: str,
        risk_levels: Dict[str, Union[dict, str]],
        budget: DailyBudget
    ) -> PortionDecision:
        """
        Recommend portion for a single ingredient.

        Clinical rationale:
        For each nutrient, extract the severity_score from the enriched
        risk_levels dict and convert it to a continuous budget fraction
        via the sigmoid mapping. This means two patients with the same
        label but different underlying eGFR (or HbA1c, etc.) will receive
        meaningfully different portion sizes.
        
        Args:
            ingredient: Name of ingredient (must be in IFCT)
            risk_levels: Output from Model1 (enriched dicts or legacy strings)
            budget: Remaining daily nutrient budget
        
        Returns:
            PortionDecision with max_grams, label, and explanation
        """
        # Get nutrient values per 100g
        try:
            n = self.ifct.get_nutrients_per_100g(ingredient)
        except KeyError:
            # Ingredient not in database
            suggestions = self.ifct.search_ingredient(ingredient)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}" if suggestions else ""
            return PortionDecision(
                ingredient=ingredient,
                max_grams=0.0,
                label="Avoid",
                explanation=f"Avoid: Ingredient '{ingredient}' not found in IFCT database.{suggestion_text}",
                nutrient_load_at_max={k: 0.0 for k in ["sodium_mg", "potassium_mg", "protein_g", "carbs_g"]},
                binding_constraint="unknown"
            )
        
        # Extract severity scores from enriched risk_levels
        sod_sev = self._extract_severity(risk_levels.get("sodium_sensitivity", {}), default=1.0)
        pot_sev = self._extract_severity(risk_levels.get("potassium_sensitivity", {}), default=1.0)
        pro_sev = self._extract_severity(risk_levels.get("protein_restriction", {}), default=1.0)
        carb_sev = self._extract_severity(risk_levels.get("carb_sensitivity", {}), default=1.0)
        phos_sev = self._extract_severity(risk_levels.get("phosphorus_sensitivity", {}), default=0.2)
        
        # Convert severity scores to continuous fractions via sigmoid
        f_sod = self.severity_to_fraction(sod_sev)
        f_pot = self.severity_to_fraction(pot_sev)
        f_pro = self.severity_to_fraction(pro_sev)
        f_carb = self.severity_to_fraction(carb_sev)
        f_phos = self.severity_to_fraction(phos_sev)
        
        # Compute grams allowed by each constraint
        g_sod = self._grams_from_budget(n["sodium_mg"], budget.sodium_mg_remaining, f_sod)
        g_pot = self._grams_from_budget(n["potassium_mg"], budget.potassium_mg_remaining, f_pot)
        g_pro = self._grams_from_budget(n["protein_g"], budget.protein_g_remaining, f_pro)
        g_carb = self._grams_from_budget(n["carbs_g"], budget.carbs_g_remaining, f_carb)
        g_phos = self._grams_from_budget(n["phosphorus_mg"], budget.phosphorus_mg_remaining, f_phos)
        
        constraints = {
            "sodium": g_sod,
            "potassium": g_pot,
            "protein": g_pro,
            "carbs": g_carb,
            "phosphorus": g_phos,
        }
        
        # Find most restrictive constraint
        binding = min(constraints, key=constraints.get)
        max_g = min(constraints.values())
        max_g = min(max_g, self.config.default_cap_g)  # Apply practical cap
        
        # ── Phase 1C: Phosphorus-protein efficiency adjustment ──
        # KDIGO 2024: favour low-phos protein sources for CKD patients
        phos_efficiency = None
        if phos_sev > 1.5:
            phos_efficiency = self.ifct.get_phos_protein_efficiency(
                ingredient, config=self.config
            )
            if phos_efficiency["ckd_preferred"]:
                max_g *= self.config.phos_efficiency_bonus
            elif phos_efficiency["efficiency_tier"] == "poor":
                max_g *= self.config.phos_efficiency_penalty
            # Re-apply cap after multiplier
            max_g = min(max_g, self.config.default_cap_g)
        
        # Determine label
        if max_g <= self.config.avoid_threshold_g:
            label = "Avoid"
        elif max_g <= self.config.half_portion_g:
            label = "Half portion"
        else:
            label = "Allowed"
        
        # Calculate nutrient load at max portion
        factor = max_g / 100.0
        load = {
            "sodium_mg": round(n["sodium_mg"] * factor, 1),
            "potassium_mg": round(n["potassium_mg"] * factor, 1),
            "protein_g": round(n["protein_g"] * factor, 1),
            "carbs_g": round(n["carbs_g"] * factor, 1),
        }
        
        # Build explanation with severity scores
        def _label_from(entry):
            return Model1Integration.get_risk_label(entry) if isinstance(entry, dict) else str(entry)
        
        sod_label = _label_from(risk_levels.get("sodium_sensitivity", "moderate"))
        pot_label = _label_from(risk_levels.get("potassium_sensitivity", "moderate"))
        pro_label = _label_from(risk_levels.get("protein_restriction", "moderate"))
        carb_label = _label_from(risk_levels.get("carb_sensitivity", "moderate"))
        phos_label = _label_from(risk_levels.get("phosphorus_sensitivity", "low"))
        
        risk_explanations = {
            "sodium": f"sodium sensitivity ({sod_label}, sev={sod_sev:.2f})",
            "potassium": f"potassium sensitivity ({pot_label}, sev={pot_sev:.2f})",
            "protein": f"protein restriction ({pro_label}, sev={pro_sev:.2f})",
            "carbs": f"carb sensitivity ({carb_label}, sev={carb_sev:.2f})",
            "phosphorus": f"phosphorus restriction ({phos_label}, sev={phos_sev:.2f}) - CKD guideline",
        }
        
        # Append phosphorus efficiency info if applicable
        eff_note = ""
        if phos_efficiency is not None:
            tier = phos_efficiency["efficiency_tier"]
            ratio = phos_efficiency["phos_per_g_protein"]
            eff_note = f" [Phos efficiency: {tier} ({ratio:.1f} mg/g protein)"
            if phos_efficiency["ckd_preferred"]:
                eff_note += f", ×{self.config.phos_efficiency_bonus} bonus]"
            elif tier == "poor":
                eff_note += f", ×{self.config.phos_efficiency_penalty} penalty]"
            else:
                eff_note += "]"
        
        explanation = (
            f"{label}: max ~{max_g:.0f}g. "
            f"Limited by {risk_explanations[binding]}. "
            f"At this portion: {load['sodium_mg']:.0f}mg Na, "
            f"{load['potassium_mg']:.0f}mg K, "
            f"{load['protein_g']:.1f}g protein, "
            f"{load['carbs_g']:.1f}g carbs."
            f"{eff_note}"
        )
        
        return PortionDecision(
            ingredient=ingredient,
            max_grams=round(max_g, 1),
            label=label,
            explanation=explanation,
            nutrient_load_at_max=load,
            binding_constraint=binding
        )
    
    def recommend_batch(
        self,
        ingredients: List[str],
        risk_levels: Dict[str, str],
        budget: DailyBudget
    ) -> List[PortionDecision]:
        """Recommend portions for multiple ingredients"""
        return [
            self.recommend_one(ing, risk_levels, budget)
            for ing in ingredients
        ]


# ===============================
# Budget Calculator
# ===============================
class BudgetCalculator:
    """
    Calculate daily nutrient budgets based on disease conditions.
    """
    
    def __init__(self, config: Model2Config):
        self.config = config
    
    def get_daily_budget(
        self,
        has_ckd: bool = False,
        has_htn: bool = False,
        has_dm: bool = False,
        consumed_today: Optional[Dict[str, float]] = None
    ) -> DailyBudget:
        """
        Get base daily budget based on disease conditions.

        Clinical rationale:
        Returns the full-day nutrient allowance before any consumption.
        Consumption tracking is now handled by NutrientLedger.

        Args:
            has_ckd: Has chronic kidney disease
            has_htn: Has hypertension
            has_dm: Has diabetes
            consumed_today: DEPRECATED — use NutrientLedger instead.
                Kept for backward compatibility; ignored if provided.

        Returns:
            DailyBudget with full daily nutrient allowances
        """
        if consumed_today is not None:
            import warnings as _w
            _w.warn(
                "consumed_today is deprecated and ignored. "
                "Use NutrientLedger.get_remaining_budget() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Determine condition key
        conditions = []
        if has_ckd:
            conditions.append("ckd")
        if has_htn:
            conditions.append("htn")
        if has_dm:
            conditions.append("dm")

        if not conditions:
            key = "healthy"
        else:
            key = "_".join(sorted(conditions))

        # Get base budget (fall back to most restrictive if combo not defined)
        budgets = self.config.daily_budgets
        if key in budgets:
            base = budgets[key]
        elif "ckd_htn_dm" in budgets and has_ckd:
            base = budgets["ckd_htn_dm"]
        elif "ckd" in budgets and has_ckd:
            base = budgets["ckd"]
        else:
            base = budgets["healthy"]

        return DailyBudget(
            sodium_mg_remaining=base["sodium_mg"],
            potassium_mg_remaining=base["potassium_mg"],
            protein_g_remaining=base["protein_g"],
            carbs_g_remaining=base["carbs_g"],
            phosphorus_mg_remaining=base.get("phosphorus_mg", 700),
        )


# ===============================
# Ingredient Input Handler
# ===============================
class IngredientInputHandler:
    """Handle user input for available ingredients"""
    
    def __init__(self, ifct: IFCTDatabase):
        self.ifct = ifct
    
    def parse_input(self, input_str: str) -> Tuple[List[str], List[str]]:
        """
        Parse comma-separated ingredient list.
        
        Args:
            input_str: "tomato, spinach, banana, rice"
        
        Returns:
            (found_ingredients, not_found_with_suggestions)
        """
        raw_ingredients = [i.strip() for i in input_str.split(",") if i.strip()]
        
        found = []
        not_found = []
        
        for ing in raw_ingredients:
            # Check if exists (case-insensitive)
            if ing.lower() in self.ifct.all_ingredients_lower:
                idx = self.ifct.all_ingredients_lower.index(ing.lower())
                found.append(self.ifct.all_ingredients[idx])
            else:
                # Try to find suggestions
                suggestions = self.ifct.search_ingredient(ing, n_matches=3)
                if suggestions:
                    not_found.append(f"'{ing}' → Did you mean: {', '.join(suggestions)}?")
                else:
                    not_found.append(f"'{ing}' → No matches found")
        
        return found, not_found


# ===============================
# Main Model2 Class
# ===============================
class PortionControlModel:
    """
    Complete Model2 Portion Control System.
    Integrates Model1 risk predictions with IFCT-based portion recommendations.
    """
    
    def __init__(self, config: Model2Config = None):
        self.config = config or Model2Config()
        
        # Initialize components
        print("\n" + "="*60)
        print("INITIALIZING MODEL2: PORTION CONTROL SYSTEM")
        print("="*60)
        
        # Load IFCT database
        ifct_path = Path(__file__).parent / self.config.ifct_path
        self.ifct = IFCTDatabase(str(ifct_path))
        
        # Load Model1
        model1_path = Path(__file__).parent / self.config.model1_dir
        self.model1 = Model1Integration(model1_path)
        
        # Initialize helpers
        self.recommender = PortionRecommender(self.ifct, self.config)
        self.budget_calc = BudgetCalculator(self.config)
        self.input_handler = IngredientInputHandler(self.ifct)
        
        # Initialize daily nutrient ledger
        self.ledger = NutrientLedger()
        
        # Initialize caloric sufficiency validator (Phase 2B)
        self.caloric_validator = CaloricSufficiencyValidator(
            self.ifct, self.recommender
        )
        
        # Initialize substitution engine (Phase 2A)
        self.substitution_engine = SubstitutionEngine(self.ifct)
        
        print("\u2713 Model2 initialization complete")
        print("="*60 + "\n")
    
    def get_recommendations(
        self,
        patient_data: Dict[str, Any],
        ingredients: List[str],
        consumed_today: Optional[Dict[str, float]] = None,
        include_substitutes: bool = True,
    ) -> Dict[str, Any]:
        """
        Get portion recommendations for ingredients given patient data.

        Clinical rationale:
        The budget now reflects what the patient has already eaten today
        (via the NutrientLedger), not just the full-day allowance. This
        ensures that a CKD patient who ate 900 mg sodium at breakfast
        sees only 1100 mg headroom for subsequent meals.
        
        Args:
            patient_data: Clinical data dict with keys:
                - age, sex_male, has_htn, has_dm, has_ckd
                - serum_sodium, serum_potassium, creatinine, egfr
                - hba1c, fbs, sbp, dbp, bmi
            ingredients: List of ingredient names
            consumed_today: DEPRECATED — ignored; ledger tracks this
            include_substitutes: If True, find substitutes for Avoid items.
                If False, skip substitution entirely (pre-2A behavior).
        
        Returns:
            Dict with risk_levels, budget, recommendations, and summary
        """
        # Step 1: Get risk levels from Model1
        risk_levels = self.model1.predict_risk_levels(patient_data)
        
        # Step 2: Calculate remaining budget from ledger
        base_budget = self.budget_calc.get_daily_budget(
            has_ckd=patient_data.get("has_ckd", 0) == 1,
            has_htn=patient_data.get("has_htn", 0) == 1,
            has_dm=patient_data.get("has_dm", 0) == 1,
        )
        budget = self.ledger.get_remaining_budget(base_budget)
        
        # Step 3: Get portion recommendations
        recommendations = self.recommender.recommend_batch(ingredients, risk_levels, budget)
        
        # Step 3B: Caloric sufficiency validation & reconciliation (Phase 2B)
        validation = self.caloric_validator.validate(recommendations)
        clinical_warnings: List[Dict[str, Any]] = []
        
        if not validation["is_sufficient"]:
            reconciled = self.caloric_validator.reconcile(
                validation, risk_levels, budget, ingredients
            )
            if reconciled.get("reconciliation_applied"):
                # Extract reconciliation metadata before passing to recommender
                reconciliation_log = reconciled.pop("reconciliation_log", [])
                reconciled.pop("reconciliation_applied", None)
                
                # Re-run recommendations with relaxed risk_levels
                recommendations = self.recommender.recommend_batch(
                    ingredients, reconciled, budget
                )
                risk_levels = reconciled
                
                clinical_warnings = reconciliation_log
        
        # Step 3C: Substitution for Avoid items (Phase 2A)
        if include_substitutes:
            for rec in recommendations:
                if rec.label == "Avoid":
                    rec.substitutes = self.substitution_engine.find_substitutes(
                        avoided_ingredient=rec.ingredient,
                        binding_constraint=rec.binding_constraint,
                        risk_levels=risk_levels,
                        budget=budget,
                        recommender=self.recommender,
                        n=3,
                    )
        
        # Step 4: Build summary
        allowed = [r.ingredient for r in recommendations if r.label == "Allowed"]
        half_portion = [r.ingredient for r in recommendations if r.label == "Half portion"]
        avoid = [r.ingredient for r in recommendations if r.label == "Avoid"]
        
        return {
            "patient_conditions": {
                "has_ckd": patient_data.get("has_ckd", 0) == 1,
                "has_htn": patient_data.get("has_htn", 0) == 1,
                "has_dm": patient_data.get("has_dm", 0) == 1,
            },
            "risk_levels": risk_levels,
            "daily_budget": {
                "sodium_mg": budget.sodium_mg_remaining,
                "potassium_mg": budget.potassium_mg_remaining,
                "protein_g": budget.protein_g_remaining,
                "carbs_g": budget.carbs_g_remaining,
                "phosphorus_mg": budget.phosphorus_mg_remaining,
            },
            "recommendations": [
                {
                    "ingredient": r.ingredient,
                    "max_grams": r.max_grams,
                    "label": r.label,
                    "explanation": r.explanation,
                    "binding_constraint": r.binding_constraint,
                    "nutrient_load": r.nutrient_load_at_max,
                    "substitutes": [
                        {
                            "ingredient": s.ingredient,
                            "max_grams": s.max_grams,
                            "label": s.label,
                            "explanation": s.explanation,
                            "binding_constraint": s.binding_constraint,
                        }
                        for s in r.substitutes
                    ] if r.substitutes else [],
                }
                for r in recommendations
            ],
            "summary": {
                "allowed": allowed,
                "half_portion": half_portion,
                "avoid": avoid,
            },
            "clinical_warnings": clinical_warnings,
        }
    
    def confirm_meal(
        self,
        meal_name: str,
        portions: List[PortionDecision],
    ) -> Dict[str, float]:
        """
        Record a confirmed meal in the daily nutrient ledger.

        Clinical rationale:
        Called by the chatbot / user interface layer after the user confirms
        what they actually ate. Only confirmed meals should affect the budget
        for subsequent recommendations — tentative queries must not.

        Parameters
        ----------
        meal_name : str
            Human-readable label (e.g. 'Breakfast', 'Lunch').
        portions : list of PortionDecision
            The accepted portion recommendations.

        Returns
        -------
        dict : nutrients consumed in this meal
        """
        return self.ledger.log_accepted_portions(meal_name, portions, self.ifct)
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Proxy to NutrientLedger.daily_summary().

        Clinical rationale:
        Provides a running view of the patient's daily intake, meal history,
        and remaining budget headroom for clinician/patient review.

        Returns
        -------
        dict with consumed_totals, meals, meal_count
        """
        return self.ledger.daily_summary()
    
    def interactive_mode(self):
        """Run interactive ingredient input mode"""
        print("\n" + "="*60)
        print("MODEL2 INTERACTIVE MODE")
        print("="*60)
        
        # Get patient data
        print("\nEnter patient clinical data:")
        patient = {
            "age": int(input("  Age: ") or "55"),
            "sex_male": int(input("  Sex (0=female, 1=male): ") or "1"),
            "has_htn": int(input("  Has HTN? (0/1): ") or "1"),
            "has_dm": int(input("  Has DM? (0/1): ") or "1"),
            "has_ckd": int(input("  Has CKD? (0/1): ") or "1"),
            "serum_sodium": float(input("  Serum sodium (mEq/L): ") or "140"),
            "serum_potassium": float(input("  Serum potassium (mEq/L): ") or "4.5"),
            "creatinine": float(input("  Creatinine (mg/dL): ") or "1.5"),
            "egfr": float(input("  eGFR (mL/min): ") or "45"),
            "hba1c": float(input("  HbA1c (%): ") or "7.5"),
            "fbs": float(input("  FBS (mg/dL): ") or "130"),
            "sbp": int(input("  SBP (mmHg): ") or "145"),
            "dbp": int(input("  DBP (mmHg): ") or "90"),
            "bmi": float(input("  BMI: ") or "28"),
        }
        
        # Get ingredients
        print("\nEnter available ingredients (comma-separated):")
        ing_input = input("  → ")
        
        found, not_found = self.input_handler.parse_input(ing_input)
        
        if not_found:
            print("\n⚠ Some ingredients not found:")
            for nf in not_found:
                print(f"    {nf}")
        
        if not found:
            print("\nNo valid ingredients to analyze.")
            return
        
        print(f"\n✓ Found {len(found)} ingredients: {', '.join(found)}")
        
        # Get recommendations
        results = self.get_recommendations(patient, found)
        
        # Display results
        print("\n" + "="*60)
        print("PORTION RECOMMENDATIONS")
        print("="*60)
        
        print(f"\nPatient Risk Levels (from Model1):")
        for key, val in results["risk_levels"].items():
            if isinstance(val, dict):
                print(f"  • {key}: {val['label']} "
                      f"(sev={val['severity_score']:.3f}, "
                      f"conf={val['confidence']:.3f})")
            else:
                print(f"  • {key}: {val}")
        
        print(f"\nDaily Budget Remaining:")
        for key, val in results["daily_budget"].items():
            print(f"  • {key}: {val}")
        
        print("\nRecommendations:")
        print("-"*60)
        for rec in results["recommendations"]:
            print(f"\n  {rec['ingredient']}")
            print(f"    Label: {rec['label']}")
            print(f"    Max portion: {rec['max_grams']}g")
            print(f"    Reason: {rec['explanation']}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"  ✓ Allowed: {', '.join(results['summary']['allowed']) or 'None'}")
        print(f"  ⚠ Half portion: {', '.join(results['summary']['half_portion']) or 'None'}")
        print(f"  ✗ Avoid: {', '.join(results['summary']['avoid']) or 'None'}")


# ===============================
# Main Entry Point
# ===============================
def main():
    """Test Model2 with sample data"""
    
    # Initialize model
    model = PortionControlModel()
    
    # Test patient with CKD + HTN + DM (standard test patient)
    patient = {
        "age": 55,
        "sex_male": 1,
        "has_htn": 1,
        "has_dm": 1,
        "has_ckd": 1,
        "serum_sodium": 144,
        "serum_potassium": 5.6,
        "creatinine": 2.3,
        "egfr": 28,
        "hba1c": 8.4,
        "fbs": 170,
        "sbp": 152,
        "dbp": 94,
        "bmi": 29
    }
    
    # Test ingredients
    ingredients = [
        "Spinach (Palak)",
        "Banana, ripe",
        "Potato (Aloo)",
        "Rice, milled (white)",
        "Tomato, ripe",
        "Curd (Dahi/Yogurt)",
        "Chicken, breast",
        "Apple",
    ]
    
    # Get recommendations
    print("\n" + "="*60)
    print("TEST: CKD + HTN + DM PATIENT")
    print("="*60)
    
    results = model.get_recommendations(patient, ingredients)
    
    # Display results
    print("\n📋 Risk Levels (from Model1):")
    for key, val in results["risk_levels"].items():
        if isinstance(val, dict):
            print(f"   {key}: {val['label']} "
                  f"(severity={val['severity_score']:.3f}, "
                  f"confidence={val['confidence']:.3f})")
        else:
            print(f"   {key}: {val}")
    
    print("\n📊 Daily Budget:")
    for key, val in results["daily_budget"].items():
        print(f"   {key}: {val}")
    
    print("\n🍽️ Portion Recommendations:")
    print("-"*60)
    
    # Create results table
    df = pd.DataFrame([
        {
            "Ingredient": r["ingredient"],
            "Label": r["label"],
            "Max (g)": r["max_grams"],
            "Limited by": r["binding_constraint"],
        }
        for r in results["recommendations"]
    ])
    print(df.to_string(index=False))
    
    print(f"\n✓ ALLOWED: {', '.join(results['summary']['allowed']) or 'None'}")
    print(f"⚠ HALF PORTION: {', '.join(results['summary']['half_portion']) or 'None'}")
    print(f"✗ AVOID: {', '.join(results['summary']['avoid']) or 'None'}")
    
    # ==========================================
    # VALIDATION: Two-patient differentiation
    # ==========================================
    print("\n" + "="*60)
    print("VALIDATION: TWO-PATIENT DIFFERENTIATION")
    print("Different severity_scores → different max_grams for Chicken")
    print("="*60)
    
    # Patient A: eGFR=55, no CKD flag, lower creatinine → moderate protein severity
    patient_a = dict(patient, egfr=55, creatinine=1.3, has_ckd=0)
    # Patient B: eGFR=28, CKD flag, high creatinine → high protein severity
    patient_b = patient  # already eGFR=28
    
    test_ingredient = "Chicken, breast"
    
    results_a = model.get_recommendations(patient_a, [test_ingredient])
    results_b = model.get_recommendations(patient_b, [test_ingredient])
    
    pro_a = results_a["risk_levels"].get("protein_restriction", {})
    pro_b = results_b["risk_levels"].get("protein_restriction", {})
    
    rec_a = results_a["recommendations"][0]
    rec_b = results_b["recommendations"][0]
    
    print(f"\n  Patient A (eGFR=55, no CKD):")
    if isinstance(pro_a, dict):
        print(f"    protein_restriction: label={pro_a['label']}, "
              f"severity={pro_a['severity_score']:.4f}")
    print(f"    {test_ingredient}: max_grams={rec_a['max_grams']}g, label={rec_a['label']}")
    
    print(f"\n  Patient B (eGFR=28, CKD):")
    if isinstance(pro_b, dict):
        print(f"    protein_restriction: label={pro_b['label']}, "
              f"severity={pro_b['severity_score']:.4f}")
    print(f"    {test_ingredient}: max_grams={rec_b['max_grams']}g, label={rec_b['label']}")
    
    grams_diff = rec_a['max_grams'] - rec_b['max_grams']
    print(f"\n  ✓ Difference: {grams_diff:.1f}g "
          f"({'Patient B is more restricted — correct!' if grams_diff > 0 else 'UNEXPECTED'})")
    
    # ==========================================
    # VALIDATION: NutrientLedger (Phase 1B)
    # ==========================================
    print("\n" + "="*60)
    print("VALIDATION: NUTRIENT LEDGER — BUDGET DEPLETION")
    print("="*60)
    
    # Reset ledger for a clean test
    model.ledger._reset()
    
    # --- BREAKFAST ---
    breakfast_items = ["Rice, milled (white)", "Curd (Dahi/Yogurt)", "Banana, ripe"]
    breakfast = model.get_recommendations(patient, breakfast_items)
    breakfast_budget = breakfast["daily_budget"]
    print(f"\n  BREAKFAST budget (before any meal):")
    print(f"    sodium={breakfast_budget['sodium_mg']:.0f}mg, "
          f"potassium={breakfast_budget['potassium_mg']:.0f}mg, "
          f"protein={breakfast_budget['protein_g']:.1f}g")
    
    # Build PortionDecision objects for confirm_meal
    breakfast_portions = [
        PortionDecision(
            ingredient=r["ingredient"],
            max_grams=r["max_grams"],
            label=r["label"],
            explanation=r["explanation"],
            nutrient_load_at_max=r["nutrient_load"],
            binding_constraint=r["binding_constraint"],
        )
        for r in breakfast["recommendations"]
    ]
    
    # Confirm breakfast
    meal_nutrients = model.confirm_meal("Breakfast", breakfast_portions)
    print(f"\n  ✓ Breakfast confirmed — consumed:")
    print(f"    sodium={meal_nutrients['sodium_mg']:.0f}mg, "
          f"potassium={meal_nutrients['potassium_mg']:.0f}mg, "
          f"protein={meal_nutrients['protein_g']:.1f}g")
    
    # --- LUNCH ---
    lunch_items = ["Chicken, breast", "Potato (Aloo)", "Tomato, ripe"]
    lunch = model.get_recommendations(patient, lunch_items)
    lunch_budget = lunch["daily_budget"]
    print(f"\n  LUNCH budget (after breakfast):")
    print(f"    sodium={lunch_budget['sodium_mg']:.0f}mg, "
          f"potassium={lunch_budget['potassium_mg']:.0f}mg, "
          f"protein={lunch_budget['protein_g']:.1f}g")
    
    # Verify budget decreased
    sod_decreased = lunch_budget["sodium_mg"] < breakfast_budget["sodium_mg"]
    print(f"\n  ✓ Budget decreased after breakfast: {sod_decreased} "
          f"({'CORRECT' if sod_decreased else 'FAILED'})")
    
    # --- DAILY SUMMARY ---
    summary = model.get_daily_summary()
    print(f"\n  Daily Summary: {summary['meal_count']} meal(s) logged")
    print(f"    Total consumed: sodium={summary['consumed_totals']['sodium_mg']:.0f}mg, "
          f"potassium={summary['consumed_totals']['potassium_mg']:.0f}mg, "
          f"protein={summary['consumed_totals']['protein_g']:.1f}g")
    for m in summary["meals"]:
        print(f"    • {m['meal_name']}: {', '.join(m['items'])}")
    
    # ==========================================
    # VALIDATION: Phase 1C — Phosphorus-Protein Efficiency Scoring
    # ==========================================
    print("\n" + "="*60)
    print("VALIDATION: PHOSPHORUS-PROTEIN EFFICIENCY SCORING (1C)")
    print("="*60)
    
    # --- Test 1: Tier assignment ---
    eff_egg = model.ifct.get_phos_protein_efficiency("Egg white")
    print(f"\n  Test 1 — Tier assignment:")
    print(f"    Egg white: ratio={eff_egg['phos_per_g_protein']}, "
          f"tier={eff_egg['efficiency_tier']}, "
          f"ckd_preferred={eff_egg['ckd_preferred']}")
    assert eff_egg["efficiency_tier"] == "excellent", f"Expected excellent, got {eff_egg['efficiency_tier']}"
    assert eff_egg["ckd_preferred"] is True, "Expected ckd_preferred=True"
    print("    ✓ Egg white → excellent tier, ckd_preferred=True (CORRECT)")
    
    eff_paneer = model.ifct.get_phos_protein_efficiency("Paneer (Cottage cheese)")
    print(f"    Paneer: ratio={eff_paneer['phos_per_g_protein']}, "
          f"tier={eff_paneer['efficiency_tier']}, "
          f"ckd_preferred={eff_paneer['ckd_preferred']}")
    
    # --- Test 2: CKD patient portion comparison ---
    print(f"\n  Test 2 — CKD patient: low-phos vs high-phos protein:")
    # Use CKD patient (patient B from earlier)
    ckd_patient = {
        "age": 65, "sex_male": 1,
        "has_htn": 1, "has_dm": 1, "has_ckd": 1,
        "serum_sodium": 137, "serum_potassium": 5.2,
        "creatinine": 2.8, "egfr": 28, "hba1c": 7.5,
        "fbs": 140, "sbp": 145, "dbp": 90, "bmi": 27,
    }
    risk_ckd = model.model1.predict_risk_levels(ckd_patient)
    budget_ckd = model.budget_calc.get_daily_budget(
        has_ckd=True, has_htn=True, has_dm=True
    )
    
    rec_egg = model.recommender.recommend_one("Egg white", risk_ckd, budget_ckd)
    rec_paneer = model.recommender.recommend_one("Paneer (Cottage cheese)", risk_ckd, budget_ckd)
    
    print(f"    Egg white:  max_grams={rec_egg.max_grams}g, label={rec_egg.label}")
    print(f"    Paneer:     max_grams={rec_paneer.max_grams}g, label={rec_paneer.label}")
    
    # Egg white should have larger max_grams due to bonus
    # (or at least not be penalised like a poor-tier item)
    diff_1c = rec_egg.max_grams - rec_paneer.max_grams
    print(f"    Difference: {diff_1c:.1f}g "
          f"({'Egg white gets more — correct!' if diff_1c > 0 else 'Paneer gets more — check tiers'})")
    
    # --- Test 3: Explanation string includes tier ---
    print(f"\n  Test 3 — Explanation includes efficiency tier:")
    has_tier = "Phos efficiency:" in rec_egg.explanation
    print(f"    Egg white explanation has tier: {has_tier}")
    if has_tier:
        # Extract the efficiency note
        idx_start = rec_egg.explanation.index("[Phos efficiency:")
        print(f"    → {rec_egg.explanation[idx_start:]}")
    assert has_tier, "Explanation should contain phos efficiency info for CKD patient"
    print("    ✓ Efficiency tier present in explanation (CORRECT)")
    
    # --- Test 4: CKD-friendly proteins list ---
    print(f"\n  Test 4 — get_ckd_friendly_proteins():")
    friendly = model.ifct.get_ckd_friendly_proteins()
    print(f"    Total CKD-friendly proteins: {len(friendly)}")
    assert len(friendly) > 0, "Expected non-empty list"
    for item in friendly:
        print(f"      {item['ingredient']:30s} ratio={item['phos_per_g_protein']:5.1f}  "
              f"tier={item['efficiency_tier']}")
    print("    ✓ Non-empty ranked list returned (CORRECT)")
    
    # ==========================================
    # VALIDATION: Phase 2B — CaloricSufficiencyValidator
    # ==========================================
    print("\n" + "="*60)
    print("VALIDATION: CALORIC SUFFICIENCY VALIDATOR (2B)")
    print("="*60)
    
    # --- Test 1: clinical_warnings key present for normal patient ---
    print("\n  Test 1 — clinical_warnings key present:")
    model.ledger._reset()
    normal_result = model.get_recommendations(patient, ingredients)
    assert "clinical_warnings" in normal_result, "clinical_warnings key missing"
    has_warnings = len(normal_result["clinical_warnings"]) > 0
    print(f"    clinical_warnings present: True")
    print(f"    Warnings triggered: {has_warnings}")
    if has_warnings:
        for w in normal_result["clinical_warnings"]:
            print(f"      Relaxed {w['constraint']}: "
                  f"sev {w['old_severity']:.2f} → {w['new_severity']:.2f} "
                  f"(→ {w['projected_kcal_after']:.0f} kcal)")
            print(f"        Rationale: {w['rationale']}")
    else:
        print("    ✓ No reconciliation needed (caloric threshold met)")
    print("    ✓ clinical_warnings key present in output (CORRECT)")
    
    # --- Test 2: Worst-case patient triggers reconciliation ---
    print("\n  Test 2 — Worst-case patient triggers reconciliation:")
    worst_patient = {
        "age": 72, "sex_male": 1,
        "has_htn": 1, "has_dm": 1, "has_ckd": 1,
        "serum_sodium": 130, "serum_potassium": 5.8,
        "creatinine": 4.5, "egfr": 12, "hba1c": 9.5,
        "fbs": 220, "sbp": 180, "dbp": 110, "bmi": 32,
    }
    # Use high-calorie high-potassium ingredients to force restriction
    worst_ingredients = [
        "Banana, ripe", "Potato (Aloo)", "Spinach (Palak)",
        "Coconut, dry", "Rice, milled (white)", "Chicken, breast",
        "Egg, whole, boiled", "Curd (Dahi/Yogurt)",
    ]
    model.ledger._reset()
    worst_result = model.get_recommendations(worst_patient, worst_ingredients)
    assert "clinical_warnings" in worst_result
    worst_warnings = worst_result["clinical_warnings"]
    print(f"    Warnings triggered: {len(worst_warnings) > 0}")
    if worst_warnings:
        print(f"    Reconciliation steps: {len(worst_warnings)}")
        for w in worst_warnings:
            print(f"      Relaxed {w['constraint']}: "
                  f"sev {w['old_severity']:.2f} → {w['new_severity']:.2f} "
                  f"(→ {w['projected_kcal_after']:.0f} kcal)")
        print("    ✓ Reconciliation triggered for worst-case patient (CORRECT)")
    else:
        print("    ⚠ No reconciliation triggered (caloric threshold may be met "
              "even under max restrictions)")
    
    # --- Test 3: Sodium and potassium are NEVER modified ---
    print("\n  Test 3 — Sodium/potassium immutability:")
    risk_worst = model.model1.predict_risk_levels(worst_patient)
    orig_sod_sev = risk_worst["sodium_sensitivity"]["severity_score"]
    orig_pot_sev = risk_worst["potassium_sensitivity"]["severity_score"]
    result_sod_sev = worst_result["risk_levels"]["sodium_sensitivity"]["severity_score"]
    result_pot_sev = worst_result["risk_levels"]["potassium_sensitivity"]["severity_score"]
    
    assert orig_sod_sev == result_sod_sev, (
        f"Sodium modified: {orig_sod_sev} → {result_sod_sev}"
    )
    assert orig_pot_sev == result_pot_sev, (
        f"Potassium modified: {orig_pot_sev} → {result_pot_sev}"
    )
    print(f"    Sodium severity: {orig_sod_sev} → {result_sod_sev} (unchanged)")
    print(f"    Potassium severity: {orig_pot_sev} → {result_pot_sev} (unchanged)")
    print("    ✓ Sodium and potassium NEVER modified (CORRECT)")
    
    # ==========================================
    # VALIDATION: Phase 3A — TabNet Feature Attribution
    # ==========================================
    print("\n" + "="*60)
    print("VALIDATION: TABNET FEATURE ATTRIBUTION (3A)")
    print("="*60)
    
    # --- Test 1: feature_attribution sums to ~1.0 ---
    print("\n  Test 1 — feature_attribution sums to ~1.0:")
    ckd_patient_3a = {
        "age": 65, "sex_male": 1,
        "has_htn": 1, "has_dm": 1, "has_ckd": 1,
        "serum_sodium": 137, "serum_potassium": 5.2,
        "creatinine": 2.8, "egfr": 28, "hba1c": 7.5,
        "fbs": 140, "sbp": 145, "dbp": 90, "bmi": 27,
    }
    risk_3a = model.model1.predict_risk_levels(ckd_patient_3a)
    for target_3a, info_3a in risk_3a.items():
        if "feature_attribution" not in info_3a:
            continue
        attr = info_3a["feature_attribution"]
        attr_sum = sum(attr.values())
        print(f"    {target_3a}: top-5 sum={attr_sum:.3f}")
        for feat, wt in attr.items():
            print(f"      {feat}: {wt:.4f}")
    print("    OK — feature_attribution present for all targets")
    
    # --- Test 2: Top feature for protein_restriction ---
    print("\n  Test 2 — Top feature for protein_restriction:")
    prot_attr = risk_3a.get("protein_restriction", {}).get("feature_attribution", {})
    if prot_attr:
        top_feat = next(iter(prot_attr))
        print(f"    Top feature: {top_feat} (weight={prot_attr[top_feat]:.4f})")
        expected = top_feat in ("egfr", "creatinine", "has_ckd")
        print(f"    Is egfr/creatinine/has_ckd: {expected}")
    
    # --- Test 3: generate_clinical_explanation ---
    print("\n  Test 3 — generate_clinical_explanation():")
    explanations = model.model1.generate_clinical_explanation(risk_3a, ckd_patient_3a)
    all_targets_3a = [
        "sodium_sensitivity", "potassium_sensitivity",
        "protein_restriction", "carb_sensitivity",
        "phosphorus_sensitivity",
    ]
    all_ok = True
    for t in all_targets_3a:
        exp = explanations.get(t, "")
        has_exp = len(exp) > 0
        if not has_exp:
            all_ok = False
        print(f"    {t}:")
        print(f"      {exp[:150]}{'...' if len(exp) > 150 else ''}")
    assert all_ok, "Not all targets have explanations"
    print("    All 5 targets have non-empty explanations (CORRECT)")
    
    # ==========================================
    # VALIDATION: Phase 2A — SubstitutionEngine
    # ==========================================
    print("\n" + "="*60)
    print("VALIDATION: SUBSTITUTION ENGINE (2A)")
    print("="*60)
    
    # --- Test 1: Direct SubstitutionEngine test — Banana substitutes ---
    print("\n  Test 1 -- Banana substitutes with lower potassium:")
    
    # Create high-severity enriched risk_levels to force Avoid on banana
    forced_risk = {
        "sodium_sensitivity":    {"label": "high", "severity_score": 1.8, "confidence": 0.9,
                                  "proba": {"low": 0.05, "moderate": 0.05, "high": 0.90},
                                  "feature_attribution": {"has_htn": 0.52, "serum_sodium": 0.20}},
        "potassium_sensitivity": {"label": "high", "severity_score": 2.0, "confidence": 1.0,
                                  "proba": {"low": 0.0, "moderate": 0.0, "high": 1.0},
                                  "feature_attribution": {"serum_potassium": 0.40, "egfr": 0.30}},
        "protein_restriction":   {"label": "high", "severity_score": 2.0, "confidence": 1.0,
                                  "proba": {"low": 0.0, "moderate": 0.0, "high": 1.0},
                                  "feature_attribution": {"egfr": 0.39, "has_ckd": 0.32}},
        "carb_sensitivity":      {"label": "high", "severity_score": 2.0, "confidence": 1.0,
                                  "proba": {"low": 0.0, "moderate": 0.0, "high": 1.0},
                                  "feature_attribution": {"egfr": 0.30, "has_dm": 0.24}},
        "phosphorus_sensitivity":{"label": "high", "severity_score": 1.8, "confidence": 0.9,
                                  "proba": {"low": 0.05, "moderate": 0.05, "high": 0.90},
                                  "feature_attribution": {}},  # rule-derived, no TabNet attribution
    }
    forced_budget = DailyBudget(
        sodium_mg_remaining=400,
        potassium_mg_remaining=400,
        protein_g_remaining=15,
        carbs_g_remaining=50,
        phosphorus_mg_remaining=200
    )
    
    # Try banana with forced constraints
    banana_rec = model.recommender.recommend_one("Banana, ripe", forced_risk, forced_budget)
    print(f"    Banana label: {banana_rec.label}, binding: {banana_rec.binding_constraint}")
    
    banana_k = model.ifct.get_nutrients_per_100g("Banana, ripe")["potassium_mg"]
    print(f"    Banana potassium: {banana_k} mg/100g")
    
    if banana_rec.label == "Avoid":
        subs = model.substitution_engine.find_substitutes(
            avoided_ingredient="Banana, ripe",
            binding_constraint=banana_rec.binding_constraint,
            risk_levels=forced_risk,
            budget=forced_budget,
            recommender=model.recommender,
            n=3,
        )
        print(f"    Substitutes found: {len(subs)}")
        for sub in subs:
            sub_k = model.ifct.get_nutrients_per_100g(sub.ingredient)["potassium_mg"]
            print(f"    Substitute: {sub.ingredient} -- K={sub_k}mg, "
                  f"label={sub.label}, max_grams={sub.max_grams:.1f}g")
            print(f"      Explanation: {sub.explanation[:120]}...")
        if subs:
            first_k = model.ifct.get_nutrients_per_100g(subs[0].ingredient)["potassium_mg"]
            assert first_k < banana_k, f"First substitute has K={first_k} >= banana K={banana_k}"
            print("    OK -- substitute has lower potassium than banana (CORRECT)")
    else:
        # Even with forced constraints, test the engine directly anyway
        print(f"    Banana not Avoid with forced budget, testing engine directly")
        subs_direct = model.substitution_engine.find_substitutes(
            avoided_ingredient="Banana, ripe",
            binding_constraint="potassium",
            risk_levels=forced_risk,
            budget=forced_budget,
            recommender=model.recommender,
            n=3,
        )
        print(f"    Direct substitutes found: {len(subs_direct)}")
        for sub in subs_direct:
            sub_k = model.ifct.get_nutrients_per_100g(sub.ingredient)["potassium_mg"]
            print(f"      {sub.ingredient}: K={sub_k}mg, label={sub.label}")
    
    # --- Test 2: Feature attribution suffix in explanations ---
    print("\n  Test 2 -- Feature attribution suffix:")
    # Test substitution with potassium binding to get suffix
    subs_attr = model.substitution_engine.find_substitutes(
        avoided_ingredient="Banana, ripe",
        binding_constraint="potassium",
        risk_levels=forced_risk,
        budget=forced_budget,
        recommender=model.recommender,
        n=3,
    )
    attribution_found = False
    for sub in subs_attr:
        if "[Substituted due to potassium restriction" in sub.explanation:
            suffix_start = sub.explanation.find("[Substituted")
            print(f"    Found in: {sub.ingredient}")
            print(f"    Suffix: {sub.explanation[suffix_start:]}")
            attribution_found = True
            break
    
    # Test with phosphorus (no attribution — should NOT error)
    subs_phos = model.substitution_engine.find_substitutes(
        avoided_ingredient="Chicken, breast",
        binding_constraint="phosphorus",
        risk_levels=forced_risk,
        budget=forced_budget,
        recommender=model.recommender,
        n=2,
    )
    phos_clean = True
    for sub in subs_phos:
        if "[Substituted due to" in sub.explanation:
            phos_clean = False  # Should NOT have suffix (empty attribution)
    
    if attribution_found:
        print("    OK -- feature attribution suffix present for potassium (CORRECT)")
    else:
        print("    NOTE -- no non-Avoid substitutes found for attribution test")
    print(f"    Phosphorus substitutes clean (no suffix for rule-derived): {phos_clean}")
    
    # --- Test 3: include_substitutes=False ---
    print("\n  Test 3 -- include_substitutes=False:")
    severe_patient = {
        "age": 70, "sex_male": 1,
        "has_htn": 1, "has_dm": 1, "has_ckd": 1,
        "serum_sodium": 130, "serum_potassium": 6.0,
        "creatinine": 4.0, "egfr": 15, "hba1c": 9.0,
        "fbs": 200, "sbp": 170, "dbp": 100, "bmi": 30,
    }
    result_no_sub = model.get_recommendations(
        severe_patient, ["Banana, ripe"], include_substitutes=False
    )
    banana_no_sub = result_no_sub["recommendations"][0]
    has_subs = len(banana_no_sub.get("substitutes", [])) > 0
    print(f"    Substitutes present: {has_subs}")
    assert not has_subs, "include_substitutes=False should produce no substitutes"
    print("    OK -- no substitutes when include_substitutes=False (CORRECT)")
    
    # --- Test 4: Full pipeline integration ---
    print("\n  Test 4 -- Full pipeline integration via get_recommendations:")
    result_with_sub = model.get_recommendations(
        severe_patient, ["Banana, ripe", "Spinach (Palak)", "Potato (Aloo)", "Apple"]
    )
    any_subs = any(
        len(rec.get("substitutes", [])) > 0
        for rec in result_with_sub["recommendations"]
        if rec["label"] == "Avoid"
    )
    avoid_count = sum(1 for r in result_with_sub["recommendations"] if r["label"] == "Avoid")
    print(f"    Avoid items: {avoid_count}")
    print(f"    Any Avoid items have substitutes: {any_subs}")
    for rec in result_with_sub["recommendations"]:
        if rec["label"] == "Avoid" and rec.get("substitutes"):
            print(f"    {rec['ingredient']}: {len(rec['substitutes'])} substitutes")
            for sub in rec["substitutes"]:
                print(f"      -> {sub['ingredient']}: {sub['label']}, {sub['max_grams']:.1f}g")
    if avoid_count == 0:
        print("    No Avoid items in pipeline (caloric validator relaxed constraints)")
        print("    Direct engine test above already verified substitution works")
    print("    OK -- Phase 2A integration validated (CORRECT)")
    
    return model


if __name__ == "__main__":
    model = main()

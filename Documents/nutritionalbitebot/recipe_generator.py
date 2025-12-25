"""
Hybrid RAG + Edit Recipe Generator with SHARE Methodology
==========================================================
Retrieves recipes using available ingredients and adapts them to clinical
constraints using the SHARE (Substitute, Halve, Add, Remove, Emphasize) methodology.

Features:
- Recipe retrieval based on pantry inventory
- Clinical adaptation using SHARE methodology
- Explainability logs citing specific lab values
- Dyslipidemia fat substitution
- Diabetes carb counting integration
- CKD protein restriction

Author: Pantry-to-Plate Development Team
Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecipeIngredient:
    """Recipe ingredient with quantity."""
    name: str
    quantity: float
    unit: str
    fdc_id: Optional[int] = None
    nutrients: Optional[Dict] = None


@dataclass
class RecipeNutrition:
    """Nutritional information per serving."""
    calories: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    saturated_fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    phosphorus_mg: Optional[float] = None


@dataclass
class SHAREEdit:
    """SHARE methodology edit applied to recipe."""
    action: str  # "substitute", "halve", "add", "remove", "emphasize"
    original_item: Optional[str] = None
    new_item: Optional[str] = None
    reason: str = ""
    clinical_basis: str = ""
    lab_value_cited: Optional[str] = None


@dataclass
class Recipe:
    """Complete recipe with all details."""
    recipe_id: str
    name: str
    servings: int
    ingredients: List[RecipeIngredient]
    instructions: List[str]
    nutrition_per_serving: RecipeNutrition
    tags: List[str] = field(default_factory=list)
    source: str = "Generated"


@dataclass
class AdaptedRecipe:
    """Clinically adapted recipe with explainability."""
    original_recipe: Recipe
    adapted_recipe: Recipe
    share_edits: List[SHAREEdit]
    explainability_log: List[Dict]
    compliance_check: Dict
    patient_id: str
    generation_timestamp: str


class RecipeDatabase:
    """
    Mock recipe database for RAG retrieval.
    In production, this would query a real recipe database or vector store.
    """
    
    def __init__(self):
        """Initialize with sample recipes."""
        self.recipes = self._load_sample_recipes()
    
    def _load_sample_recipes(self) -> List[Recipe]:
        """Load sample recipes for demonstration."""
        
        recipes = [
            Recipe(
                recipe_id="RCP001",
                name="Roasted Vegetable Medley",
                servings=4,
                ingredients=[
                    RecipeIngredient(name="carrot", quantity=200, unit="g"),
                    RecipeIngredient(name="cabbage", quantity=150, unit="g"),
                    RecipeIngredient(name="cauliflower", quantity=200, unit="g"),
                    RecipeIngredient(name="olive_oil", quantity=20, unit="ml"),
                    RecipeIngredient(name="herbs", quantity=5, unit="g")
                ],
                instructions=[
                    "Preheat oven to 400°F (200°C)",
                    "Cut vegetables into bite-sized pieces",
                    "Toss with olive oil and herbs",
                    "Roast for 25-30 minutes until tender"
                ],
                nutrition_per_serving=RecipeNutrition(
                    calories=120,
                    protein_g=3,
                    carbohydrates_g=18,
                    fat_g=5,
                    saturated_fat_g=0.8,
                    fiber_g=5,
                    sodium_mg=45,
                    potassium_mg=380,
                    phosphorus_mg=60
                ),
                tags=["vegetable", "low_sodium", "fiber_rich"]
            ),
            
            Recipe(
                recipe_id="RCP002",
                name="Grilled Chicken with Steamed Vegetables",
                servings=4,
                ingredients=[
                    RecipeIngredient(name="chicken_breast", quantity=400, unit="g"),
                    RecipeIngredient(name="broccoli", quantity=200, unit="g"),
                    RecipeIngredient(name="carrot", quantity=150, unit="g"),
                    RecipeIngredient(name="olive_oil", quantity=15, unit="ml"),
                    RecipeIngredient(name="lemon", quantity=1, unit="whole")
                ],
                instructions=[
                    "Season chicken with herbs and lemon juice",
                    "Grill chicken for 6-7 minutes per side",
                    "Steam vegetables until tender",
                    "Serve chicken with vegetables"
                ],
                nutrition_per_serving=RecipeNutrition(
                    calories=220,
                    protein_g=28,
                    carbohydrates_g=12,
                    fat_g=7,
                    saturated_fat_g=1.5,
                    fiber_g=4,
                    sodium_mg=85,
                    potassium_mg=520,
                    phosphorus_mg=240
                ),
                tags=["protein", "low_fat", "vegetables"]
            ),
            
            Recipe(
                recipe_id="RCP003",
                name="Apple Cinnamon Oatmeal",
                servings=2,
                ingredients=[
                    RecipeIngredient(name="oats", quantity=80, unit="g"),
                    RecipeIngredient(name="apple", quantity=150, unit="g"),
                    RecipeIngredient(name="cinnamon", quantity=2, unit="g"),
                    RecipeIngredient(name="almond_milk", quantity=240, unit="ml"),
                    RecipeIngredient(name="walnuts", quantity=20, unit="g")
                ],
                instructions=[
                    "Cook oats with almond milk for 5 minutes",
                    "Dice apple and add to oats",
                    "Sprinkle with cinnamon",
                    "Top with walnuts before serving"
                ],
                nutrition_per_serving=RecipeNutrition(
                    calories=280,
                    protein_g=8,
                    carbohydrates_g=45,
                    fat_g=8,
                    saturated_fat_g=1.0,
                    fiber_g=8,
                    sodium_mg=55,
                    potassium_mg=320,
                    phosphorus_mg=180
                ),
                tags=["breakfast", "fiber_rich", "whole_grain"]
            )
        ]
        
        return recipes
    
    def search_by_ingredients(self, available_ingredients: List[str]) -> List[Recipe]:
        """
        Search recipes that can be made with available ingredients.
        
        Args:
            available_ingredients: List of ingredient names
            
        Returns:
            List of matching recipes
        """
        matching_recipes = []
        
        for recipe in self.recipes:
            recipe_ingredients = {ing.name.lower() for ing in recipe.ingredients}
            available_set = {ing.lower() for ing in available_ingredients}
            
            # Calculate match percentage
            matches = recipe_ingredients.intersection(available_set)
            match_pct = len(matches) / len(recipe_ingredients) if recipe_ingredients else 0
            
            # If >= 50% ingredients available, include recipe
            if match_pct >= 0.5:
                matching_recipes.append(recipe)
        
        return matching_recipes


class ClinicalRecipeAdapter:
    """
    Adapts recipes to clinical constraints using SHARE methodology.
    """
    
    def __init__(self):
        """Initialize adapter."""
        self.clinical_constraint = None
        self.patient_labs = None
    
    def load_clinical_data(self, constraint_file: str):
        """Load clinical constraints from Rules Engine."""
        with open(constraint_file, 'r') as f:
            self.clinical_constraint = json.load(f)
        
        # Extract key lab values for explainability
        self.patient_labs = {
            'egfr': self.clinical_constraint.get('egfr'),
            'potassium': self.clinical_constraint.get('current_potassium'),
            'ckd_stage': self.clinical_constraint.get('ckd_stage'),
            'conditions': self.clinical_constraint.get('medical_conditions', {})
        }
        
        logger.info(f"Loaded clinical data for {self.clinical_constraint['user_id']}")
    
    def apply_share_methodology(self, recipe: Recipe) -> Tuple[Recipe, List[SHAREEdit]]:
        """
        Apply SHARE methodology to adapt recipe.
        
        SHARE:
        - Substitute: Replace high-risk ingredients
        - Halve: Reduce quantities of restricted items
        - Add: Include fiber/beneficial nutrients
        - Remove: Eliminate prohibited items
        - Emphasize: Highlight important cooking/prep methods
        
        Args:
            recipe: Original recipe
            
        Returns:
            Tuple of (adapted_recipe, share_edits)
        """
        edits = []
        adapted_ingredients = []
        adapted_instructions = list(recipe.instructions)
        
        # Get clinical limits
        k_limit = self.clinical_constraint['potassium']['per_meal_max']
        na_limit = self.clinical_constraint['sodium']['per_meal_max']
        protein_target = self.clinical_constraint['protein']['per_meal_protein_g']
        
        prohibited_foods = [f['food_name'].lower() 
                          for f in self.clinical_constraint.get('prohibited_foods', [])]
        
        for ingredient in recipe.ingredients:
            ing_name = ingredient.name.lower()
            
            # REMOVE: Check for prohibited ingredients
            if any(prohibited in ing_name for prohibited in prohibited_foods):
                edits.append(SHAREEdit(
                    action="remove",
                    original_item=ingredient.name,
                    new_item=None,
                    reason=f"Prohibited due to high potassium content",
                    clinical_basis=f"eGFR {self.patient_labs['egfr']} mL/min/1.73m² (CKD Stage 3-5)",
                    lab_value_cited=f"eGFR: {self.patient_labs['egfr']}"
                ))
                
                # Find alternative
                alternative = self._get_low_k_alternative(ingredient.name)
                if alternative:
                    adapted_ingredients.append(RecipeIngredient(
                        name=alternative,
                        quantity=ingredient.quantity,
                        unit=ingredient.unit
                    ))
                    
                    edits.append(SHAREEdit(
                        action="substitute",
                        original_item=ingredient.name,
                        new_item=alternative,
                        reason=f"Low-potassium alternative to {ingredient.name}",
                        clinical_basis=f"CKD requires K+ < {k_limit}mg per meal"
                    ))
                
                continue
            
            # SUBSTITUTE: Replace saturated fats (for dyslipidemia)
            if self.patient_labs['conditions'].get('dyslipidemia'):
                if 'butter' in ing_name or 'cream' in ing_name:
                    edits.append(SHAREEdit(
                        action="substitute",
                        original_item=ingredient.name,
                        new_item="olive_oil",
                        reason="Replace saturated fat with heart-healthy oil",
                        clinical_basis="Dyslipidemia: Reduce saturated fat, increase fiber"
                    ))
                    
                    adapted_ingredients.append(RecipeIngredient(
                        name="olive_oil",
                        quantity=ingredient.quantity * 0.8,  # Slightly less
                        unit=ingredient.unit
                    ))
                    continue
            
            # HALVE: Reduce high-sodium ingredients (for HTN)
            if self.patient_labs['conditions'].get('hypertension'):
                if 'salt' in ing_name or 'soy_sauce' in ing_name:
                    edits.append(SHAREEdit(
                        action="halve",
                        original_item=ingredient.name,
                        new_item=ingredient.name,
                        reason=f"Reduce sodium for hypertension control",
                        clinical_basis=f"HTN requires Na+ < {na_limit}mg per meal"
                    ))
                    
                    adapted_ingredients.append(RecipeIngredient(
                        name=ingredient.name,
                        quantity=ingredient.quantity * 0.5,
                        unit=ingredient.unit
                    ))
                    continue
            
            # Default: Keep ingredient
            adapted_ingredients.append(ingredient)
        
        # ADD: Fiber for diabetes and dyslipidemia
        if self.patient_labs['conditions'].get('type2_diabetes') or \
           self.patient_labs['conditions'].get('dyslipidemia'):
            
            # Check if fiber-rich ingredient already present
            has_fiber = any('oat' in ing.name.lower() or 'bean' in ing.name.lower() 
                          for ing in adapted_ingredients)
            
            if not has_fiber:
                edits.append(SHAREEdit(
                    action="add",
                    original_item=None,
                    new_item="chia_seeds",
                    reason="Add fiber to improve glycemic control and lower cholesterol",
                    clinical_basis="Diabetes + Dyslipidemia: Fiber helps control blood sugar and lipids"
                ))
                
                adapted_ingredients.append(RecipeIngredient(
                    name="chia_seeds",
                    quantity=10,
                    unit="g"
                ))
        
        # EMPHASIZE: Add carb counting for diabetes
        if self.patient_labs['conditions'].get('type2_diabetes'):
            carb_per_serving = recipe.nutrition_per_serving.carbohydrates_g
            
            edits.append(SHAREEdit(
                action="emphasize",
                original_item=None,
                new_item="carb_counting_note",
                reason=f"Carb counting: {carb_per_serving}g per serving",
                clinical_basis="Diabetes management requires carbohydrate awareness",
                lab_value_cited=f"HbA1c from MIMIC-IV data"
            ))
            
            # Add to instructions
            adapted_instructions.insert(0, 
                f"⚠️ CARB COUNT: {carb_per_serving}g carbohydrates per serving "
                f"(Target: ≤{self.clinical_constraint['carbohydrates']['per_meal_max']}g)"
            )
        
        # Create adapted recipe
        adapted_recipe = Recipe(
            recipe_id=f"{recipe.recipe_id}_ADAPTED",
            name=f"{recipe.name} (Clinically Adapted)",
            servings=recipe.servings,
            ingredients=adapted_ingredients,
            instructions=adapted_instructions,
            nutrition_per_serving=self._recalculate_nutrition(adapted_ingredients, recipe.servings),
            tags=recipe.tags + ["clinically_adapted"],
            source=f"Adapted from {recipe.source}"
        )
        
        return adapted_recipe, edits
    
    def _get_low_k_alternative(self, prohibited_item: str) -> Optional[str]:
        """Get low-potassium alternative for prohibited item."""
        alternatives = {
            'potato': 'cauliflower',
            'sweet_potato': 'butternut_squash',
            'banana': 'apple',
            'orange': 'grape',
            'tomato': 'cucumber',
            'spinach': 'lettuce'
        }
        
        for key, value in alternatives.items():
            if key in prohibited_item.lower():
                return value
        
        return None
    
    def _recalculate_nutrition(self, ingredients: List[RecipeIngredient], 
                              servings: int) -> RecipeNutrition:
        """
        Recalculate nutrition after SHARE edits.
        This is simplified - in production would use full nutrient database.
        """
        # Simplified estimation
        return RecipeNutrition(
            calories=200,
            protein_g=15,
            carbohydrates_g=25,
            fat_g=6,
            saturated_fat_g=1.0,
            fiber_g=6,
            sodium_mg=300,
            potassium_mg=450,
            phosphorus_mg=150
        )
    
    def generate_explainability_log(self, share_edits: List[SHAREEdit], 
                                   adapted_recipe: Recipe) -> List[Dict]:
        """
        Generate detailed explainability log citing lab values.
        
        Args:
            share_edits: List of SHARE edits applied
            adapted_recipe: Final adapted recipe
            
        Returns:
            List of explainability entries
        """
        log = []
        
        # Header entry
        log.append({
            'type': 'clinical_context',
            'patient_id': self.clinical_constraint['user_id'],
            'ckd_stage': self.patient_labs['ckd_stage'],
            'egfr': f"{self.patient_labs['egfr']} mL/min/1.73m²",
            'current_potassium': f"{self.patient_labs['potassium']} mEq/L" if self.patient_labs['potassium'] else "N/A",
            'conditions': ', '.join([k for k, v in self.patient_labs['conditions'].items() if v])
        })
        
        # Document each SHARE edit
        for edit in share_edits:
            entry = {
                'action': edit.action,
                'reason': edit.reason,
                'clinical_basis': edit.clinical_basis
            }
            
            if edit.lab_value_cited:
                entry['lab_value_cited'] = edit.lab_value_cited
            
            if edit.action == "substitute":
                entry['substitution'] = f"{edit.original_item} → {edit.new_item}"
            elif edit.action == "remove":
                entry['removed_item'] = edit.original_item
            elif edit.action == "add":
                entry['added_item'] = edit.new_item
            elif edit.action == "halve":
                entry['reduced_item'] = edit.original_item
            
            log.append(entry)
        
        # Nutrient compliance check
        nutrition = adapted_recipe.nutrition_per_serving
        
        compliance = {
            'type': 'nutrient_compliance',
            'potassium': {
                'value': f"{nutrition.potassium_mg}mg",
                'limit': f"{self.clinical_constraint['potassium']['per_meal_max']}mg",
                'compliant': nutrition.potassium_mg <= self.clinical_constraint['potassium']['per_meal_max'],
                'citation': f"eGFR {self.patient_labs['egfr']} requires K+ restriction"
            },
            'sodium': {
                'value': f"{nutrition.sodium_mg}mg",
                'limit': f"{self.clinical_constraint['sodium']['per_meal_max']}mg",
                'compliant': nutrition.sodium_mg <= self.clinical_constraint['sodium']['per_meal_max']
            },
            'protein': {
                'value': f"{nutrition.protein_g}g",
                'target': f"{self.clinical_constraint['protein']['per_meal_protein_g']}g",
                'compliant': abs(nutrition.protein_g - self.clinical_constraint['protein']['per_meal_protein_g']) < 5
            }
        }
        
        log.append(compliance)
        
        return log
    
    def validate_recipe_compliance(self, recipe: Recipe) -> Dict:
        """
        Validate adapted recipe against clinical constraints.
        
        Returns:
            Dict with compliance status and violations
        """
        nutrition = recipe.nutrition_per_serving
        violations = []
        warnings = []
        
        # Check potassium
        k_limit = self.clinical_constraint['potassium']['per_meal_max']
        if nutrition.potassium_mg > k_limit:
            violations.append({
                'nutrient': 'Potassium',
                'value': nutrition.potassium_mg,
                'limit': k_limit,
                'severity': 'CRITICAL',
                'citation': f"eGFR {self.patient_labs['egfr']} mL/min/1.73m²"
            })
        elif nutrition.potassium_mg > k_limit * 0.9:
            warnings.append({
                'nutrient': 'Potassium',
                'value': nutrition.potassium_mg,
                'message': 'Near limit - monitor closely'
            })
        
        # Check sodium
        na_limit = self.clinical_constraint['sodium']['per_meal_max']
        if nutrition.sodium_mg > na_limit:
            violations.append({
                'nutrient': 'Sodium',
                'value': nutrition.sodium_mg,
                'limit': na_limit,
                'severity': 'HIGH'
            })
        
        # Check protein
        protein_target = self.clinical_constraint['protein']['per_meal_protein_g']
        if abs(nutrition.protein_g - protein_target) > protein_target * 0.3:
            warnings.append({
                'nutrient': 'Protein',
                'value': nutrition.protein_g,
                'target': protein_target,
                'message': 'Protein outside recommended range'
            })
        
        # Check carbohydrates (for diabetes)
        if self.patient_labs['conditions'].get('type2_diabetes'):
            carb_limit = self.clinical_constraint['carbohydrates']['per_meal_max']
            if nutrition.carbohydrates_g > carb_limit:
                violations.append({
                    'nutrient': 'Carbohydrates',
                    'value': nutrition.carbohydrates_g,
                    'limit': carb_limit,
                    'severity': 'MEDIUM'
                })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'overall_status': 'SAFE' if len(violations) == 0 else 'UNSAFE'
        }
    
    def generate_full_adaptation(self, recipe: Recipe) -> AdaptedRecipe:
        """
        Complete recipe adaptation with all components.
        
        Args:
            recipe: Original recipe
            
        Returns:
            AdaptedRecipe with explainability
        """
        logger.info(f"Adapting recipe: {recipe.name}")
        
        # Apply SHARE methodology
        adapted_recipe, share_edits = self.apply_share_methodology(recipe)
        
        # Generate explainability log
        explainability_log = self.generate_explainability_log(share_edits, adapted_recipe)
        
        # Validate compliance
        compliance_check = self.validate_recipe_compliance(adapted_recipe)
        
        # Create complete adapted recipe
        full_adaptation = AdaptedRecipe(
            original_recipe=recipe,
            adapted_recipe=adapted_recipe,
            share_edits=share_edits,
            explainability_log=explainability_log,
            compliance_check=compliance_check,
            patient_id=self.clinical_constraint['user_id'],
            generation_timestamp=datetime.now().isoformat()
        )
        
        return full_adaptation


class HybridRAGRecipeSystem:
    """
    Complete system integrating pantry inventory, recipe retrieval, and adaptation.
    """
    
    def __init__(self):
        """Initialize system components."""
        self.recipe_db = RecipeDatabase()
        self.adapter = ClinicalRecipeAdapter()
    
    def generate_meal_plan(self, 
                          pantry_summary_file: str,
                          constraint_file: str,
                          num_recipes: int = 3) -> List[AdaptedRecipe]:
        """
        Generate complete meal plan from pantry to adapted recipes.
        
        Args:
            pantry_summary_file: Pantry inventory summary JSON
            constraint_file: Clinical constraints JSON
            num_recipes: Number of recipes to generate
            
        Returns:
            List of adapted recipes
        """
        logger.info("Generating personalized meal plan...")
        
        # Load pantry inventory
        with open(pantry_summary_file, 'r') as f:
            pantry_summary = json.load(f)
        
        # Load clinical constraints
        self.adapter.load_clinical_data(constraint_file)
        
        # Get safe ingredients
        safe_ingredients = [item['name'] for item in pantry_summary['safe_items']]
        
        logger.info(f"Available safe ingredients: {', '.join(safe_ingredients)}")
        
        # Retrieve matching recipes
        candidate_recipes = self.recipe_db.search_by_ingredients(safe_ingredients)
        
        logger.info(f"Found {len(candidate_recipes)} candidate recipes")
        
        # Adapt recipes
        adapted_recipes = []
        for recipe in candidate_recipes[:num_recipes]:
            adapted = self.adapter.generate_full_adaptation(recipe)
            adapted_recipes.append(adapted)
            
            logger.info(f"✓ Adapted: {recipe.name}")
        
        return adapted_recipes
    
    def export_meal_plan(self, adapted_recipes: List[AdaptedRecipe], output_file: str):
        """Export meal plan to JSON."""
        meal_plan = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_recipes': len(adapted_recipes),
            'recipes': [asdict(recipe) for recipe in adapted_recipes]
        }
        
        with open(output_file, 'w') as f:
            json.dump(meal_plan, f, indent=2)
        
        logger.info(f"Meal plan exported to {output_file}")
    
    def print_adapted_recipe(self, adapted: AdaptedRecipe):
        """Print human-readable adapted recipe."""
        print("\n" + "="*80)
        print(f"ADAPTED RECIPE: {adapted.adapted_recipe.name}")
        print("="*80)
        
        print(f"\nPatient: {adapted.patient_id}")
        print(f"Generated: {adapted.generation_timestamp}")
        print(f"Compliance Status: {adapted.compliance_check['overall_status']}")
        
        print("\n" + "-"*80)
        print("SHARE METHODOLOGY EDITS APPLIED")
        print("-"*80)
        
        for edit in adapted.share_edits:
            print(f"\n[{edit.action.upper()}]")
            if edit.original_item and edit.new_item:
                print(f"  {edit.original_item} → {edit.new_item}")
            elif edit.original_item:
                print(f"  {edit.original_item}")
            elif edit.new_item:
                print(f"  Added: {edit.new_item}")
            
            print(f"  Reason: {edit.reason}")
            print(f"  Clinical Basis: {edit.clinical_basis}")
            
            if edit.lab_value_cited:
                print(f"  Lab Citation: {edit.lab_value_cited}")
        
        print("\n" + "-"*80)
        print("INGREDIENTS")
        print("-"*80)
        
        for ing in adapted.adapted_recipe.ingredients:
            print(f"  • {ing.quantity}{ing.unit} {ing.name}")
        
        print("\n" + "-"*80)
        print("INSTRUCTIONS")
        print("-"*80)
        
        for i, instruction in enumerate(adapted.adapted_recipe.instructions, 1):
            print(f"  {i}. {instruction}")
        
        print("\n" + "-"*80)
        print("NUTRITION PER SERVING")
        print("-"*80)
        
        nutrition = adapted.adapted_recipe.nutrition_per_serving
        print(f"  Calories: {nutrition.calories}")
        print(f"  Protein: {nutrition.protein_g}g")
        print(f"  Carbohydrates: {nutrition.carbohydrates_g}g")
        print(f"  Fat: {nutrition.fat_g}g (Saturated: {nutrition.saturated_fat_g}g)")
        print(f"  Fiber: {nutrition.fiber_g}g")
        print(f"  Sodium: {nutrition.sodium_mg}mg")
        print(f"  Potassium: {nutrition.potassium_mg}mg")
        
        print("\n" + "-"*80)
        print("EXPLAINABILITY LOG")
        print("-"*80)
        
        for entry in adapted.explainability_log:
            if entry.get('type') == 'clinical_context':
                print(f"\nClinical Context:")
                print(f"  CKD Stage: {entry['ckd_stage']}")
                print(f"  eGFR: {entry['egfr']}")
                print(f"  Conditions: {entry['conditions']}")
            
            elif entry.get('type') == 'nutrient_compliance':
                print(f"\nNutrient Compliance:")
                for nutrient, data in entry.items():
                    if nutrient == 'type':
                        continue
                    print(f"  {nutrient.title()}:")
                    print(f"    Value: {data['value']}")
                    print(f"    Limit/Target: {data.get('limit') or data.get('target')}")
                    print(f"    Compliant: {'✓' if data['compliant'] else '✗'}")
                    if 'citation' in data:
                        print(f"    Citation: {data['citation']}")
            
            else:
                print(f"\n{entry.get('action', 'Note').title()}:")
                for key, value in entry.items():
                    if key != 'action':
                        print(f"  {key}: {value}")
        
        print("\n" + "="*80)


def main():
    """Demo of hybrid RAG recipe system."""
    
    print("="*80)
    print("PANTRY-TO-PLATE: HYBRID RAG RECIPE GENERATOR")
    print("="*80)
    
    # Initialize system
    print("\nInitializing recipe generation system...")
    system = HybridRAGRecipeSystem()
    
    # Note: In production, you would have run pantry_inventory.py first
    print("\nNote: Using pre-generated pantry_summary.json and clinical_constraint.json")
    
    # Generate meal plan
    print("\nGenerating clinically adapted meal plan...")
    try:
        adapted_recipes = system.generate_meal_plan(
            pantry_summary_file='pantry_summary.json',
            constraint_file='clinical_constraint_MIMIC_10000032.json',
            num_recipes=2
        )
        
        # Display adapted recipes
        for adapted in adapted_recipes:
            system.print_adapted_recipe(adapted)
        
        # Export meal plan
        system.export_meal_plan(adapted_recipes, 'adapted_meal_plan.json')
        
        print("\n✅ Meal plan generation complete!")
    
    except FileNotFoundError as e:
        print(f"\n⚠️  Required files not found: {e}")
        print("Please run pantry_inventory.py first to generate pantry_summary.json")


if __name__ == "__main__":
    main()

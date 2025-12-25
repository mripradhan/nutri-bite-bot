"""
Hierarchical Clinical Rules Engine for Pantry-to-Plate
=======================================================
Resolves conflicting nutrition guidelines based on clinical severity hierarchy.

Clinical Priority Hierarchy (Highest to Lowest):
1. Renal Safety (CKD Stage 3-5) - Prevents hyperkalemia and uremia
2. Cardiovascular Safety (HTN) - Prevents stroke and heart failure
3. Metabolic Control (Diabetes) - Prevents long-term complications
4. Lipid Management (Dyslipidemia) - Reduces atherosclerosis risk
5. Endocrine Balance (Hypothyroidism) - Prevents medication interference

Author: Pantry-to-Plate Development Team
Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import IntEnum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalPriority(IntEnum):
    """Clinical priority levels (lower number = higher priority)."""
    CRITICAL_RENAL = 1      # Life-threatening (hyperkalemia, uremia)
    CRITICAL_CARDIAC = 2     # Life-threatening (hypertensive crisis)
    HIGH_METABOLIC = 3       # Serious (diabetic complications)
    MEDIUM_LIPID = 4         # Important (CVD risk)
    LOW_ENDOCRINE = 5        # Manageable (medication timing)
    GENERAL_HEALTH = 6       # General wellness


class CKDStage(IntEnum):
    """Chronic Kidney Disease stages based on eGFR."""
    NORMAL = 0               # eGFR >= 90
    STAGE_1 = 1              # eGFR >= 90 with kidney damage
    STAGE_2 = 2              # eGFR 60-89 (Mild)
    STAGE_3A = 3             # eGFR 45-59 (Moderate decline)
    STAGE_3B = 4             # eGFR 30-44 (Moderate to severe)
    STAGE_4 = 5              # eGFR 15-29 (Severe)
    STAGE_5 = 6              # eGFR < 15 (Kidney failure)


@dataclass
class NutrientLimit:
    """Represents a nutrient limit with rationale."""
    daily_max: Optional[float] = None
    daily_min: Optional[float] = None
    per_meal_max: Optional[float] = None
    per_meal_min: Optional[float] = None
    unit: str = "mg"
    priority: ClinicalPriority = ClinicalPriority.GENERAL_HEALTH
    rationale: str = ""
    override_reason: Optional[str] = None


@dataclass
class FoodRestriction:
    """Represents a specific food restriction."""
    food_name: str
    severity: str  # "prohibited", "limited", "warning"
    reason: str
    priority: ClinicalPriority
    alternative_foods: List[str] = None
    temporal_restriction: Optional[str] = None  # e.g., "4 hours before levothyroxine"
    
    def __post_init__(self):
        if self.alternative_foods is None:
            self.alternative_foods = []


@dataclass
class ProteinCalculation:
    """Protein requirements based on weight and conditions."""
    weight_kg: float
    target_grams_per_kg: float
    daily_protein_min_g: float
    daily_protein_max_g: float
    per_meal_protein_g: float
    rationale: str


@dataclass
class ClinicalConstraint:
    """Complete clinical constraints for meal generation."""
    user_id: str
    generation_timestamp: str
    
    # Medical context
    medical_conditions: Dict[str, bool]
    ckd_stage: Optional[str]
    egfr: Optional[float]
    current_potassium: Optional[float]
    
    # Macronutrient constraints
    protein: ProteinCalculation
    
    # Micronutrient constraints (daily limits)
    sodium: NutrientLimit
    potassium: NutrientLimit
    phosphorus: NutrientLimit
    carbohydrates: NutrientLimit
    
    # Food restrictions
    prohibited_foods: List[FoodRestriction]
    limited_foods: List[FoodRestriction]
    warning_foods: List[FoodRestriction]
    
    # Temporal restrictions (medication interactions)
    temporal_warnings: List[Dict[str, str]]
    
    # Conflict resolutions applied
    conflict_resolutions: List[Dict[str, str]]
    
    # Clinical safety notes
    safety_notes: List[str]


class HierarchicalClinicalRulesEngine:
    """
    Hierarchical Clinical Rules Engine for resolving conflicting nutrition guidelines.
    
    This engine applies evidence-based clinical decision rules to generate safe,
    personalized nutrition constraints from MIMIC-IV patient data.
    """
    
    def __init__(self):
        """Initialize the clinical rules engine with reference values."""
        
        # CKD Stage Classification (based on KDIGO guidelines)
        self.ckd_stages = {
            'normal': {'egfr_min': 90, 'stage': CKDStage.NORMAL},
            'stage_1': {'egfr_min': 90, 'stage': CKDStage.STAGE_1},
            'stage_2': {'egfr_min': 60, 'stage': CKDStage.STAGE_2},
            'stage_3a': {'egfr_min': 45, 'stage': CKDStage.STAGE_3A},
            'stage_3b': {'egfr_min': 30, 'stage': CKDStage.STAGE_3B},
            'stage_4': {'egfr_min': 15, 'stage': CKDStage.STAGE_4},
            'stage_5': {'egfr_min': 0, 'stage': CKDStage.STAGE_5}
        }
        
        # Reference nutrient limits (KDOQI, DASH, ADA guidelines)
        self.reference_limits = {
            'potassium': {
                'general': {'max': 4700, 'unit': 'mg'},  # DRI
                'htn_target': {'min': 4700, 'unit': 'mg'},  # DASH diet
                'ckd_stage_3': {'max': 2000, 'unit': 'mg'},  # KDOQI
                'ckd_stage_4': {'max': 2000, 'unit': 'mg'},  # KDOQI
                'ckd_stage_5': {'max': 2000, 'unit': 'mg'}   # KDOQI
            },
            'sodium': {
                'general': {'max': 2300, 'unit': 'mg'},  # DRI
                'htn': {'max': 1500, 'unit': 'mg'},      # AHA recommendation
                'ckd': {'max': 2000, 'unit': 'mg'}       # KDOQI
            },
            'phosphorus': {
                'general': {'max': 1250, 'unit': 'mg'},  # DRI
                'ckd_stage_3': {'max': 1000, 'unit': 'mg'},
                'ckd_stage_4': {'max': 800, 'unit': 'mg'},
                'ckd_stage_5': {'max': 800, 'unit': 'mg'}
            },
            'protein': {
                'general': {'grams_per_kg': 0.8},
                'ckd_dm': {'min_grams_per_kg': 0.6, 'max_grams_per_kg': 0.8},  # KDOQI
                'ckd_no_dm': {'grams_per_kg': 0.8}
            },
            'carbohydrates': {
                'diabetes': {'percent_calories': (45, 65)}  # ADA
            }
        }
        
        # High-potassium foods requiring restriction
        self.high_potassium_foods = {
            'potatoes': {
                'potassium_per_100g': 425,
                'severity': 'prohibited',
                'alternatives': ['cauliflower', 'turnips', 'radishes']
            },
            'sweet_potatoes': {
                'potassium_per_100g': 475,
                'severity': 'prohibited',
                'alternatives': ['butternut_squash', 'carrots']
            },
            'bananas': {
                'potassium_per_100g': 358,
                'severity': 'prohibited',
                'alternatives': ['berries', 'apples', 'grapes']
            },
            'oranges': {
                'potassium_per_100g': 181,
                'severity': 'limited',
                'alternatives': ['lemons', 'limes']
            },
            'tomatoes': {
                'potassium_per_100g': 237,
                'severity': 'limited',
                'alternatives': ['cucumber', 'bell_peppers']
            },
            'spinach': {
                'potassium_per_100g': 558,
                'severity': 'limited',
                'alternatives': ['lettuce', 'cabbage']
            }
        }
        
        # Goitrogenic foods (for hypothyroidism with levothyroxine)
        self.goitrogenic_foods = {
            'soy': {
                'interaction': 'levothyroxine_absorption',
                'temporal_gap': '4 hours',
                'severity': 'warning',
                'note': 'Soy products can reduce levothyroxine absorption by up to 50%'
            },
            'cabbage': {
                'interaction': 'iodine_deficiency_only',
                'severity': 'conditional',
                'note': 'Only restrict if iodine deficiency confirmed'
            },
            'broccoli': {
                'interaction': 'iodine_deficiency_only',
                'severity': 'conditional',
                'note': 'Only restrict if iodine deficiency confirmed'
            }
        }
    
    def classify_ckd_stage(self, egfr: Optional[float]) -> Tuple[str, CKDStage]:
        """
        Classify CKD stage based on eGFR.
        
        Args:
            egfr: Estimated glomerular filtration rate (mL/min/1.73m²)
            
        Returns:
            Tuple of (stage_name, CKDStage enum)
        """
        if egfr is None:
            return "unknown", None
        
        if egfr >= 90:
            return "Stage 1-2 (Mild or Normal)", CKDStage.STAGE_2
        elif egfr >= 60:
            return "Stage 2 (Mild)", CKDStage.STAGE_2
        elif egfr >= 45:
            return "Stage 3a (Moderate)", CKDStage.STAGE_3A
        elif egfr >= 30:
            return "Stage 3b (Moderate-Severe)", CKDStage.STAGE_3B
        elif egfr >= 15:
            return "Stage 4 (Severe)", CKDStage.STAGE_4
        else:
            return "Stage 5 (Kidney Failure)", CKDStage.STAGE_5
    
    def resolve_potassium_conflict(
        self,
        has_htn: bool,
        has_ckd: bool,
        egfr: Optional[float],
        current_potassium: Optional[float]
    ) -> Tuple[NutrientLimit, List[Dict[str, str]]]:
        """
        CRITICAL RULE: Resolve Renal vs. HTN potassium conflict.
        
        Clinical Logic:
        - HTN alone: Encourage high potassium (DASH diet: 4700 mg/day)
        - CKD Stage 3-5: HARD CAP at 2000 mg/day (prevents hyperkalemia)
        - CKD overrides HTN due to life-threatening hyperkalemia risk
        
        Args:
            has_htn: Hypertension diagnosis
            has_ckd: Chronic kidney disease diagnosis
            egfr: Estimated GFR (mL/min/1.73m²)
            current_potassium: Current serum potassium (mEq/L)
            
        Returns:
            Tuple of (NutrientLimit, conflict_resolutions)
        """
        conflict_resolutions = []
        
        # Determine CKD stage
        stage_name, ckd_stage = self.classify_ckd_stage(egfr)
        
        # Default limit
        limit = NutrientLimit(
            daily_max=4700,
            daily_min=2000,
            per_meal_max=1500,
            unit="mg",
            priority=ClinicalPriority.GENERAL_HEALTH,
            rationale="General population recommendation"
        )
        
        # CRITICAL: CKD Stage 3-5 overrides all other recommendations
        if has_ckd and egfr is not None and egfr < 60:
            # HARD CAP for CKD Stage 3-5
            limit = NutrientLimit(
                daily_max=2000,
                daily_min=None,
                per_meal_max=650,  # ~3 meals/day
                unit="mg",
                priority=ClinicalPriority.CRITICAL_RENAL,
                rationale=f"CKD {stage_name}: Strict potassium restriction to prevent hyperkalemia",
                override_reason="Renal safety overrides HTN potassium recommendation"
            )
            
            if has_htn:
                conflict_resolutions.append({
                    'conflict': 'HTN vs CKD Potassium Requirements',
                    'htn_recommendation': 'High potassium (≥4700 mg/day) for blood pressure control',
                    'ckd_recommendation': 'Low potassium (≤2000 mg/day) to prevent hyperkalemia',
                    'resolution': 'CKD restriction applied (Priority 1: Renal Safety)',
                    'rationale': 'Hyperkalemia is life-threatening in CKD. Cardiac arrest risk > HTN benefit.',
                    'alternative_htn_management': 'Use DASH diet principles with low-K foods, ACE inhibitors'
                })
            
            # Additional warning if current K+ is elevated
            if current_potassium is not None and current_potassium > 5.0:
                limit.daily_max = 1500  # Even stricter if K+ elevated
                limit.rationale += f" (Current K+: {current_potassium} mEq/L - Elevated)"
                conflict_resolutions.append({
                    'alert': 'Elevated Serum Potassium',
                    'value': f'{current_potassium} mEq/L',
                    'action': 'Reduced potassium limit to 1500 mg/day',
                    'urgency': 'HIGH - Risk of cardiac arrhythmia'
                })
        
        # HTN alone (no significant CKD)
        elif has_htn and (not has_ckd or egfr >= 60):
            limit = NutrientLimit(
                daily_max=None,  # No upper limit
                daily_min=4700,
                per_meal_max=None,
                unit="mg",
                priority=ClinicalPriority.CRITICAL_CARDIAC,
                rationale="HTN: DASH diet recommends high potassium for blood pressure control"
            )
        
        return limit, conflict_resolutions
    
    def resolve_food_restrictions(
        self,
        has_htn: bool,
        has_ckd: bool,
        egfr: Optional[float],
        has_hypothyroidism: bool = False
    ) -> Tuple[List[FoodRestriction], List[FoodRestriction], List[FoodRestriction]]:
        """
        Resolve specific food restrictions based on conditions.
        
        Returns:
            Tuple of (prohibited_foods, limited_foods, warning_foods)
        """
        prohibited = []
        limited = []
        warnings = []
        
        # Potato restriction for HTN or CKD (as per medical professional's note)
        if has_htn or (has_ckd and egfr is not None and egfr < 60):
            potato_data = self.high_potassium_foods['potatoes']
            
            if has_ckd and egfr < 60:
                # Complete prohibition for CKD Stage 3-5
                prohibited.append(FoodRestriction(
                    food_name="Potatoes (all varieties)",
                    severity="prohibited",
                    reason=f"High potassium content (425 mg/100g). CKD Stage 3-5 requires K+ restriction.",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    alternative_foods=potato_data['alternatives']
                ))
                
                # Also restrict other high-K vegetables
                prohibited.append(FoodRestriction(
                    food_name="Sweet Potatoes",
                    severity="prohibited",
                    reason="Very high potassium (475 mg/100g)",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    alternative_foods=['butternut_squash', 'carrots']
                ))
                
                prohibited.append(FoodRestriction(
                    food_name="Bananas",
                    severity="prohibited",
                    reason="High potassium (358 mg/100g)",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    alternative_foods=['berries', 'apples', 'grapes']
                ))
                
                # Limited foods (can have small portions)
                limited.append(FoodRestriction(
                    food_name="Tomatoes",
                    severity="limited",
                    reason="Moderate potassium (237 mg/100g). Limit to <50g per meal",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    alternative_foods=['cucumber', 'bell_peppers']
                ))
                
                limited.append(FoodRestriction(
                    food_name="Spinach",
                    severity="limited",
                    reason="Very high potassium (558 mg/100g). Limit to <30g per meal",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    alternative_foods=['lettuce', 'bok_choy']
                ))
            
            elif has_htn:
                # For HTN alone, provide warning but don't prohibit
                warnings.append(FoodRestriction(
                    food_name="Potatoes (especially fried/processed)",
                    severity="warning",
                    reason="Often prepared with high sodium. Prefer baked/steamed without salt.",
                    priority=ClinicalPriority.CRITICAL_CARDIAC,
                    alternative_foods=['sweet_potatoes_baked', 'cauliflower_mash']
                ))
        
        # Hypothyroidism - Soy-Levothyroxine interaction
        if has_hypothyroidism:
            warnings.append(FoodRestriction(
                food_name="Soy Products (tofu, soy milk, edamame)",
                severity="warning",
                reason="Soy interferes with levothyroxine absorption (reduces efficacy by up to 50%)",
                priority=ClinicalPriority.LOW_ENDOCRINE,
                alternative_foods=['almond_milk', 'oat_milk', 'chicken', 'fish'],
                temporal_restriction="Consume ≥4 hours after levothyroxine dose"
            ))
            
            # Note: Cabbage/goitrogens only restricted if iodine deficient
            warnings.append(FoodRestriction(
                food_name="Cabbage, Broccoli, Cauliflower (Cruciferous vegetables)",
                severity="warning",
                reason="Goitrogenic effect only significant with iodine deficiency. Generally safe when cooked.",
                priority=ClinicalPriority.LOW_ENDOCRINE,
                alternative_foods=[],
                temporal_restriction="Only restrict if iodine deficiency confirmed"
            ))
        
        return prohibited, limited, warnings
    
    def calculate_protein_requirements(
        self,
        weight_kg: Optional[float],
        has_ckd: bool,
        has_diabetes: bool,
        egfr: Optional[float]
    ) -> ProteinCalculation:
        """
        Calculate protein requirements based on weight and conditions.
        
        Clinical Guidelines:
        - CKD + Diabetes: 0.6-0.8 g/kg/day (KDOQI)
        - CKD without DM: 0.8 g/kg/day
        - General population: 0.8 g/kg/day
        
        Args:
            weight_kg: Patient weight in kg
            has_ckd: CKD diagnosis
            has_diabetes: Diabetes diagnosis
            egfr: Estimated GFR
            
        Returns:
            ProteinCalculation with targets
        """
        # Default weight if missing
        if weight_kg is None:
            weight_kg = 70.0  # Standard reference weight
            weight_note = " (using reference weight - actual weight unavailable)"
        else:
            weight_note = ""
        
        # Determine protein target based on conditions
        if has_ckd and has_diabetes and egfr is not None and egfr < 60:
            # CKD Stage 3-5 with Diabetes: 0.6-0.8 g/kg/day
            min_g_per_kg = 0.6
            max_g_per_kg = 0.8
            rationale = (
                f"CKD with Diabetes: Moderate protein restriction (0.6-0.8 g/kg/day) "
                f"to slow CKD progression while maintaining adequate nutrition{weight_note}"
            )
        elif has_ckd and egfr is not None and egfr < 45:
            # CKD Stage 3b-5 without DM: 0.6-0.75 g/kg/day
            min_g_per_kg = 0.6
            max_g_per_kg = 0.75
            rationale = (
                f"Advanced CKD (Stage 3b-5): Protein restriction (0.6-0.75 g/kg/day) "
                f"to reduce uremic toxins{weight_note}"
            )
        else:
            # General population or mild CKD
            min_g_per_kg = 0.8
            max_g_per_kg = 1.0
            rationale = f"Standard protein intake (0.8-1.0 g/kg/day){weight_note}"
        
        # Calculate daily protein targets
        daily_min = weight_kg * min_g_per_kg
        daily_max = weight_kg * max_g_per_kg
        
        # Per-meal target (assuming 3 meals + 1-2 snacks)
        per_meal = (daily_min + daily_max) / 2 / 3  # Average divided by 3 meals
        
        return ProteinCalculation(
            weight_kg=weight_kg,
            target_grams_per_kg=(min_g_per_kg + max_g_per_kg) / 2,
            daily_protein_min_g=round(daily_min, 1),
            daily_protein_max_g=round(daily_max, 1),
            per_meal_protein_g=round(per_meal, 1),
            rationale=rationale
        )
    
    def calculate_sodium_limit(
        self,
        has_htn: bool,
        has_ckd: bool,
        egfr: Optional[float]
    ) -> NutrientLimit:
        """Calculate sodium limit based on conditions."""
        
        if has_htn and (has_ckd and egfr is not None and egfr < 60):
            # Both HTN and CKD: Use lower limit
            return NutrientLimit(
                daily_max=1500,
                per_meal_max=500,
                unit="mg",
                priority=ClinicalPriority.CRITICAL_CARDIAC,
                rationale="HTN + CKD: Strict sodium restriction (AHA/KDOQI)"
            )
        elif has_htn:
            return NutrientLimit(
                daily_max=1500,
                per_meal_max=500,
                unit="mg",
                priority=ClinicalPriority.CRITICAL_CARDIAC,
                rationale="HTN: Sodium restriction for blood pressure control (AHA)"
            )
        elif has_ckd:
            return NutrientLimit(
                daily_max=2000,
                per_meal_max=650,
                unit="mg",
                priority=ClinicalPriority.CRITICAL_RENAL,
                rationale="CKD: Moderate sodium restriction (KDOQI)"
            )
        else:
            return NutrientLimit(
                daily_max=2300,
                per_meal_max=750,
                unit="mg",
                priority=ClinicalPriority.GENERAL_HEALTH,
                rationale="General population recommendation (DRI)"
            )
    
    def calculate_phosphorus_limit(
        self,
        has_ckd: bool,
        egfr: Optional[float]
    ) -> NutrientLimit:
        """Calculate phosphorus limit for CKD patients."""
        
        if has_ckd and egfr is not None:
            if egfr < 30:  # Stage 4-5
                return NutrientLimit(
                    daily_max=800,
                    per_meal_max=265,
                    unit="mg",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    rationale="CKD Stage 4-5: Strict phosphorus restriction to prevent bone disease"
                )
            elif egfr < 60:  # Stage 3
                return NutrientLimit(
                    daily_max=1000,
                    per_meal_max=330,
                    unit="mg",
                    priority=ClinicalPriority.CRITICAL_RENAL,
                    rationale="CKD Stage 3: Moderate phosphorus restriction"
                )
        
        return NutrientLimit(
            daily_max=1250,
            per_meal_max=415,
            unit="mg",
            priority=ClinicalPriority.GENERAL_HEALTH,
            rationale="General population recommendation"
        )
    
    def generate_clinical_constraints(
        self,
        user_profile: Dict
    ) -> ClinicalConstraint:
        """
        Generate complete clinical constraints from MIMIC-IV user profile.
        
        Args:
            user_profile: User medical record from MIMIC-IV extraction
            
        Returns:
            ClinicalConstraint object with all nutrition rules
        """
        logger.info(f"Generating clinical constraints for {user_profile['user_id']}")
        
        # Extract patient data
        user_id = user_profile['user_id']
        conditions = user_profile['medical_conditions']
        demographics = user_profile['demographics']
        labs = user_profile['laboratory_results']
        
        # Extract key parameters
        weight_kg = demographics.get('weight_kg')
        egfr = labs['renal_profile'].get('egfr')
        current_k = labs['electrolytes'].get('potassium')
        
        # Classify CKD stage
        ckd_stage_name, ckd_stage = self.classify_ckd_stage(egfr)
        
        # Check for hypothyroidism (not in MIMIC-IV base extraction, but included for completeness)
        has_hypothyroidism = conditions.get('hypothyroidism', False)
        
        conflict_resolutions = []
        
        # 1. CRITICAL: Resolve Potassium Conflict
        potassium_limit, k_conflicts = self.resolve_potassium_conflict(
            has_htn=conditions.get('hypertension', False),
            has_ckd=conditions.get('chronic_kidney_disease', False),
            egfr=egfr,
            current_potassium=current_k
        )
        conflict_resolutions.extend(k_conflicts)
        
        # 2. Calculate Protein Requirements
        protein_calc = self.calculate_protein_requirements(
            weight_kg=weight_kg,
            has_ckd=conditions.get('chronic_kidney_disease', False),
            has_diabetes=conditions.get('type2_diabetes', False),
            egfr=egfr
        )
        
        # 3. Calculate other nutrient limits
        sodium_limit = self.calculate_sodium_limit(
            has_htn=conditions.get('hypertension', False),
            has_ckd=conditions.get('chronic_kidney_disease', False),
            egfr=egfr
        )
        
        phosphorus_limit = self.calculate_phosphorus_limit(
            has_ckd=conditions.get('chronic_kidney_disease', False),
            egfr=egfr
        )
        
        # 4. Carbohydrate limits for diabetes
        carb_limit = NutrientLimit(
            daily_max=None,
            daily_min=None,
            per_meal_max=60,  # ~45-60g per meal for diabetes
            unit="g",
            priority=ClinicalPriority.HIGH_METABOLIC,
            rationale="Diabetes: Distribute carbs evenly across meals, prefer low GI (ADA)"
        )
        
        # 5. Resolve Food Restrictions
        prohibited, limited, warning_foods = self.resolve_food_restrictions(
            has_htn=conditions.get('hypertension', False),
            has_ckd=conditions.get('chronic_kidney_disease', False),
            egfr=egfr,
            has_hypothyroidism=has_hypothyroidism
        )
        
        # 6. Temporal Warnings
        temporal_warnings = []
        if has_hypothyroidism:
            temporal_warnings.append({
                'medication': 'Levothyroxine',
                'food_interaction': 'Soy products',
                'timing': 'Take levothyroxine on empty stomach; wait ≥4 hours before consuming soy',
                'severity': 'MEDIUM',
                'reason': 'Soy reduces levothyroxine absorption by up to 50%'
            })
        
        # 7. Safety Notes
        safety_notes = []
        
        if egfr is not None and egfr < 30:
            safety_notes.append(
                "CRITICAL: Advanced CKD (Stage 4-5). Strict dietary compliance essential. "
                "Monitor K+, phosphorus, and fluid status closely."
            )
        
        if current_k is not None and current_k > 5.5:
            safety_notes.append(
                f"ALERT: Elevated serum potassium ({current_k} mEq/L). "
                "Risk of cardiac arrhythmia. Immediate potassium restriction required."
            )
        
        if conditions.get('type2_diabetes') and labs['diabetes'].get('hba1c', 0) > 9.0:
            safety_notes.append(
                f"ALERT: Poor glycemic control (HbA1c {labs['diabetes']['hba1c']}%). "
                "Strict carbohydrate management and meal timing critical."
            )
        
        # Create comprehensive constraint object
        constraint = ClinicalConstraint(
            user_id=user_id,
            generation_timestamp=datetime.now().isoformat(),
            medical_conditions=conditions,
            ckd_stage=ckd_stage_name if egfr else None,
            egfr=egfr,
            current_potassium=current_k,
            protein=protein_calc,
            sodium=sodium_limit,
            potassium=potassium_limit,
            phosphorus=phosphorus_limit,
            carbohydrates=carb_limit,
            prohibited_foods=prohibited,
            limited_foods=limited,
            warning_foods=warning_foods,
            temporal_warnings=temporal_warnings,
            conflict_resolutions=conflict_resolutions,
            safety_notes=safety_notes
        )
        
        logger.info(f"Clinical constraints generated successfully for {user_id}")
        
        return constraint
    
    def export_constraint_json(
        self,
        constraint: ClinicalConstraint,
        output_path: str
    ):
        """
        Export clinical constraint to JSON file.
        
        Args:
            constraint: ClinicalConstraint object
            output_path: Path to save JSON
        """
        # Convert dataclass to dict
        constraint_dict = asdict(constraint)
        
        with open(output_path, 'w') as f:
            json.dump(constraint_dict, f, indent=2)
        
        logger.info(f"Clinical constraint exported to {output_path}")
    
    def generate_constraint_summary(self, constraint: ClinicalConstraint) -> str:
        """Generate human-readable summary of constraints."""
        
        summary = []
        summary.append("=" * 80)
        summary.append("CLINICAL NUTRITION CONSTRAINTS SUMMARY")
        summary.append("=" * 80)
        summary.append(f"\nPatient ID: {constraint.user_id}")
        summary.append(f"Generated: {constraint.generation_timestamp}")
        
        if constraint.ckd_stage:
            summary.append(f"\nCKD Stage: {constraint.ckd_stage} (eGFR: {constraint.egfr} mL/min/1.73m²)")
        
        # Safety alerts
        if constraint.safety_notes:
            summary.append("\n" + "=" * 80)
            summary.append("⚠️  SAFETY ALERTS")
            summary.append("=" * 80)
            for note in constraint.safety_notes:
                summary.append(f"  • {note}")
        
        # Conflict resolutions
        if constraint.conflict_resolutions:
            summary.append("\n" + "=" * 80)
            summary.append("🔄 CONFLICT RESOLUTIONS APPLIED")
            summary.append("=" * 80)
            for resolution in constraint.conflict_resolutions:
                summary.append(f"\nConflict: {resolution.get('conflict', 'N/A')}")
                for key, value in resolution.items():
                    if key != 'conflict':
                        summary.append(f"  {key}: {value}")
        
        # Macronutrients
        summary.append("\n" + "=" * 80)
        summary.append("MACRONUTRIENT TARGETS")
        summary.append("=" * 80)
        
        p = constraint.protein
        summary.append(f"\nProtein (based on {p.weight_kg}kg body weight):")
        summary.append(f"  Daily target: {p.daily_protein_min_g}-{p.daily_protein_max_g}g")
        summary.append(f"  Per meal: ~{p.per_meal_protein_g}g")
        summary.append(f"  Rationale: {p.rationale}")
        
        summary.append(f"\nCarbohydrates:")
        summary.append(f"  Per meal: ≤{constraint.carbohydrates.per_meal_max}g")
        summary.append(f"  Rationale: {constraint.carbohydrates.rationale}")
        
        # Micronutrients
        summary.append("\n" + "=" * 80)
        summary.append("MICRONUTRIENT LIMITS")
        summary.append("=" * 80)
        
        for nutrient_name, nutrient in [
            ('Sodium', constraint.sodium),
            ('Potassium', constraint.potassium),
            ('Phosphorus', constraint.phosphorus)
        ]:
            summary.append(f"\n{nutrient_name}:")
            if nutrient.daily_max:
                summary.append(f"  Daily maximum: {nutrient.daily_max} {nutrient.unit}")
            if nutrient.daily_min:
                summary.append(f"  Daily minimum: {nutrient.daily_min} {nutrient.unit}")
            if nutrient.per_meal_max:
                summary.append(f"  Per meal maximum: {nutrient.per_meal_max} {nutrient.unit}")
            summary.append(f"  Priority: {nutrient.priority.name}")
            summary.append(f"  Rationale: {nutrient.rationale}")
            if nutrient.override_reason:
                summary.append(f"  ⚠️  Override: {nutrient.override_reason}")
        
        # Food restrictions
        summary.append("\n" + "=" * 80)
        summary.append("FOOD RESTRICTIONS")
        summary.append("=" * 80)
        
        if constraint.prohibited_foods:
            summary.append("\n🚫 PROHIBITED FOODS:")
            for food in constraint.prohibited_foods:
                summary.append(f"  • {food.food_name}")
                summary.append(f"    Reason: {food.reason}")
                if food.alternative_foods:
                    summary.append(f"    Alternatives: {', '.join(food.alternative_foods)}")
        
        if constraint.limited_foods:
            summary.append("\n⚠️  LIMITED FOODS:")
            for food in constraint.limited_foods:
                summary.append(f"  • {food.food_name}")
                summary.append(f"    Reason: {food.reason}")
        
        if constraint.warning_foods:
            summary.append("\n💡 FOODS WITH WARNINGS:")
            for food in constraint.warning_foods:
                summary.append(f"  • {food.food_name}")
                summary.append(f"    Reason: {food.reason}")
                if food.temporal_restriction:
                    summary.append(f"    Timing: {food.temporal_restriction}")
        
        # Temporal warnings
        if constraint.temporal_warnings:
            summary.append("\n" + "=" * 80)
            summary.append("⏰ MEDICATION-FOOD INTERACTIONS")
            summary.append("=" * 80)
            for warning in constraint.temporal_warnings:
                summary.append(f"\n  Medication: {warning['medication']}")
                summary.append(f"  Food: {warning['food_interaction']}")
                summary.append(f"  Timing: {warning['timing']}")
                summary.append(f"  Reason: {warning['reason']}")
        
        summary.append("\n" + "=" * 80)
        
        return "\n".join(summary)


def main():
    """Main function demonstrating the clinical rules engine."""
    
    print("=" * 80)
    print("HIERARCHICAL CLINICAL RULES ENGINE - PANTRY-TO-PLATE")
    print("=" * 80)
    
    # Load sample user profiles
    print("\nLoading MIMIC-IV user profiles...")
    with open('sample_output.json', 'r') as f:
        data = json.load(f)
    
    # Initialize rules engine
    engine = HierarchicalClinicalRulesEngine()
    
    # Process each patient
    for i, profile in enumerate(data['user_profiles'], 1):
        print(f"\n{'='*80}")
        print(f"Processing Patient {i}/{len(data['user_profiles'])}: {profile['user_id']}")
        print(f"{'='*80}")
        
        # Generate clinical constraints
        constraint = engine.generate_clinical_constraints(profile)
        
        # Print summary
        summary = engine.generate_constraint_summary(constraint)
        print(summary)
        
        # Export to JSON
        output_file = f"clinical_constraint_{profile['user_id']}.json"
        engine.export_constraint_json(constraint, output_file)
        print(f"\n✅ Constraint exported to: {output_file}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

"""
Ingredient-to-Nutrient Mapping and Inventory Check Module
==========================================================
Processes pantry scans, maps to USDA FDC nutrients, and validates against
clinical constraints from the Rules Engine.

Features:
- CV-simulated pantry scan processing
- USDA FoodData Central API integration
- Quantity validation against clinical rules
- Safe vs. High-Risk ingredient classification
- Warning generation for restricted foods

Author: Pantry-to-Plate Development Team
Version: 1.0.0
"""

import json
import requests
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NutrientProfile:
    """Nutrient profile from USDA FDC."""
    fdc_id: int
    description: str
    potassium_mg: Optional[float] = None
    sodium_mg: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    phosphorus_mg: Optional[float] = None
    calories: Optional[float] = None
    fiber_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    source: str = "USDA FDC"


@dataclass
class PantryItem:
    """Item detected in pantry scan."""
    cv_label: str           # From CV system: "Tubes", "Fruits", "Cabbage"
    normalized_name: str    # Standardized: "potato", "apple", "cabbage"
    quantity_g: float       # Amount user has
    fdc_id: Optional[int] = None
    nutrient_profile: Optional[NutrientProfile] = None


@dataclass
class InventoryCheck:
    """Result of inventory validation against clinical rules."""
    item_name: str
    quantity_available_g: float
    quantity_allowed_g: Optional[float]
    status: str  # "safe", "restricted", "prohibited"
    risk_level: str  # "low", "medium", "high", "critical"
    warning_message: Optional[str] = None
    nutrient_concerns: List[str] = None
    
    def __post_init__(self):
        if self.nutrient_concerns is None:
            self.nutrient_concerns = []


@dataclass
class PantrySummary:
    """Complete pantry analysis summary."""
    scan_timestamp: str
    total_items: int
    safe_items: List[Dict]
    restricted_items: List[Dict]
    prohibited_items: List[Dict]
    warnings: List[str]
    recommendations: List[str]


class USDAFoodDataClient:
    """
    Client for USDA FoodData Central API.
    Handles food search and nutrient profile retrieval.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize USDA FDC client.
        
        Args:
            api_key: USDA FDC API key (get from https://fdc.nal.usda.gov/api-key-signup.html)
                    If None, uses DEMO_KEY with rate limits
        """
        self.api_key = api_key or "DEMO_KEY"
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        
        # Common food mappings (CV label -> search terms)
        self.cv_to_search = {
            'tubes': 'potato',
            'tubers': 'potato',
            'potato': 'potato raw',
            'sweet_potato': 'sweet potato raw',
            'fruits': 'apple',
            'apple': 'apple raw',
            'banana': 'banana raw',
            'orange': 'orange raw',
            'cabbage': 'cabbage raw',
            'spinach': 'spinach raw',
            'tomato': 'tomato raw',
            'carrot': 'carrot raw',
            'broccoli': 'broccoli raw',
            'cauliflower': 'cauliflower raw',
            'lettuce': 'lettuce raw',
            'chicken': 'chicken breast raw',
            'fish': 'salmon raw',
            'tofu': 'tofu raw',
            'beans': 'beans kidney raw',
            'rice': 'rice white cooked',
            'bread': 'bread whole wheat'
        }
        
        # Cache to avoid repeated API calls
        self.cache = {}
    
    def search_food(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for food in USDA database.
        
        Args:
            query: Search term
            max_results: Maximum results to return
            
        Returns:
            List of food items with FDC IDs
        """
        # Check cache
        cache_key = f"search_{query}"
        if cache_key in self.cache:
            logger.info(f"Using cached search results for: {query}")
            return self.cache[cache_key]
        
        url = f"{self.base_url}/foods/search"
        params = {
            'api_key': self.api_key,
            'query': query,
            'pageSize': max_results,
            'dataType': ['Foundation', 'SR Legacy']  # Most reliable data
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            foods = data.get('foods', [])
            
            results = []
            for food in foods:
                results.append({
                    'fdc_id': food.get('fdcId'),
                    'description': food.get('description'),
                    'data_type': food.get('dataType')
                })
            
            # Cache results
            self.cache[cache_key] = results
            
            # Rate limiting for DEMO_KEY
            if self.api_key == "DEMO_KEY":
                time.sleep(0.5)
            
            return results
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching food '{query}': {e}")
            return []
    
    def get_nutrient_profile(self, fdc_id: int) -> Optional[NutrientProfile]:
        """
        Get detailed nutrient profile for a food item.
        
        Args:
            fdc_id: USDA FDC ID
            
        Returns:
            NutrientProfile or None
        """
        # Check cache
        cache_key = f"nutrient_{fdc_id}"
        if cache_key in self.cache:
            logger.info(f"Using cached nutrient profile for FDC ID: {fdc_id}")
            return self.cache[cache_key]
        
        url = f"{self.base_url}/food/{fdc_id}"
        params = {'api_key': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract nutrients (per 100g)
            nutrients = {}
            for nutrient in data.get('foodNutrients', []):
                name = nutrient.get('nutrient', {}).get('name', '').lower()
                value = nutrient.get('amount')
                
                # Map nutrient names to our fields
                if 'potassium' in name:
                    nutrients['potassium_mg'] = value
                elif 'sodium' in name:
                    nutrients['sodium_mg'] = value
                elif 'protein' in name:
                    nutrients['protein_g'] = value
                elif 'carbohydrate' in name:
                    nutrients['carbohydrates_g'] = value
                elif 'phosphorus' in name:
                    nutrients['phosphorus_mg'] = value
                elif 'energy' in name and 'kcal' in name.lower():
                    nutrients['calories'] = value
                elif 'fiber' in name:
                    nutrients['fiber_g'] = value
                elif 'saturated' in name and 'fat' in name:
                    nutrients['saturated_fat_g'] = value
            
            profile = NutrientProfile(
                fdc_id=fdc_id,
                description=data.get('description', ''),
                **nutrients
            )
            
            # Cache profile
            self.cache[cache_key] = profile
            
            # Rate limiting for DEMO_KEY
            if self.api_key == "DEMO_KEY":
                time.sleep(0.5)
            
            return profile
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting nutrient profile for FDC ID {fdc_id}: {e}")
            return None


class PantryInventoryManager:
    """
    Manages pantry inventory and validates against clinical constraints.
    """
    
    def __init__(self, usda_api_key: Optional[str] = None):
        """Initialize manager with USDA client."""
        self.usda_client = USDAFoodDataClient(api_key=usda_api_key)
        self.pantry_items: List[PantryItem] = []
        self.clinical_constraint: Optional[Dict] = None
    
    def load_clinical_constraint(self, constraint_file: str):
        """
        Load clinical constraints from Rules Engine output.
        
        Args:
            constraint_file: Path to ClinicalConstraint JSON
        """
        with open(constraint_file, 'r') as f:
            self.clinical_constraint = json.load(f)
        
        logger.info(f"Loaded clinical constraints for {self.clinical_constraint['user_id']}")
    
    def normalize_cv_label(self, cv_label: str) -> str:
        """
        Normalize CV system label to standardized food name.
        
        Args:
            cv_label: Raw label from CV system
            
        Returns:
            Normalized food name
        """
        cv_label_lower = cv_label.lower().strip()
        
        # Direct mapping
        if cv_label_lower in self.usda_client.cv_to_search:
            return cv_label_lower
        
        # Fuzzy matching for common variations
        if 'tube' in cv_label_lower or 'potato' in cv_label_lower:
            return 'potato'
        elif 'fruit' in cv_label_lower or 'apple' in cv_label_lower:
            return 'apple'
        elif 'cabbage' in cv_label_lower or 'lettuce' in cv_label_lower:
            return cv_label_lower
        elif 'tomato' in cv_label_lower:
            return 'tomato'
        elif 'banana' in cv_label_lower:
            return 'banana'
        
        return cv_label_lower
    
    def process_pantry_scan(self, scan_data: List[Dict]) -> List[PantryItem]:
        """
        Process CV pantry scan and map to USDA nutrients.
        
        Args:
            scan_data: List of dicts with 'cv_label' and 'quantity_g'
            
        Returns:
            List of PantryItem objects with nutrient profiles
        """
        logger.info(f"Processing pantry scan with {len(scan_data)} items...")
        
        pantry_items = []
        
        for item_data in scan_data:
            cv_label = item_data['cv_label']
            quantity_g = item_data['quantity_g']
            
            # Normalize label
            normalized_name = self.normalize_cv_label(cv_label)
            
            # Search USDA database
            search_term = self.usda_client.cv_to_search.get(normalized_name, normalized_name)
            search_results = self.usda_client.search_food(search_term)
            
            if not search_results:
                logger.warning(f"No USDA match found for: {cv_label}")
                pantry_items.append(PantryItem(
                    cv_label=cv_label,
                    normalized_name=normalized_name,
                    quantity_g=quantity_g
                ))
                continue
            
            # Use first (best) result
            fdc_id = search_results[0]['fdc_id']
            
            # Get nutrient profile
            nutrient_profile = self.usda_client.get_nutrient_profile(fdc_id)
            
            item = PantryItem(
                cv_label=cv_label,
                normalized_name=normalized_name,
                quantity_g=quantity_g,
                fdc_id=fdc_id,
                nutrient_profile=nutrient_profile
            )
            
            pantry_items.append(item)
            logger.info(f"✓ Mapped '{cv_label}' to FDC ID {fdc_id}")
        
        self.pantry_items = pantry_items
        return pantry_items
    
    def validate_item_against_constraints(self, item: PantryItem) -> InventoryCheck:
        """
        Validate pantry item against clinical constraints.
        
        Args:
            item: PantryItem to validate
            
        Returns:
            InventoryCheck with status and warnings
        """
        if not self.clinical_constraint:
            raise ValueError("Clinical constraints not loaded. Call load_clinical_constraint() first.")
        
        # Check if item is prohibited
        prohibited_foods = self.clinical_constraint.get('prohibited_foods', [])
        for prohibited in prohibited_foods:
            if item.normalized_name.lower() in prohibited['food_name'].lower():
                return InventoryCheck(
                    item_name=item.normalized_name,
                    quantity_available_g=item.quantity_g,
                    quantity_allowed_g=0,
                    status="prohibited",
                    risk_level="critical",
                    warning_message=f"⛔ PROHIBITED: {prohibited['reason']}",
                    nutrient_concerns=[prohibited['reason']]
                )
        
        # Check if item is limited
        limited_foods = self.clinical_constraint.get('limited_foods', [])
        for limited in limited_foods:
            if item.normalized_name.lower() in limited['food_name'].lower():
                # Extract allowed quantity from reason (e.g., "Limit to <50g per meal")
                allowed_per_meal = self._extract_allowed_quantity(limited['reason'])
                
                return InventoryCheck(
                    item_name=item.normalized_name,
                    quantity_available_g=item.quantity_g,
                    quantity_allowed_g=allowed_per_meal,
                    status="restricted",
                    risk_level="high",
                    warning_message=f"⚠️ RESTRICTED: {limited['reason']}",
                    nutrient_concerns=[limited['reason']]
                )
        
        # Calculate nutrient concerns based on item quantity
        if item.nutrient_profile:
            concerns = self._check_nutrient_concerns(item)
            
            if concerns:
                return InventoryCheck(
                    item_name=item.normalized_name,
                    quantity_available_g=item.quantity_g,
                    quantity_allowed_g=None,
                    status="restricted",
                    risk_level="medium",
                    warning_message=f"⚠️ High in: {', '.join(concerns)}",
                    nutrient_concerns=concerns
                )
        
        # Item is safe
        return InventoryCheck(
            item_name=item.normalized_name,
            quantity_available_g=item.quantity_g,
            quantity_allowed_g=None,
            status="safe",
            risk_level="low",
            warning_message=None,
            nutrient_concerns=[]
        )
    
    def _extract_allowed_quantity(self, reason: str) -> Optional[float]:
        """Extract allowed quantity from restriction reason."""
        import re
        match = re.search(r'<(\d+)g', reason)
        if match:
            return float(match.group(1))
        return None
    
    def _check_nutrient_concerns(self, item: PantryItem) -> List[str]:
        """
        Check if item's nutrients exceed daily limits.
        
        Args:
            item: PantryItem with nutrient profile
            
        Returns:
            List of nutrient concerns
        """
        concerns = []
        profile = item.nutrient_profile
        
        # Calculate nutrients in available quantity (per 100g * quantity/100)
        scaling_factor = item.quantity_g / 100.0
        
        # Potassium check
        if profile.potassium_mg:
            total_k = profile.potassium_mg * scaling_factor
            k_limit = self.clinical_constraint['potassium']['daily_max']
            
            if k_limit and total_k > k_limit * 0.3:  # >30% of daily limit
                concerns.append(f"Potassium ({total_k:.0f}mg)")
        
        # Sodium check
        if profile.sodium_mg:
            total_na = profile.sodium_mg * scaling_factor
            na_limit = self.clinical_constraint['sodium']['daily_max']
            
            if na_limit and total_na > na_limit * 0.3:
                concerns.append(f"Sodium ({total_na:.0f}mg)")
        
        # Phosphorus check
        if profile.phosphorus_mg:
            total_p = profile.phosphorus_mg * scaling_factor
            p_limit = self.clinical_constraint['phosphorus']['daily_max']
            
            if p_limit and total_p > p_limit * 0.3:
                concerns.append(f"Phosphorus ({total_p:.0f}mg)")
        
        return concerns
    
    def generate_pantry_summary(self) -> PantrySummary:
        """
        Generate comprehensive pantry analysis summary.
        
        Returns:
            PantrySummary with safe/restricted/prohibited items
        """
        logger.info("Generating pantry summary...")
        
        safe_items = []
        restricted_items = []
        prohibited_items = []
        warnings = []
        recommendations = []
        
        for item in self.pantry_items:
            check = self.validate_item_against_constraints(item)
            
            item_dict = {
                'name': item.normalized_name,
                'cv_label': item.cv_label,
                'quantity_g': item.quantity_g,
                'fdc_id': item.fdc_id,
                'status': check.status,
                'risk_level': check.risk_level
            }
            
            if item.nutrient_profile:
                item_dict['nutrients'] = {
                    'potassium_mg_per_100g': item.nutrient_profile.potassium_mg,
                    'sodium_mg_per_100g': item.nutrient_profile.sodium_mg,
                    'protein_g_per_100g': item.nutrient_profile.protein_g
                }
            
            if check.status == "safe":
                safe_items.append(item_dict)
            elif check.status == "restricted":
                item_dict['allowed_quantity_g'] = check.quantity_allowed_g
                item_dict['warning'] = check.warning_message
                restricted_items.append(item_dict)
                warnings.append(check.warning_message)
                
                # Generate recommendation
                if check.quantity_allowed_g:
                    excess = item.quantity_g - check.quantity_allowed_g
                    recommendations.append(
                        f"Reduce {item.normalized_name} usage from {item.quantity_g}g to "
                        f"{check.quantity_allowed_g}g (remove {excess}g)"
                    )
            
            elif check.status == "prohibited":
                item_dict['warning'] = check.warning_message
                prohibited_items.append(item_dict)
                warnings.append(check.warning_message)
                recommendations.append(
                    f"DO NOT USE {item.normalized_name}. Consider alternatives: "
                    f"{self._get_alternatives(item.normalized_name)}"
                )
        
        summary = PantrySummary(
            scan_timestamp=datetime.now().isoformat(),
            total_items=len(self.pantry_items),
            safe_items=safe_items,
            restricted_items=restricted_items,
            prohibited_items=prohibited_items,
            warnings=warnings,
            recommendations=recommendations
        )
        
        return summary
    
    def _get_alternatives(self, food_name: str) -> str:
        """Get alternative foods from clinical constraints."""
        prohibited_foods = self.clinical_constraint.get('prohibited_foods', [])
        
        for prohibited in prohibited_foods:
            if food_name.lower() in prohibited['food_name'].lower():
                alternatives = prohibited.get('alternative_foods', [])
                return ', '.join(alternatives) if alternatives else 'none specified'
        
        return 'none specified'
    
    def export_summary(self, output_file: str):
        """Export pantry summary to JSON."""
        summary = self.generate_pantry_summary()
        
        with open(output_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        logger.info(f"Pantry summary exported to {output_file}")
        
        return summary
    
    def print_summary(self):
        """Print human-readable summary."""
        summary = self.generate_pantry_summary()
        
        print("\n" + "="*80)
        print("PANTRY INVENTORY ANALYSIS")
        print("="*80)
        print(f"Scan Time: {summary.scan_timestamp}")
        print(f"Total Items Scanned: {summary.total_items}")
        
        print(f"\n✅ SAFE ITEMS ({len(summary.safe_items)}):")
        for item in summary.safe_items:
            print(f"  • {item['name']}: {item['quantity_g']}g available")
        
        print(f"\n⚠️  RESTRICTED ITEMS ({len(summary.restricted_items)}):")
        for item in summary.restricted_items:
            print(f"  • {item['name']}: {item['quantity_g']}g available")
            if item.get('allowed_quantity_g'):
                print(f"    Allowed: {item['allowed_quantity_g']}g per meal")
            print(f"    {item['warning']}")
        
        print(f"\n⛔ PROHIBITED ITEMS ({len(summary.prohibited_items)}):")
        for item in summary.prohibited_items:
            print(f"  • {item['name']}: {item['quantity_g']}g available")
            print(f"    {item['warning']}")
        
        print(f"\n📋 RECOMMENDATIONS ({len(summary.recommendations)}):")
        for i, rec in enumerate(summary.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Demo of pantry inventory system."""
    
    print("="*80)
    print("PANTRY-TO-PLATE: INVENTORY MANAGEMENT SYSTEM")
    print("="*80)
    
    # Simulated CV pantry scan
    pantry_scan = [
        {'cv_label': 'Tubes', 'quantity_g': 500},         # Potatoes (prohibited for CKD)
        {'cv_label': 'Fruits', 'quantity_g': 300},        # Apples (safe)
        {'cv_label': 'Cabbage', 'quantity_g': 400},       # Cabbage (safe)
        {'cv_label': 'Banana', 'quantity_g': 200},        # Banana (prohibited for CKD)
        {'cv_label': 'Tomato', 'quantity_g': 150},        # Tomato (limited for CKD)
        {'cv_label': 'Spinach', 'quantity_g': 100},       # Spinach (limited for CKD)
        {'cv_label': 'Chicken', 'quantity_g': 600},       # Chicken (safe, protein source)
        {'cv_label': 'Carrot', 'quantity_g': 300}         # Carrot (safe)
    ]
    
    # Initialize manager
    print("\nInitializing Pantry Inventory Manager...")
    manager = PantryInventoryManager(usda_api_key=None)  # Using DEMO_KEY
    
    # Load clinical constraints
    print("Loading clinical constraints...")
    manager.load_clinical_constraint('clinical_constraint_MIMIC_10000032.json')
    
    # Process pantry scan
    print("\nProcessing pantry scan...")
    items = manager.process_pantry_scan(pantry_scan)
    
    print(f"\n✓ Processed {len(items)} items")
    
    # Generate and display summary
    print("\n" + "-"*80)
    manager.print_summary()
    
    # Export to JSON
    print("\nExporting summary to JSON...")
    manager.export_summary('pantry_summary.json')
    
    print("\n✅ Pantry inventory analysis complete!")


if __name__ == "__main__":
    main()

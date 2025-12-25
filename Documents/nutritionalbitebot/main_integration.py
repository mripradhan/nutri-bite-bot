"""
Pantry-to-Plate: Complete End-to-End System Integration
========================================================
Master script integrating all components:
1. MIMIC-IV patient extraction
2. Clinical Rules Engine (conflict resolution)
3. Pantry Inventory Management (CV scan + USDA mapping)
4. Recipe Generation with SHARE methodology
5. Explainability and compliance validation

Author: Pantry-to-Plate Development Team
Version: 1.0.0
"""

import json
import logging
import sys

# Ensure console encoding supports UTF-8 (Windows cp1252 workaround)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Fallback: wrap stdout with UTF-8 writer
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, errors='replace', closefd=False)

from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pantry_to_plate.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PantryToPlateSystem:
    """
    Complete end-to-end system for personalized clinical meal planning.
    """
    
    def __init__(self, config: dict):
        """
        Initialize system with configuration.
        
        Args:
            config: Configuration dictionary with:
                - mimic_path: Path to MIMIC-IV data
                - usda_api_key: USDA FDC API key (optional, uses DEMO_KEY)
                - output_dir: Directory for outputs
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Component modules (imported lazily)
        self.mimic_extractor = None
        self.rules_engine = None
        self.pantry_manager = None
        self.recipe_system = None
    
    def run_full_pipeline(self, patient_id: str = None, pantry_scan: list = None):
        """
        Execute complete pipeline from patient data to adapted recipes.
        
        Args:
            patient_id: Optional specific patient ID (e.g., "MIMIC_10000032")
            pantry_scan: Optional custom pantry scan data
            
        Returns:
            dict with all outputs
        """
        logger.info("="*80)
        logger.info("PANTRY-TO-PLATE: FULL PIPELINE EXECUTION")
        logger.info("="*80)
        
        results = {
            'pipeline_start': datetime.now().isoformat(),
            'steps_completed': [],
            'outputs': {}
        }
        
        try:
            # STEP 1: Extract Patient Data from MIMIC-IV
            logger.info("\n" + "-"*80)
            logger.info("STEP 1: MIMIC-IV Patient Extraction")
            logger.info("-"*80)
            
            patient_profile = self._step1_extract_patient(patient_id)
            results['steps_completed'].append('patient_extraction')
            results['outputs']['patient_profile'] = patient_profile
            
            # STEP 2: Generate Clinical Constraints
            logger.info("\n" + "-"*80)
            logger.info("STEP 2: Clinical Rules Engine (Conflict Resolution)")
            logger.info("-"*80)
            
            clinical_constraint = self._step2_generate_constraints(patient_profile)
            results['steps_completed'].append('clinical_constraints')
            results['outputs']['clinical_constraint'] = clinical_constraint
            
            # STEP 3: Process Pantry Scan
            logger.info("\n" + "-"*80)
            logger.info("STEP 3: Pantry Inventory Management")
            logger.info("-"*80)
            
            pantry_summary = self._step3_process_pantry(
                clinical_constraint, 
                pantry_scan
            )
            results['steps_completed'].append('pantry_inventory')
            results['outputs']['pantry_summary'] = pantry_summary
            
            # STEP 4: Generate Adapted Recipes
            logger.info("\n" + "-"*80)
            logger.info("STEP 4: Recipe Generation with SHARE Methodology")
            logger.info("-"*80)
            
            adapted_recipes = self._step4_generate_recipes(
                pantry_summary,
                clinical_constraint
            )
            results['steps_completed'].append('recipe_generation')
            results['outputs']['adapted_recipes'] = adapted_recipes
            
            # STEP 5: Generate Final Report
            logger.info("\n" + "-"*80)
            logger.info("STEP 5: Final Report Generation")
            logger.info("-"*80)
            
            final_report = self._step5_generate_report(results)
            results['outputs']['final_report'] = final_report
            
            results['pipeline_end'] = datetime.now().isoformat()
            results['status'] = 'SUCCESS'
            
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['status'] = 'FAILED'
            results['error'] = str(e)
            return results
    
    def _step1_extract_patient(self, patient_id: str = None) -> dict:
        """Extract patient data from MIMIC-IV."""
        
        # If patient_id provided, load existing profile
        if patient_id:
            logger.info(f"Loading existing patient profile: {patient_id}")
            profile_file = f'clinical_constraint_{patient_id}.json'
            
            if Path(profile_file).exists():
                with open(profile_file, 'r') as f:
                    return json.load(f)
        
        # Otherwise, use sample data
        logger.info("Using sample patient profile from MIMIC-IV extraction")
        
        sample_file = 'sample_output.json'
        if not Path(sample_file).exists():
            raise FileNotFoundError(
                f"{sample_file} not found. Please run mimic_cohort_extraction.py first."
            )
        
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        patient = data['user_profiles'][0]
        logger.info(f"Loaded patient: {patient['user_id']}")
        logger.info(f"  Age: {patient['demographics']['anchor_age']}")
        logger.info(f"  Conditions: HTN, T2DM, CKD, Dyslipidemia")
        logger.info(f"  eGFR: {patient['laboratory_results']['renal_profile']['egfr']}")
        
        return patient
    
    def _step2_generate_constraints(self, patient_profile: dict) -> dict:
        """Generate clinical constraints using Rules Engine."""
        from clinical_rules_engine import HierarchicalClinicalRulesEngine
        
        if self.rules_engine is None:
            self.rules_engine = HierarchicalClinicalRulesEngine()
        
        # Generate constraints
        constraint = self.rules_engine.generate_clinical_constraints(patient_profile)
        
        # Export to file
        constraint_file = self.output_dir / f"constraint_{patient_profile['user_id']}.json"
        self.rules_engine.export_constraint_json(constraint, str(constraint_file))
        
        logger.info("Clinical constraints generated:")
        logger.info(f"  Potassium limit: {constraint.potassium.daily_max}mg/day")
        logger.info(f"  Sodium limit: {constraint.sodium.daily_max}mg/day")
        logger.info(f"  Protein target: {constraint.protein.per_meal_protein_g}g/meal")
        logger.info(f"  Prohibited foods: {len(constraint.prohibited_foods)}")
        
        # Convert to dict for return
        from dataclasses import asdict
        return asdict(constraint)
    
    def _step3_process_pantry(self, clinical_constraint: dict, 
                              custom_scan: list = None) -> dict:
        """Process pantry scan and validate against constraints."""
        from pantry_inventory import PantryInventoryManager
        
        if self.pantry_manager is None:
            usda_key = self.config.get('usda_api_key')
            self.pantry_manager = PantryInventoryManager(usda_api_key=usda_key)
        
        # Load clinical constraints
        constraint_file = self.output_dir / f"constraint_{clinical_constraint['user_id']}.json"
        self.pantry_manager.load_clinical_constraint(str(constraint_file))
        
        # Use custom scan or default
        if custom_scan is None:
            # Default pantry scan
            custom_scan = [
                {'cv_label': 'Fruits', 'quantity_g': 300},        # Apples (safe)
                {'cv_label': 'Cabbage', 'quantity_g': 400},       # Cabbage (safe)
                {'cv_label': 'Carrot', 'quantity_g': 300},        # Carrot (safe)
                {'cv_label': 'Chicken', 'quantity_g': 600},       # Chicken (protein)
                {'cv_label': 'Tomato', 'quantity_g': 150},        # Tomato (limited)
            ]
        
        logger.info(f"Processing pantry scan with {len(custom_scan)} items...")
        
        # Process scan
        items = self.pantry_manager.process_pantry_scan(custom_scan)
        
        # Generate summary
        summary = self.pantry_manager.generate_pantry_summary()
        
        # Export
        summary_file = self.output_dir / 'pantry_summary.json'
        self.pantry_manager.export_summary(str(summary_file))
        
        logger.info("Pantry inventory analysis:")
        logger.info(f"  Safe items: {len(summary.safe_items)}")
        logger.info(f"  Restricted items: {len(summary.restricted_items)}")
        logger.info(f"  Prohibited items: {len(summary.prohibited_items)}")
        
        from dataclasses import asdict
        return asdict(summary)
    
    def _step4_generate_recipes(self, pantry_summary: dict, 
                                clinical_constraint: dict) -> list:
        """Generate adapted recipes using SHARE methodology."""
        from recipe_generator import HybridRAGRecipeSystem
        
        if self.recipe_system is None:
            self.recipe_system = HybridRAGRecipeSystem()
        
        # Save files for recipe system
        pantry_file = self.output_dir / 'pantry_summary.json'
        constraint_file = self.output_dir / f"constraint_{clinical_constraint['user_id']}.json"
        
        # Generate meal plan
        adapted_recipes = self.recipe_system.generate_meal_plan(
            str(pantry_file),
            str(constraint_file),
            num_recipes=2
        )
        
        # Export
        meal_plan_file = self.output_dir / 'adapted_meal_plan.json'
        self.recipe_system.export_meal_plan(adapted_recipes, str(meal_plan_file))
        
        logger.info(f"Generated {len(adapted_recipes)} adapted recipes:")
        for recipe in adapted_recipes:
            logger.info(f"  - {recipe.adapted_recipe.name}")
            logger.info(f"    Compliance: {recipe.compliance_check['overall_status']}")
        
        from dataclasses import asdict
        return [asdict(r) for r in adapted_recipes]
    
    def _step5_generate_report(self, results: dict) -> dict:
        """Generate final comprehensive report."""
        
        report = {
            'report_id': f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'patient_id': results['outputs']['patient_profile']['user_id'],
            'generation_timestamp': datetime.now().isoformat(),
            'pipeline_summary': {
                'total_steps': len(results['steps_completed']),
                'steps_completed': results['steps_completed'],
                'execution_time': self._calculate_execution_time(
                    results['pipeline_start'],
                    results.get('pipeline_end', datetime.now().isoformat())
                )
            },
            'clinical_summary': self._generate_clinical_summary(results),
            'safety_alerts': self._generate_safety_alerts(results),
            'recipe_summary': self._generate_recipe_summary(results),
            'next_steps': self._generate_next_steps(results)
        }
        
        # Export report
        report_file = self.output_dir / 'final_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("\nFinal Report Summary:")
        logger.info(f"  Patient: {report['patient_id']}")
        logger.info(f"  Safe recipes generated: {len(results['outputs']['adapted_recipes'])}")
        logger.info(f"  Safety alerts: {len(report['safety_alerts'])}")
        
        return report
    
    def _calculate_execution_time(self, start: str, end: str) -> str:
        """Calculate pipeline execution time."""
        from datetime import datetime
        
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        duration = end_dt - start_dt
        
        return f"{duration.total_seconds():.2f} seconds"
    
    def _generate_clinical_summary(self, results: dict) -> dict:
        """Generate clinical summary from results."""
        constraint = results['outputs']['clinical_constraint']
        patient = results['outputs']['patient_profile']
        
        return {
            'patient_id': patient['user_id'],
            'ckd_stage': constraint.get('ckd_stage'),
            'egfr': constraint.get('egfr'),
            'key_restrictions': {
                'potassium_daily_max': constraint['potassium']['daily_max'],
                'sodium_daily_max': constraint['sodium']['daily_max'],
                'protein_per_meal': constraint['protein']['per_meal_protein_g']
            },
            'conflict_resolutions': len(constraint.get('conflict_resolutions', [])),
            'prohibited_foods_count': len(constraint.get('prohibited_foods', []))
        }
    
    def _generate_safety_alerts(self, results: dict) -> list:
        """Generate safety alerts from results."""
        alerts = []
        
        # Check for safety notes in constraints
        constraint = results['outputs']['clinical_constraint']
        if constraint.get('safety_notes'):
            for note in constraint['safety_notes']:
                alerts.append({
                    'type': 'CRITICAL',
                    'source': 'Clinical Rules Engine',
                    'message': note
                })
        
        # Check for pantry warnings
        pantry = results['outputs']['pantry_summary']
        if pantry.get('warnings'):
            for warning in pantry['warnings']:
                alerts.append({
                    'type': 'WARNING',
                    'source': 'Pantry Inventory',
                    'message': warning
                })
        
        # Check recipe compliance
        recipes = results['outputs']['adapted_recipes']
        for recipe in recipes:
            if not recipe['compliance_check']['compliant']:
                alerts.append({
                    'type': 'WARNING',
                    'source': 'Recipe Generator',
                    'message': f"Recipe '{recipe['adapted_recipe']['name']}' has violations"
                })
        
        return alerts
    
    def _generate_recipe_summary(self, results: dict) -> dict:
        """Generate recipe summary."""
        recipes = results['outputs']['adapted_recipes']
        
        return {
            'total_recipes': len(recipes),
            'compliant_recipes': sum(1 for r in recipes if r['compliance_check']['compliant']),
            'share_edits_applied': sum(len(r['share_edits']) for r in recipes),
            'recipes': [
                {
                    'name': r['adapted_recipe']['name'],
                    'compliance': r['compliance_check']['overall_status'],
                    'share_edits': len(r['share_edits'])
                }
                for r in recipes
            ]
        }
    
    def _generate_next_steps(self, results: dict) -> list:
        """Generate recommended next steps."""
        steps = []
        
        # Check for prohibited items in pantry
        pantry = results['outputs']['pantry_summary']
        if pantry.get('prohibited_items'):
            steps.append({
                'priority': 'HIGH',
                'action': 'Remove prohibited items from pantry',
                'items': [item['name'] for item in pantry['prohibited_items']]
            })
        
        # Check for recipe violations
        recipes = results['outputs']['adapted_recipes']
        violations_exist = any(not r['compliance_check']['compliant'] for r in recipes)
        if violations_exist:
            steps.append({
                'priority': 'MEDIUM',
                'action': 'Review recipe violations and adjust portions'
            })
        
        # General recommendations
        steps.append({
            'priority': 'LOW',
            'action': 'Schedule follow-up lab work to monitor eGFR and potassium levels'
        })
        
        return steps


def main():
    """Main execution function."""
    
    # Configuration
    config = {
        'mimic_path': r'C:\Users\pradh\Downloads\mimic-iv-clinical-database-demo-2.2\mimic-iv-clinical-database-demo-2.2',
        'usda_api_key': "JmcQCJYOXNeWwhth8SLXicsi6cg0pHu63D3d7PFM",
        'output_dir': 'outputs'
    }
    
    # Initialize system
    system = PantryToPlateSystem(config)
    
    # Run full pipeline
    results = system.run_full_pipeline(
        patient_id="MIMIC_10000032",  # Optional: specify patient
        pantry_scan=None  # Optional: custom pantry scan
    )
    
    # Print summary
    if results['status'] == 'SUCCESS':
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"Status: {results['status']}")
        print(f"Steps Completed: {len(results['steps_completed'])}")
        print(f"Output Directory: {config['output_dir']}")
        print("\nGenerated Files:")
        print("  - clinical_constraint_*.json")
        print("  - pantry_summary.json")
        print("  - adapted_meal_plan.json")
        print("  - final_report.json")
        print("="*80)
    else:
        print(f"\n❌ Pipeline failed: {results.get('error')}")


if __name__ == "__main__":
    main()

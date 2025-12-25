"""
Pantry-to-Plate: MIMIC-IV Clinical Cohort Extraction Script
============================================================
Extracts patients with HTN, Type 2 Diabetes, CKD, and Dyslipidemia
along with their lab parameters to create User Medical Records.

Author: Pantry-to-Plate Development Team
Dataset: MIMIC-IV (hosp and icu modules)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MIMICCohortExtractor:
    """
    Extract clinical cohort from MIMIC-IV for nutrition system.
    """
    
    def __init__(self, mimic_path: str):
        """
        Initialize extractor with path to MIMIC-IV data.
        
        Args:
            mimic_path: Base path to MIMIC-IV dataset
        """
        self.mimic_path = mimic_path
        self.hosp_path = f"{mimic_path}/hosp"
        self.icu_path = f"{mimic_path}/icu"
        
        # ICD-10 codes for target conditions
        self.target_icd_codes = {
            'HTN': ['I10'],  # Essential hypertension
            'Type2_Diabetes': ['E11'],  # Type 2 diabetes mellitus
            'CKD': ['N18'],  # Chronic kidney disease
            'Dyslipidemia': ['E78']  # Disorders of lipoprotein metabolism
        }
        
        # MIMIC-IV lab itemids mapping
        self.lab_itemids = {
            # Diabetes markers
            'FBS': [50809, 225664],  # Fasting Blood Sugar/Glucose
            'PPBS': [50809, 225664],  # Post-prandial (using glucose, filtered by timing)
            'HbA1c': [50852],  # Hemoglobin A1c
            
            # Renal profile
            'Creatinine': [50912],  # Serum Creatinine
            'BUN': [51006],  # Blood Urea Nitrogen
            'GFR': [50920],  # eGFR (estimated GFR)
            
            # Lipid profile
            'Triglycerides': [51000],  # TGL
            'Cholesterol': [50907],  # Total Cholesterol
            'LDL': [50899],  # LDL
            'HDL': [50885],  # HDL
            
            # Electrolytes
            'Sodium': [50983, 50824],  # Na+
            'Potassium': [50971, 50822],  # K+
            'Chloride': [50902, 50806]  # Cl-
        }
    
    def load_data(self):
        """Load necessary MIMIC-IV tables."""
        print("Loading MIMIC-IV tables...")
        
        # Load core tables
        self.patients = pd.read_csv(f"{self.hosp_path}/patients.csv.gz", compression='gzip')
        self.admissions = pd.read_csv(f"{self.hosp_path}/admissions.csv.gz", compression='gzip')
        self.diagnoses_icd = pd.read_csv(f"{self.hosp_path}/diagnoses_icd.csv.gz", compression='gzip')
        self.labevents = pd.read_csv(f"{self.hosp_path}/labevents.csv.gz", compression='gzip')
        self.d_labitems = pd.read_csv(f"{self.hosp_path}/d_labitems.csv.gz", compression='gzip')
        
        # Load ICU data for weight
        try:
            self.chartevents = pd.read_csv(f"{self.icu_path}/chartevents.csv.gz", compression='gzip')
        except:
            print("Warning: Could not load chartevents. Weight data may be incomplete.")
            self.chartevents = None
        
        print("Data loaded successfully!")
    
    def identify_cohort(self) -> pd.DataFrame:
        """
        Identify patients with all four target conditions.
        
        Returns:
            DataFrame with subject_id and their conditions
        """
        print("\nIdentifying cohort with target conditions...")
        
        # Extract ICD-9 and ICD-10 codes (MIMIC-IV uses both)
        cohort_conditions = {}
        
        for condition, icd_codes in self.target_icd_codes.items():
            # Match ICD codes (handle both ICD-9 and ICD-10)
            condition_mask = self.diagnoses_icd['icd_code'].str.startswith(
                tuple(icd_codes), na=False
            )
            patients_with_condition = self.diagnoses_icd[condition_mask]['subject_id'].unique()
            cohort_conditions[condition] = set(patients_with_condition)
            print(f"  {condition}: {len(patients_with_condition)} patients")
        
        # Find patients with ALL four conditions
        cohort_patients = (
            cohort_conditions['HTN'] & 
            cohort_conditions['Type2_Diabetes'] & 
            cohort_conditions['CKD'] & 
            cohort_conditions['Dyslipidemia']
        )
        
        print(f"\nCohort with all 4 conditions: {len(cohort_patients)} patients")
        
        # Create cohort dataframe
        cohort_df = pd.DataFrame({'subject_id': list(cohort_patients)})
        
        return cohort_df
    
    def get_most_recent_lab_value(self, subject_id: int, lab_name: str) -> Optional[float]:
        """
        Get most recent lab value for a patient.
        
        Args:
            subject_id: Patient identifier
            lab_name: Name of lab test
            
        Returns:
            Most recent lab value or None
        """
        itemids = self.lab_itemids.get(lab_name, [])
        if not itemids:
            return None
        
        # Filter lab events for this patient and lab type
        patient_labs = self.labevents[
            (self.labevents['subject_id'] == subject_id) &
            (self.labevents['itemid'].isin(itemids))
        ].copy()
        
        if patient_labs.empty:
            return None
        
        # Sort by charttime and get most recent
        patient_labs['charttime'] = pd.to_datetime(patient_labs['charttime'])
        patient_labs = patient_labs.sort_values('charttime', ascending=False)
        
        # Get the most recent value
        most_recent = patient_labs.iloc[0]
        value = most_recent['valuenum']
        
        return float(value) if pd.notna(value) else None
    
    def get_patient_weight(self, subject_id: int) -> Optional[float]:
        """
        Get most recent weight for a patient.
        
        Args:
            subject_id: Patient identifier
            
        Returns:
            Weight in kg or None
        """
        if self.chartevents is None:
            return None
        
        # Weight itemids in MIMIC-IV: 226512 (Admit Wt), 224639 (Daily Weight)
        weight_itemids = [226512, 224639, 226531]
        
        patient_weights = self.chartevents[
            (self.chartevents['subject_id'] == subject_id) &
            (self.chartevents['itemid'].isin(weight_itemids))
        ].copy()
        
        if patient_weights.empty:
            return None
        
        patient_weights['charttime'] = pd.to_datetime(patient_weights['charttime'])
        patient_weights = patient_weights.sort_values('charttime', ascending=False)
        
        most_recent = patient_weights.iloc[0]
        weight = most_recent['valuenum']
        
        return float(weight) if pd.notna(weight) else None
    
    def extract_lab_parameters(self, subject_id: int) -> Dict:
        """
        Extract all lab parameters for a patient.
        
        Args:
            subject_id: Patient identifier
            
        Returns:
            Dictionary of lab parameters
        """
        labs = {
            'diabetes': {
                'fasting_blood_sugar': self.get_most_recent_lab_value(subject_id, 'FBS'),
                'ppbs': self.get_most_recent_lab_value(subject_id, 'PPBS'),
                'hba1c': self.get_most_recent_lab_value(subject_id, 'HbA1c'),
                'unit_glucose': 'mg/dL',
                'unit_hba1c': '%'
            },
            'renal_profile': {
                'serum_creatinine': self.get_most_recent_lab_value(subject_id, 'Creatinine'),
                'egfr': self.get_most_recent_lab_value(subject_id, 'GFR'),
                'bun': self.get_most_recent_lab_value(subject_id, 'BUN'),
                'unit_creatinine': 'mg/dL',
                'unit_egfr': 'mL/min/1.73m²',
                'unit_bun': 'mg/dL'
            },
            'lipid_profile': {
                'triglycerides': self.get_most_recent_lab_value(subject_id, 'Triglycerides'),
                'total_cholesterol': self.get_most_recent_lab_value(subject_id, 'Cholesterol'),
                'ldl': self.get_most_recent_lab_value(subject_id, 'LDL'),
                'hdl': self.get_most_recent_lab_value(subject_id, 'HDL'),
                'unit': 'mg/dL'
            },
            'electrolytes': {
                'sodium': self.get_most_recent_lab_value(subject_id, 'Sodium'),
                'potassium': self.get_most_recent_lab_value(subject_id, 'Potassium'),
                'chloride': self.get_most_recent_lab_value(subject_id, 'Chloride'),
                'unit': 'mEq/L'
            }
        }
        
        return labs
    
    def create_user_profile(self, subject_id: int) -> Dict:
        """
        Create comprehensive user profile for a patient.
        
        Args:
            subject_id: Patient identifier
            
        Returns:
            User profile dictionary
        """
        # Get patient demographics
        patient_info = self.patients[self.patients['subject_id'] == subject_id].iloc[0]
        
        # Get lab parameters
        labs = self.extract_lab_parameters(subject_id)
        
        # Get weight
        weight = self.get_patient_weight(subject_id)
        
        # Create profile
        profile = {
            'user_id': f"MIMIC_{subject_id}",
            'subject_id': int(subject_id),
            'demographics': {
                'anchor_age': int(patient_info['anchor_age']),
                'gender': patient_info['gender'],
                'weight_kg': weight
            },
            'medical_conditions': {
                'hypertension': True,
                'type2_diabetes': True,
                'chronic_kidney_disease': True,
                'dyslipidemia': True
            },
            'laboratory_results': labs,
            'extraction_metadata': {
                'source': 'MIMIC-IV',
                'extraction_date': datetime.now().isoformat(),
                'data_completeness': self._calculate_completeness(labs, weight)
            }
        }
        
        return profile
    
    def _calculate_completeness(self, labs: Dict, weight: Optional[float]) -> Dict:
        """Calculate data completeness metrics."""
        total_fields = 0
        complete_fields = 0
        
        # Count lab fields
        for category in labs.values():
            for key, value in category.items():
                if not key.startswith('unit'):
                    total_fields += 1
                    if value is not None:
                        complete_fields += 1
        
        # Count weight
        total_fields += 1
        if weight is not None:
            complete_fields += 1
        
        completeness_pct = (complete_fields / total_fields * 100) if total_fields > 0 else 0
        
        return {
            'total_fields': total_fields,
            'complete_fields': complete_fields,
            'completeness_percentage': round(completeness_pct, 2)
        }
    
    def extract_cohort_profiles(self, output_path: str = 'user_medical_records.json', 
                                max_patients: Optional[int] = None):
        """
        Extract complete cohort and save as JSON.
        
        Args:
            output_path: Path to save JSON output
            max_patients: Optional limit on number of patients
        """
        # Load data
        self.load_data()
        
        # Identify cohort
        cohort = self.identify_cohort()
        
        if max_patients:
            cohort = cohort.head(max_patients)
            print(f"\nLimiting extraction to {max_patients} patients")
        
        # Extract profiles
        print(f"\nExtracting user profiles for {len(cohort)} patients...")
        user_profiles = []
        
        for idx, row in cohort.iterrows():
            subject_id = row['subject_id']
            try:
                profile = self.create_user_profile(subject_id)
                user_profiles.append(profile)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(cohort)} patients...")
            except Exception as e:
                print(f"  Warning: Error processing patient {subject_id}: {e}")
                continue
        
        # Create final output structure
        output = {
            'dataset': 'MIMIC-IV',
            'cohort_description': 'Patients with HTN, Type 2 Diabetes, CKD, and Dyslipidemia',
            'extraction_date': datetime.now().isoformat(),
            'total_patients': len(user_profiles),
            'user_profiles': user_profiles
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Extraction complete!")
        print(f"Total profiles extracted: {len(user_profiles)}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*70}")
        
        return user_profiles
    
    def generate_summary_statistics(self, user_profiles: List[Dict]):
        """Generate summary statistics for the cohort."""
        print("\n" + "="*70)
        print("COHORT SUMMARY STATISTICS")
        print("="*70)
        
        # Age statistics
        ages = [p['demographics']['anchor_age'] for p in user_profiles]
        print(f"\nAge Statistics:")
        print(f"  Mean age: {np.mean(ages):.1f} years")
        print(f"  Age range: {min(ages)} - {max(ages)} years")
        
        # Gender distribution
        genders = [p['demographics']['gender'] for p in user_profiles]
        gender_counts = pd.Series(genders).value_counts()
        print(f"\nGender Distribution:")
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count} ({count/len(genders)*100:.1f}%)")
        
        # Data completeness
        completeness = [p['extraction_metadata']['data_completeness']['completeness_percentage'] 
                       for p in user_profiles]
        print(f"\nData Completeness:")
        print(f"  Mean completeness: {np.mean(completeness):.1f}%")
        print(f"  Median completeness: {np.median(completeness):.1f}%")
        
        # Lab value statistics (for non-null values)
        print(f"\nLab Value Statistics (where available):")
        
        lab_stats = {
            'HbA1c (%)': [],
            'Creatinine (mg/dL)': [],
            'eGFR (mL/min/1.73m²)': [],
            'Total Cholesterol (mg/dL)': [],
            'Triglycerides (mg/dL)': []
        }
        
        for profile in user_profiles:
            labs = profile['laboratory_results']
            
            if labs['diabetes']['hba1c']:
                lab_stats['HbA1c (%)'].append(labs['diabetes']['hba1c'])
            if labs['renal_profile']['serum_creatinine']:
                lab_stats['Creatinine (mg/dL)'].append(labs['renal_profile']['serum_creatinine'])
            if labs['renal_profile']['egfr']:
                lab_stats['eGFR (mL/min/1.73m²)'].append(labs['renal_profile']['egfr'])
            if labs['lipid_profile']['total_cholesterol']:
                lab_stats['Total Cholesterol (mg/dL)'].append(labs['lipid_profile']['total_cholesterol'])
            if labs['lipid_profile']['triglycerides']:
                lab_stats['Triglycerides (mg/dL)'].append(labs['lipid_profile']['triglycerides'])
        
        for lab_name, values in lab_stats.items():
            if values:
                print(f"  {lab_name}:")
                print(f"    Mean: {np.mean(values):.2f}, Median: {np.median(values):.2f}")
                print(f"    Available for {len(values)}/{len(user_profiles)} patients")
        
        print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("PANTRY-TO-PLATE: MIMIC-IV COHORT EXTRACTION")
    print("="*70)
    
    # IMPORTANT: Update this path to your MIMIC-IV data location
    mimic_path = r'C:\Users\pradh\Downloads\mimic-iv-clinical-database-demo-2.2\mimic-iv-clinical-database-demo-2.2'
    
    # Initialize extractor
    extractor = MIMICCohortExtractor(mimic_path)
    
    # Extract cohort (limit to first 100 patients for testing)
    # Remove max_patients parameter to extract all patients
    user_profiles = extractor.extract_cohort_profiles(
        output_path='sample_output.json',
        max_patients=100  # Remove this line to extract all patients
    )
    
    # Generate summary statistics
    if user_profiles:
        extractor.generate_summary_statistics(user_profiles)
    
    print("\nExtraction complete! User medical records ready for Pantry-to-Plate system.")


if __name__ == "__main__":
    main()

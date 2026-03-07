import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Fetch environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # Ensure inserted row is returned
    }

def save_patient_data(patient_data: dict) -> str:
    """
    Saves the patient's medical clinical parameters to the Supabase 'patients' table via REST API.
    Bypasses RLS because we are using the service_role key.
    
    Returns:
        The UUID string of the inserted patient, or None if failed.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not found in environment variables. Database features disabled.")
        return None
        
    try:
        # Extract features ensuring we only send what matches the schema
        payload = {
            "age": int(float(patient_data.get("age", 0))),
            "sex_male": bool(patient_data.get("sex_male", False)),
            "has_htn": bool(patient_data.get("has_htn", False)),
            "has_dm": bool(patient_data.get("has_dm", False)),
            "has_ckd": bool(patient_data.get("has_ckd", False)),
            "serum_sodium": float(patient_data.get("serum_sodium", 0.0)),
            "serum_potassium": float(patient_data.get("serum_potassium", 0.0)),
            "creatinine": float(patient_data.get("creatinine", 0.0)),
            "egfr": float(patient_data.get("egfr", 0.0)),
            "hba1c": float(patient_data.get("hba1c", 0.0)),
            "fbs": float(patient_data.get("fbs", 0.0)),
            "sbp": float(patient_data.get("sbp", 0.0)),
            "dbp": float(patient_data.get("dbp", 0.0)),
            "bmi": float(patient_data.get("bmi", 0.0))
        }
        
        url = f"{SUPABASE_URL}/rest/v1/patients"
        response = requests.post(url, headers=get_headers(), json=payload)
        
        if response.status_code in (200, 201):
            data = response.json()
            if data and len(data) > 0:
                patient_id = data[0]["id"]
                logger.info(f"Successfully saved patient data with ID: {patient_id}")
                return patient_id
        else:
            logger.error(f"Failed to save patient. Code: {response.status_code}, Body: {response.text}")
            
    except Exception as e:
        logger.error(f"Error saving patient data to Supabase: {e}")
        
    return None


def save_recipe(patient_id: str, ingredients_used: list, recipe_content: str) -> str:
    """
    Saves the generated recipe and associates it with a specific patient_id via REST API.
    
    Returns:
        The UUID string of the inserted recipe, or None if failed.
    """
    if not SUPABASE_URL or not SUPABASE_KEY or not patient_id:
        return None
        
    try:
        payload = {
            "patient_id": patient_id,
            "ingredients_used": ingredients_used,
            "recipe_content": recipe_content
        }
        
        url = f"{SUPABASE_URL}/rest/v1/recipes"
        response = requests.post(url, headers=get_headers(), json=payload)
        
        if response.status_code in (200, 201):
            data = response.json()
            if data and len(data) > 0:
                recipe_id = data[0]["id"]
                logger.info(f"Successfully saved recipe for patient {patient_id} with ID: {recipe_id}")
                return recipe_id
        else:
            logger.error(f"Failed to save recipe. Code: {response.status_code}, Body: {response.text}")
            
    except Exception as e:
        logger.error(f"Error saving recipe to Supabase: {e}")
        
    return None

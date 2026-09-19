"""
NutriBiteBot — Flask API Backend
=================================
Serves clinical risk stratification with one TabNet model per target:
  - sodium_sensitivity
  - potassium_sensitivity
  - protein_restriction
  - carb_sensitivity

Model 1 predictions come from the artifact set in artifacts/models/ written by
clinical-models/train_model1.py (see clinical-models/model1_artifacts.py).
"""

import json
import math
import os
import sys
import base64
import tempfile
import requests as http_requests
from difflib import get_close_matches

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from supabase_client import save_patient_data, save_recipe
from groq import Groq
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(MODEL_DIR, "reports")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
IFCT_CSV = os.path.join(BASE_DIR, "clinical-models", "ifct_database.csv")

# ── load ML models at startup ─────────────────────────────────────
# One TabNet model + one monotonic transformer per target, loaded from the
# portable artifact set written by clinical-models/train_model1.py.
sys.path.insert(0, os.path.join(BASE_DIR, "clinical-models"))
from model1_artifacts import Model1Predictor, TARGETS  # noqa: E402

print("Loading Model 1 (per-target TabNet) …")
PREDICTOR = Model1Predictor(MODEL_DIR)
TARGET_NAMES = list(TARGETS)
FEATURE_NAMES = PREDICTOR.feature_names
MODEL_MANIFEST = PREDICTOR.manifest
print(f"  ✓ {len(TARGET_NAMES)} targets, {len(FEATURE_NAMES)} features, "
      f"trained {MODEL_MANIFEST['created_utc']} on {MODEL_MANIFEST['data']['source']}\n")


def predict_risk(patient: dict) -> dict:
    """Per-target {label, severity_score, confidence, proba, feature_attribution}."""
    return PREDICTOR.predict({f: patient.get(f) for f in FEATURE_NAMES})


# ── load reference data ────────────────────────────────────────────
THRESHOLDS = {}
thresholds_path = os.path.join(REPORTS_DIR, "nutrient_thresholds_reference.json")
if os.path.exists(thresholds_path):
    with open(thresholds_path, "r") as f:
        THRESHOLDS = json.load(f)


# Clinical descriptions for each risk domain
RISK_DESCRIPTIONS = {
    "sodium_sensitivity": {
        "name": "Sodium Sensitivity",
        "low": "Normal sodium tolerance — standard dietary sodium is acceptable.",
        "moderate": "Moderate sodium sensitivity — consider reducing processed food intake.",
        "high": "High sodium sensitivity — strict sodium restriction recommended (<2000 mg/day).",
    },
    "potassium_sensitivity": {
        "name": "Potassium Sensitivity",
        "low": "Normal potassium handling — no dietary restrictions needed.",
        "moderate": "Moderate potassium concern — monitor intake of high-K foods.",
        "high": "High potassium risk — avoid bananas, oranges, potatoes; risk of hyperkalemia.",
    },
    "protein_restriction": {
        "name": "Protein Restriction",
        "low": "No protein restriction needed — normal dietary protein is safe.",
        "moderate": "Moderate protein restriction — reduce to 0.8 g/kg/day.",
        "high": "Strict protein restriction — limit to 0.6–0.8 g/kg/day to protect kidneys.",
    },
    "carb_sensitivity": {
        "name": "Carbohydrate Sensitivity",
        "low": "Normal carbohydrate tolerance — balanced intake is fine.",
        "moderate": "Moderate carb sensitivity — prefer complex carbs, reduce simple sugars.",
        "high": "High carb sensitivity — strict glycemic control needed; limit to low-GI foods.",
    },
}


# ══════════════════════════════════════════════════════════════════
#  MODEL 2 — PORTION RECOMMENDATION ENGINE
#  Extracted from train_model2.py: PortionRecommender + BudgetCalc
# ══════════════════════════════════════════════════════════════════

# ── IFCT nutritional database ─────────────────────────────────────
print("Loading IFCT nutritional database …")
_ifct_df = pd.read_csv(IFCT_CSV)
_ifct_df["ingredient_norm"] = _ifct_df["ingredient"].str.lower().str.strip()
_ifct_idx = _ifct_df.set_index("ingredient_norm")
IFCT_INGREDIENTS = _ifct_df["ingredient"].tolist()
_ifct_ingredients_lower = [i.lower() for i in IFCT_INGREDIENTS]
print(f"  ✓ IFCT database: {len(_ifct_df)} ingredients loaded\n")


def ifct_get_nutrients(ingredient: str) -> dict:
    """Get nutrient values per 100g for an ingredient."""
    key = ingredient.lower().strip()
    if key not in _ifct_idx.index:
        raise KeyError(f"Ingredient not found: {ingredient}")
    row = _ifct_idx.loc[key]
    return {
        "sodium_mg": float(row.get("sodium_mg_per_100g", 0)),
        "potassium_mg": float(row.get("potassium_mg_per_100g", 0)),
        "protein_g": float(row.get("protein_g_per_100g", 0)),
        "carbs_g": float(row.get("carbs_g_per_100g", 0)),
        "phosphorus_mg": float(row.get("phosphorus_mg_per_100g", 0)),
        "fat_g": float(row.get("fat_g_per_100g", 0)),
        "fiber_g": float(row.get("fiber_g_per_100g", 0)),
        "calories": float(row.get("calories_per_100g", 0)),
        "category": str(row.get("category", "")),
    }


def ifct_search(query: str, n: int = 8) -> list:
    """Fuzzy search for ingredients matching query."""
    q = query.lower().strip()
    # Exact match
    if q in _ifct_ingredients_lower:
        idx = _ifct_ingredients_lower.index(q)
        return [IFCT_INGREDIENTS[idx]]
    # Partial match
    matches = [
        IFCT_INGREDIENTS[i]
        for i, low in enumerate(_ifct_ingredients_lower)
        if q in low or low in q
    ]
    if matches:
        return matches[:n]
    # Fuzzy
    close = get_close_matches(q, _ifct_ingredients_lower, n=n, cutoff=0.4)
    return [IFCT_INGREDIENTS[_ifct_ingredients_lower.index(m)] for m in close]


# ── Daily nutrient budgets by condition (from Model2Config) ───────
DAILY_BUDGETS = {
    "healthy": {
        "sodium_mg": 2300, "potassium_mg": 4700,
        "protein_g": 56, "carbs_g": 275, "phosphorus_mg": 1250,
    },
    "ckd": {
        "sodium_mg": 2000, "potassium_mg": 2000,
        "protein_g": 42, "carbs_g": 275, "phosphorus_mg": 800,
    },
    "htn": {
        "sodium_mg": 1500, "potassium_mg": 4700,
        "protein_g": 56, "carbs_g": 275, "phosphorus_mg": 1250,
    },
    "dm": {
        "sodium_mg": 2300, "potassium_mg": 4700,
        "protein_g": 56, "carbs_g": 180, "phosphorus_mg": 1250,
    },
    "ckd_htn": {
        "sodium_mg": 1500, "potassium_mg": 2000,
        "protein_g": 42, "carbs_g": 275, "phosphorus_mg": 800,
    },
    "ckd_dm": {
        "sodium_mg": 2000, "potassium_mg": 2000,
        "protein_g": 42, "carbs_g": 150, "phosphorus_mg": 800,
    },
    "ckd_htn_dm": {
        "sodium_mg": 1500, "potassium_mg": 2000,
        "protein_g": 42, "carbs_g": 150, "phosphorus_mg": 800,
    },
}

# Portion thresholds (from Model2Config)
DEFAULT_CAP_G = 300.0      # max grams per ingredient
HALF_PORTION_G = 75.0      # below this → "Half portion"
AVOID_THRESHOLD_G = 5.0    # below this → "Avoid"

# Sigmoid constants for severity→fraction mapping (from PortionRecommender)
_SIG_L = 0.42       # upper asymptote
_SIG_K = 1.60       # steepness
_SIG_S0 = 1.0       # midpoint
_SIG_FLOOR = 0.05   # minimum fraction


def get_daily_budget(has_ckd: bool, has_htn: bool, has_dm: bool) -> dict:
    """Get daily nutrient budget based on conditions."""
    conditions = sorted(
        [c for c, v in [("ckd", has_ckd), ("dm", has_dm), ("htn", has_htn)] if v]
    )
    key = "_".join(conditions) if conditions else "healthy"
    return DAILY_BUDGETS.get(key, DAILY_BUDGETS.get("ckd_htn_dm", DAILY_BUDGETS["healthy"]))


def severity_to_fraction(severity_score: float) -> float:
    """
    Map severity_score ∈ [0.0, 2.0] to a budget fraction via sigmoid.
    severity=0.0 → ~0.40 (generous), severity=2.0 → ~0.08 (restrictive).
    """
    fraction = _SIG_L / (1.0 + math.exp(_SIG_K * (severity_score - _SIG_S0)))
    return max(_SIG_FLOOR, fraction)


def grams_from_budget(
    nutrient_per_100g: float, remaining_budget: float, risk_fraction: float
) -> float:
    """Compute max grams: (budget * fraction / nutrient_per_100g) * 100."""
    if nutrient_per_100g <= 0:
        return float("inf")
    allowed = max(0.0, remaining_budget) * risk_fraction
    return max(0.0, (allowed / nutrient_per_100g) * 100.0)


def recommend_ingredient(ingredient: str, severity_scores: dict, budget: dict) -> dict:
    """
    Recommend a safe portion for one ingredient.
    Returns: {ingredient, max_grams, label, binding_constraint, nutrient_load, nutrients_per_100g}
    """
    try:
        n = ifct_get_nutrients(ingredient)
    except KeyError:
        suggestions = ifct_search(ingredient)
        return {
            "ingredient": ingredient,
            "max_grams": 0,
            "label": "Not Found",
            "binding_constraint": "unknown",
            "explanation": f"Not in IFCT database. Try: {', '.join(suggestions)}" if suggestions else "Not in IFCT database.",
            "nutrient_load": {},
            "nutrients_per_100g": {},
            "suggestions": suggestions,
        }

    # Convert severity scores to budget fractions via sigmoid
    f_sod = severity_to_fraction(severity_scores.get("sodium_sensitivity", 1.0))
    f_pot = severity_to_fraction(severity_scores.get("potassium_sensitivity", 1.0))
    f_pro = severity_to_fraction(severity_scores.get("protein_restriction", 1.0))
    f_carb = severity_to_fraction(severity_scores.get("carb_sensitivity", 1.0))
    f_phos = severity_to_fraction(severity_scores.get("phosphorus_sensitivity", 0.2))

    # Compute max grams per constraint
    constraints = {
        "sodium": grams_from_budget(n["sodium_mg"], budget["sodium_mg"], f_sod),
        "potassium": grams_from_budget(n["potassium_mg"], budget["potassium_mg"], f_pot),
        "protein": grams_from_budget(n["protein_g"], budget["protein_g"], f_pro),
        "carbs": grams_from_budget(n["carbs_g"], budget["carbs_g"], f_carb),
        "phosphorus": grams_from_budget(n["phosphorus_mg"], budget["phosphorus_mg"], f_phos),
    }

    binding = min(constraints, key=constraints.get)
    max_g = min(constraints.values())
    max_g = min(max_g, DEFAULT_CAP_G)

    # Label
    if max_g <= AVOID_THRESHOLD_G:
        label = "Avoid"
    elif max_g <= HALF_PORTION_G:
        label = "Half Portion"
    else:
        label = "Allowed"

    # Nutrient load at recommended portion
    factor = max_g / 100.0
    load = {
        "sodium_mg": round(n["sodium_mg"] * factor, 1),
        "potassium_mg": round(n["potassium_mg"] * factor, 1),
        "protein_g": round(n["protein_g"] * factor, 2),
        "carbs_g": round(n["carbs_g"] * factor, 2),
        "phosphorus_mg": round(n["phosphorus_mg"] * factor, 1),
        "calories": round(n["calories"] * factor, 0),
    }

    return {
        "ingredient": ingredient,
        "category": n.get("category", ""),
        "max_grams": round(max_g, 1),
        "label": label,
        "binding_constraint": binding,
        "nutrient_load": load,
        "nutrients_per_100g": {
            "sodium_mg": n["sodium_mg"],
            "potassium_mg": n["potassium_mg"],
            "protein_g": n["protein_g"],
            "carbs_g": n["carbs_g"],
            "phosphorus_mg": n["phosphorus_mg"],
            "calories": n["calories"],
        },
        "constraint_grams": {k: round(v, 1) if v != float("inf") else None for k, v in constraints.items()},
    }


# ── Flask app ──────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

load_dotenv()
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    groq_client = None

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")


@app.route("/")
def serve_frontend():
    """Serve the main frontend page."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend files (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run all 4 ML models on patient data.

    Expects JSON body with keys:
      age, sex_male, has_htn, has_dm, has_ckd,
      serum_sodium, serum_potassium, creatinine, egfr,
      hba1c, fbs, sbp, dbp, bmi

    Returns risk_levels from MODEL PREDICTIONS (not thresholds).
    """
    data = request.get_json(force=True)

    # Validate all required features are present
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    patient = {f: float(data[f]) for f in FEATURE_NAMES}
    risk_levels = {}
    for target, r in predict_risk(patient).items():
        label = r["label"]
        risk_levels[target] = {
            "label": label,
            "confidence": round(r["confidence"] * 100, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in r["proba"].items()},
            "feature_attribution": r["feature_attribution"],
            "display_name": RISK_DESCRIPTIONS[target]["name"],
            "clinical_note": RISK_DESCRIPTIONS[target][label],
        }

    # Look up applicable nutrient thresholds based on conditions
    condition_key = (
        f"htn={int(data.get('has_htn', 0))}_"
        f"dm={int(data.get('has_dm', 0))}_"
        f"ckd={int(data.get('has_ckd', 0))}"
    )
    applicable_thresholds = THRESHOLDS.get(condition_key, {})

    return jsonify({
        "risk_levels": risk_levels,
        "patient_summary": {
            "age": data.get("age"),
            "sex": "Male" if data.get("sex_male", 0) == 1 else "Female",
            "conditions": {
                "Hypertension": bool(int(data.get("has_htn", 0))),
                "Diabetes": bool(int(data.get("has_dm", 0))),
                "CKD": bool(int(data.get("has_ckd", 0))),
            },
            "key_labs": {
                "eGFR": data.get("egfr"),
                "HbA1c": data.get("hba1c"),
                "Serum Potassium": data.get("serum_potassium"),
                "Serum Sodium": data.get("serum_sodium"),
                "Creatinine": data.get("creatinine"),
                "FBS": data.get("fbs"),
                "SBP/DBP": f"{data.get('sbp')}/{data.get('dbp')}",
                "BMI": data.get("bmi"),
            },
        },
        "nutrient_thresholds": applicable_thresholds,
        "condition_key": condition_key,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata; metrics come from the training manifest (held-out real patients)."""
    test = (MODEL_MANIFEST.get("metrics") or {}).get("test") or {}

    def mean_of(key):
        vals = [m[key] for m in test.values() if key in m]
        return round(float(np.mean(vals)), 4) if vals else None

    return jsonify({
        "models": {
            target: {
                "display_name": RISK_DESCRIPTIONS[target]["name"],
                "type": "TabNetClassifier",
                "classes": ["low", "moderate", "high"],
                "features_used": list(FEATURE_NAMES),
                "test_metrics": test.get(target),
            }
            for target in TARGET_NAMES
        },
        "accuracy_metrics": {
            "evaluated_on": "held-out real MIMIC-IV patients",
            "mean_f1_macro": mean_of("f1_macro"),
            "mean_balanced_accuracy": mean_of("balanced_accuracy"),
            "mean_auroc_ovr": mean_of("auroc_ovr_macro"),
            "mean_accuracy": mean_of("accuracy"),
            "mean_cohen_kappa": mean_of("cohen_kappa"),
        },
        "trained": MODEL_MANIFEST.get("created_utc"),
        "data_source": MODEL_MANIFEST["data"]["source"],
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "preprocessing": ["median imputation", "z-score scaling", "per-target isotonic monotonic transform"],
        "model_backend": "TabNet (pytorch_tabnet), one model per target",
    })


@app.route("/api/thresholds", methods=["GET"])
def thresholds():
    """
    Return nutrient thresholds reference (supplementary info only).
    These are NOT used for prediction — they are clinical reference
    values displayed alongside ML model predictions.
    """
    return jsonify(THRESHOLDS)


# ══════════════════════════════════════════════════════════════════
#  MODEL 2 API — PORTION RECOMMENDATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/api/ingredients", methods=["GET"])
def ingredients_list():
    """Return full ingredient list + search for autocomplete."""
    query = request.args.get("q", "").strip()
    if query:
        results = ifct_search(query)
        return jsonify({"ingredients": results})
    # Full list grouped by category
    grouped = {}
    for _, row in _ifct_df.iterrows():
        cat = row["category"]
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(row["ingredient"])
    return jsonify({"ingredients": IFCT_INGREDIENTS, "by_category": grouped})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Full Model1→Model2 pipeline: predict risk, then recommend portions.

    Expects JSON body:
    {
        "patient": { age, sex_male, has_htn, has_dm, has_ckd, ... },
        "ingredients": ["Banana, ripe", "Rice, milled (white)", ...]
    }

    Returns risk_levels + portion recommendations for each ingredient.
    """
    data = request.get_json(force=True)
    patient_data = data.get("patient", {})
    ingredient_list = data.get("ingredients", [])

    if not ingredient_list:
        return jsonify({"error": "No ingredients provided"}), 400

    # Validate patient features
    missing = [f for f in FEATURE_NAMES if f not in patient_data]
    if missing:
        return jsonify({"error": f"Missing patient features: {missing}"}), 400

    # ── Step 1: Run Model1 — risk predictions ──
    patient = {f: float(patient_data[f]) for f in FEATURE_NAMES}
    risk_levels = {}
    severity_scores = {}
    for target, r in predict_risk(patient).items():
        risk_levels[target] = {
            "label": r["label"],
            "confidence": round(r["confidence"] * 100, 2),
            "display_name": RISK_DESCRIPTIONS[target]["name"],
        }
        # severity_score = probability-weighted class index ∈ [0.0, 2.0]
        severity_scores[target] = r["severity_score"]

    # ── Step 2: Get daily budget from conditions ──
    has_ckd = bool(int(patient_data.get("has_ckd", 0)))
    has_htn = bool(int(patient_data.get("has_htn", 0)))
    has_dm = bool(int(patient_data.get("has_dm", 0)))
    budget = get_daily_budget(has_ckd, has_htn, has_dm)

    # ── Step 3: Recommend portion for each ingredient ──
    recommendations = []
    for ing in ingredient_list:
        rec = recommend_ingredient(ing, severity_scores, budget)
        recommendations.append(rec)

    # Sort: Avoid first, then Half Portion, then Allowed
    label_order = {"Avoid": 0, "Half Portion": 1, "Allowed": 2, "Not Found": 3}
    recommendations.sort(key=lambda r: (label_order.get(r["label"], 99), r["max_grams"]))

    return jsonify({
        "risk_levels": risk_levels,
        "severity_scores": {k: round(v, 4) for k, v in severity_scores.items()},
        "daily_budget": budget,
        "recommendations": recommendations,
        "patient_conditions": {
            "has_ckd": has_ckd, "has_htn": has_htn, "has_dm": has_dm,
        },
    })


# ══════════════════════════════════════════════════════════════════
#  PHASE 3 — FRIDGE DETECTION & RECIPE GENERATION
# ══════════════════════════════════════════════════════════════════

@app.route("/api/detect", methods=["POST"])
def detect_ingredients():
    """Takes an image, runs Roboflow CV, and maps to IFCT database."""
    if not ROBOFLOW_API_KEY:
        return jsonify({"error": "Roboflow API key not configured"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image = Image.open(image_file).convert("RGB")
            image.save(tmp.name)

            with open(tmp.name, "rb") as bf:
                encoded_image = base64.b64encode(bf.read()).decode("ascii")

        # Call Roboflow HTTP API (Model: ingredient-detection-5uzov/5)
        upload_url = "".join([
            "https://detect.roboflow.com/ingredient-detection-5uzov/5",
            f"?api_key={ROBOFLOW_API_KEY}",
            "&name=image.jpg"
        ])

        response = http_requests.post(
            upload_url,
            data=encoded_image,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code != 200:
            return jsonify({"error": f"Roboflow API error: {response.text}"}), 500

        result = response.json()
        predictions = result.get("predictions", [])
        
        # Unique classes detected by CV
        detected_names = list({pred["class"] for pred in predictions})
        
        # Match to IFCT database
        mapped_ingredients = []
        for name in detected_names:
            matches = ifct_search(name, n=1)
            if matches:
                mapped_ingredients.append(matches[0])

        os.remove(tmp.name)

        return jsonify({
            "detected_raw": detected_names,
            "mapped_ifct": mapped_ingredients
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/generate-recipe", methods=["POST"])
def generate_recipe():
    """Generates a recipe constrained by the patient's safe portion limits."""
    if not groq_client:
        return jsonify({"error": "Groq client not configured"}), 500

    data = request.json
    patient_data = data.get("patient", {})
    ingredients = data.get("ingredients", [])
    equipment = data.get("equipment", "Standard kitchen")
    time_limit = data.get("time_limit", "Any")
    cuisine = data.get("cuisine", "Any")

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    try:
        # 1-2. Model 1 risk prediction → severity scores (features matched by name)
        severity_scores = {t: r["severity_score"] for t, r in predict_risk(patient_data).items()}

        # 3. Calculate nutrient budget
        has_ckd = bool(int(patient_data.get("has_ckd", 0)))
        has_htn = bool(int(patient_data.get("has_htn", 0)))
        has_dm = bool(int(patient_data.get("has_dm", 0)))
        daily_budget = get_daily_budget(has_ckd, has_htn, has_dm)

        # 4. Determine safe maximum quantity for each ingredient
        safe_ingredients = []
        for ing in ingredients:
            rec = recommend_ingredient(ing, severity_scores, daily_budget)
            if rec:
                safe_ingredients.append(rec)

        # 5. Build prompt with constraints
        ingredient_lines = []
        for rec in safe_ingredients:
            line = f"- {rec['ingredient']}: MAXIMUM {rec['max_grams']}g permitted"
            if rec['binding_constraint']:
                line += f" (limiting constraint: {rec['binding_constraint']})"
            ingredient_lines.append(line)
            
        clinical_context = [
            f"- CKD Status: {'Positive' if has_ckd else 'Negative'}",
            f"- HTN Status: {'Positive' if has_htn else 'Negative'}",
            f"- DM Status: {'Positive' if has_dm else 'Negative'}"
        ]

        system_prompt = (
            "You are a specialized clinical nutritionist and chef. "
            "Your task is to generate a recipe strictly adhering to provided ingredient quantity limits.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST NOT exceed the 'MAXIMUM permitted' grams for ANY ingredient.\n"
            "2. If an ingredient has a very low maximum (e.g. <15g), use it only as a garnish or minor flavoring.\n"
            "3. You must ONLY use the provided ingredients, optionally adding generic water/salt/pepper/oil "
            "(unless hypertension is flagged bounding sodium in which case minimize salt).\n"
            "4. Output format: Clean Markdown with a Title, short description of clinical benefits, "
            "precise Ingredients list (in grams), and step-by-step cooking instructions."
        )

        user_prompt = (
            "Patient Clinical Context:\n" + "\n".join(clinical_context) + "\n\n"
            "Available Ingredients with Safety Limits:\n" + "\n".join(ingredient_lines) + "\n\n"
            f"Available Equipment: {equipment}\n"
            f"Cuisine Preference: {cuisine}\n"
            f"Time Constraint: {time_limit}\n\n"
            "Generate a recipe now."
        )

        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.6,
        )
        recipe = completion.choices[0].message.content
        
        # Save to Local Supabase DB
        patient_id = save_patient_data(patient_data)
        if patient_id:
            save_recipe(patient_id, safe_ingredients, recipe)
        
        return jsonify({
            "recipe": recipe,
            "portions_used": safe_ingredients
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting NutriBiteBot server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)

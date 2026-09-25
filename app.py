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
import os
import re
import sys
import base64
import tempfile
import requests as http_requests
from difflib import get_close_matches

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from supabase_client import save_patient_data, save_recipe, save_recipe_adherence
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
from model1_artifacts import TARGETS  # noqa: E402
from train_model2 import PortionControlModel  # noqa: E402

# Model 2 (portion engine) owns the Model 1 predictor, so both endpoints and the
# research code share one implementation. Requests use it statelessly (use_ledger=False).
print("Loading Model 1 + Model 2 (portion engine) …")
ENGINE = PortionControlModel()
PREDICTOR = ENGINE.model1.predictor
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
#  MODEL 2 — PORTION RECOMMENDATION (clinical-models/train_model2.py)
# ══════════════════════════════════════════════════════════════════

# ── IFCT nutritional database ─────────────────────────────────────
print("Loading IFCT nutritional database …")
_ifct_df = pd.read_csv(IFCT_CSV)
_ifct_df["ingredient_norm"] = _ifct_df["ingredient"].str.lower().str.strip()
_ifct_idx = _ifct_df.set_index("ingredient_norm")
IFCT_INGREDIENTS = _ifct_df["ingredient"].tolist()
_ifct_ingredients_lower = [i.lower() for i in IFCT_INGREDIENTS]
print(f"  ✓ IFCT database: {len(_ifct_df)} ingredients loaded\n")


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


# ── Portion recommendations via the single Model 2 engine ────────
_LABEL_API = {"Half portion": "Half Portion"}
_NUTRIENT_KEYS = ["sodium_mg", "potassium_mg", "protein_g", "carbs_g", "phosphorus_mg", "calories"]


def _api_decision(rec: dict) -> dict:
    """Engine PortionDecision dict → the API shape the frontends expect."""
    n = ENGINE.ifct.get_nutrients_per_100g(rec["ingredient"])
    factor = rec["max_grams"] / 100.0
    key = rec["ingredient"].lower().strip()
    return {
        "ingredient": rec["ingredient"],
        "category": str(_ifct_idx.loc[key].get("category", "")) if key in _ifct_idx.index else "",
        "max_grams": round(float(rec["max_grams"]), 1),
        "label": _LABEL_API.get(rec["label"], rec["label"]),
        "binding_constraint": rec["binding_constraint"],
        "explanation": rec.get("explanation", ""),
        "nutrient_load": {k: round(n[k] * factor, 1) for k in _NUTRIENT_KEYS},
        "nutrients_per_100g": {k: n[k] for k in _NUTRIENT_KEYS},
        "substitutes": [{**sub, "label": _LABEL_API.get(sub["label"], sub["label"])}
                        for sub in rec.get("substitutes", [])],
    }


def _not_found(ingredient: str) -> dict:
    suggestions = ifct_search(ingredient)
    return {
        "ingredient": ingredient, "max_grams": 0, "label": "Not Found", "binding_constraint": "unknown",
        "explanation": f"Not in IFCT database. Try: {', '.join(suggestions)}" if suggestions else "Not in IFCT database.",
        "nutrient_load": {}, "nutrients_per_100g": {}, "suggestions": suggestions,
    }


def portion_recommendations(patient_data: dict, ingredients: list) -> dict:
    """Model 1 risk → daily budget → per-ingredient portions, all from train_model2.PortionControlModel."""
    patient = {f: (None if patient_data.get(f) is None else float(patient_data[f])) for f in FEATURE_NAMES}
    known = [i for i in ingredients if i.lower().strip() in ENGINE.ifct.idx.index]
    if known:
        res = ENGINE.get_recommendations(patient, known, include_substitutes=True, use_ledger=False)
        risk, budget, warnings_ = res["risk_levels"], res["daily_budget"], res["clinical_warnings"]
        by_name = {r["ingredient"]: _api_decision(r) for r in res["recommendations"]}
    else:
        risk, by_name, warnings_ = ENGINE.model1.predict_risk_levels(patient), {}, []
        b = ENGINE.budget_calc.get_daily_budget(has_ckd=patient.get("has_ckd") == 1,
                                                has_htn=patient.get("has_htn") == 1,
                                                has_dm=patient.get("has_dm") == 1)
        budget = {"sodium_mg": b.sodium_mg_remaining, "potassium_mg": b.potassium_mg_remaining,
                  "protein_g": b.protein_g_remaining, "carbs_g": b.carbs_g_remaining,
                  "phosphorus_mg": b.phosphorus_mg_remaining}
    recs = [by_name[i] if i in by_name else _not_found(i) for i in ingredients]
    return {"risk_levels": risk, "daily_budget": budget, "recommendations": recs, "clinical_warnings": warnings_}


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
            # calibrated outcome probabilities; the tier label comes from the balanced decision scores
            "probabilities": {k: round(v * 100, 2) for k, v in r["proba"].items()},
            "severity_score": r["severity_score"],
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

    result = portion_recommendations(patient_data, ingredient_list)
    # severity scores are those the engine actually used (after any caloric reconciliation)
    risk_levels, severity_scores = {}, {}
    for target in TARGET_NAMES:
        r = result["risk_levels"][target]
        risk_levels[target] = {
            "label": r["label"],
            "confidence": round(r["confidence"] * 100, 2),
            "display_name": RISK_DESCRIPTIONS[target]["name"],
        }
        severity_scores[target] = r["severity_score"]
    recommendations = result["recommendations"]
    has_ckd = bool(int(patient_data.get("has_ckd", 0)))
    has_htn = bool(int(patient_data.get("has_htn", 0)))
    has_dm = bool(int(patient_data.get("has_dm", 0)))

    # Sort: Avoid first, then Half Portion, then Allowed
    label_order = {"Avoid": 0, "Half Portion": 1, "Allowed": 2, "Not Found": 3}
    recommendations.sort(key=lambda r: (label_order.get(r["label"], 99), r["max_grams"]))

    return jsonify({
        "risk_levels": risk_levels,
        "severity_scores": {k: round(v, 4) for k, v in severity_scores.items()},
        "daily_budget": result["daily_budget"],
        "recommendations": recommendations,
        "clinical_warnings": result["clinical_warnings"],
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


# Stated grams within this fraction over the limit aren't flagged as violations, since the
# extraction call itself has some rounding noise (e.g. "150g" vs a 148.6g limit).
# Groq decommissioned llama-3.3-70b-versatile for free/developer tiers on 2026-08-16;
# openai/gpt-oss-120b is Groq's recommended replacement. The rules engine bounds the
# gram limits upstream of this call, so the constraint layer is model-agnostic.
RECIPE_LLM_MODEL = "openai/gpt-oss-120b"

RECIPE_ADHERENCE_TOLERANCE = 0.05


def _normalize_ingredient_name(name: str) -> str:
    return " ".join(str(name).lower().split())


def _ingredient_name_tokens(name: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(name).lower()))


def check_recipe_adherence(recipe_text: str, safe_ingredients: list, client) -> list:
    """
    Second structured-output Groq call: asks the model to extract the gram quantity the
    recipe actually used for each ingredient, then checks each one against
    safe_ingredients[]['max_grams']. Returns one adherence record per ingredient
    (whether or not the extraction matched it), for logging and for surfacing to the
    caller — the first (recipe-writing) call is never trusted to have followed its own
    prompt limits.
    """
    ingredient_names = [rec["ingredient"] for rec in safe_ingredients]
    extraction_prompt = (
        "Extract the quantity in grams actually used for each ingredient in the recipe "
        "below. Respond only in JSON, of the form "
        '{"ingredients": [{"name": "<ingredient name>", "grams": <number>}, ...]}. '
        "Include exactly one entry per ingredient in this list, using these exact names: "
        f"{ingredient_names}. If an ingredient from the list is not used in the recipe, "
        "report its grams as 0.\n\nRecipe:\n" + recipe_text
    )

    stated_by_name = {}
    try:
        extraction = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You extract structured ingredient quantities "
                                              "from recipes and respond only in JSON."},
                {"role": "user", "content": extraction_prompt},
            ],
            model=RECIPE_LLM_MODEL,
            max_tokens=500,
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(extraction.choices[0].message.content)
        for item in parsed.get("ingredients", []):
            stated_by_name[_normalize_ingredient_name(item.get("name", ""))] = item.get("grams")
    except Exception as exc:
        app.logger.error(f"Recipe adherence extraction failed: {exc}")

    records = []
    for rec in safe_ingredients:
        name = rec["ingredient"]
        max_grams = rec["max_grams"]
        norm_name = _normalize_ingredient_name(name)

        stated_grams = stated_by_name.get(norm_name)
        if stated_grams is None:
            # extraction may have paraphrased the name slightly; fall back to a substring match
            for k, v in stated_by_name.items():
                if norm_name in k or k in norm_name:
                    stated_grams = v
                    break
        if stated_grams is None and stated_by_name:
            # still unmatched (e.g. reordered words: "white rice" vs "rice, milled (white)"):
            # word-set containment is more robust here than difflib's char-sequence ratio,
            # which misses exactly these reordering / extra-descriptor cases.
            target_tokens = _ingredient_name_tokens(name)
            best_score, best_grams = 0.6, None  # 0.6 acts as the match threshold
            for k, v in stated_by_name.items():
                k_tokens = _ingredient_name_tokens(k)
                if not target_tokens or not k_tokens:
                    continue
                overlap = len(target_tokens & k_tokens) / min(len(target_tokens), len(k_tokens))
                if overlap >= best_score:
                    best_score, best_grams = overlap, v
            stated_grams = best_grams

        matched = stated_grams is not None
        stated_grams = float(stated_grams) if matched else None
        violated = bool(matched and max_grams is not None
                        and stated_grams > max_grams * (1 + RECIPE_ADHERENCE_TOLERANCE))
        records.append({
            "ingredient": name,
            "max_grams": max_grams,
            "stated_grams": stated_grams,
            "matched": matched,
            "violated": violated,
            "overage_grams": round(stated_grams - max_grams, 2) if matched and max_grams is not None else None,
        })
    return records


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
        # 1-4. Model 1 risk → budget → per-ingredient gram limits (single Model 2 engine)
        has_ckd = bool(int(patient_data.get("has_ckd", 0)))
        has_htn = bool(int(patient_data.get("has_htn", 0)))
        has_dm = bool(int(patient_data.get("has_dm", 0)))
        portion_result = portion_recommendations(patient_data, ingredients)
        safe_ingredients = portion_result["recommendations"]
        clinical_warnings = portion_result["clinical_warnings"]

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
        if clinical_warnings:
            # The gram limits above already reflect these relaxed severities (Algorithm 2's
            # caloric-floor relaxation fired upstream) — this is context for the recipe's
            # clinical-benefits description, not an additional constraint to enforce.
            for w in clinical_warnings:
                clinical_context.append(
                    f"- Caloric-adequacy relaxation applied to {w['constraint']}: "
                    f"severity {w['old_severity']:.2f}→{w['new_severity']:.2f} "
                    f"(sodium/potassium were never relaxed)"
                )

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
            model=RECIPE_LLM_MODEL,
            max_tokens=800,
            temperature=0.6,
        )
        recipe = completion.choices[0].message.content

        # The recipe-writing call is never trusted to have followed its own prompt limits:
        # a second, structured-output call extracts what it actually wrote and checks it
        # against safe_ingredients[]['max_grams'].
        adherence = check_recipe_adherence(recipe, safe_ingredients, groq_client)

        # Save to Local Supabase DB
        patient_id = save_patient_data(patient_data)
        recipe_id = save_recipe(patient_id, safe_ingredients, recipe) if patient_id else None
        if recipe_id:
            save_recipe_adherence(recipe_id, adherence)

        return jsonify({
            "recipe": recipe,
            "portions_used": safe_ingredients,
            "adherence": adherence,
            "clinical_warnings": clinical_warnings
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting NutriBiteBot server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)

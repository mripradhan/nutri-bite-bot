# NutriBiteBot: A Clinical Nutrition Decision Support System

NutriBiteBot is an end-to-end clinical nutrition platform that generates safe, personalized recipes for patients with chronic conditions (CKD, Hypertension, Type 2 Diabetes). It combines a fine-tuned TabNet deep-learning model for risk stratification, a hierarchical clinical rules engine for conflict resolution, computer vision for fridge scanning, and an LLM for recipe generation — all running locally with a Dockerized Supabase database for data privacy.

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [Using the App](#using-the-app)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Stopping the Services](#stopping-the-services)
10. [Troubleshooting](#troubleshooting)

---

## Features

- **Clinical Risk Stratification (TabNet ML)** — Predicts sodium sensitivity, potassium sensitivity, protein restriction need, and carbohydrate sensitivity from 14 lab/demographic features. No hardcoded thresholds; all risk levels come from model predictions.
- **Hierarchical Clinical Rules Engine** — Resolves conflicting dietary guidelines automatically (e.g., CKD potassium cap overrides HTN DASH recommendation). Priority order: Renal > Cardiac > Metabolic.
- **Fridge Scanner / Pantry Inventory** — Upload a photo of your fridge; Roboflow computer vision detects ingredients and maps them to the IFCT nutritional database.
- **Safe Portion Calculation** — Determines the maximum safe grams per ingredient per meal based on the patient's ML-derived risk profile and daily nutrient budget (sodium, potassium, protein, carbs, phosphorus).
- **Recipe Generation (Groq / Llama 3.3)** — Generates recipes that strictly respect per-ingredient portion limits, using SHARE methodology (Substitute, Halve, Add, Remove, Emphasize).
- **Local Secure Storage** — A fully local Dockerized Supabase (PostgreSQL) instance stores patient data and generated recipes, ensuring no data leaves your machine.

---

## System Architecture

```
[Browser / Frontend UI]
         |  HTTP
         v
[Flask API — app.py :5000]
    |           |           |
    v           v           v
[TabNet     [Portion    [Groq LLM]
 Model1]     Engine]    (Recipe Gen)
    |           |
    v           v
[IFCT Nutrient DB]   [Roboflow CV API]
         |
         v
[Local Supabase (Docker)] ← stores patients + recipes
```

**Stack**
| Layer | Technology |
|---|---|
| Frontend | HTML / CSS / Vanilla JS (SPA) |
| Backend | Python 3.9+, Flask |
| ML Models | TabNet (`pytorch_tabnet`), PyTorch |
| Nutritional DB | IFCT CSV database |
| Computer Vision | Roboflow API |
| LLM | Groq API (Llama-3.3-70b-versatile) |
| Database | Local Supabase (PostgreSQL via Docker) |

---

## Prerequisites

Install the following before proceeding:

| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.9 or later | Backend and ML models |
| **Docker Desktop** | Latest | Runs the local Supabase database stack |
| **Node.js** | 18 or later | Required to run the Supabase CLI via `npx` |
| **pip** | Bundled with Python | Python package manager |

You also need API keys for two external services:

| Service | Where to get it | Required for |
|---|---|---|
| **Groq API** | https://console.groq.com | Recipe generation |
| **Roboflow API** | https://app.roboflow.com | Fridge ingredient detection |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd nutri-bite-bot
```

### 2. Create your environment file

Create a `.env` file in the project root. Start with your external API keys — the Supabase credentials will be filled in after the next step.

```env
# External APIs
GROQ_API_KEY="your_groq_api_key_here"
ROBOFLOW_API_KEY="your_roboflow_api_key_here"

# Local Supabase (fill in after running: npx supabase start)
SUPABASE_URL="http://127.0.0.1:54331"
SUPABASE_ANON_KEY="your_local_anon_key"
SUPABASE_SERVICE_ROLE_KEY="your_local_service_role_key"
```

### 3. Start the local Supabase database

Make sure **Docker Desktop is open and running**, then execute:

```bash
npx --yes supabase start
```

This will pull and start all required containers (Postgres, PostgREST, Studio, etc.). The first run takes a few minutes to download images.

When it finishes, the CLI prints output similar to:

```
API URL:         http://127.0.0.1:54331
Studio URL:      http://127.0.0.1:54333
anon key:        eyJhbGci...
service_role key: eyJhbGci...
```

Copy the `anon key` and `service_role key` into your `.env` file under `SUPABASE_ANON_KEY` and `SUPABASE_SERVICE_ROLE_KEY`.

> The Supabase Studio dashboard (table viewer, SQL editor) is available at http://127.0.0.1:54333

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, PyTorch, pytorch_tabnet, scikit-learn, Groq, Pillow, pandas, and all other required packages.

> If you see torch or tabnet compilation errors, ensure your Python version is 3.9–3.11 and that you have a compatible pip version (`pip install --upgrade pip`).

---

## Running the Application

Start the Flask backend (which also serves the frontend):

```bash
python app.py
```

You should see output like:

```
Loading ML model weights (TabNet) …
  ✓ TabNet model loaded from model_params.json + network.pt
  ✓ sodium_sensitivity → TabNet model
  ✓ potassium_sensitivity → TabNet model
  ✓ protein_restriction → TabNet model
  ✓ carb_sensitivity → TabNet model
  ✓ imputer, scaler, feature_names (14 features) loaded
Loading IFCT nutritional database …
  ✓ IFCT database: N ingredients loaded

Starting NutriBiteBot server on http://localhost:5000
```

Open your browser and navigate to **http://localhost:5000**

---

## Using the App

The frontend is a single-page application with three main phases:

### Phase 1 — Clinical Risk Assessment

Enter the patient's demographic and lab values:

| Field | Description |
|---|---|
| Age | Patient age in years |
| Sex | Male / Female |
| Has HTN | Hypertension diagnosis |
| Has DM | Type 2 Diabetes diagnosis |
| Has CKD | Chronic Kidney Disease diagnosis |
| Serum Sodium | mEq/L |
| Serum Potassium | mEq/L |
| Creatinine | mg/dL |
| eGFR | mL/min/1.73m² |
| HbA1c | % |
| FBS | mg/dL (Fasting Blood Sugar) |
| SBP / DBP | mmHg (Systolic / Diastolic) |
| BMI | kg/m² |

Click **Run Risk Assessment** to get TabNet model predictions for sodium sensitivity, potassium sensitivity, protein restriction, and carbohydrate sensitivity (each classified as Low / Moderate / High with confidence scores).

### Phase 2 — Fridge Scan / Ingredient Selection

Either:
- **Upload a fridge photo** — Roboflow CV detects ingredients and maps them to the IFCT database automatically.
- **Select ingredients manually** — Use the searchable ingredient list pulled from the IFCT database.

### Phase 3 — Portion Recommendations & Recipe Generation

Click **Get Portions** to compute the maximum safe grams for each ingredient based on the risk assessment. Ingredients are labelled:
- **Allowed** — Full portion safe
- **Half Portion** — Use sparingly (≤75g)
- **Avoid** — Clinically unsafe (≤5g)

Then select optional preferences (cuisine, equipment, time limit) and click **Generate Recipe**. The Groq LLM generates a recipe that strictly respects every per-ingredient limit. The result is saved to your local Supabase database.

---

## API Reference

The Flask backend exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend SPA |
| `POST` | `/api/predict` | Run TabNet risk assessment on patient data |
| `GET` | `/api/model-info` | Model metadata and accuracy metrics |
| `GET` | `/api/thresholds` | Clinical nutrient reference thresholds |
| `GET` | `/api/ingredients?q=<query>` | Search or list IFCT ingredients |
| `POST` | `/api/recommend` | Full Model1→Model2 pipeline: risk + portions |
| `POST` | `/api/detect` | Fridge image → Roboflow CV → IFCT mapping |
| `POST` | `/api/generate-recipe` | Generate a safe recipe via Groq LLM |

**`POST /api/predict` — required fields:**
```json
{
  "age": 68, "sex_male": 1, "has_htn": 1, "has_dm": 1, "has_ckd": 1,
  "serum_sodium": 138, "serum_potassium": 4.2, "creatinine": 1.8,
  "egfr": 52, "hba1c": 7.8, "fbs": 145, "sbp": 148, "dbp": 88, "bmi": 27.4
}
```

**`POST /api/recommend` — required fields:**
```json
{
  "patient": { <same fields as /api/predict> },
  "ingredients": ["Banana, ripe", "Rice, milled (white)", "Carrot"]
}
```

---

## Project Structure

```
nutri-bite-bot/
├── app.py                        # Flask backend + ML inference
├── supabase_client.py            # Supabase REST API bindings
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed)
│
├── frontend/
│   ├── index.html                # Single-page application
│   ├── script.js                 # Frontend logic
│   └── style.css
│
├── artifacts/
│   └── models/
│       ├── model_params.json     # TabNet architecture config
│       ├── network.pt            # Trained TabNet weights
│       ├── imputer.joblib        # Fitted median imputer
│       ├── scaler.joblib         # Fitted StandardScaler
│       ├── feature_names.joblib  # Ordered feature list
│       └── reports/
│           ├── accuracy_summary.txt
│           └── nutrient_thresholds_reference.json
│
├── clinical models/
│   ├── ifct_database.csv         # Indian Food Composition Table
│   ├── clinicalbb_model#1.ipynb  # Model 1 training notebook
│   ├── clinicalbb_model#2.ipynb  # Model 2 training notebook
│   ├── train_model1.py
│   ├── train_model2.py
│   └── ...
│
├── supabase/
│   ├── config.toml               # Supabase project config
│   └── migrations/               # Database schema migrations
│
├── clinical_rules_engine.py      # Hierarchical conflict resolver
├── pantry_inventory.py           # Pantry management module
├── recipe_generator.py           # SHARE recipe adaptation
├── main_integration.py           # Standalone CLI pipeline
└── mimic_cohort_extraction.py    # MIMIC-IV patient extraction
```

---

## Stopping the Services

When finished, stop the app with `Ctrl+C` in the terminal, then stop the Supabase containers to free up Docker resources:

```bash
npx --yes supabase stop
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pytorch_tabnet'`**
Run `pip install pytorch_tabnet torch` separately, then retry `pip install -r requirements.txt`.

**TabNet model fails to load on startup**
Verify that `artifacts/models/model_params.json` and `artifacts/models/network.pt` both exist. These are the trained model files and must not be moved.

**Supabase containers fail to start**
Ensure Docker Desktop is running and that ports 54331–54333 are not occupied by another process. Run `docker ps` to check for conflicts.

**Groq API error during recipe generation**
Double-check `GROQ_API_KEY` in your `.env` file. Get a key at https://console.groq.com.

**Roboflow returns no detections**
Ensure `ROBOFLOW_API_KEY` is set correctly and that the uploaded image is a clear, well-lit photo. The model used is `ingredient-detection-5uzov/5`.

**Supabase credentials missing warning (non-fatal)**
If you see `Supabase credentials not found`, the app still works — patient/recipe storage is simply skipped. Add the `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` from `npx supabase start` output to fix this.

---

## Clinical Guidelines Referenced

- **KDOQI 2020** — CKD nutrition guidelines (potassium, protein, phosphorus)
- **AHA/ACC 2017** — Hypertension management (sodium limits)
- **ADA 2024** — Diabetes nutrition therapy (carbohydrate targets)
- **KDIGO 2017** — CKD staging and eGFR classification

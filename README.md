---
title: Nutri Bite Bot
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# NutriBiteBot: A Clinical Nutrition Decision Support System

NutriBiteBot is an end-to-end clinical nutrition platform that generates safe, personalised recipes for patients with chronic conditions (CKD, Hypertension, Type 2 Diabetes). It combines a multi-target TabNet deep-learning model for risk stratification, a sigmoid-based portion computation engine, a hierarchical clinical rules engine for conflict resolution, computer vision for fridge scanning, and a bounded LLM for recipe generation.

**Live deployment:** Flask backend on [Hugging Face Spaces](https://gayathri-27-nutri-bite-bot.hf.space) · Frontend on Vercel

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [TabNet Model Architecture](#tabnet-model-architecture)
4. [Deployment](#deployment)
5. [Running Locally](#running-locally)
6. [Using the App](#using-the-app)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [Clinical Guidelines Referenced](#clinical-guidelines-referenced)

---

## Features

- **Clinical Risk Stratification (TabNet)** — Predicts sodium sensitivity, potassium sensitivity, protein restriction need, and carbohydrate sensitivity from 14 EHR-derived features, trained on real MIMIC-IV v3.1 observed outcomes. Mean macro-F1 0.417 and mean macro-AUROC 0.731 on a 74,663-admission held-out test set, outperforming a deterministic guideline-rule baseline (macro-F1 0.349) on every target, confirmed by 5×5 repeated patient-grouped cross-validation.
- **Sigmoid Portion Engine** — Converts continuous severity scores into ingredient-specific maximum safe gram quantities against a session-persistent daily nutrient budget, grounded in IFCT 2017.
- **Hierarchical Clinical Rules Engine** — Automatically resolves conflicting dietary guidelines across co-existing conditions. Priority: Renal (KDIGO) > Cardiac (AHA/ACC) > Metabolic (ADA).
- **Fridge Scanner** — Upload a fridge photo; Roboflow CV detects ingredients and maps them to the IFCT nutritional database.
- **Bounded Recipe Generation (Groq / `openai/gpt-oss-120b`)** — Generates recipes strictly within pre-computed per-ingredient gram limits. The LLM cannot override clinical constraints.
- **Quantitative Recipe Adherence Check** — A second, structured-output Groq call independently extracts the gram quantities the generated recipe actually used and checks them against the computed limits; violations are logged, not just prompted against.
- **Caloric Sufficiency Safeguard** — If a recommendation set would fall below 1,200 kcal, severity is progressively relaxed (protein → carbohydrate → phosphorus, KDIGO-prioritised) until adequacy is restored. Sodium and potassium severity are never relaxed.
- **Optional Local Storage** — Supabase (PostgreSQL) instance for persisting patient data, recipes, and clinical-warning/adherence logs. Gracefully disabled if not configured.

---

## System Architecture

```
[Vercel Frontend (SPA)]
         |  HTTPS
         v
[HF Spaces — Flask API :7860]
    |           |           |
    v           v           v
[TabNet     [Portion    [Groq LLM]
 Classifier] Engine]    (Recipe Gen)
    |           |
    v           v
[IFCT Nutrient DB]   [Roboflow CV API]
         |
         v
[Supabase] ← optional, stores patients + recipes
```

**Stack**

| Layer | Technology | Hosting |
|---|---|---|
| Frontend | HTML / CSS / Vanilla JS (SPA) | Vercel |
| Backend | Python 3.9, Flask, Gunicorn | Hugging Face Spaces (Docker) |
| ML Model | TabNet (`pytorch_tabnet`), PyTorch | Bundled in Docker image |
| Nutritional DB | IFCT 2017 CSV (101 ingredients) | Bundled in Docker image |
| Computer Vision | Roboflow API | External API |
| LLM | Groq API (`openai/gpt-oss-120b`) | External API |
| Database | Supabase (PostgreSQL) | Optional / local |

---

## TabNet Model Architecture

### Overview

Four independent `TabNetClassifier` models are trained, one per clinical sensitivity target. Each model takes the same 14-feature input vector and outputs a 3-class probability distribution (Low / Moderate / High).

### Input Features (14)

| Feature | Description | Clinical Basis |
|---|---|---|
| `age` | Patient age in years | Demographics |
| `sex_male` | Binary sex flag | Demographics |
| `bmi` | Body mass index (kg/m²) | Metabolic |
| `sbp` / `dbp` | Systolic / diastolic BP (mmHg) | AHA/ACC |
| `has_htn` | Hypertension diagnosis flag | AHA/ACC |
| `has_dm` | Type 2 Diabetes flag | ADA 2024 |
| `has_ckd` | Chronic Kidney Disease flag | KDIGO 2024 |
| `serum_sodium` | Serum Na (mEq/L) | AHA/ACC |
| `serum_potassium` | Serum K (mmol/L) | KDIGO 2024 |
| `creatinine` | Serum creatinine (mg/dL) | KDIGO 2024 |
| `egfr` | Estimated GFR (mL/min/1.73m²) | KDIGO 2024 |
| `hba1c` | Glycated haemoglobin (%) | ADA 2024 |
| `fbs` | Fasting blood sugar (mg/dL) | ADA 2024 |

### Output Targets (4)

| Target | Dominant Features | Guideline |
|---|---|---|
| `sodium_sensitivity` | serum Na, SBP, has\_htn | AHA/ACC |
| `potassium_sensitivity` | eGFR, serum K | KDIGO 2024 |
| `protein_restriction` | eGFR, creatinine, has\_ckd | KDIGO 2024 |
| `carb_sensitivity` | HbA1c, FBS, has\_dm | ADA 2024 |

Each model outputs class probabilities `[P(Low), P(Moderate), P(High)]`. A continuous severity score is derived as:

```
severity = 0·P(Low) + 1·P(Moderate) + 2·P(High)   →   s ∈ [0, 2]
```

This score feeds directly into the sigmoid portion engine, preserving sub-threshold severity gradients.

### Preprocessing Pipeline

1. **Median imputation** — missing lab values replaced with training-set medians
2. **Z-score scaling** — all features standardised (mean 0, std 1)
3. **Per-target monotonic transformer** — isotonic regression enforces physiologically correct feature directionality (e.g. declining eGFR → higher restriction, rising HbA1c → higher carb restriction). Each target gets an independently fitted transformer to prevent cross-target contamination.

### Attention Mechanism

TabNet uses a sparsemax attention mask at each sequential decision step, selecting a sparse subset of features per step. Per-step masks are averaged to produce a per-patient feature attribution — native interpretability without SHAP or other post-hoc methods.

### Training Configuration

| Parameter | Value |
|---|---|
| Architecture | `TabNetClassifier` (pytorch_tabnet) |
| Training device | GPU where available |
| Inference device | CPU (Docker) |
| Early stopping patience | 10 epochs |
| Max epochs | 100 |
| Optimiser | Adam, lr 0.02, batch 4096, virtual batch 256 |
| Training data | MIMIC-IV v3.1, patient-level `StratifiedGroupKFold` split (no patient in both train and test) |
| Targets | Clinically observed outcomes measured after a 24 h baseline window (hyperkalaemia, AKI, hyperglycaemia, follow-up BP), not rule-derived labels |
| Classes per target | 3 (Low / Moderate / High) |

### Performance (Held-Out Test Split — 74,663 admissions / 32,330 patients)

TabNet vs. a deterministic guideline-rule baseline, macro-F1 [95% CI]; every paired
difference excludes zero and is confirmed by 5×5 repeated patient-grouped
cross-validation.

| Target | TabNet macro-F1 | Rules macro-F1 | TabNet AUROC |
|---|---|---|---|
| sodium\_sensitivity | 0.390 [0.383, 0.397] | 0.371 [0.361, 0.382] | 0.756 |
| potassium\_sensitivity | 0.414 [0.408, 0.419] | 0.386 [0.380, 0.392] | 0.721 |
| protein\_restriction | 0.379 [0.374, 0.384] | 0.321 [0.316, 0.325] | 0.673 |
| carb\_sensitivity | 0.486 [0.479, 0.492] | 0.317 [0.312, 0.322] | 0.775 |
| **Mean** | **0.417** | **0.349** | **0.731** |

A gradient-boosting reference model on the same validation fold reaches AUROC
0.70–0.77, the same range as TabNet — performance is limited by what the 14
admission-time features can predict about later outcomes, not by architecture choice.
TabNet's advantage over the reference model is native per-instance attribution at no
accuracy cost. Full methodology and robustness results (measurement-noise sweep,
missing-data test, subgroup/COVID-era breakdowns) are in
`clinical-models/evaluate_model1.py`, `repeated_cv.py` and `robustness_test.py`.

### Model Artifacts

One portable format shared by training and deployment (`clinical-models/model1_artifacts.py`) — no joblib/sklearn pickles, so loading doesn't depend on the sklearn version, and a missing file is a hard error rather than a silent fallback:

```
artifacts/models/
├── manifest.json                          # provenance, training config, held-out test metrics
├── preprocessing.json                     # feature order, imputer medians, scaler params,
│                                           # per-target isotonic (monotonic) maps, class priors
├── tabnet/
│   ├── sodium_sensitivity/
│   │   ├── model_params.json              # TabNet architecture config
│   │   └── network.pt                     # Trained weights (~218 KB)
│   ├── potassium_sensitivity/  ...
│   ├── protein_restriction/    ...
│   └── carb_sensitivity/       ...
└── reports/                               # accuracy summary, confusion matrices,
                                            # evaluation/ (CIs, calibration, subgroups),
                                            # robustness/ (noise + missing-data tests)
```

---

## Deployment

### Current Setup

| Component | Platform | URL |
|---|---|---|
| Backend (Flask + TabNet) | Hugging Face Spaces (Docker) | `https://gayathri-27-nutri-bite-bot.hf.space` |
| Frontend (SPA) | Vercel | Your Vercel project URL |

### HF Spaces — Backend

The backend runs as a Docker container on HF Spaces. The `Dockerfile` at the project root handles the full build:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libpq-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
```

**Required Secrets** (set in Space Settings → Variables and Secrets):

| Secret | Purpose |
|---|---|
| `GROQ_API_KEY` | Recipe generation via `openai/gpt-oss-120b` |
| `ROBOFLOW_API_KEY` | Fridge ingredient detection |
| `SUPABASE_URL` | Optional — cloud Supabase instance |
| `SUPABASE_SERVICE_ROLE_KEY` | Optional — cloud Supabase auth |

### Vercel — Frontend

The `frontend/` directory is a static SPA deployed to Vercel. `frontend/vercel.json` configures it as a static site with no build step.

The API base URL in `frontend/script.js` is set to the HF Space:

```javascript
const API_BASE = "https://gayathri-27-nutri-bite-bot.hf.space";
```

To deploy the frontend:

```bash
npm i -g vercel
cd frontend
vercel --prod
```

Or connect the repo to Vercel via the dashboard, setting the **Root Directory** to `frontend`.

---

## Running Locally

Supabase is optional — the app is fully functional without it (patient/recipe storage is simply skipped).

### 1. Clone and install

```bash
git clone <repo-url>
cd nutri-bite-bot
pip install -r requirements.txt
```

### 2. Create `.env`

```env
GROQ_API_KEY="your_groq_api_key"
ROBOFLOW_API_KEY="your_roboflow_api_key"

# Optional — leave blank to skip storage
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

### 3. Run the backend

```bash
python app.py
```

Expected startup output:

```
Loading Model 1 + Model 2 (portion engine) …
============================================================
INITIALIZING MODEL2: PORTION CONTROL SYSTEM
============================================================
✓ Loaded IFCT database with 101 ingredients
✓ Loaded Model1 components from ../artifacts/models
✓ Model2 initialization complete
============================================================
  ✓ 4 targets, 14 features, trained <timestamp> on MIMIC-IV v3.1 (PhysioNet credentialed access)

Loading IFCT nutritional database …
  ✓ IFCT database: 101 ingredients loaded

Starting NutriBiteBot server on http://localhost:5000
```

Open **http://localhost:5000** — the backend serves the frontend SPA directly.

### 4. (Optional) Local Supabase database

If you want local persistent storage, start Supabase with Docker Desktop running:

```bash
npx --yes supabase start
```

Copy the printed `API URL` and `service_role key` into `.env`:

```env
SUPABASE_URL="http://127.0.0.1:54331"
SUPABASE_SERVICE_ROLE_KEY="<printed service_role key>"
```

Restart `app.py`. Stop the containers when done:

```bash
npx --yes supabase stop
```

---

## Using the App

### Phase 1 — Clinical Risk Assessment

Enter the patient's lab values and click **Run Risk Assessment**. TabNet predicts sodium sensitivity, potassium sensitivity, protein restriction, and carbohydrate sensitivity (Low / Moderate / High) with confidence scores and a continuous severity score per nutrient.

| Field | Unit |
|---|---|
| Age | years |
| Sex | Male / Female |
| Has HTN / DM / CKD | checkbox |
| Serum Sodium | mEq/L |
| Serum Potassium | mEq/L |
| Creatinine | mg/dL |
| eGFR | mL/min/1.73m² |
| HbA1c | % |
| FBS | mg/dL |
| SBP / DBP | mmHg |
| BMI | kg/m² |

### Phase 2 — Ingredient Selection

- **Upload a fridge photo** — Roboflow CV detects ingredients and maps them to IFCT automatically.
- **Select manually** — search the 101-ingredient IFCT database.

### Phase 3 — Portions & Recipe

Click **Get Portions** to compute per-ingredient maximum safe grams. Each ingredient is labelled:

| Label | Condition |
|---|---|
| Allowed | g* > 75 g |
| Half Portion | 5 g < g* ≤ 75 g |
| Avoid | g* ≤ 5 g |

Click **Generate Recipe**. The Groq-hosted model generates a recipe that strictly respects every gram limit. Results are saved to Supabase if configured.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend SPA |
| `POST` | `/api/predict` | TabNet risk assessment |
| `GET` | `/api/model-info` | Model metadata and accuracy |
| `GET` | `/api/thresholds` | Clinical nutrient reference thresholds |
| `GET` | `/api/ingredients?q=<query>` | Search IFCT ingredient list |
| `POST` | `/api/recommend` | Full pipeline: risk + portions |
| `POST` | `/api/detect` | Fridge image → Roboflow → IFCT |
| `POST` | `/api/generate-recipe` | Bounded recipe via Groq LLM |

**`POST /api/predict` payload:**
```json
{
  "age": 68, "sex_male": 1, "has_htn": 1, "has_dm": 1, "has_ckd": 1,
  "serum_sodium": 138, "serum_potassium": 4.2, "creatinine": 1.8,
  "egfr": 52, "hba1c": 7.8, "fbs": 145, "sbp": 148, "dbp": 88, "bmi": 27.4
}
```

**`POST /api/recommend` payload:**
```json
{
  "patient": { "age": 68, "sex_male": 1, ... },
  "ingredients": ["Banana, ripe", "Rice, milled (white)", "Carrot"]
}
```

Both `/api/recommend` and `/api/generate-recipe` return a `clinical_warnings` array —
non-empty only when the caloric-sufficiency safety mechanism fired (projected session
calories fell below 1,200 kcal), logging which constraint's severity was relaxed and by
how much. Sodium and potassium severity are never relaxed by this mechanism.

**`POST /api/generate-recipe` response:**
```json
{
  "recipe": "## Spiced Paneer & Dal Bowl\n...",
  "portions_used": [ { "ingredient": "Paneer", "max_grams": 24.9, "binding_constraint": "phosphorus", ... } ],
  "clinical_warnings": [
    { "constraint": "protein_restriction", "old_severity": 1.8, "new_severity": 1.5,
      "projected_kcal_after": 1240.0, "rationale": "KDIGO 2024: malnutrition risk outweighs CKD progression; protein restriction is the first target to relax." }
  ],
  "adherence": [
    { "ingredient": "Paneer", "max_grams": 24.9, "stated_grams": 25.0, "matched": true,
      "violated": false, "overage_grams": 0.1 }
  ]
}
```
`adherence` is a second, structured-output Groq call that independently extracts what
the recipe text actually says it used and checks it against `max_grams` — the
recipe-writing call is never trusted to have followed its own prompt limits. Both
`clinical_warnings` and `adherence` entries are also logged to Supabase
(`recipes` and `recipe_adherence` tables) when configured.

---

## Project Structure

```
nutri-bite-bot/
├── app.py                        # Flask backend — inference, routing, startup
├── supabase_client.py            # Supabase REST bindings (optional)
├── requirements.txt
├── Dockerfile                    # HF Spaces Docker build
├── .env                          # API keys — never committed
│
├── frontend/                     # Deployed vanilla-JS SPA (served by app.py)
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── vercel.json
├── frontend-next/                # Next.js/React/TypeScript frontend (actively developed;
│                                  # see remaining_work.md — not yet reconciled with app.py's
│                                  # static-serving, which still points at frontend/)
│
├── artifacts/models/             # Model 1 (TabNet) portable artifact set — see
│                                  # "Model Artifacts" above for the real layout
│
├── clinical-models/              # Training/evaluation pipeline (not needed to run the app)
│   ├── ifct_database.csv         # IFCT 2017 — 102 Indian food ingredients
│   ├── mimic_extract.py          # Raw MIMIC-IV v3.1 -> cohort parquet (DuckDB)
│   ├── cohort_data.py            # Patient-level split, India reweighting, synthetic ablation
│   ├── train_model1.py           # TabNet training (Model 1: risk stratification)
│   ├── train_model2.py           # Portion engine + caloric-sufficiency validator (Model 2)
│   ├── evaluate_model1.py        # Held-out test metrics, bootstrap CIs, calibration
│   ├── repeated_cv.py            # 5x5 repeated patient-grouped cross-validation
│   ├── robustness_test.py        # Measurement-noise + missing-data robustness comparison
│   ├── test_model2.py            # Portion engine / caloric reconciliation test scenarios
│   ├── model1_artifacts.py       # Shared artifact read/write format (training <-> deployment)
│   └── output_formatter.py       # Pretty-printer for Model 2 output (manual testing)
│
├── supabase/
│   ├── config.toml
│   └── migrations/               # PostgreSQL schema (patients, recipes, recipe_adherence)
│
└── remaining_work.md             # Repo audit + open items (legacy files, deployment
                                   # questions, manuscript/response-letter status)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pytorch_tabnet'`**
Run `pip install pytorch_tabnet torch` separately, then retry.

**TabNet model fails to load on startup**
Ensure `artifacts/models/manifest.json`, `artifacts/models/preprocessing.json`, and each `artifacts/models/tabnet/<target>/{model_params.json,network.pt}` exist and have not been moved — `Model1Predictor` (`clinical-models/model1_artifacts.py`) treats any missing file as a hard error. These are plain git-tracked files (~218 KB each), not Git LFS.

**UnicodeEncodeError on Windows at startup**
Set `PYTHONIOENCODING=utf-8` before running: `set PYTHONIOENCODING=utf-8 && python app.py`

**Groq API error during recipe generation**
Check `GROQ_API_KEY` in `.env`. Free-tier keys at https://console.groq.com.

**Roboflow returns no detections**
Ensure `ROBOFLOW_API_KEY` is set and the image is a clear, well-lit photo of raw ingredients. Model: `ingredient-detection-5uzov/5`.

**HF Space cold-start timeout**
HF free-tier Spaces sleep after inactivity. The frontend shows a wake-up notice automatically — wait 20–30 seconds for the container to restart.

**Supabase credentials missing (non-fatal)**
`Supabase credentials not found` is a warning, not an error. The app works fully without Supabase; storage is silently skipped.

---

## Clinical Guidelines Referenced

| Guideline | Scope |
|---|---|
| KDIGO 2024 | CKD staging, potassium and protein limits, eGFR thresholds |
| KDOQI 2020 | CKD nutrition — phosphorus and protein |
| AHA/ACC 2017 | Hypertension — sodium limits, BP targets |
| ADA 2024 | Type 2 Diabetes — carbohydrate and glycaemic targets |

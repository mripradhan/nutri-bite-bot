# Implementation Plan — Mrida's Track (Data & Pipeline)

Companion to `feedbackplan.md`. Covers Task 1 (use-case bug) and Task 3
(dataset rebuild), plus the retraining pieces Gayathri's CV and
robustness work depend on.

---

## 0. Data check (done 2026-09-19)

**Source:** `physionet.org/files/mimiciv/3.1/` — full MIMIC-IV v3.1, 8.9 GB.

| Table | Needed for | Present |
|---|---|---|
| `hosp/patients` | age, sex, `anchor_year_group` (for the COVID-era check) | ✓ |
| `hosp/admissions` | admission time/window, length of stay | ✓ |
| `hosp/diagnoses_icd` + `d_icd_diagnoses` | HTN / T2DM / CKD flags | ✓ |
| `hosp/labevents` + `d_labitems` | Na, K, creatinine, glucose, HbA1c | ✓ (2.6 GB) |
| `hosp/omr` | BMI, outpatient BP | ✓ |
| `icu/chartevents` + `d_items` | fallback BP for ICU stays | ✓ (3.5 GB) |

- **Incomplete download.** 6 `hosp` tables are missing (`poe_detail`,
  `prescriptions`, `procedures_icd`, `provider`, `services`,
  `transfers`), and `poe.csv.gz` is likely truncated. **None of these
  are needed.** No re-download is required unless scope changes.
- **Checksums** for the needed tables: running (result recorded below
  once complete).
- **Fixed:** `physionet.org/` and `mimic-iv-clinical-database-demo-*/`
  were **not** git-ignored. The old `*.csv` rule doesn't match
  `.csv.gz`, and `origin` is the public GitHub repo. Both folders and
  `*.csv.gz` have been added to `.gitignore`. Derived extracts go in
  `data/derived/`, which also needs to stay ignored (`*.parquet` already
  is). Never commit anything row-level from MIMIC.

**Lab item IDs (serum/blood, Chemistry):** sodium `50983`, potassium
`50971`, creatinine `50912`, glucose `50931`, HbA1c `50852`.
**OMR `result_name`s:** `BMI (kg/m2)`, `Blood Pressure` (+ sitting/lying/standing variants).

---

## Decisions — RESOLVED 2026-09-19 (Mrida)

- **D1:** observed-outcome targets as primary; guideline rules as baseline. ✅
- **D2:** code is the source of truth — the paper's rule table is
  corrected to match the code (carb Moderate: `bmi >= 25` alone). ✅
- **D3:** synthetic data is used for training only (as an ablation);
  every reported metric comes from real held-out patients. ✅
- **General rule:** where the paper and the code disagree, fix the paper
  to match the code (unless the code is clinically wrong, e.g. the eGFR
  formula, which gets fixed in code and then described accurately).

**Checksums:** all needed tables OK. `poe.csv.gz` fails as expected
(truncated, and not used).

**Data-handling rule (PhysioNet LLM guidance):** row-level MIMIC data
must not be sent through third-party APIs. Claude writes code that runs
locally and only sees aggregates (counts, missingness, metrics, public
data dictionaries). Counts < 10 are suppressed. The supervising
professor (credentialed DUA holder) reviewed the handling setup and
approved use on 2026-09-19.

**Implementation note — sodium target:** inpatient BP only exists for
ICU stays, so the sodium/BP outcome is the **mean outpatient SBP 30–365
days after discharge** (OMR). This is a better fit for a dietary sodium
question than acute ICU BP anyway.

## Phase 1 status

- `clinical-models/mimic_extract.py` written (DuckDB; caches filtered
  labs/BP to parquet).
- Demo run OK: 275 admissions / 100 patients. Sanity checks pass:
  eGFR median 92 (no CKD) vs 33 (CKD); HbA1c median 5.6 (no DM) vs 8.3
  (DM). HbA1c missing 77% (ordered rarely); morning glucose missing 54%.
- Full v3.1 run ✅ (6 min): **545,848 admissions / 223,380 patients**.
  Comorbidity mix: none 237,902 · HTN only 142,270 · DM only 20,534 ·
  CKD only 6,586 · HTN+DM 62,297 · HTN+CKD 36,870 · DM+CKD 2,814 ·
  **all three 36,575**. COVID-era share 7.8%. Missingness: HbA1c 82.9%,
  morning glucose 56.8%, BMI 48.8%, BP 41.1%, baseline chemistry ~33%.
  Labelled per target: sodium 280,739 · potassium 278,435 · renal
  246,843 · carb 276,964. Aggregates are in
  `data/derived/cohort_card_v3.1.json` (committable).

## Phase 2 status ✅ — `clinical-models/cohort_data.py`

- **Eligibility:** ≥ 1 baseline chemistry result (Na, K or creatinine
  in the first 24 h). 173,562 admissions excluded → **372,286 eligible**.
- **Patient-level split** (`StratifiedGroupKFold` on subject_id,
  stratified by comorbidity group): **train 297,623 admissions / 129,230
  patients; real held-out test 74,663 admissions / 32,330 patients.**
  No patient appears in both (asserted). 5 grouped CV folds inside
  train (~59–60k admissions each). Saved as `split_v3.1.parquet`.
- **Indian reweighting** (raking to ICMR-INDIAB HTN 35.5% / DM 11.4%;
  CKD 13.24% from Talukdar et al. 2025): hits the targets exactly.
  Weights range 0.34–1.75; effective sample size 228,921. The CKD source
  is a **new citation** for the manuscript. Config in
  `clinical-models/config/indian_prevalence.json`.
- **Synthetic ablation** (Gaussian copula per comorbidity group; copies
  real missingness patterns; eGFR recomputed): 61,932 rows (17.2% of
  training) for CKD only / DM only / DM+CKD. Fidelity vs real rows of the
  same group: max standardised mean difference ≤ 0.024, mean
  correlation difference ≤ 0.03. It must be regenerated inside each CV
  fold by the training code; the saved card is for reporting only.
- Aggregate card: `data/derived/cohort_card_v3.1_split.json`.

**Phase 3 compute — measured 2026-09-19** (carb target, 157,677 train
rows, CV fold 0 as validation; RTX 3050 Ti Laptop 4 GB, CUDA PyTorch
2.12.1+cu130):

| batch / virtual batch | GPU s/epoch | CPU (20 cores) s/epoch |
|---|---|---|
| 4096 / 256 (old config) | 6.6 | 9.3 |
| 16384 / 1024 | 3.3 | 4.1 |
| 32768 / 2048 | 2.0 | 3.1 |

The earlier "30–60 min per fit on CPU" estimate was wrong. TabNet at
this size is overhead-bound, not compute-bound: GPU peak memory is 0.19
GB, and the GPU is only 1.3–1.6× faster than CPU. At the large batch
sizes a 60-epoch fit takes ~2–4 min, so **the full 5×5 repeated grouped
CV on the full training set (100 fits) is ~4–6 h** on this laptop alone,
less if GPU and CPU jobs run in parallel. No subsample and no second
machine needed. Large batches may need more epochs or a different
learning rate to converge: choose the batch size on CV fold 0 by
validation macro-F1, not by speed alone.

## Decisions (original options, kept for the response letter)

### D1 — What the targets are (the core Reviewer 2 fix)

**Recommended:** switch the primary targets to **clinically observed
outcomes** measured *after* the feature window, so the labels are no
longer a formula of the inputs. Keep the guideline rules as the
**baseline** to beat, scored on the same observed outcomes. This is
exactly the comparison Reviewer 2 asked for ("compare TabNet against the
original deterministic rule system" on "clinically observed targets").

Features come from the **first 24 h** of the admission, plus recent
outpatient BMI/BP. Labels come from the **worst value from 24 h to
discharge**:

| Target | Low (0) | Moderate (1) | High (2) | Basis |
|---|---|---|---|---|
| potassium | max K ≤ 5.0 | 5.1–5.5 | > 5.5 mmol/L | hyperkalaemia grading |
| protein / renal | no creatinine rise | ≥ 0.3 mg/dL rise (KDIGO AKI stage 1) | ≥ 2× baseline (stage 2+) | KDIGO AKI |
| carb | max glucose ≤ 180 | 181–250 | > 250 mg/dL | ADA inpatient targets |
| sodium | mean follow-up SBP ≤ 140 | 141–160 | > 160 mmHg | AHA/ACC BP stages (outpatient SBP 30–365 d post-discharge) |

- Admissions with length of stay < 48 h, or with no measurements after
  24 h, are excluded as censored. Exclusion counts are reported.
- **Consequence:** accuracy will fall well below 99.9%. That is the
  point: it becomes a real prediction number. Gayathri's Methods and
  Discussion reframing has to follow this ("risk of nutrient-related
  metabolic derangement" rather than a restatement of guideline
  thresholds).
- **Fallback (lower effort, weaker answer):** keep the rule-derived
  labels on real MIMIC features. Reframe TabNet as a learned surrogate
  for the rules, and rely on the missing-data and noise robustness
  comparison. This still leaves Reviewer 2 ¶1 largely standing.

Diet orders in `poe` were checked as a possible label source. The diet
*type* (renal, cardiac, diabetic) is not recorded in `poe_detail`, so
they can't be used as per-nutrient labels.

### D2 — Which carb rule is the truth

Code (`train_model1.py:709`): Moderate if `bmi >= 25` (alone). Manuscript
table: Moderate if `BMI ≥ 30 and HbA1c ≥ 5.5`. Pick one and make code
and paper match. Under D1 this becomes the *baseline* rule, but it still
has to be stated correctly.

### D3 — Synthetic data's role

**Recommended:** synthetic rows are **training-only and an ablation,
never test data.** Primary Indian-population adjustment = **importance
reweighting** of real MIMIC rows to ICMR-INDIAB comorbidity prevalence
(`anjana2023`, already cited). Synthetic generation (Gaussian copula fit
per comorbidity stratum on the training fold only) is reported as a
second arm: *real-only* vs *real + synthetic* training, both evaluated
on the same real held-out patients. CKD prevalence for India needs a
citation (to confirm with Gayathri).

---

## Phase 1 — Extraction (`clinical-models/mimic_extract.py`) · ~3 days

Tooling: `pip install duckdb`. It reads `.csv.gz` directly with filter
pushdown, so `labevents` (2.6 GB) and `chartevents` (3.5 GB) fit
comfortably in 15 GB RAM. Output: `data/derived/cohort_v3.1.parquet`,
one row per `hadm_id`.

1. **Cohort:** adults (age ≥ 18) with ≥ 1 hospital admission. Keep
   admissions *without* HTN/DM/CKD too, so the Low class is real rather
   than synthetic.
2. **Age:** `anchor_age + (year(admittime) − anchor_year)`. **Sex:**
   `gender`.
3. **Comorbidity flags** from `diagnoses_icd`:
   - HTN: ICD-10 `I10–I16` / ICD-9 `401–405`
   - T2DM: ICD-10 `E11` / ICD-9 `250.x0`, `250.x2`
   - CKD: ICD-10 `N18` / ICD-9 `585`
4. **Baseline labs (first 24 h):** first Na, K, creatinine; morning
   glucose (draws 04:00–08:00) as the **fasting proxy**. MIMIC has no
   fasting flag, so this is documented as a limitation. The column name
   stays `fbs` to preserve the app's 14-feature schema. HbA1c = most
   recent value within 90 days before admission or during the first
   24 h (expect high missingness; report it).
5. **BMI / BP:** nearest `omr` value within 365 days before admission.
   Fallback BP: first 24 h of `chartevents` (non-invasive BP items
   `220179`/`220180`; confirm IDs in `d_items`).
6. **eGFR:** CKD-EPI 2021 race-free equation from creatinine, age and
   sex. This replaces the current placeholder formula at
   `train_model1.py:608`, which has no sex term and is not a real
   CKD-EPI equation.
7. **Outcome labels** (if D1 is accepted): worst post-24 h K,
   creatinine delta, glucose and SBP, per the D1 table.
8. **Era flag:** `covid_era = anchor_year_group == "2020 - 2022"`.
9. **Plausibility filters:** drop physiologically impossible values
   (e.g. K < 1.5 or > 10) before aggregation; log the counts.
10. **Dataset card** (`data/derived/cohort_card.json`, safe to commit
    because it holds aggregates only): N admissions and N patients,
    per-flag counts, co-occurrence counts (HTN+DM, HTN+CKD, DM+CKD, all
    three — this answers Reviewer 3 major #3), missingness per feature,
    class balance per target, and the COVID-era share.

**Build against the demo first** (`mimic-iv-clinical-database-demo-2.2`,
~100 patients, same schema), then run on v3.1.

## Phase 2 — Leakage-safe split and augmentation · ~2 days

1. **Split by `subject_id`**, not by row. Patients have multiple
   admissions, and the current row-level `train_test_split` lets the
   same patient appear in both train and test. Use `StratifiedGroupKFold`.
2. Hold out a fixed **real-only test set** (20% of patients) before
   anything else touches the data. The imputer, scaler, monotonic
   transformer, reweighting and synthetic generator are all fitted on
   training folds only.
3. Implement D3: reweighting, plus the optional copula synthetic arm
   (training folds only; synthetic rows get a `is_synthetic` flag and
   are never scored).
4. Log N real / N synthetic / ratio per class / seed into the dataset card.

## Phase 3 — Training pipeline fixes (`train_model1.py`) · ~2 days

1. Replace `MIMICDataLoader`, `ClinicalFeatureExtractor` and
   `_add_synthetic_demographics` in the main path with a loader for the
   Phase 1 parquet. Delete the synthetic-demographics generator from
   the training path; it can't coexist with the paper's data claims.
2. **Per-target monotonic transformers**, as the manuscript already
   claims (Eq. `isotonic_exp`). Currently `train_model1.py:1270` fits a
   *single* transformer on the mean label across all targets.
   Save as `monotonic_transformer_{target}.joblib`. Update the loader in
   `train_model2.py:1004`, and make a missing transformer a **hard
   error** instead of the current silent skip (`:1011`). That skip is
   how the deployed app ended up running without it.
3. Save a **coherent artifact set** from one run, plus
   `artifacts/models/manifest.json`: MIMIC version, cohort-card hash,
   git commit, date, per-target metrics. This prevents a repeat of the
   mixed March/May artifact set found in Task 1.
4. Update the `.gitignore` force-includes if new artifact files must
   ship to HF Spaces.
5. Fix the stale "NGBoost" docstring on `ClinicalRiskStratifier` (`:1224`).

## Phase 3 status — code done, tuning running (2026-09-19)

- **`clinical-models/model1_artifacts.py`** (new): one portable artifact
  format shared by training and deployment. Layout: `manifest.json`,
  `preprocessing.json` (imputer medians, scaler, per-target isotonic
  maps), `tabnet/<target>/{model_params.json, network.pt}`. No
  joblib/sklearn pickles; any missing file is a hard error.
- **`train_model1.py`** rewritten:
  - old `MIMICDataLoader` / `ClinicalFeatureExtractor` /
    `_add_synthetic_demographics` removed;
  - `LabelGenerator` kept as the non-mutating guideline-rule baseline
    (thresholds unchanged);
  - per-target monotonic transformers;
  - class-balanced sampling combined with optional India weights and
    synthetic rows (training folds only);
  - `--tune` grid on CV fold 0; `--evaluate-test` for a single held-out
    evaluation;
  - reports macro-F1, balanced accuracy, AUROC, log-loss and quadratic κ;
  - rule baseline scored on the same test patients;
  - after saving, artifacts are reloaded and must reproduce the in-memory
    predictions.
- **`train_model2.py`** `Model1Integration` and **`app.py`** now both
  load through `Model1Predictor`. Old artifacts moved to
  `artifacts/models/_legacy_pre_revision/` (git-ignored; kept as
  evidence).
- Demo smoke tests pass for final fit, ablation flags, the tuning grid,
  Model 2 integration, and missing-lab input.

**New findings in the deployed app (for the response letter):**
8. `app.py` loaded **one** TabNet network and used it for **all four
   targets**, so the live app returned the same model's output for
   sodium, potassium, protein and carb. It also applied no monotonic
   transformer, and `/api/model-info` returned hardcoded 0.9991 accuracy.
9. `app.py`'s recipe endpoint preprocessed patient data by column
   *position*, so features could be misaligned depending on the order the
   frontend sent them. Fixed: `Model1Predictor` matches features by name.

## Phase 3 results ✅ (2026-09-19)

**Tuning** (CV fold 0 only; `artifacts/models/reports/tuning_results_fold0.csv`).
Best: lr 0.02, batch 4096 / virtual batch 256, class-balanced sampling →
mean validation macro-F1 0.421, AUROC 0.718. The manuscript's original
settings (lr 1e-3, no balancing) scored lowest but one (macro-F1 0.299).
Reference check: sklearn HistGradientBoosting on the same fold reaches
AUROC 0.70–0.77, the same range as TabNet (0.66–0.77). So this is the
predictability ceiling of these 14 features, not TabNet under-training.

**Final fit:** train 238,418 admissions (CV folds 1–4); early stopping on
fold 0 (59,205); best epochs 21–27. **Held-out test: 74,663 real
admissions / 32,330 patients, scored once.**

| Target | n test | Rules macro-F1 | TabNet macro-F1 | Rules bal-acc | TabNet bal-acc | Rules κ | TabNet κ | TabNet AUROC |
|---|---|---|---|---|---|---|---|---|
| sodium | 40,645 | 0.371 | **0.390** | 0.458 | **0.514** | 0.120 | **0.175** | 0.739 |
| potassium | 49,803 | 0.386 | **0.414** | 0.464 | **0.476** | 0.133 | **0.171** | 0.711 |
| protein/renal | 49,705 | 0.321 | **0.379** | 0.359 | **0.472** | 0.092 | **0.142** | 0.661 |
| carb | 49,530 | 0.317 | **0.486** | 0.425 | **0.527** | 0.088 | **0.299** | 0.766 |
| **mean** | | 0.349 | **0.417** | 0.427 | **0.497** | 0.108 | **0.197** | **0.719** |

TabNet beats the guideline rules on every target and on every
unweighted metric. **Caveat for Phase 4:** on protein/renal, TabNet's
*quadratic* κ is lower than the rules' (0.064 vs 0.149). Class-balanced
sampling likely makes it over-call "high" on a target that is mostly low.
Look at its confusion matrix and calibration before writing it up.

**App check:** `/api/predict`, `/api/recommend` and `/api/model-info`
work with the new artifacts; results don't depend on field order; the
model-info endpoint reports the held-out metrics above. Representative
patient (CKD3, HTN, no DM, HbA1c 5.2): carb **LOW** (bmi/fbs/has_dm),
sodium HIGH (egfr/has_htn/sbp), protein HIGH (bmi/egfr/has_ckd),
potassium MODERATE. This resolves Reviewer 2 ¶2.

**Provenance ✅:** Phase 1–3 code committed as `d60cd8a` (local, not
pushed; the planning docs are deliberately not committed because the
repo is public and they quote the confidential reviews). The final fit
was re-run from that commit: **bit-for-bit identical results** (same best
epochs and test metrics to 4 d.p.), so the pipeline is deterministic and
reproducible. The deployed manifest references `d60cd8a`.

## Phase 4 results ✅ (2026-09-19, commit `b53f336`)

`clinical-models/evaluate_model1.py` → `artifacts/models/reports/evaluation/`
(summary.json, metrics_ci.csv, subgroups.csv, confusion_matrices.csv,
reliability_*.png). Patient-level cluster bootstrap, 1000 resamples.

**TabNet vs guideline rules, held-out test, macro-F1 [95% CI]:**

| Target | TabNet | Rules | TabNet − rules | TabNet AUROC |
|---|---|---|---|---|
| sodium | 0.390 [0.383, 0.397] | 0.371 [0.361, 0.382] | [+0.009, +0.027] | 0.739 [0.732, 0.747] |
| potassium | 0.414 [0.408, 0.419] | 0.386 [0.380, 0.392] | [+0.023, +0.032] | 0.711 [0.704, 0.718] |
| protein/renal | 0.379 [0.374, 0.384] | 0.321 [0.316, 0.325] | [+0.052, +0.065] | 0.661 [0.655, 0.669] |
| carb | 0.486 [0.479, 0.492] | 0.317 [0.312, 0.322] | [+0.162, +0.175] | 0.766 [0.760, 0.771] |

Every difference CI excludes zero.

- **Indian-reweighted test (deployed model):** TabNet > rules on
  potassium [+0.017, +0.026], protein [+0.051, +0.065] and carb
  [+0.165, +0.179]. **Sodium is a tie** [−0.012, +0.005].
- **Ablations** (same test set, real patients only):
  - *Synthetic training rows:* protein 0.396 (+0.017), carb 0.508
    (+0.022), sodium 0.394, potassium 0.412. There was no gain on
    validation fold 0, so the deployed model stays real-only (switching
    because of test results would be test-set selection). Report as an
    ablation: synthetic data gave a small, inconsistent benefit.
  - *Indian-weighted training:* **worse** (sodium 0.369, potassium 0.391,
    protein 0.344, carb 0.484). Report as a negative result; the deployed
    model is unweighted.
- **COVID era:** 2020–22 vs 2008–19 AUROC within 0.01–0.02 for every
  target, so no era confound.
- **Comorbidity subgroups** (TabNet − rules macro-F1): TabNet wins in
  most groups. Exceptions to report: protein in non-CKD groups (DM only
  −0.084, HTN+DM −0.083, HTN only −0.050, none −0.022) and carb in
  DM+CKD (−0.016, n = 374).
- **Protein κ caveat explained:** TabNet detects far more renal
  deterioration than the rules (moderate recall 51% vs 11%, high 37% vs
  28%) at the cost of more low→high false alarms (10,929 vs 8,273), which
  quadratic κ penalises. It's a sensitivity/specificity trade-off, not a
  defect.
- **Calibration — DECISION PENDING:** class-balanced training inflates raw
  probabilities. Mean severity score is 0.80–0.96 vs observed mean level
  0.21–0.35; ECE is 0.07–0.17. Prior correction (× training class prior)
  gives ECE ≤ 0.008 and matches observed levels, but using corrected
  probabilities for *labels* lowers macro-F1. **Proposal:** labels from
  the balanced model, severity score (the portion engine's input) from
  prior-corrected probabilities. This changes deployed portion sizes and
  the paper's use-case table.

**Still with Gayathri:** 5×5 repeated grouped CV and the missing-data /
noise-robustness experiments. Both can reuse `train_model1.ClinicalRiskStratifier`
and `evaluate_model1.metrics`.

## Calibrated severity ✅ (2026-09-19)

- Artifact schema v3: `preprocessing.json` stores a per-target correction
  factor (training prior ÷ sampler prior).
- `Model1Predictor.predict` now returns:
  - `label` — tier, argmax of the balanced scores;
  - `decision_proba` — the balanced scores;
  - `proba` — calibrated probabilities;
  - `severity_score` — from the calibrated probabilities (the portion
    engine's input);
  - `confidence`.
- Deployed and ablation models retrained (identical networks,
  deterministic) and re-evaluated. Calibration error 0.004–0.008 (deployed).
  AUROC on calibrated probabilities: 0.756 / 0.721 / 0.673 / 0.775,
  mean 0.731.

## Phase 5 status (2026-09-19)

- `clinical-models/usecase_report.py` regenerates Section 5 (Model 1 +
  both portion engines) → `artifacts/models/reports/usecase_report.{json,md}`,
  `usecase_table.tex`.
- `clinical-models/attribution_summary.py` → test-set mean attributions,
  which are clinically coherent: sodium→SBP/eGFR/HTN, potassium→K/eGFR/CKD,
  protein→eGFR/age/CKD, carb→FBS/HbA1c/DM.
- `manuscript_edits.md`: section-by-section edit list for Gayathri, with
  exact replacement numbers and the list of corrected errors for the
  response letter.
- **Representative patient:** carb now **LOW** (resolves R2 ¶2); sodium
  HIGH, potassium MODERATE, protein HIGH.
- **✅ Engines unified (2026-09-20).** `app.py` now calls
  `train_model2.PortionControlModel` (stateless: `use_ledger=False`) for
  `/api/recommend` and `/api/generate-recipe`, with a thin adapter that
  keeps the frontend contract (`Half Portion`, `Not Found`, full
  `nutrient_load`). The app's duplicate engine code was removed.
  `train_model2`'s TFT import is now lazy, so the web app doesn't need the
  training stack. Use case regenerated: 0 disagreements;
  `test_model2.py` passes. *(Original note below.)*
- ~~DECISION PENDING — two portion engines.~~ `app.py` has its own
  simplified copy of the portion engine: no phosphorus constraint, 300 g
  cap, no caloric reconciliation. For the same patient it disagrees with
  `train_model2.PortionControlModel` (what the paper describes) on 6 of 8
  ingredients, e.g. paneer 67.6 g protein-bound (app) vs 24.9 g
  phosphorus-bound (train_model2). **Recommendation:** make `app.py`
  delegate to `train_model2` so there is one engine. Watch out: its
  `NutrientLedger` is per-instance state and must not be shared across
  users in the Flask process. Then regenerate the Section 5 table.
- **Draft vs submitted manuscript:** the `.tex` in the worktree may not be
  the submitted version (see the note at the top of `manuscript_edits.md`).

## Phase 4 — Evaluation (hand-off to Gayathri) · ~2 days

Produce these as scripts and CSVs, so Gayathri's CV and robustness work
plugs straight in:

1. **Rules baseline vs TabNet** on the real held-out set: macro-F1, AUROC
   (one-vs-rest), Cohen's κ, with bootstrap 95% CIs.
2. Real-only training vs real + synthetic training (D3 ablation).
3. COVID-era sensitivity: metrics with vs without 2020–2022 admissions.
4. Calibration (ECE, reliability plot). This is cheap and gives
   Reviewer 1 ¶5 (uncertainty) something concrete.
5. Per-comorbidity-subgroup metrics (HTN only, HTN+CKD, all three, …).

Gayathri owns: the 5×5 repeated grouped CV wrapper, the missing-data and
noise-injection experiments, and the write-up.

## Phase 5 — Use case and consistency · ~1 day

1. Re-run the representative patient through the **retrained** pipeline
   with a committed script (`clinical-models/usecase_report.py`), not
   hand transcription. Regenerate the Section 5 tables from its output.
2. Check every TabNet attribution claimed in the manuscript against the
   script's output.
3. Hand Gayathri the list of manuscript edits: dataset version 2.2 → 3.1,
   the D2 rule table, FBS → morning-glucose proxy, the 14-feature
   description, per-target transformers (now actually true), and the
   new label definitions if D1 is accepted.

---

## Findings to date (for the response letter)

1. **Use-case outputs were not reproducible** from the saved artifacts.
   Protein and carb come out LOW; the paper says HIGH. Attributions for
   all four targets differ from the manuscript's.
2. **Deployed artifacts were a mixed set:** TabNet `.zip` files dated
   May 4; imputer, scaler and feature names dated March 7; no
   monotonic transformer on disk at all.
3. **The monotonic transformer is shared across targets** in code, but
   the paper says it is per-target.
4. **Carb rule differs** between the code (BMI ≥ 25) and the paper (BMI ≥ 30 + HbA1c).
5. **All clinical features except BP were synthetic** (`np.random`),
   not MIMIC-IV.
6. **The eGFR formula was not CKD-EPI.**
7. **Train/test split was row-level**, not patient-level.

Items 5–7 must be disclosed plainly in the response letter as corrected
errors, not framed as improvements.

## Estimated total: ~10 days (Phases 1–5)

Critical path: Phase 1 → 2 → 3. Phase 4 can start on the demo extract
while the full v3.1 extraction runs.

# Manuscript Edits — from the rebuilt pipeline (for Gayathri)

Every number below comes from committed code and artifacts. Rule agreed
with Mrida: **where the paper and the code disagree, the paper follows the
code.**

> **⚠ Check which file you're editing.** Line numbers below refer to
> `.claude/worktrees/agent-ab3420d601fb4c55e/finalpaperdraft.tex` (dated
> 2 May). That draft may **not** be the submitted version. Reviewer 3
> quotes "no significant difference from dietitian diets at p > 0.05", and
> the submitted title contains "Privacy-Preserving"; neither appears in
> this draft. Apply these edits to the **submitted** manuscript, using
> tracked changes as the editor requires.

Sources (all in the repo):

| What | Where |
|---|---|
| Cohort counts, definitions | `data/derived/cohort_card_v3.1.json`, `cohort_card_v3.1_split.json` |
| Training config, per-target training info | `artifacts/models/manifest.json` |
| Test metrics + 95% CIs | `artifacts/models/reports/evaluation/summary.json`, `metrics_ci.csv` |
| Subgroups, confusion matrices | `…/evaluation/subgroups.csv`, `confusion_matrices.csv` |
| Calibration plots | `…/evaluation/reliability_*.png` |
| Attribution table | `…/evaluation/attribution_summary.json` |
| Ablations | `artifacts/experiments/{synthetic,india_weights}/reports/evaluation/` |
| Tuning grid | `artifacts/models/reports/tuning_results_fold0.csv` |
| Use case (Section 5) | `artifacts/models/reports/usecase_report.md`, `usecase_table.tex` — **pending the portion-engine decision (§9)** |

---

## 1. Abstract (≈ lines 60–90) — R2 ¶1, R3 major #1

Replace "99.91% mean accuracy and 99.91% weighted F1 … κ = 0.9983" with
real-patient results, e.g.:

> Trained on 238,418 adult admissions from MIMIC-IV v3.1 and evaluated on
> 74,663 held-out admissions from 32,330 unseen patients, the per-target
> TabNet classifiers predict clinically observed outcomes (hyperkalaemia,
> acute kidney injury, hyperglycaemia and follow-up blood pressure) with a
> mean macro-AUROC of 0.731 and outperform the deterministic guideline
> rules on every target (macro-F1 0.417 vs 0.349; all paired 95% CIs
> exclude zero).

Also replace "to ensure compliance and data privacy" (line 81) with the
data-residency wording in §8.

## 2. Contributions list (≈ lines 220–245)

- Lines 226–229: delete the 99.91% / κ 0.9983 claim; use the numbers above.
- Line 243 ("DPDPA 2023 requirements"): reword per §8.
- **Add a contribution:** a head-to-head evaluation of TabNet against the
  guideline-rule system on independent, clinically observed outcomes. This
  is the comparison Reviewer 2 asked for.

## 3. Dataset (§ Dataset Specification, lines 528–556) — R2 ¶3, R1 ¶1, R3 major #3

Rewrite completely. Content to include:

- **Source:** MIMIC-IV **v3.1** (not 2.2); update the citation. Accessed
  under PhysioNet credentialed access by the supervising author (DUA).
- **Unit:** one row per adult (≥ 18 y) hospital admission. 545,848
  admissions / 223,380 patients extracted. **Eligibility:** ≥ 1 baseline
  chemistry result (Na, K or creatinine in the first 24 h). 173,562
  excluded → **372,286 eligible**.
- **Features** (first 24 h unless stated):
  - age (from `anchor_age`), sex;
  - HTN / T2DM / CKD from ICD-9/10 codes (HTN I10–I16 / 401–405; T2DM E11
    / 250.x0, 250.x2; CKD N18 / 585);
  - first serum Na, K and creatinine;
  - **eGFR by the CKD-EPI 2021 race-free equation**;
  - **"FBS" = first morning (04:00–07:59) plasma glucose as a fasting
    proxy** (MIMIC has no fasting flag — state this as a limitation);
  - HbA1c = most recent within 90 d before admission or in the first 24 h;
  - BMI and BP = most recent outpatient value (OMR) within 365 d before
    admission; BP falls back to first ICU non-invasive BP.
- **Missingness:** HbA1c 82.9%, morning glucose 56.8%, BMI 48.8%, BP
  41.1%, baseline chemistry ~33% (before eligibility). Median imputation,
  fitted on training rows.
- **Comorbidity mix** (answers R3 major #3), all 545,848 admissions:

  | Group | n |
  |---|---|
  | none | 237,902 |
  | HTN only | 142,270 |
  | DM only | 20,534 |
  | CKD only | 6,586 |
  | HTN+DM | 62,297 |
  | HTN+CKD | 36,870 |
  | DM+CKD | 2,814 |
  | **HTN+DM+CKD** | **36,575** |

  About 138,556 admissions have ≥ 2 of the three conditions.
- **COVID era:** 7.8% of admissions are from 2020–2022.

## 4. Labels — replace Table `tab:label_rules` and its text (lines 542–610) — R2 ¶1, R3 major #1

The targets are **no longer generated from the input features**. They are
outcomes observed **after** the 24-hour feature window:

| Target | Low (0) | Moderate (1) | High (2) | Basis |
|---|---|---|---|---|
| potassium_sensitivity | max serum K after 24 h ≤ 5.0 | 5.1–5.5 | > 5.5 mmol/L | hyperkalaemia grading |
| protein_restriction | no KDIGO AKI | AKI stage 1 (↑ ≥ 0.3 mg/dL or ≥ 1.5× baseline creatinine) | stage ≥ 2 (≥ 2× baseline) | KDIGO AKI |
| carb_sensitivity | max glucose after 24 h ≤ 180 | 181–250 | > 250 mg/dL | ADA inpatient targets |
| sodium_sensitivity | mean outpatient SBP 30–365 d after discharge ≤ 140 | 141–160 | > 160 mmHg | AHA/ACC |

- Inpatient outcomes require length of stay ≥ 48 h (otherwise censored,
  label missing).
- Labelled test admissions per target: 40,645 / 49,803 / 49,705 / 49,530.
- **Keep the old threshold table**, retitled "guideline-rule baseline",
  and correct it to match the code (Methods §6).
  - **Carb row:** Moderate = HbA1c ≥ 5.7 **or** FBS ≥ 100 **or BMI ≥ 25**
    **or** DM. High = HbA1c ≥ 7.0 or FBS ≥ 126 **or BMI ≥ 30** or
    (DM and HbA1c ≥ 6.5). The draft's "(BMI ≥ 30 and HbA1c ≥ 5.5)" and
    "(BMI ≥ 35 and HbA1c ≥ 6.0)" are **not** what the code does.
  - Delete "No renal variables appear in carbohydrate rules" only if it
    becomes untrue. It is still true; keep it.
- **Discussion point (limitation):** these are inpatient/short-term
  outcome proxies for dietary sensitivity, not direct measures of dietary
  response.

## 5. Delete § "Data Integrity and Spurious Correlation Mitigation" (lines 612–637)

Its premises no longer exist: synthetic HbA1c sampling, a shared
monotonic preprocessor, BMI co-occurrence patterns in synthetic CKD+HTN.
Its claims were also not true of the code: the transformer was shared,
not per-target. Replace it with a short "Data leakage safeguards"
paragraph:

- patient-level split (no patient in train and test);
- all preprocessing, isotonic maps, class-balancing weights, reweighting
  and synthetic rows fitted on training rows only;
- test set scored once.

## 6. Preprocessing & monotonic constraints (lines 639–690)

- Per-target isotonic transformers (Eq. `isotonic_exp`) are **now actually
  implemented**, fitted on each target's labelled training rows. Keep the
  equation. Delete "eliminates the cross-target contamination identified
  in Section …" (lines 658–662).
- Positive constraint list: creatinine, serum K, serum Na, HbA1c, FBS,
  SBP, DBP, BMI; negative: eGFR. This matches the draft.

## 7. Training & evaluation setup (lines 740–750, 975–1000) — R2 ¶4

- **Split:** replace "80/20 stratified on sodium_sensitivity" with:
  - patient-level `StratifiedGroupKFold`, stratified by comorbidity group;
  - **test = 74,663 admissions / 32,330 patients**, held out before any
    fitting;
  - training patients divided into 5 grouped CV folds; fold 0 (59,205
    admissions) used for tuning and early stopping; the final model is
    trained on folds 1–4 (**238,418 admissions**).
- **Optimisation:** Adam **lr 0.02** (was 1e-3), batch **4096**, virtual
  batch 256, **patience 10** (was 15), max 100 epochs, StepLR (50, 0.9),
  weight decay 1e-5, seed 42. Best epochs 21–27. Architecture unchanged
  (n_d = n_a = 16, n_steps = 5, γ = 1.3, sparsemax).
  - Chosen from a 7-setting grid on fold 0 (tuning_results_fold0.csv).
    The original settings (lr 1e-3, no balancing) scored mean validation
    macro-F1 0.299 vs 0.421 for the chosen setting.
- **Class-balanced sampling** (inverse class frequency), because outcomes
  are ~80% "low".
- **Calibration (new subsection, also answers R1 ¶5):** balanced sampling
  inflates raw class scores. The risk **tier** (label) is the argmax of
  the balanced scores. The reported **probabilities and the severity
  score s_i** use prior-corrected probabilities,
  p̃_k ∝ p_k · π_k / π̂_k (training prior over sampler prior).
  - Test calibration error falls from 0.068–0.166 to **0.004–0.008**.
  - Mean calibrated severity matches the observed mean level:

    | Target | Calibrated severity | Observed level |
    |---|---|---|
    | sodium | 0.233 | 0.236 |
    | potassium | 0.209 | 0.212 |
    | protein | 0.223 | 0.217 |
    | carb | 0.360 | 0.348 |

  - Update Eq. `severity` to use calibrated probabilities. Add the
    reliability plots as a figure.
- **Metrics:** replace accuracy / weighted-F1 as primary with
  **macro-F1, balanced accuracy, macro one-vs-rest AUROC, Cohen's κ** and
  quadratic κ. Accuracy is misleading with ~80% low class.
  - **95% CIs by patient-level cluster bootstrap (1000 resamples)**, plus
    paired TabNet − rules differences.
- **Reproducibility:** re-running the final fit from the committed code
  reproduces identical results (deterministic seeds).
- Gayathri's 5×5 repeated grouped CV and the missing-data / noise
  robustness results go here when done.

## 8. Claims to moderate everywhere — R2 ¶5, R3 major #2, editor

| Where | Current | Change to |
|---|---|---|
| Title / abstract (submitted version) | "Privacy-Preserving" | Drop, or "locally deployed / data-resident". Nothing in the system is a privacy-preserving learning mechanism (R3 major #2). |
| lines 81, 243, 384, 427–430, 449, 472 | "ensures DPDPA compliance", "full DPDPA 2023 compliance" | "designed for local data residency consistent with DPDPA 2023 data-localisation principles; no formal compliance audit has been performed" |
| lines 75, 143, 238, 754, 898, 1193 | "safe gram quantities", "safe alternatives" | "guideline-bounded gram limits" / "guideline-compliant alternatives" |
| lines 424, 1292 | "ensuring clinical safety" | "constraining generated recipes to the rules engine's gram limits" |
| line 150 | "designed and validated in the Indian clinical context" | "designed for the Indian clinical context; evaluated retrospectively on a US cohort (MIMIC-IV) reweighted to Indian comorbidity prevalence" |
| lines 1313–1314 | "directly deployable", "validated and immediately deployable" | Delete. Replace with "requires prospective clinical validation before deployment" |
| wherever it appears | "generalized rather than memorized" | Delete |
| Section 5 | "verified manually against KDIGO/AHA" | "generated by `usecase_report.py` from the deployed model" |

## 9. Results (lines 1003–1128) — R2 ¶1, ¶4; R3 major #1

**Replace Table `tab:tabnet_perf`** with TabNet vs rules on the held-out
test set (point estimate [95% CI]):

| Target | n | TabNet macro-F1 | Rules macro-F1 | Δ (95% CI) | TabNet bal-acc | Rules bal-acc | TabNet κ | Rules κ | TabNet AUROC |
|---|---|---|---|---|---|---|---|---|---|
| sodium | 40,645 | 0.390 [0.383, 0.397] | 0.371 [0.361, 0.382] | +0.009 to +0.027 | 0.514 | 0.458 | 0.175 | 0.120 | 0.756 [0.748, 0.763] |
| potassium | 49,803 | 0.414 [0.408, 0.419] | 0.386 [0.380, 0.392] | +0.023 to +0.032 | 0.476 | 0.464 | 0.171 | 0.133 | 0.721 [0.714, 0.728] |
| protein/renal | 49,705 | 0.379 [0.374, 0.384] | 0.321 [0.316, 0.325] | +0.052 to +0.065 | 0.472 | 0.359 | 0.142 | 0.092 | 0.673 [0.666, 0.680] |
| carb | 49,530 | 0.486 [0.479, 0.492] | 0.317 [0.312, 0.322] | +0.162 to +0.175 | 0.527 | 0.425 | 0.299 | 0.088 | 0.775 [0.769, 0.780] |
| **mean** | | **0.417** | **0.349** | | **0.497** | **0.427** | **0.197** | **0.108** | **0.731** |

(Rules have no probabilities, so no AUROC.)

**Text to add:**

- **Ceiling check:** a gradient-boosting model (sklearn
  HistGradientBoosting) on the same validation fold reaches AUROC
  0.70–0.77. Performance is limited by what 14 admission-time features can
  predict about later outcomes, not by the architecture. This is also the
  honest answer to "why TabNet": comparable accuracy plus native
  per-patient attribution.
- **Protein/renal caveat:** quadratic κ is lower for TabNet (0.064 vs
  0.149). TabNet detects far more deterioration: moderate recall 51% vs
  11%, high 37% vs 28%. The cost is more low→high false alarms (10,929 vs
  8,273). Present as a sensitivity/specificity trade-off.
- **Delete** "all misclassifications are confined to adjacent ordinal
  classes" (lines 1008–1012) and the matching figure caption. It is no
  longer true. Replace Fig. `fig:tabnet_cms` with the new confusion
  matrices (`artifacts/models/reports/confusion_matrix_*.png`).
- **Indian-prevalence-reweighted test set:** TabNet > rules on potassium
  (Δ +0.017 to +0.026), protein (+0.051 to +0.065) and carb (+0.165 to
  +0.179). **Sodium is a tie** (−0.012 to +0.005).
- **Subgroups** (subgroups.csv): TabNet ≥ rules in most comorbidity
  groups. Exceptions: protein in non-CKD groups (DM only −0.084, HTN+DM
  −0.083, HTN only −0.050, none −0.022) and carb in DM+CKD (−0.016,
  n = 374).
- **COVID-era sensitivity:** 2020–22 vs 2008–19 AUROC within 0.01–0.02
  for every target.
- **Ablations** (same real test set):

  | Target | Real-only (deployed) | + synthetic rows | India-weighted training |
  |---|---|---|---|
  | sodium | 0.390 | 0.394 | 0.369 |
  | potassium | 0.414 | 0.412 | 0.391 |
  | protein | 0.379 | 0.396 | 0.344 |
  | carb | 0.486 | 0.508 | 0.484 |

  - Synthetic rows: 61,932 Gaussian-copula rows for CKD only, DM only and
    DM+CKD (17.2% of training). Fidelity: max standardised mean
    difference ≤ 0.024, correlation difference ≤ 0.03. Used only in
    training; never evaluated on. They gave a small, inconsistent gain,
    with no gain on validation, so the deployed model stays real-only.
  - Training with India weights **hurt** performance. Report as a
    negative result.

**Replace Table `tab:attribution`** (lines ~1095–1121) with test-set mean
attributions (`attribution_summary.json`, top 3):

| Target | Top attributions (mean weight) |
|---|---|
| sodium | SBP 0.377, eGFR 0.210, has_htn 0.147 |
| potassium | serum K 0.361, eGFR 0.156, has_ckd 0.118 |
| protein/renal | eGFR 0.272, age 0.140, has_ckd 0.134 |
| carb | FBS 0.250, HbA1c 0.181, has_dm 0.155 |

These align with guideline expectations, which strengthens the
interpretability claim. Also delete lines 1123–1128 (spurious-correlation
claim).

## 10. Clinical use case, Section 5 (lines 1130–1256) — R2 ¶2

- **Model 1 outputs** for the manuscript's patient (58 M, CKD3 eGFR 45,
  HTN SBP 162, no DM, HbA1c 5.2, FBS 85) from `usecase_report.md`:

  | Target | Tier | Rules | Calibrated P(high) | Severity |
  |---|---|---|---|---|
  | sodium | HIGH | high | 0.11 | 0.60 |
  | potassium | MODERATE | high | 0.17 | 0.61 |
  | protein | HIGH | high | 0.07 | 0.43 |
  | carb | **LOW** | moderate | 0.03 | 0.13 |

  - **Carb is now LOW**, consistent with normal glycaemia. This resolves
    Reviewer 2 ¶2.
  - Replace "all four targets HIGH at 100% model confidence". The new
    text should explain tier vs calibrated probability: a HIGH tier means
    the patient is in the highest-risk group relative to the base rate,
    not that the outcome is likely.
  - Per-patient attributions are noisier than the test-set averages
    (e.g. protein's top feature for this patient is HbA1c). Either report
    the test-set table (§9) instead, or show and discuss.
- **Portion table:** replace Table `tab:usecase` with
  `artifacts/models/reports/usecase_table.tex`. It comes from the single
  unified engine (`train_model2.PortionControlModel`, which the app now
  calls). The draft's numbers (12.4 g, 15.9 g, …) do not reproduce.

  | Ingredient | Decision | Max (g) | Binding |
  |---|---|---|---|
  | Green gram dal | Half Portion | 34.2 | phosphorus |
  | Paneer | Half Portion | 24.9 | phosphorus |
  | Egg, whole, boiled | Half Portion | 56.7 | phosphorus |
  | Rice, milled (white) | Allowed | 116.3 | phosphorus |
  | Banana, ripe | Allowed | 114.6 | potassium |
  | Apple | Allowed | 225.0 | potassium |
  | Pineapple | Allowed | 300.0 | 300 g practical cap |
  | Bottle gourd | Allowed | 300.0 | 300 g practical cap |

  Rewrite the surrounding narrative to match:
  - For this CKD patient, **phosphorus** (not protein) binds every
    protein source and rice. This comes from the KDIGO phosphorus
    constraint and the phosphorus-efficiency adjustment in Section
    `sec:portion`.
  - Potassium binds the fruits.
  - Pineapple and bottle gourd reach the 300 g practical cap.
- **Delete** "the cumulative carbohydrate load across the session causes
  the carbohydrate budget to bind … at the rice step" (lines 1237–1241).
  Each ingredient is scored independently against the daily budget;
  nothing accumulates within a request.
- Replace "verified manually against KDIGO 2024 and AHA/ACC" (line 1255)
  with "generated by `usecase_report.py`".

## 11. Limitations / Future work (lines 1294–1307) — R1 ¶1, R2 ¶4–5

Add:

- retrospective, single-centre US data (BIDMC) despite reweighting;
- outcome proxies are inpatient / short-term;
- fasting-glucose proxy;
- high HbA1c missingness;
- ICD-based comorbidity flags (under-coding);
- sodium target only for patients with outpatient follow-up (40,645 of
  74,663 test admissions);
- no prospective or dietitian-labelled validation yet;
- no formal privacy/security or regulatory assessment.

## 12. Declarations & data availability — editor

- New **"Declarations"** section with subheadings: Ethical approval /
  Consent to participate / Consent to publish. MIMIC-IV is de-identified
  and publicly available under PhysioNet credentialing; the original
  collection was approved by the BIDMC and MIT IRBs with a waiver of
  informed consent. **Confirm exact wording with the professor.**
- **Data availability:** MIMIC-IV v3.1 is available via PhysioNet
  credentialed access. Code (and aggregate cohort cards) is in the GitHub
  repo; row-level derived data cannot be shared under the DUA. **Use
  identical text in the submission system and the manuscript**, and
  select "yes" for the data availability declaration.

## 13. New citations

- Talukdar R et al. *Nephrology* 2025;30(1):e14420. doi:10.1111/nep.14420
  (India CKD prevalence 13.24%).
- MIMIC-IV v3.1 (PhysioNet), replacing the v2.2 citation.
- CKD-EPI 2021 race-free equation (Inker LA et al., NEJM 2021).
- KDIGO AKI definition (KDIGO 2012 AKI guideline).
- ICMR-INDIAB-17 (`anjana2023`) is already cited; it is now also used for
  reweighting.

---

## Corrected errors to disclose in the response letter

State these plainly as corrections, not improvements:

1. Clinical features other than BP were generated with `np.random`, not
   extracted from MIMIC-IV.
2. Labels were deterministic functions of the model's own inputs
   (circularity).
3. The eGFR formula was not CKD-EPI.
4. The train/test split was row-level, not patient-level.
5. The monotonic transformer was shared across targets, contrary to the
   Methods.
6. The carb rule in the paper differed from the code.
7. The deployed app used one network for all four targets, applied no
   monotonic transformer, reported hardcoded 0.9991 accuracy, and
   preprocessed input by column position.
8. The deployed model artifacts were a mixed set from two training runs.
9. The use-case outputs did not reproduce from the saved models.
10. The "cumulative carbohydrate load" explanation in Section 5 did not
    match the code.
11. "All misclassifications adjacent-class" is not true of the corrected
    model.
12. Two different portion engines existed. The deployed app had a
    simplified copy with no phosphorus constraint, no caloric
    reconciliation and no substitutions, and it disagreed with the
    described engine on 6 of 8 use-case ingredients. It is now unified:
    the app calls the same `train_model2` engine the paper describes.

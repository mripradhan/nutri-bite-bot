# Reviewer 2 Feedback — Remediation Plan

Submission ID: `48b78bef-dfde-405c-a072-59ef194461ba`
Journal: Springer Discover Computing (Q2) — Major Revision, 10-day window

## Grounding: what the code actually does

Before planning fixes, the training code and the manuscript
(`finalpaperdraft.tex`, found in `.claude/worktrees/agent-ab3420d601fb4c55e/`)
were checked directly. Two load-bearing findings:

### Finding A — the circularity is worse than the paper admits

In `clinical-models/train_model1.py`, `_add_synthetic_demographics()`
(line 574) generates **every clinical feature** — age, sex, HTN/DM/CKD
flags, serum sodium, serum potassium, creatinine, eGFR, HbA1c, FBS, BMI —
via `np.random` with a fixed seed, correlated by construction with the
disease flags. Only `sbp`/`dbp` fall back to real MIMIC-IV vitals when
present; everything else is synthetic regardless of whether real MIMIC-IV
data existed for that patient.

`LabelGenerator.generate_labels()` (line 651) then derives all four
targets as deterministic threshold functions of those *same* synthetic
variables, and `ClinicalModelConfig.feature_cols` (line 52) feeds the
identical 14 variables into TabNet as input.

So the reported 99.9%+ accuracy isn't just "labels are rule-derived from
the same inputs the model sees" (Reviewer 2's stated concern) — it's
TabNet learning a fixed formula over its own synthetic noise. This is a
bigger problem than the paper describes, and it is the underlying reason
Reviewer 2 and Reviewer 3 both flagged the accuracy as a red flag.

### Finding B — the "internal inconsistency" is a checkable, real bug/error

The manuscript's own label-rule table (`finalpaperdraft.tex` line ~597)
states: Carbohydrate = Moderate only if HbA1c ≥ 5.7, or FBS ≥ 100, or
DM = 1, or (BMI ≥ 30 and HbA1c ≥ 5.5); High requires even more extreme
values.

The representative patient (line ~1143) has HbA1c = 5.2, FBS = 85,
DM = 0, BMI = 26 — none of those conditions hold, so by the paper's own
stated rule the correct label is **LOW**. But the use-case section
(line ~1170) reports "Carbohydrate HIGH... driven by HbA1c=5.2 and
BMI=26 in the presence of CKD." This is not a subtle contradiction —
it is arithmetically impossible under the rule table two sections
earlier. It strongly suggests the use-case numbers were hand-written or
copied rather than regenerated from the final pipeline before
submission, exactly as Reviewer 2 suspects.

---

## 1. Fix the verifiable bug first (cheap, must happen regardless)

- Re-run `ClinicalRiskStratifier.predict()` /
  `Model1Integration.predict_risk_levels()` on the exact
  representative-patient vector (age 58, CKD3, eGFR 45, HbA1c 5.2,
  FBS 85, no DM) against the currently saved models in
  `artifacts/models/`.
- Whatever it actually outputs becomes the corrected Section 5 use
  case. If it still outputs HIGH, that is a live bug — likely in
  `MonotonicFeatureTransformer` or in `generate_clinical_explanation`'s
  attribution-suffix text (`train_model2.py` ~line 940) letting CKD
  flags leak into the carb explanation — and needs an actual code fix,
  not just a text rewrite.
- Regenerate every number in the use-case table from a fresh script
  run, not by hand-transcription.

## 2. Address circularity honestly (Reviewer 2 ¶1 / Reviewer 3 major #1)

Full independent clinical ground truth (real dietitian labels) is not
obtainable in a 10-day window. Two moves that are:

- **Reframe the claim.** Stop describing this as "prediction of
  independent clinical outcomes." State plainly that TabNet is a
  learned, differentiable surrogate for the deterministic rule engine,
  and justify why a learned surrogate is needed over the rules
  themselves:
  - continuous severity scores feed the sigmoid portion mapping
    directly, avoiding the discrete dose-cliffs the rules produce;
  - TabNet degrades gracefully under missing/partial labs via learned
    imputation, where the hard rule engine simply cannot fire;
  - per-instance attention gives an explanation surface the if/else
    engine does not.
- **Add the comparison Reviewer 2 explicitly asked for**: TabNet vs.
  the deterministic rule engine directly, on:
  1. a perturbed/noisy-label robustness test — inject clinically
     plausible measurement noise into labs and compare whose decision
     boundary is smoother/more stable near thresholds (e.g. HbA1c
     5.65–5.75);
  2. a missing-data test — drop 1–3 labs per patient; the rules engine
     fails closed, TabNet still predicts.

  Report agreement rate and where the two diverge — divergence near
  boundaries is the evidence that TabNet learned something beyond
  memorizing the lookup table.

## 3. Fix and document the synthetic dataset (Reviewer 2 ¶3 / Reviewer 1 ¶1)

**Dataset version decision:** use **MIMIC-IV v3.1**, not the v2.2 cited
in the current manuscript draft. v3.0 added ICU/hospital stays from
2020–2022 and grew the cohort to 364,627 patients; v3.1 fixed known
data-quality issues (inconsistent lab `itemid` mappings, orphaned
`subject_id`s present in v2.2/v3.0). The larger, cleaner cohort
directly strengthens the real-MIMIC-only holdout number this section
produces, and the extraction pipeline is being rewritten from scratch
regardless, so there is no cost to using the better release. Update the
Methods citation from "MIMIC-IV (version 2.2)" to "(version 3.1)" as
part of this change.

Caveat: the 2020–2022 stays added in v3.0 fall in the COVID-19 period,
where ICU surges and altered admission patterns could skew lab-value
distributions unrelated to the HTN/T2DM/CKD focus. Report what
proportion of the extracted cohort comes from 2020–2022, and run a
sensitivity check (real-only accuracy with vs. without those years) to
confirm it doesn't materially change the headline real-data number.

Access note: full MIMIC-IV is PhysioNet **credentialed** data (CITI
training + signed DUA, per-individual, non-transferable — sharing a
credentialed login, including a supervisor's, is a DUA violation).
Mrida's own credentialing is in progress; the ~100-patient
`mimic-iv-clinical-database-demo-2.2` already in the repo is v2.2
schema-only and fine for prototyping the extraction code in the
meantime, but not a substitute for the real holdout numbers below.

Rebuild `_add_synthetic_demographics` / `LabelGenerator` into a
documented, leakage-safe pipeline:

- Pull real MIMIC-IV labs/vitals directly from the standard `hosp`/
  `icu` tables (`labevents` + `d_labitems` for sodium, potassium,
  creatinine, HbA1c; `diagnoses_icd` + `d_icd_diagnoses` for HTN/DM/CKD
  flags; `chartevents` for BP) — the current loader's expected
  `B_EventLog.csv`/`E_ActivityAttributes.csv` format does not match
  this schema and needs to be replaced, not just repointed.
- For patients/fields MIMIC-IV cannot supply (Indian-comorbidity
  augmentation), generate synthetic rows from distributions calibrated
  to published Indian cohort statistics (e.g. ICMR-INDIAB) rather than
  arbitrary `np.random.normal` constants — this also answers
  Reviewer 1 ¶1 on Indian-population validity.
- Augment strictly **after** the train/test split (fit the split on
  real MIMIC-IV encounter IDs first, synthesize training-side rows
  only) so synthetic rows can never land in the test set.
- Log and report: N real, N synthetic, ratio per class, generation
  method per field, random seed, and — critically — **report accuracy
  separately on the real-MIMIC-only held-out subset vs. the full
  augmented test set.** If real-only accuracy is meaningfully lower
  than 99.9%, that is the honest number to publish, and it directly
  defuses Reviewer 2/3's "red flag."

## 4. Strengthen the validation protocol (Reviewer 2 ¶4)

- Replace/supplement the single 80/20 split in
  `ClinicalRiskStratifier.fit()` with stratified 5×5 repeated k-fold,
  reporting mean ± 95% CI per target per metric.
- Add the missing-data / label-noise robustness runs from Section 2
  above as a de facto "external validation" stand-in, clearly labeled
  as such (not claimed as clinical validation).
- If feasible in the time available, get even 20–30 real cases
  informally reviewed by a dietitian/clinician (RVCE's CHTR center, or
  the corresponding author's clinical contacts) as a small preliminary
  concordance check — explicitly flagged "preliminary, n=X, not a
  substitute for prospective validation," matching the framing
  Reviewer 3 already accepted for the existing dietitian-diet
  comparison.

## 5. Moderate the overstated claims (Reviewer 2 ¶5 / editor's note)

Concrete edits needed in the conclusion
(`finalpaperdraft.tex` ~line 1313):

- "satisfying DPDPA... making NutriBiteBot directly deployable" →
  reframe as local data residency / DPDPA-aligned architecture, not a
  deployment-readiness claim.
- Strike "validated and immediately deployable."
- Strike "generalized rather than memorized" outright — this cannot be
  claimed without the real-data-only accuracy number from Section 3.
- Soften "ensuring clinical safety" to "constraining outputs to
  guideline-derived bounds" throughout.
- Same treatment for Reviewer 3 ¶2 on privacy: reframe as data
  residency, not privacy-preserving machine learning.

---

## Deliverables for resubmission

1. Updated `train_model1.py` — documented synthetic pipeline, real-only
   eval split, k-fold CV, missing-data/noise robustness test, TabNet-
   vs-rules comparison.
2. Corrected, re-run use-case section (Section 5) with numbers
   generated by the actual current pipeline.
3. Tracked-changes manuscript with the claim-moderation edits.
4. Point-by-point response-to-reviewers letter.
5. New "Declarations" section (ethical approval / consent to
   participate / consent to publish) and a reconciled Data
   Availability Statement — required by the editor's letter
   regardless of Reviewer 2.

## Logistics note

This is a lot of new experimentation for a 10-day window. The editor's
letter explicitly invites contacting them with the submission ID if
more time is needed — worth doing now rather than rushing the
real-data analysis in Section 3, since a rushed number there risks
rejection on resubmission rather than acceptance.

## Suggested next step

Start with Section 1 (rerun the actual pipeline on the representative
patient) — it is fast and immediately tells us whether we are dealing
with a writing error or a live bug in the current model/pipeline.

---

## Work Split — Mrida & Gayathri

Split by dependency chain rather than by even task count, so each
person owns a track that can mostly move independently until the
final merge. Estimated effort balances to roughly the same total on
each side (~10–11 days).

### Mrida — Data & Pipeline Track (~11 days)

| Task | Section | Est. |
|---|---|---|
| Rerun current pipeline on the representative patient; confirm whether Finding B is a live bug or a stale write-up | 1 | 1 day |
| If live bug: fix `MonotonicFeatureTransformer` / attribution-suffix leak in `train_model2.py`; if stale write-up: regenerate Section 5 use-case numbers from a fresh script run | 1 | included above |
| Rebuild the synthetic dataset pipeline: audit real MIMIC-IV lab/vital coverage, calibrate synthetic fields to Indian cohort literature, move augmentation to strictly after the train/test split, add real-only vs. augmented-test accuracy reporting | 3 | 7–10 days |
| Owns: corrected use-case section (Section 5) and the new dataset-documentation subsection of the Methods | 1, 3 | — |

### Gayathri — Model Comparison & Manuscript Track (~10.5 days)

| Task | Section | Est. |
|---|---|---|
| Reframe the ML contribution (surrogate-model argument) in the Methods/Discussion | 2 | 1 day |
| Build TabNet-vs-rules-engine comparison: noisy-label robustness test + missing-data test, on the current pipeline (rerun once Mrida's rebuilt dataset lands) | 2 | 3 days |
| Add stratified 5×5 repeated k-fold CV + 95% CI reporting (depends on Mrida's rebuilt pipeline being in place) | 4 | 1.5 days |
| Claims-moderation edit pass across the manuscript (DPDPA/deployability/"generalized not memorized"/"clinical safety" language) | 5 | 2 days |
| Draft new Declarations section (ethical approval / consent to participate / consent to publish) and reconcile the Data Availability Statement between system and manuscript | Editor's letter | 1.5 days |
| First draft of the point-by-point response-to-reviewers letter | Deliverables | 1.5 days |

### Joint / final assembly (both, ~2 days, after both tracks land)

- Merge Mrida's dataset/use-case numbers into Gayathri's manuscript edit pass.
- Cross-check that every claim the response letter makes matches what's actually in the resubmitted manuscript.
- Produce the tracked-changes manuscript and finalize the response letter together.
- If an informal dietitian/clinician concordance check (Section 4, optional) is pursued, assign it to whichever of the two has the clinical contact — treat it as best-effort, drop it without blocking the resubmission if scheduling doesn't work out in time.

Note: the two tracks can start in parallel on day 1, but Gayathri's k-fold CV step and the final robustness-comparison rerun should wait for Mrida's rebuilt dataset pipeline to avoid computing numbers that get thrown away.

# Reviewer 2 Feedback — Remediation Plan

Submission ID: `48b78bef-dfde-405c-a072-59ef194461ba`
Journal: Springer Discover Computing (Q2) — Major Revision, 10-day window

This is now the single tracking doc for the reviewer-feedback remediation (folds in
what used to be `implementation_plan.md` and `manuscript_edits.md`, both retired —
their content is merged below; see `remaining_work.md` for everything else, i.e. the
non-manuscript code/repo work).

---

## Status (2026-09-23)

**Done** — both tracks' code-side work is complete and committed:

- Dataset rebuilt on real MIMIC-IV **v3.1** (545,848 admissions / 223,380 patients,
  372,286 eligible after baseline-chemistry filtering), patient-level
  `StratifiedGroupKFold` split, India reweighting (ICMR-INDIAB) + Gaussian-copula
  synthetic-row ablation, CKD-EPI 2021 eGFR.
- Labels switched from rule-derived (circular) to **observed clinical outcomes**
  measured after a 24 h baseline window (hyperkalaemia, AKI via KDIGO, hyperglycaemia,
  follow-up BP). Guideline rules kept only as the comparison baseline.
- TabNet retrained per-target with per-target monotonic transformers (previously
  shared — a real bug), class-balanced sampling + prior-correction calibration
  (test ECE 0.004–0.008), portable artifact format (`model1_artifacts.py`,
  schema v3) with a reload-reproduces-in-memory-predictions check.
- Held-out test evaluation (74,663 admissions / 32,330 patients, scored once):
  TabNet beats the rules baseline on **every target, every metric**, mean macro-F1
  0.417 vs 0.349, mean macro-AUROC 0.731. Bootstrap 95% CIs, Indian-reweighted
  variant, COVID-era sensitivity check, comorbidity-subgroup breakdown, synthetic/
  India-weighting ablations — all done (`clinical-models/evaluate_model1.py`).
- **5×5 repeated patient-grouped cross-validation** (training patients only, 100
  TabNet fits) confirms the single-split result: TabNet ahead of rules on all four
  targets, every paired-difference CI excludes zero (`clinical-models/repeated_cv.py`).
- **Robustness comparison** vs. the rules baseline (`clinical-models/robustness_test.py`):
  measurement-noise sweep (flip rate + agreement rate by magnitude), threshold-boundary
  stability (continuous TabNet probability vs. the rules' step function near each
  guideline threshold), and a missing-data test showing the rules failing closed
  (an increasing share of decisions pulled to a lower tier as inputs go missing) while
  TabNet's F1 stays materially stable.
- Two portion-engine code paths (the deployed app's simplified copy vs. the one
  described in the paper) unified — `app.py` now calls the same
  `train_model2.PortionControlModel` the manuscript describes; use-case table
  regenerated with zero disagreements.
- Manuscript (`manuscript/sn-article.tex`, gitignored — confidential review context,
  lives only on this machine + Mrida's) updated end-to-end: abstract, contributions,
  dataset section, label tables, leakage-safeguards section (replacing the old
  "spurious correlation" section), preprocessing, training/calibration, results
  (new TabNet-vs-rules table, repeated-CV table, ablations, subgroups, attribution),
  claims moderated throughout (DPDPA/deployability/"generalized not memorized"/
  "clinical safety" language), use case corrected (carb now LOW, resolves Reviewer 2
  ¶2), limitations expanded, Declarations skeleton added, new citations. Also caught
  and fixed two additional paper-vs-code mismatches found while doing this pass (not
  originally flagged by reviewers, but same reproducibility risk): the sigmoid
  portion-fraction equation's constants, and Algorithm 2's caloric-relaxation order —
  both now match the deployed code exactly (see `remaining_work.md` if the code side
  of either needs revisiting).

**Still pending** (manuscript/coordination only — no more code work on this track):

- [ ] **Declarations wording** — ethics-approval paragraph is a placeholder pending
      the corresponding author's exact wording.
- [ ] **Talukdar et al. citation** — CKD-prevalence citation for India; only DOI/
      journal/volume were available, not the full author list/title. Don't guess it.
- [ ] **Response-to-reviewers letter** — not drafted yet. Input is the "Corrected
      errors to disclose" list below.
- [ ] **Claims-moderation human read-through** — the edit list below was applied
      mechanically; worth a tone/framing pass before submission, especially the
      surrogate-model reframing (§2) and limitations section.
- [ ] **Figure check** — `sigmoid_severity.png` regenerated and correct; do a final
      visual pass on the other figures (confusion matrices, reliability plots) against
      the manuscript text before submission.
- [ ] **LaTeX build check** — no LaTeX toolchain on this machine; the manuscript
      hasn't been compiled since the edits. Do a full `pdflatex`/`latexmk` pass
      somewhere before submission.
- [ ] Confirm which `.tex` is actually the submitted version — the draft this edit
      pass was based on may not match what's on the journal's system (see the
      "Draft vs submitted manuscript" note under Decisions below).

---

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

**Resolved:** carb now correctly outputs LOW for this patient, regenerated by
`clinical-models/usecase_report.py` from the deployed pipeline, not hand-transcribed.

---

## 1. Fix the verifiable bug first (cheap, must happen regardless) — ✅ done

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

## 2. Address circularity honestly (Reviewer 2 ¶1 / Reviewer 3 major #1) — ✅ done

Full independent clinical ground truth (real dietitian labels) is not
obtainable in a 10-day window. Two moves that are:

- **Reframe the claim.** Stop describing this as "prediction of
  independent clinical outcomes." ~~State plainly that TabNet is a
  learned, differentiable surrogate for the deterministic rule engine~~
  — **superseded by D1 below**: rather than reframing as a surrogate, the
  labels themselves were switched to real observed outcomes, so TabNet is now
  genuinely predicting independent clinical outcomes and beating the rules
  baseline at it. The surrogate-model argument (continuous severity feeding the
  sigmoid mapping, graceful missing-data degradation, native attention) is kept
  in the Discussion as a secondary justification for TabNet over the rules
  engine even on equal footing, not as the primary framing.
- **Add the comparison Reviewer 2 explicitly asked for**: TabNet vs.
  the deterministic rule engine directly, on:
  1. a perturbed/noisy-label robustness test — inject clinically
     plausible measurement noise into labs and compare whose decision
     boundary is smoother/more stable near thresholds (e.g. HbA1c
     5.65–5.75); — done, `robustness_test.py`.
  2. a missing-data test — drop 1–3 labs per patient; the rules engine
     fails closed, TabNet still predicts. — done, `robustness_test.py`.

  Report agreement rate and where the two diverge — divergence near
  boundaries is the evidence that TabNet learned something beyond
  memorizing the lookup table.

## 3. Fix and document the synthetic dataset (Reviewer 2 ¶3 / Reviewer 1 ¶1) — ✅ done

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
— **Resolved:** 7.8% of admissions are 2020–2022; AUROC within 0.01–0.02
with vs. without, no material confound.

Access note: full MIMIC-IV is PhysioNet **credentialed** data (CITI
training + signed DUA, per-individual, non-transferable — sharing a
credentialed login, including a supervisor's, is a DUA violation).
Row-level MIMIC data must not be sent through third-party APIs; the
extraction/training code runs locally and only ever sees aggregates
(counts, missingness, metrics, public data dictionaries); counts < 10
are suppressed throughout. The supervising professor (credentialed DUA
holder) reviewed and approved this handling setup.

Rebuilt `_add_synthetic_demographics` / `LabelGenerator` into a
documented, leakage-safe pipeline (`clinical-models/mimic_extract.py`,
`cohort_data.py`):

- Real MIMIC-IV labs/vitals pulled directly from `hosp`/`icu` tables
  (`labevents`+`d_labitems`, `diagnoses_icd`+`d_icd_diagnoses`, `chartevents`,
  `omr`) — the old `B_EventLog.csv`/`E_ActivityAttributes.csv` loader replaced
  entirely, not repointed.
- Indian-population adjustment via **importance reweighting** to ICMR-INDIAB
  prevalence (primary) plus a **Gaussian-copula synthetic-row ablation**
  (secondary, training-only, fidelity-checked against real rows of the same
  comorbidity stratum) — answers Reviewer 1 ¶1.
- Split is **patient-level** (`StratifiedGroupKFold` on `subject_id`), fit before
  any augmentation touches the data; synthetic rows generated per training fold
  only, never scored.
- Logged: N real (372,286 eligible) / N synthetic (61,932, 17.2% of training) /
  class ratios / generation method / seed — in `data/derived/cohort_card_v3.1*.json`.
  **Real-only vs. augmented-test accuracy reported separately** (§ Results below) —
  real accuracy is the honest, much-lower-than-99.9% number, exactly as intended.

## 4. Strengthen the validation protocol (Reviewer 2 ¶4) — ✅ done (first two bullets)

- Replace/supplement the single 80/20 split in
  `ClinicalRiskStratifier.fit()` with stratified 5×5 repeated k-fold,
  reporting mean ± 95% CI per target per metric. — done, patient-grouped
  (not just stratified), `clinical-models/repeated_cv.py`.
- Add the missing-data / label-noise robustness runs from Section 2
  above as a de facto "external validation" stand-in, clearly labeled
  as such (not claimed as clinical validation). — done, `robustness_test.py`.
- If feasible in the time available, get even 20–30 real cases
  informally reviewed by a dietitian/clinician (RVCE's CHTR center, or
  the corresponding author's clinical contacts) as a small preliminary
  concordance check — explicitly flagged "preliminary, n=X, not a
  substitute for prospective validation." — **not pursued**; treated as
  best-effort/optional per the original plan, dropped without blocking
  resubmission (no clinical contact was available in the window).

## 5. Moderate the overstated claims (Reviewer 2 ¶5 / editor's note) — ✅ done

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

All five applied throughout `manuscript/sn-article.tex`.

---

## Decisions (resolved 2026-09-19, Mrida — kept for the response letter)

- **D1:** observed-outcome targets as primary; guideline rules as baseline. ✅
- **D2:** code is the source of truth — the paper's rule table is
  corrected to match the code (carb Moderate: `bmi >= 25` alone). ✅
- **D3:** synthetic data is used for training only (as an ablation);
  every reported metric comes from real held-out patients. ✅
- **General rule** applied throughout this whole remediation, including the two
  additional mismatches found later (sigmoid constants, Algorithm 2 order): where
  the paper and the code disagree, fix the paper to match the code, unless the code
  is clinically wrong (e.g. the original eGFR formula, which got fixed in code and
  then described accurately — it was a placeholder, not real CKD-EPI).

**Draft vs. submitted manuscript:** the `.tex` this edit pass was based on may not be
the submitted version — Reviewer 3's review quotes "no significant difference from
dietitian diets at p > 0.05" and the submitted title contains "Privacy-Preserving"
(now corrected in this draft too), neither of which appeared in the original working
draft found in the worktree. **Confirm which file is authoritative before
resubmission.**

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
8. The deployed model artifacts were a mixed set from two training runs
   (TabNet `.zip` files dated May 4; imputer/scaler/feature names dated March 7;
   no monotonic transformer on disk at all).
9. The use-case outputs did not reproduce from the saved models.
10. The "cumulative carbohydrate load" explanation in Section 5 did not
    match the code (each ingredient is scored independently against the
    daily budget; nothing accumulates within a request).
11. "All misclassifications adjacent-class" is not true of the corrected
    model.
12. Two different portion engines existed. The deployed app had a
    simplified copy with no phosphorus constraint, no caloric
    reconciliation and no substitutions, and it disagreed with the
    described engine on 6 of 8 use-case ingredients. It is now unified:
    the app calls the same `train_model2` engine the paper describes.
13. The manuscript's sigmoid severity-to-fraction equation (Eq. 7) described
    different constants, and its algorithm for caloric-sufficiency relaxation
    (Algorithm 2) described a different formula and a different relaxation
    order, than what the deployed portion engine actually computes. Both
    corrected to match the code exactly (found during this remediation pass,
    not originally flagged by a reviewer).

---

## Logistics note

This is a lot of new experimentation for a 10-day window. The editor's
letter explicitly invites contacting them with the submission ID if
more time is needed — worth doing now rather than rushing the
real-data analysis in Section 3, since a rushed number there risks
rejection on resubmission rather than acceptance.

---

## Work Split — Mrida & Gayathri (both tracks complete)

Split by dependency chain rather than by even task count, so each
person owns a track that can mostly move independently until the
final merge.

### Mrida — Data & Pipeline Track — ✅ complete

| Task | Section | Status |
|---|---|---|
| Rerun current pipeline on the representative patient; confirm whether Finding B is a live bug or a stale write-up | 1 | done |
| Regenerate Section 5 use-case numbers from a fresh script run | 1 | done |
| Rebuild the synthetic dataset pipeline: audit real MIMIC-IV lab/vital coverage, calibrate synthetic fields to Indian cohort literature, move augmentation to strictly after the train/test split, add real-only vs. augmented-test accuracy reporting | 3 | done |
| Owns: corrected use-case section (Section 5) and the new dataset-documentation subsection of the Methods | 1, 3 | done |

### Gayathri — Model Comparison & Manuscript Track — ✅ complete

| Task | Section | Status |
|---|---|---|
| Reframe the ML contribution in the Methods/Discussion | 2 | done (superseded by D1 — see §2 above) |
| Build TabNet-vs-rules-engine comparison: noisy-label robustness test + missing-data test | 2 | done, `robustness_test.py` |
| Add stratified 5×5 repeated k-fold CV + 95% CI reporting | 4 | done, `repeated_cv.py` |
| Claims-moderation edit pass across the manuscript | 5 | done |
| Draft new Declarations section and reconcile the Data Availability Statement | Editor's letter | skeleton done; exact ethics wording still pending author sign-off — see Status |
| First draft of the point-by-point response-to-reviewers letter | Deliverables | **not started** — see Status |

### Joint / final assembly — partially done

- [x] Merge Mrida's dataset/use-case numbers into the manuscript edit pass.
- [ ] Cross-check that every claim the response letter makes matches what's actually
      in the resubmitted manuscript — blocked on the letter itself being drafted.
- [ ] Produce the tracked-changes manuscript and finalize the response letter together.
- [ ] Informal dietitian/clinician concordance check — dropped, no clinical contact
      available in the window (per §4 above, this was always optional/best-effort).

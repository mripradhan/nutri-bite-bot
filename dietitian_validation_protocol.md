# Clinical validation protocol for the computed gram ceilings

**Status: protocol only. No rating, linkage or pilot data have been collected, and nothing in this repository
claims clinical validity of the gram ceilings.** This document states what would have to be shown before any
clinical use, in the order it would have to be shown. It answers the reviewers' request to "state concretely what
evaluation would be required" (Reviewer 3 point 3; Reviewer 2 points 1, 2 and 15).

Thresholds marked **[to be fixed with clinical collaborators]** are proposals. They must be fixed, and the
analysis plan time-stamped, **before** any rating is collected.

## 0. What is and is not established today

| Question | Evidence today | Status |
|---|---|---|
| Does the classifier predict later clinical outcomes better than the guideline rules? | Retrospective MIMIC-IV v3.1 held-out test set, patient-cluster bootstrap CIs, 5x5 repeated grouped CV | Established, with modest absolute performance (macro-F1 0.38-0.49, AUROC 0.67-0.78) |
| Does the engine compute the ceilings it is specified to compute? | Unit tests; regression against the committed use case | Established (computational correctness) |
| Does generated text respect the ceilings and exclusions? | Deterministic verifier; adherence experiment (`recipe_adherence_eval.py`) | Preliminary; "no violations observed in the evaluated sample", not a guarantee |
| **Are the ceilings clinically appropriate or safe for a patient?** | **None** | **Not established** |
| Do the ceilings improve patient outcomes? | None | Not established |

The four classifier targets are later clinical outcomes (follow-up blood pressure, hyperkalaemia, acute kidney
injury, hyperglycaemia). They are **proxies** for dietary sensitivity, not measured responses to sodium, potassium,
protein or carbohydrate intake. The mapping from outcome risk to a budget fraction (a sigmoid in
`train_model2.PortionRecommender`) is a modelling assumption that this protocol is designed to test.

## Stage 1: blinded dietitian review of the ceilings (cross-sectional)

**Aim.** Estimate how often qualified dietitians judge the model's ceilings inappropriate or unsafe, how closely
the ceilings agree with the ceilings dietitians would set, and whether the learned component changes either
compared with a rules-only engine.

**Materials.** `clinical-models/dietitian_rating_sheet.py` generates the sheet. Patient profiles are **synthetic**
(drawn from plausible ranges across all eight HTN/DM/CKD strata). Real MIMIC-IV records must never be shown to
raters: the data-use agreement forbids sharing row-level credentialed data.

**Conditions (hidden from raters).** (A) the deployed engine, with severity from the TabNet classifier;
(B) the same engine with severity from the guideline-rule label. Where A and B agree within 10% the item is shown
once. About 10% of items are repeated verbatim to measure each rater's own consistency. Order is randomised;
the engine key is kept by the analyst only.

**Raters.** At least 3 independent registered dietitians with renal and/or diabetes experience, not involved in
building the system. Each rater sees every item, rates independently, and sees no other rater's responses.

**Per item** (columns of `rating_sheet.csv`):
1. Rating of the proposed ceiling: 1 too low / 2 appropriate / 3 too high / 4 unsafe.
2. The maximum grams per meal the rater would set.
3. Free-text comment.

**Sample size.** Default sheet: 6 patients per stratum x 8 ingredients = 48 patients, about 400 items per rater
(about 2 hours). Items from the same patient are correlated, so the effective sample is smaller than the number of
items. If no unsafe rating is observed among *n* effectively independent items, the one-sided 95% upper bound on the
rate is 1 - 0.05^(1/n):

| effective n | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| upper bound | 5.8% | 3.0% | 1.5% | 0.75% |

**Primary endpoints.**
- **P1: unsafe or too-high rate.** Item-level majority rating (>= 2 of 3 raters) of 3 or 4, per engine, with
  patient-cluster bootstrap 95% CIs.
- **P2: unsafe rate.** Majority rating of 4, per engine, with the upper 95% bound.

**Secondary endpoints.**
- **S1: agreement.** Fleiss' kappa across raters on the 4-level rating; intra-rater kappa on repeated items.
- **S2: concordance with the rater-set ceiling.** Per-item log-ratio of the model ceiling to the median rater
  ceiling; median ratio, 95% limits of agreement, Lin's concordance correlation coefficient.
- **S3: learned vs rules.** Paired difference between engines A and B in P1 and in S2, on items where they differ.
- **S4: subgroups.** All endpoints by stratum (CKD, diabetes, hypertension) and by binding nutrient.

**Proposed acceptance criteria [to be fixed with clinical collaborators].** Stage 1 is passed only if all of:
the upper 95% bound of P2 is <= 5%; the upper 95% bound of P1 is <= 15%; Fleiss' kappa >= 0.6 (otherwise the
ground truth itself is unreliable and the stage is inconclusive, not passed); and no stratum shows P2 above 10%.
Failure of the CKD stratum fails the stage, because that is where an over-permissive ceiling is most dangerous.

**What a pass means.** The ceilings are acceptable to dietitians for synthetic patients. It does not show that they
are safe for real patients or that they change outcomes.

## Stage 2: retrospective diet-outcome linkage

**Aim.** Test the assumption chain directly: do higher predicted outcome risk and a tighter ceiling correspond to
a worse response to the corresponding nutrient?

**Data needed.** A cohort with quantified dietary intake (food-frequency questionnaire or 24-hour recall mapped to
IFCT) **and** later laboratory or clinical outcomes in the same patients, ideally Indian. Candidate sources
(for example population cohorts such as ICMR-INDIAB or CARRS) have **not** been approached; access, consent
scope and data-sharing terms are all unknown.

**Pre-stated hypotheses.**
- H1: among patients with high predicted potassium risk, higher potassium intake is associated with later serum
  potassium above the cohort's hyperkalaemia threshold, with a larger slope than among low-risk patients
  (interaction test).
- H2, H3, H4: the analogous statements for protein intake and creatinine rise, carbohydrate intake and
  hyperglycaemia, and sodium intake and follow-up blood pressure.
- H5: patients whose reported intake exceeded the model-implied ceiling more often had adverse outcomes than those
  below it, after adjustment for age, sex, comorbidity, baseline laboratory values and energy intake.

**Reading the result.** A null result on H1-H4 means the proxy-to-prescription mapping is unsupported and the
ceilings should not be presented as nutrient-specific limits. Confounding by indication and measurement error in
dietary recall are expected; effect sizes, not only significance, must be reported.

## Stage 3: supervised prospective pilot

**Prerequisites.** Stage 1 passed; Stage 2 not contradicting the mapping; institutional ethics approval; trial
registry entry; a named responsible clinician. The tool is used only **with a dietitian in the loop**: the
dietitian sees each ceiling, may override it, and every override is logged.

**Design.** Single-site feasibility pilot, about 60 adults with CKD stage 3-4 and/or diabetes and hypertension,
followed for 12 weeks.

**Endpoints.** Primary: safety (serum potassium >= 5.5 mmol/L, eGFR fall >= 25%, hypoglycaemia events, symptomatic
hypotension). Secondary: dietitian override rate and direction, patient-reported usability, laboratory trajectories.

**Stopping rules [to be fixed with clinical collaborators].** Stop for a single serum potassium >= 6.0 mmol/L or
any event attributable to a ceiling the dietitian did not override.

## Gates

Do not proceed to Stage 2 or 3 unless the preceding stage meets its pre-registered criterion. Do not describe the
system as "clinically validated", "safe", or "for unsupervised use" until after Stage 3 and an external,
independent confirmation.

## Limitations of this plan

- Synthetic profiles in Stage 1 cannot reveal failures that depend on real-world data messiness.
- Dietitian judgement is itself variable; Stage 1 measures acceptability, not truth.
- Stage 2 depends on a data source that may not exist.
- Stage 3 is a feasibility pilot, not an efficacy trial.

# Remaining Work Roadmap

Covers repo hygiene (legacy/redundant files) and non-manuscript code work
(recipe-adherence feature, caloric-sufficiency safeguard). For the reviewer-feedback
remediation and manuscript/response-letter status, see `feedbackplan.md` instead —
that's the single tracker for anything reviewer-facing. Not a memory file — update or
delete sections as they're resolved.

---

## 1. Legacy / redundant files — needs a decision

| Path | Issue |
|---|---|
| `frontend/` vs `frontend-next/` | `app.py` (`FRONTEND_DIR`, line 37) still serves the **old** vanilla-JS `frontend/` via Flask's static routes. `frontend-next/` (Next.js/React/TS) has the recent commits (`da880f5`, `4fcbbd7`) and matches the architecture described in the manuscript. Either Flask's static-serving of `frontend/` is dead code (if Vercel hosts `frontend-next/` independently and only calls the API), or `python app.py` alone currently serves a stale UI. Confirm which is actually true of the live deployment before touching either directory. |
| `bitebot-recipe-generator/` | A **git submodule** (gitlink to commit `40f028f`), not a plain directory — appears empty on disk because it's never been initialized (`git submodule update --init`). Decide whether it's still needed; if not, remove the submodule entry properly (`git submodule deinit` + `git rm`), not just `rm -rf`. |
| `data/derived/cohort_card_demo.json` / `cohort_card_demo_split.json` | From the 100-patient demo-scale extraction. Superseded by the real `cohort_card_v3.1*.json`. Keep only if you want the demo run documented as a build-smoke-test reference. |
| `inspo_sites/` | Three design-reference screenshots, not code. Harmless clutter; move out of the repo if it was only ever meant as local reference material. |

---

## 2. Recipe-adherence feature (reviewer #7 ask)

Code is written and unit-tested (mocked Groq client); **not yet exercised against real
infrastructure**:

- [ ] **Apply the Supabase migration.** `supabase/migrations/20260923120000_recipe_adherence.sql`
      creates the `recipe_adherence` table — it hasn't been pushed to the actual Supabase
      instance yet (`supabase db push`, or apply manually via the dashboard SQL editor).
      `/api/generate-recipe` will fail silently on the adherence-logging step until this
      exists (`save_recipe_adherence` just logs an error and returns `False` if the insert
      fails — it won't crash the request, but nothing will be recorded).
- [ ] **Real Groq smoke test.** The extraction call (`response_format={"type": "json_object"}`)
      has only been tested with a mocked client. Generate a handful of real recipes and
      inspect `adherence` in the API response to confirm: (a) the model actually returns
      valid JSON in this mode for `llama-3.3-70b-versatile`, (b) the name-matching
      (exact → substring → token-overlap) handles real Groq phrasing, not just the
      synthetic cases in the smoke test.
- [ ] **Baseline the violation rate.** Once real recipes are flowing, pull the first
      batch of `recipe_adherence` rows and compute violation rate / mean overage —
      this is the actual number reviewer #7 wants, and it doesn't exist yet since no
      real recipe has been generated with this code live.
- [ ] **Decide on the 5% tolerance.** `RECIPE_ADHERENCE_TOLERANCE = 0.05` in `app.py` was
      a reasonable-default judgment call to absorb extraction rounding noise, not a
      clinically-derived number. Revisit once real violation-rate data exists — if
      real violations cluster just above 5% over the limit, the tolerance may be masking
      genuinely unsafe outputs.
- [ ] **Frontend surfacing.** The API now returns `"adherence"` and `"clinical_warnings"`
      in the `/api/generate-recipe` response, but nothing in `frontend/` or
      `frontend-next/` displays either yet. Decide whether/how to surface them in the UI
      (e.g. a warning banner) rather than leaving them API-only.

---

## 3. Caloric-sufficiency safeguard (Algorithm 2)

- [ ] The no-known-ingredients fallback branch in `portion_recommendations()`
      (`app.py`) skips the caloric validator entirely — low priority, only matters
      when zero requested ingredients exist in the IFCT database (nothing to compute
      calories for anyway).

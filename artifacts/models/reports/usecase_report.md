# Clinical use case — regenerated from the deployed system

Model trained 2026-09-23T15:42:37+00:00, commit cc8112e.

## Model 1

| Target | Tier (TabNet) | Guideline rules | P(low / moderate / high), calibrated | Severity | Top attributions |
|---|---|---|---|---|---|
| sodium_sensitivity | HIGH | high | 0.51 / 0.39 / 0.11 | 0.60 | serum_potassium (0.36), sbp (0.33), has_htn (0.13) |
| potassium_sensitivity | MODERATE | high | 0.56 / 0.27 / 0.17 | 0.61 | serum_potassium (0.38), egfr (0.29), sex_male (0.16) |
| protein_restriction | HIGH | high | 0.64 / 0.29 / 0.07 | 0.43 | hba1c (0.32), egfr (0.16), sex_male (0.15) |
| carb_sensitivity | LOW | moderate | 0.90 / 0.07 / 0.03 | 0.13 | fbs (0.25), hba1c (0.25), has_dm (0.19) |

## Model 2 — portion decisions (deployed app.py)

| Ingredient | Decision | Max (g) | Binding constraint |
|---|---|---|---|
| Green gram dal (Moong dal) | Half Portion | 34.2 | phosphorus |
| Paneer (Cottage cheese) | Half Portion | 24.9 | phosphorus |
| Egg, whole, boiled | Half Portion | 56.7 | phosphorus |
| Rice, milled (white) | Allowed | 116.3 | phosphorus |
| Banana, ripe | Allowed | 114.6 | potassium |
| Apple | Allowed | 225.0 | potassium |
| Pineapple (Ananas) | Allowed | 300.0 | potassium |
| Bottle gourd (Lauki) | Allowed | 300.0 | potassium |

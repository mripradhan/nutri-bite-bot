-- Recipe Adherence Table
-- Logs the post-hoc, per-ingredient check of whether the LLM-generated recipe's stated
-- gram quantity respects the safe_ingredients[].max_grams limit that was put in its
-- prompt. One row per ingredient per recipe, so violation rate / mean overage can be
-- computed in aggregate (quantitative-adherence reporting).
CREATE TABLE recipe_adherence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recipe_id UUID REFERENCES recipes(id) ON DELETE CASCADE,
    ingredient TEXT,
    max_grams NUMERIC,
    stated_grams NUMERIC,
    matched BOOLEAN,
    violated BOOLEAN,
    overage_grams NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE recipe_adherence DISABLE ROW LEVEL SECURITY;

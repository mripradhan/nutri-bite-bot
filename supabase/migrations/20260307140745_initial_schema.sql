-- Patients Table (Medical Information)
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    age INT,
    sex_male BOOLEAN,
    has_htn BOOLEAN,
    has_dm BOOLEAN,
    has_ckd BOOLEAN,
    serum_sodium NUMERIC,
    serum_potassium NUMERIC,
    creatinine NUMERIC,
    egfr NUMERIC,
    hba1c NUMERIC,
    fbs NUMERIC,
    sbp NUMERIC,
    dbp NUMERIC,
    bmi NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recipes Table
CREATE TABLE recipes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    ingredients_used JSONB,
    recipe_content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

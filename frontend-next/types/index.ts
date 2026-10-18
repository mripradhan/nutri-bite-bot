// ─── Patient & Form Types ───────────────────────────────────────────────────

export interface PatientData {
  age: number | ''
  sex_male: number | ''          // 0 = Female, 1 = Male
  has_htn: boolean
  has_dm: boolean
  has_ckd: boolean
  serum_sodium: number | ''
  serum_potassium: number | ''
  creatinine: number | ''
  egfr: number | ''
  hba1c: number | ''
  fbs: number | ''
  sbp: number | ''
  dbp: number | ''
  bmi: number | ''
}

// ─── API Payload Helpers ────────────────────────────────────────────────────

export interface PatientPayload extends Omit<PatientData, 'has_htn' | 'has_dm' | 'has_ckd'> {
  has_htn: number   // 0 | 1
  has_dm: number    // 0 | 1
  has_ckd: number   // 0 | 1
}

// ─── Prediction Result ──────────────────────────────────────────────────────

export type RiskLabel = 'low' | 'moderate' | 'high'

export interface RiskLevel {
  label: RiskLabel
  confidence: number
  probabilities: {
    low: number
    moderate: number
    high: number
  }
  display_name: string
  clinical_note: string
  /** Calibrated severity on a 0-2 scale (the value the portion engine uses). */
  severity_score?: number
}

export type TargetName =
  | 'sodium_sensitivity'
  | 'potassium_sensitivity'
  | 'protein_restriction'
  | 'carb_sensitivity'

export interface PredictionResult {
  risk_levels: Record<TargetName, RiskLevel>
  patient_summary: {
    age: number
    sex: string
    conditions: Record<string, boolean>
    key_labs: Record<string, number | string>
  }
  nutrient_thresholds: Record<string, unknown>
  condition_key: string
}

// ─── Recommendation Result ──────────────────────────────────────────────────

export interface NutrientLoad {
  sodium_mg: number
  potassium_mg: number
  protein_g: number
  carbs_g: number
  phosphorus_mg: number
  calories: number
}

export interface IngredientRecommendation {
  ingredient: string
  category: string
  max_grams: number
  label: 'Allowed' | 'Half Portion' | 'Avoid' | 'Not Found' | 'Excluded'
  excluded_reason?: string
  binding_constraint: string
  nutrient_load: NutrientLoad
  nutrients_per_100g: NutrientLoad
  constraint_grams?: Record<string, number | null>
  explanation?: string
  suggestions?: string[]
}

export interface DailyBudget {
  sodium_mg: number
  potassium_mg: number
  protein_g: number
  carbs_g: number
  phosphorus_mg: number
}

export interface ExclusionSummary {
  groups: string[]
  terms: string[]
  excluded_ingredients: Record<string, string>
}

export interface RecommendResult {
  risk_levels: Record<TargetName, { label: RiskLabel; confidence: number; display_name: string }>
  severity_scores: Record<TargetName, number>
  daily_budget: DailyBudget
  recommendations: IngredientRecommendation[]
  patient_conditions: { has_ckd: boolean; has_htn: boolean; has_dm: boolean }
  exclusions?: ExclusionSummary
  disclaimer?: string
}

// ─── Exclusions (allergies / intolerances / preferences) ────────────────────

export interface Exclusions {
  groups: string[]   // allergen group ids, e.g. 'dairy'
  terms: string[]    // free-text avoid terms
}

export interface AllergenGroup {
  id: string
  label: string
}

// ─── Recipe Result ──────────────────────────────────────────────────────────

export interface AdherenceRecord {
  ingredient: string
  max_grams: number
  stated_grams: number | null
  matched: boolean
  violated: boolean
  overage_grams: number | null
}

export type VerificationStatus = 'verified' | 'verified_after_regeneration' | 'template_fallback'

export interface RecipeResult {
  recipe: string
  portions_used?: IngredientRecommendation[]
  adherence?: AdherenceRecord[]
  verification?: {
    method: string
    status: VerificationStatus
    attempts: number
    rejected_issues: Array<Record<string, unknown>>
    note?: string
  }
  exclusions?: ExclusionSummary
  disclaimer?: string
}

// ─── Detect Result ──────────────────────────────────────────────────────────

export interface DetectResult {
  detected_raw: string[]
  mapped_ifct: string[]
}

// ─── Model Info ─────────────────────────────────────────────────────────────

export interface ModelInfo {
  models: Record<
    TargetName,
    {
      display_name: string
      type: string
      classes: string[]
      features_used: string[]
    }
  >
  accuracy_metrics: {
    evaluated_on?: string
    mean_f1_macro: number | null
    mean_balanced_accuracy: number | null
    mean_auroc_ovr: number | null
    mean_accuracy?: number | null
    mean_cohen_kappa?: number | null
  }
  targets_note?: string
  disclaimer?: string
  feature_count: number
  feature_names: string[]
  preprocessing: string[]
  model_backend: string
}

// ─── Ingredients List ───────────────────────────────────────────────────────

export interface IngredientsListResult {
  ingredients: string[]
  by_category?: Record<string, string[]>
}

import type {
  PatientData,
  PatientPayload,
  PredictionResult,
  RecommendResult,
  DetectResult,
  ModelInfo,
  IngredientsListResult,
} from '@/types'

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? 'https://nutri-bite-bot.onrender.com'

// ─── Wake-up Banner ─────────────────────────────────────────────────────────
// Exported so the hook can subscribe
type WakeUpListener = (show: boolean) => void
const wakeUpListeners: WakeUpListener[] = []

export function subscribeToWakeUp(fn: WakeUpListener) {
  wakeUpListeners.push(fn)
  return () => {
    const idx = wakeUpListeners.indexOf(fn)
    if (idx !== -1) wakeUpListeners.splice(idx, 1)
  }
}

function notifyWakeUp(show: boolean) {
  wakeUpListeners.forEach((fn) => fn(show))
}

// ─── Core fetch wrapper ──────────────────────────────────────────────────────
async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let timerFired = false
  const timer = setTimeout(() => {
    timerFired = true
    notifyWakeUp(true)
  }, 4000)

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.headers ?? {}),
        // Don't set Content-Type for FormData — browser sets it with boundary
        ...(init?.body instanceof FormData
          ? {}
          : { 'Content-Type': 'application/json' }),
      },
    })

    clearTimeout(timer)
    if (timerFired) notifyWakeUp(false)

    if (!res.ok) {
      const text = await res.text()
      throw new Error(`API ${res.status}: ${text}`)
    }

    return (await res.json()) as T
  } catch (err) {
    clearTimeout(timer)
    if (timerFired) notifyWakeUp(false)
    throw err
  }
}

// ─── Patient payload conversion ──────────────────────────────────────────────
export function toPayload(p: PatientData): PatientPayload {
  return {
    ...p,
    has_htn: p.has_htn ? 1 : 0,
    has_dm: p.has_dm ? 1 : 0,
    has_ckd: p.has_ckd ? 1 : 0,
  }
}

// ─── API calls ────────────────────────────────────────────────────────────────

export async function apiPredict(patient: PatientData): Promise<PredictionResult> {
  return apiFetch<PredictionResult>('/api/predict', {
    method: 'POST',
    body: JSON.stringify(toPayload(patient)),
  })
}

export async function apiRecommend(
  patient: PatientData,
  ingredients: string[],
): Promise<RecommendResult> {
  return apiFetch<RecommendResult>('/api/recommend', {
    method: 'POST',
    body: JSON.stringify({ patient: toPayload(patient), ingredients }),
  })
}

export interface RecipeRequestBody {
  patient: PatientPayload
  ingredients: string[]
  equipment: string
  cuisine: string
  time_constraint: string
  portions_used?: Record<string, number>
}

export async function apiGenerateRecipe(body: RecipeRequestBody): Promise<{ recipe: string }> {
  // The backend reads `time_limit`, not `time_constraint`
  const backendBody = {
    patient: body.patient,
    ingredients: body.ingredients,
    equipment: body.equipment,
    cuisine: body.cuisine,
    time_limit: body.time_constraint,
    portions_used: body.portions_used,
  }
  return apiFetch<{ recipe: string }>('/api/generate-recipe', {
    method: 'POST',
    body: JSON.stringify(backendBody),
  })
}

export async function apiDetect(imageFile: File): Promise<DetectResult> {
  const fd = new FormData()
  fd.append('image', imageFile)
  return apiFetch<DetectResult>('/api/detect', {
    method: 'POST',
    body: fd,
  })
}

export async function apiIngredients(query?: string): Promise<IngredientsListResult> {
  const qs = query ? `?q=${encodeURIComponent(query)}` : ''
  return apiFetch<IngredientsListResult>(`/api/ingredients${qs}`)
}

export async function apiModelInfo(): Promise<ModelInfo> {
  return apiFetch<ModelInfo>('/api/model-info')
}

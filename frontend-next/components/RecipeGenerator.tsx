'use client'

import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useApp } from '@/context/AppContext'
import { apiGenerateRecipe, toPayload } from '@/lib/api'
import type { RecipeResult } from '@/types'

// The API wraps errors as `API <status>: {json}`; surface the server's message.
function friendlyError(err: unknown): string {
  const raw = err instanceof Error ? err.message : 'Recipe generation failed'
  const m = raw.match(/^API \d+: ([\s\S]*)$/)
  if (!m) return raw
  try {
    const j = JSON.parse(m[1])
    return j.error ?? raw
  } catch {
    return raw
  }
}

export default function RecipeGenerator() {
  const { patientData, selectedIngredients, recipeOutput, setRecipeOutput, exclusions } = useApp()
  const [result, setResult] = useState<RecipeResult | null>(null)
  const [equipment, setEquipment] = useState('Standard kitchen (stovetop, oven)')
  const [cuisine, setCuisine] = useState('Any')
  const [timeConstraint, setTimeConstraint] = useState('30 minutes')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    if (selectedIngredients.length === 0) {
      setError('Please add ingredients in the Portion Analysis section first.')
      return
    }

    setLoading(true)
    setError(null)
    setRecipeOutput('')
    setResult(null)

    try {
      const res = await apiGenerateRecipe({
        patient: toPayload(patientData),
        ingredients: selectedIngredients,
        equipment,
        cuisine,
        time_constraint: timeConstraint,
        exclusions,
      })

      setResult(res)
      setRecipeOutput(res.recipe ?? JSON.stringify(res))
    } catch (err) {
      setError(friendlyError(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="recipe-generator" className="px-6 md:px-8 py-10">
      <div className="section-label mb-2">Phase 3</div>
      <h2 className="text-[20px] font-bold text-text-primary mb-1">Recipe Generator</h2>
      <p className="text-text-secondary text-[13px] mb-6">
        Generate a recipe from your selected ingredients within your computed gram ceilings.
        Every recipe is checked by a deterministic parser (not a language model) against the
        ceilings and your exclusions; failing recipes are regenerated, then replaced by a simple
        template. Quantities are verified as written in the ingredient list only.
      </p>
      <p className="text-[11px] text-text-muted mb-6">
        Data notice: recipe generation sends your comorbidity flags, gram ceilings and exclusion
        list (not your lab values) to the Groq API.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Equipment */}
        <div>
          <label className="block text-[12px] font-medium text-text-secondary mb-1.5">
            Kitchen Equipment
          </label>
          <input
            type="text"
            className="w-full"
            placeholder="e.g. Stovetop, oven, pressure cooker"
            value={equipment}
            onChange={(e) => setEquipment(e.target.value)}
          />
        </div>

        {/* Cuisine */}
        <div>
          <label className="block text-[12px] font-medium text-text-secondary mb-1.5">
            Cuisine Preference
          </label>
          <input
            type="text"
            className="w-full"
            placeholder="e.g. Indian, Mediterranean, Any"
            value={cuisine}
            onChange={(e) => setCuisine(e.target.value)}
          />
        </div>

        {/* Time */}
        <div>
          <label className="block text-[12px] font-medium text-text-secondary mb-1.5">
            Time Constraint
          </label>
          <select
            className="w-full"
            value={timeConstraint}
            onChange={(e) => setTimeConstraint(e.target.value)}
          >
            <option>15 minutes</option>
            <option>30 minutes</option>
            <option>45 minutes</option>
            <option>1 hour</option>
            <option>Any</option>
          </select>
        </div>
      </div>

      {/* Selected ingredients preview */}
      {selectedIngredients.length > 0 && (
        <div className="mb-5 p-3 rounded-[4px] bg-[#0d0d0d] border border-[#303236]">
          <div className="text-[11px] text-text-muted mb-2">
            Using {selectedIngredients.length} ingredient{selectedIngredients.length !== 1 ? 's' : ''}:
          </div>
          <div className="flex flex-wrap gap-1.5">
            {selectedIngredients.map((ing) => (
              <span key={ing} className="chip text-[11px]">{ing}</span>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 rounded-[4px] bg-risk-high/10 border border-risk-high/30 text-risk-high text-[13px]">
          {error}
        </div>
      )}

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={loading || selectedIngredients.length === 0}
        className={[
          'w-full py-3.5 rounded-full font-semibold text-[14px] transition-all duration-200 mb-8',
          loading || selectedIngredients.length === 0
            ? 'bg-accent/30 text-black/40 cursor-not-allowed'
            : 'bg-accent text-[#000000] hover:bg-accent-hover glow-accent hover:scale-[1.005]',
        ].join(' ')}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Generating and verifying recipe…
          </span>
        ) : (
          'Generate recipe'
        )}
      </button>

      {/* Recipe output */}
      {recipeOutput && result?.verification && (
        <div
          className={[
            'mb-4 p-3 rounded-[4px] border text-[12px] leading-relaxed',
            result.verification.status === 'template_fallback'
              ? 'bg-risk-moderate/10 border-risk-moderate/40 text-risk-moderate'
              : 'bg-accent/10 border-accent/30 text-accent',
          ].join(' ')}
        >
          <div className="font-semibold mb-0.5">
            {result.verification.status === 'verified' && 'Passed deterministic verification'}
            {result.verification.status === 'verified_after_regeneration' &&
              `Passed deterministic verification after ${result.verification.attempts} attempts`}
            {result.verification.status === 'template_fallback' &&
              'No generated recipe passed verification: showing a basic template built from your ceilings'}
          </div>
          <div className="opacity-90">
            {result.adherence?.filter((a) => a.matched).length ?? 0} ingredient quantities checked,{' '}
            {result.adherence?.filter((a) => a.violated).length ?? 0} over a ceiling. Amounts
            implied only in the cooking steps are not checked. Research prototype; not dietitian-reviewed.
          </div>
        </div>
      )}
      {recipeOutput && (
        <div className="card animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="section-label">Generated Recipe</div>
            <button
              onClick={() => {
                navigator.clipboard.writeText(recipeOutput).catch(() => {})
              }}
              className="text-[11px] text-text-muted hover:text-accent transition-colors flex items-center gap-1.5"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              Copy
            </button>
          </div>
          <div className="recipe-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {recipeOutput}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </section>
  )
}

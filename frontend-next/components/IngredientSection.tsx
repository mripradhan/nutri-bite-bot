'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useApp } from '@/context/AppContext'
import { apiIngredients, apiRecommend } from '@/lib/api'
import type { IngredientRecommendation, DailyBudget } from '@/types'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'

// ─── Label badge ─────────────────────────────────────────────────────────────
const LABEL_STYLES = {
  Allowed: 'badge-low',
  'Half Portion': 'badge-moderate',
  Avoid: 'badge-high',
  'Not Found': 'bg-[#303236] text-text-muted border border-[#303236]',
}

function LabelBadge({ label }: { label: IngredientRecommendation['label'] }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold ${LABEL_STYLES[label] ?? ''}`}>
      {label}
    </span>
  )
}

// ─── Nutrient bar ─────────────────────────────────────────────────────────────
function NutrientBar({
  label,
  value,
  budget,
  unit,
  color,
}: {
  label: string
  value: number
  budget: number
  unit: string
  color: string
}) {
  const pct = Math.min(100, (value / budget) * 100)
  return (
    <div>
      <div className="flex justify-between text-[11px] mb-1">
        <span className="text-text-muted">{label}</span>
        <span className="font-medium text-text-primary">
          {value.toLocaleString()}{unit}
          <span className="text-text-muted"> / {budget.toLocaleString()}</span>
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-[#303236] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

// ─── Recommendation card ──────────────────────────────────────────────────────
function RecommendationCard({
  rec,
  budget,
}: {
  rec: IngredientRecommendation
  budget: DailyBudget
}) {
  const labelColor =
    rec.label === 'Allowed'
      ? '#34d399'
      : rec.label === 'Half Portion'
      ? '#fbbf24'
      : '#f87171'

  return (
    <div
      className="rounded-[4px] border p-4 space-y-3"
      style={{
        backgroundColor: 'rgba(21,22,23,0.9)',
        borderColor: labelColor + '30',
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-[14px] font-semibold text-text-primary">{rec.ingredient}</div>
          {rec.category && (
            <div className="text-[11px] text-text-muted mt-0.5">{rec.category}</div>
          )}
        </div>
        <LabelBadge label={rec.label} />
      </div>

      {rec.label === 'Not Found' ? (
        <p className="text-[12px] text-risk-moderate">{rec.explanation}</p>
      ) : (
        <>
          <div className="flex items-center gap-4">
            <div className="text-center">
              <div
                className="text-[24px] font-extrabold leading-none"
                style={{ color: labelColor }}
              >
                {rec.max_grams}g
              </div>
              <div className="text-[10px] text-text-muted mt-0.5">max portion</div>
            </div>
            <div className="flex-1 space-y-1.5">
              <NutrientBar
                label="Sodium"
                value={rec.nutrient_load?.sodium_mg ?? 0}
                budget={budget.sodium_mg}
                unit=" mg"
                color="#8a8f98"
              />
              <NutrientBar
                label="Potassium"
                value={rec.nutrient_load?.potassium_mg ?? 0}
                budget={budget.potassium_mg}
                unit=" mg"
                color="#8a8f98"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-2 pt-1">
            <span className="chip text-[10px]">
              <span className="text-text-muted mr-1">Binding:</span>
              <span className="text-accent capitalize">{rec.binding_constraint}</span>
            </span>
            {rec.nutrient_load?.protein_g != null && (
              <span className="chip text-[10px]">
                Protein {rec.nutrient_load.protein_g}g
              </span>
            )}
            {rec.nutrient_load?.carbs_g != null && (
              <span className="chip text-[10px]">
                Carbs {rec.nutrient_load.carbs_g}g
              </span>
            )}
            {rec.nutrient_load?.calories != null && (
              <span className="chip text-[10px]">
                {rec.nutrient_load.calories} kcal
              </span>
            )}
          </div>
        </>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function IngredientSection() {
  const {
    patientData,
    selectedIngredients,
    addIngredient,
    removeIngredient,
    recommendResult,
    setRecommendResult,
  } = useApp()

  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const suggestionRef = useRef<HTMLDivElement>(null)
  const searchDebounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setSuggestions([])
      return
    }
    setLoadingSuggestions(true)
    try {
      const result = await apiIngredients(q)
      setSuggestions(result.ingredients.slice(0, 8))
    } catch {
      // silently ignore autocomplete errors
    } finally {
      setLoadingSuggestions(false)
    }
  }, [])

  useEffect(() => {
    if (searchDebounce.current) clearTimeout(searchDebounce.current)
    searchDebounce.current = setTimeout(() => fetchSuggestions(query), 300)
    return () => {
      if (searchDebounce.current) clearTimeout(searchDebounce.current)
    }
  }, [query, fetchSuggestions])

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (suggestionRef.current && !suggestionRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSelect = (ing: string) => {
    addIngredient(ing)
    setQuery('')
    setSuggestions([])
    setShowSuggestions(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && suggestions.length > 0) {
      handleSelect(suggestions[0])
    }
  }

  const handleRecommend = async () => {
    if (!selectedIngredients.length) return
    setLoading(true)
    setError(null)
    try {
      const result = await apiRecommend(patientData, selectedIngredients)
      setRecommendResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Recommendation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="portion-analysis" className="px-6 md:px-8 py-10">
      <div className="section-label mb-2">Model 2</div>
      <h2 className="text-[20px] font-bold text-text-primary mb-1">Portion Analysis</h2>
      <p className="text-text-secondary text-[13px] mb-6">
        Search for ingredients and get Deep Learning-driven safe portion sizes tailored to your lab values.
      </p>

      {/* Search */}
      <div className="relative mb-4" ref={suggestionRef}>
        <div className="relative flex items-center">
          <svg
            className="absolute left-3 text-text-muted pointer-events-none"
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            className="w-full pl-9 pr-4 py-2.5 rounded-[4px]"
            placeholder="Search IFCT 2017 database… (e.g. Rice, Banana, Chicken)"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setShowSuggestions(true)
            }}
            onFocus={() => setShowSuggestions(true)}
            onKeyDown={handleKeyDown}
          />
          {loadingSuggestions && (
            <svg className="absolute right-3 animate-spin w-4 h-4 text-accent" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
        </div>

        {/* Autocomplete dropdown */}
        {showSuggestions && suggestions.length > 0 && (
          <div className="absolute z-20 top-full left-0 right-0 mt-1 bg-bg-card border border-[#303236] rounded-[4px] shadow-xl overflow-hidden">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => handleSelect(s)}
                className="w-full text-left px-4 py-2.5 text-[13px] text-text-secondary hover:bg-[#151617] hover:text-text-primary transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Selected ingredients */}
      {selectedIngredients.length > 0 && (
        <div className="mb-5">
          <div className="section-label mb-2">
            Selected ({selectedIngredients.length})
          </div>
          <div className="flex flex-wrap gap-2">
            {selectedIngredients.map((ing) => (
              <span
                key={ing}
                className="chip chip-accent pr-1.5 flex items-center gap-1.5"
              >
                {ing}
                <button
                  onClick={() => removeIngredient(ing)}
                  className="w-4 h-4 rounded-full hover:bg-accent/20 flex items-center justify-center text-accent transition-colors"
                  aria-label={`Remove ${ing}`}
                >
                  <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </span>
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

      {/* Recommend button */}
      <button
        onClick={handleRecommend}
        disabled={loading || selectedIngredients.length === 0}
        className={[
          'w-full py-3 rounded-[4px] font-semibold text-[14px] transition-all duration-200 mb-8',
          loading || selectedIngredients.length === 0
            ? 'bg-accent/30 text-white/40 cursor-not-allowed'
            : 'bg-accent text-[#000000] hover:bg-accent-hover glow-accent hover:scale-[1.005]',
        ].join(' ')}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Computing Portions…
          </span>
        ) : (
          'Get Portion Recommendations'
        )}
      </button>

      {/* Results */}
      {recommendResult && (
        <div className="space-y-6 animate-fade-in">
          {/* Daily Budget Breakdown */}
          <div className="card">
            <div className="section-label mb-4">Daily Nutrient Budget & Consumption</div>
            
            {/* Value grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 mb-6">
              {[
                { label: 'Sodium', value: recommendResult.daily_budget.sodium_mg, unit: 'mg', color: '#8a8f98' },
                { label: 'Potassium', value: recommendResult.daily_budget.potassium_mg, unit: 'mg', color: '#8a8f98' },
                { label: 'Protein', value: recommendResult.daily_budget.protein_g, unit: 'g', color: '#8a8f98' },
                { label: 'Carbs', value: recommendResult.daily_budget.carbs_g, unit: 'g', color: '#8a8f98' },
                { label: 'Phosphorus', value: recommendResult.daily_budget.phosphorus_mg, unit: 'mg', color: '#8a8f98' },
              ].map(({ label, value, unit }) => (
                <div key={label} className="text-center p-3 rounded-[4px] bg-[#0d0d0d] border border-[#303236]">
                  <div className="text-[18px] font-bold text-accent">{value.toLocaleString()}</div>
                  <div className="text-[10px] text-text-muted mt-0.5">{unit}/day limit</div>
                  <div className="text-[11px] text-text-secondary mt-0.5">{label}</div>
                </div>
              ))}
            </div>

            {/* Consumption Chart */}
            <div className="mt-4">
              <h3 className="text-[13px] font-bold text-text-primary mb-2">Total Portion Load vs Budget</h3>
              <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      {
                        name: 'Sodium (mg)',
                        budget: recommendResult.daily_budget.sodium_mg,
                        consumed: recommendResult.recommendations.reduce((acc, r) => acc + (r.nutrient_load?.sodium_mg || 0), 0),
                      },
                      {
                        name: 'Potassium (mg)',
                        budget: recommendResult.daily_budget.potassium_mg,
                        consumed: recommendResult.recommendations.reduce((acc, r) => acc + (r.nutrient_load?.potassium_mg || 0), 0),
                      },
                      {
                        name: 'Protein (g)',
                        budget: recommendResult.daily_budget.protein_g,
                        consumed: recommendResult.recommendations.reduce((acc, r) => acc + (r.nutrient_load?.protein_g || 0), 0),
                      },
                      {
                        name: 'Carbs (g)',
                        budget: recommendResult.daily_budget.carbs_g,
                        consumed: recommendResult.recommendations.reduce((acc, r) => acc + (r.nutrient_load?.carbs_g || 0), 0),
                      },
                    ].map(d => ({ ...d, pct: Math.min(100, (d.consumed / (d.budget || 1)) * 100) }))}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: '#8a8f98', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" tick={{ fill: '#d1d5db', fontSize: 11 }} width={90} axisLine={false} tickLine={false} />
                    <Tooltip 
                      cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const d = payload[0].payload;
                          return (
                            <div className="bg-bg-card border border-[#303236] p-2 rounded shadow-lg text-[11px]">
                              <div className="font-bold text-text-primary mb-1">{d.name}</div>
                              <div className="text-accent">Consumed: {Math.round(d.consumed)}</div>
                              <div className="text-text-muted">Budget: {d.budget}</div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar dataKey="pct" radius={[0, 4, 4, 0]} maxBarSize={16}>
                      {
                        [0, 1, 2, 3].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill="#34d59a" fillOpacity={0.8} />
                        ))
                      }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div>
            <div className="section-label mb-4">
              Ingredient Recommendations ({recommendResult.recommendations.length})
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendResult.recommendations.map((rec) => (
                <RecommendationCard
                  key={rec.ingredient}
                  rec={rec}
                  budget={recommendResult.daily_budget}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

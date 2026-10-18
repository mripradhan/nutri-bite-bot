'use client'

import React, { useEffect, useState } from 'react'
import { useApp } from '@/context/AppContext'
import { apiAllergenGroups } from '@/lib/api'
import type { AllergenGroup } from '@/types'

// Used if the API is unreachable; the backend list is the source of truth.
const FALLBACK_GROUPS: AllergenGroup[] = [
  { id: 'dairy', label: 'Dairy / milk' },
  { id: 'egg', label: 'Egg' },
  { id: 'fish', label: 'Fish' },
  { id: 'shellfish', label: 'Shellfish / prawn' },
  { id: 'peanut', label: 'Peanut' },
  { id: 'tree_nuts', label: 'Tree nuts' },
  { id: 'sesame', label: 'Sesame' },
  { id: 'gluten', label: 'Gluten (wheat, barley, rye)' },
  { id: 'soy', label: 'Soy' },
  { id: 'no_meat_fish', label: 'Vegetarian (no meat, poultry or fish)' },
]

/** Allergy / intolerance / preference exclusions, enforced server-side as hard filters. */
export default function ExclusionSelector() {
  const { exclusions, setExclusions } = useApp()
  const [groups, setGroups] = useState<AllergenGroup[]>(FALLBACK_GROUPS)
  const [termInput, setTermInput] = useState('')

  useEffect(() => {
    apiAllergenGroups()
      .then((r) => r.groups?.length && setGroups(r.groups))
      .catch(() => {})
  }, [])

  const toggle = (id: string) =>
    setExclusions((prev) => ({
      ...prev,
      groups: prev.groups.includes(id) ? prev.groups.filter((g) => g !== id) : [...prev.groups, id],
    }))

  const addTerm = () => {
    const t = termInput.trim().toLowerCase()
    if (t.length >= 2 && !exclusions.terms.includes(t)) {
      setExclusions((prev) => ({ ...prev, terms: [...prev.terms, t] }))
    }
    setTermInput('')
  }

  return (
    <fieldset className="card">
      <legend className="section-label mb-4 text-accent">Allergies, intolerances &amp; preferences</legend>
      <p className="text-[11px] text-text-muted mb-3">
        Selected items are removed before portions, substitutes and recipes are computed, and any
        recipe mentioning them is rejected. Matching is keyword-based and not exhaustive: add
        free-text terms for anything not covered, and always confirm with a clinician.
      </p>
      <div className="flex flex-wrap gap-2 mb-4">
        {groups.map((g) => {
          const on = exclusions.groups.includes(g.id)
          return (
            <button
              key={g.id}
              type="button"
              onClick={() => toggle(g.id)}
              aria-pressed={on}
              className={[
                'px-3 py-1.5 rounded-full text-[12px] border transition-colors',
                on
                  ? 'bg-risk-high/15 border-risk-high/50 text-risk-high'
                  : 'bg-[#0d0d0d] border-[#303236] text-text-secondary hover:border-accent/40',
              ].join(' ')}
            >
              {g.label}
            </button>
          )
        })}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1"
          placeholder="Other ingredient to avoid (e.g. mustard)"
          value={termInput}
          onChange={(e) => setTermInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              addTerm()
            }
          }}
        />
        <button
          type="button"
          onClick={addTerm}
          className="px-4 rounded-full border border-[#303236] text-[12px] text-text-secondary hover:border-accent/40"
        >
          Add
        </button>
      </div>
      {exclusions.terms.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-3">
          {exclusions.terms.map((t) => (
            <span key={t} className="chip text-[11px]">
              {t}
              <button
                type="button"
                aria-label={`Remove ${t}`}
                className="ml-1.5 text-text-muted hover:text-risk-high"
                onClick={() =>
                  setExclusions((prev) => ({ ...prev, terms: prev.terms.filter((x) => x !== t) }))
                }
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </fieldset>
  )
}

'use client'

import React, { createContext, useContext, useState, useCallback } from 'react'
import type { PatientData, PredictionResult, RecommendResult } from '@/types'

// ─── Default patient (healthy preset) ────────────────────────────────────────
const DEFAULT_PATIENT: PatientData = {
  age: 35,
  sex_male: 0,
  has_htn: false,
  has_dm: false,
  has_ckd: false,
  serum_sodium: 140,
  serum_potassium: 4.0,
  creatinine: 0.9,
  egfr: 110,
  hba1c: 5.2,
  fbs: 88,
  sbp: 118,
  dbp: 75,
  bmi: 22,
}

// ─── Context shape ────────────────────────────────────────────────────────────
interface AppContextValue {
  patientData: PatientData
  setPatientData: React.Dispatch<React.SetStateAction<PatientData>>

  predictionResult: PredictionResult | null
  setPredictionResult: React.Dispatch<React.SetStateAction<PredictionResult | null>>

  selectedIngredients: string[]
  setSelectedIngredients: React.Dispatch<React.SetStateAction<string[]>>
  addIngredient: (name: string) => void
  removeIngredient: (name: string) => void

  recommendResult: RecommendResult | null
  setRecommendResult: React.Dispatch<React.SetStateAction<RecommendResult | null>>

  recipeOutput: string
  setRecipeOutput: React.Dispatch<React.SetStateAction<string>>

  detectedIngredients: string[]
  setDetectedIngredients: React.Dispatch<React.SetStateAction<string[]>>

  addDetectedToIngredients: () => void
}

const AppContext = createContext<AppContextValue | null>(null)

// ─── Provider ─────────────────────────────────────────────────────────────────
export function AppProvider({ children }: { children: React.ReactNode }) {
  const [patientData, setPatientData] = useState<PatientData>(DEFAULT_PATIENT)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [selectedIngredients, setSelectedIngredients] = useState<string[]>([])
  const [recommendResult, setRecommendResult] = useState<RecommendResult | null>(null)
  const [recipeOutput, setRecipeOutput] = useState<string>('')
  const [detectedIngredients, setDetectedIngredients] = useState<string[]>([])

  const addIngredient = useCallback((name: string) => {
    setSelectedIngredients((prev) =>
      prev.includes(name) ? prev : [...prev, name],
    )
  }, [])

  const removeIngredient = useCallback((name: string) => {
    setSelectedIngredients((prev) => prev.filter((i) => i !== name))
  }, [])

  const addDetectedToIngredients = useCallback(() => {
    setSelectedIngredients((prev) => {
      const next = [...prev]
      detectedIngredients.forEach((ing) => {
        if (!next.includes(ing)) next.push(ing)
      })
      return next
    })
  }, [detectedIngredients])

  return (
    <AppContext.Provider
      value={{
        patientData,
        setPatientData,
        predictionResult,
        setPredictionResult,
        selectedIngredients,
        setSelectedIngredients,
        addIngredient,
        removeIngredient,
        recommendResult,
        setRecommendResult,
        recipeOutput,
        setRecipeOutput,
        detectedIngredients,
        setDetectedIngredients,
        addDetectedToIngredients,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>')
  return ctx
}

'use client'

import React, { useState } from 'react'
import { useApp } from '@/context/AppContext'
import { apiPredict } from '@/lib/api'
import type { PatientData } from '@/types'

// ─── Presets ──────────────────────────────────────────────────────────────────
const PRESETS: Record<string, PatientData> = {
  healthy: {
    age: 35, sex_male: 0, has_htn: false, has_dm: false, has_ckd: false,
    serum_sodium: 140, serum_potassium: 4.0, creatinine: 0.9, egfr: 110,
    hba1c: 5.2, fbs: 88, sbp: 118, dbp: 75, bmi: 22,
  },
  ckd: {
    age: 62, sex_male: 1, has_htn: false, has_dm: false, has_ckd: true,
    serum_sodium: 138, serum_potassium: 5.8, creatinine: 3.5, egfr: 22,
    hba1c: 5.5, fbs: 95, sbp: 135, dbp: 82, bmi: 26,
  },
  diabetes: {
    age: 48, sex_male: 1, has_htn: false, has_dm: true, has_ckd: false,
    serum_sodium: 140, serum_potassium: 4.0, creatinine: 1.0, egfr: 92,
    hba1c: 9.2, fbs: 210, sbp: 125, dbp: 80, bmi: 32,
  },
  htn: {
    age: 58, sex_male: 0, has_htn: true, has_dm: false, has_ckd: false,
    serum_sodium: 143, serum_potassium: 3.8, creatinine: 1.1, egfr: 78,
    hba1c: 5.6, fbs: 102, sbp: 165, dbp: 98, bmi: 29,
  },
  multi: {
    age: 65, sex_male: 1, has_htn: true, has_dm: true, has_ckd: true,
    serum_sodium: 145, serum_potassium: 5.9, creatinine: 4.0, egfr: 18,
    hba1c: 8.8, fbs: 190, sbp: 170, dbp: 100, bmi: 34,
  },
}

const PRESET_LABELS: Array<{ key: string; label: string; color: string }> = [
  { key: 'healthy', label: 'Healthy', color: 'text-risk-low border-risk-low/30' },
  { key: 'ckd', label: 'CKD', color: 'text-risk-moderate border-risk-moderate/30' },
  { key: 'diabetes', label: 'Diabetes', color: 'text-risk-moderate border-risk-moderate/30' },
  { key: 'htn', label: 'HTN', color: 'text-risk-moderate border-risk-moderate/30' },
  { key: 'multi', label: 'CKD+HTN+DM', color: 'text-risk-high border-risk-high/30' },
]

// ─── Toggle component ─────────────────────────────────────────────────────────
function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={[
        'relative inline-flex items-center h-6 rounded-full w-11 transition-colors duration-200 focus:outline-none',
        checked ? 'bg-accent' : 'bg-[#303236]',
      ].join(' ')}
      aria-label={label}
      aria-checked={checked}
      role="switch"
    >
      <span
        className={[
          'inline-block w-4 h-4 transform bg-white rounded-full shadow-sm transition-transform duration-200',
          checked ? 'translate-x-6' : 'translate-x-1',
        ].join(' ')}
      />
    </button>
  )
}

// ─── Field wrapper ─────────────────────────────────────────────────────────────
function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className="block text-[12px] font-medium text-text-secondary mb-1.5">
        {label}
        {hint && <span className="ml-1 text-text-muted font-normal">{hint}</span>}
      </label>
      {children}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function PatientForm() {
  const { patientData, setPatientData, setPredictionResult } = useApp()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const update = <K extends keyof PatientData>(key: K, val: PatientData[K]) => {
    setPatientData((prev) => ({ ...prev, [key]: val }))
  }

  const applyPreset = (key: string) => {
    const p = PRESETS[key]
    if (p) setPatientData(p)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const result = await apiPredict(patientData)
      setPredictionResult(result)
      // Scroll to results
      setTimeout(() => {
        document.getElementById('risk-analysis')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        })
      }, 100)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="patient-data" className="px-6 md:px-8 py-10">
      <div className="section-label mb-2">Clinical Input</div>
      <h2 className="text-[20px] font-bold text-text-primary mb-1">Patient Data</h2>
      <p className="text-text-secondary text-[13px] mb-6">
        Enter patient lab values to run ML-based nutrient risk stratification.
      </p>

      {/* Preset bar */}
      <div className="flex flex-wrap gap-2 mb-6">
        <span className="text-[11px] text-text-muted self-center mr-1">Presets:</span>
        {PRESET_LABELS.map(({ key, label, color }) => (
          <button
            key={key}
            type="button"
            onClick={() => applyPreset(key)}
            className={[
              'px-3 py-1.5 rounded-full text-[12px] font-medium border bg-transparent transition-all duration-150 hover:bg-[#151617]',
              color,
            ].join(' ')}
          >
            {label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Demographics */}
        <fieldset className="card">
          <legend className="section-label mb-4 text-accent">Demographics</legend>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <Field label="Age" hint="years">
              <input
                type="number"
                className="w-full"
                min={18}
                max={120}
                step={1}
                value={patientData.age}
                onChange={(e) => update('age', Number(e.target.value))}
              />
            </Field>
            <Field label="Sex">
              <select
                className="w-full"
                value={patientData.sex_male}
                onChange={(e) => update('sex_male', Number(e.target.value))}
              >
                <option value={0}>Female</option>
                <option value={1}>Male</option>
              </select>
            </Field>
            <Field label="BMI" hint="kg/m²">
              <input
                type="number"
                className="w-full"
                min={10}
                max={60}
                step={0.1}
                value={patientData.bmi}
                onChange={(e) => update('bmi', Number(e.target.value))}
              />
            </Field>
          </div>
        </fieldset>

        {/* Conditions */}
        <fieldset className="card">
          <legend className="section-label mb-4 text-accent">Conditions</legend>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
            {[
              { key: 'has_htn' as const, label: 'Hypertension', abbr: 'HTN' },
              { key: 'has_dm' as const, label: 'Type 2 Diabetes', abbr: 'T2DM' },
              { key: 'has_ckd' as const, label: 'Chronic Kidney Disease', abbr: 'CKD' },
            ].map(({ key, label, abbr }) => (
              <div key={key} className="flex items-center justify-between p-3 rounded-[4px] bg-[#0d0d0d] border border-[#303236]">
                <div>
                  <div className="text-[13px] font-medium text-text-primary">{abbr}</div>
                  <div className="text-[11px] text-text-muted">{label}</div>
                </div>
                <Toggle
                  checked={patientData[key]}
                  onChange={(v) => update(key, v)}
                  label={label}
                />
              </div>
            ))}
          </div>
        </fieldset>

        {/* Lab Values */}
        <fieldset className="card">
          <legend className="section-label mb-4 text-accent">Lab Values</legend>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <Field label="Serum Sodium" hint="mEq/L">
              <input type="number" className="w-full" step={0.1} min={100} max={180}
                value={patientData.serum_sodium}
                onChange={(e) => update('serum_sodium', Number(e.target.value))} />
            </Field>
            <Field label="Serum Potassium" hint="mEq/L">
              <input type="number" className="w-full" step={0.1} min={1} max={10}
                value={patientData.serum_potassium}
                onChange={(e) => update('serum_potassium', Number(e.target.value))} />
            </Field>
            <Field label="Creatinine" hint="mg/dL">
              <input type="number" className="w-full" step={0.1} min={0.1} max={20}
                value={patientData.creatinine}
                onChange={(e) => update('creatinine', Number(e.target.value))} />
            </Field>
            <Field label="eGFR" hint="mL/min/1.73m²">
              <input type="number" className="w-full" step={1} min={1} max={200}
                value={patientData.egfr}
                onChange={(e) => update('egfr', Number(e.target.value))} />
            </Field>
            <Field label="HbA1c" hint="%">
              <input type="number" className="w-full" step={0.1} min={3} max={20}
                value={patientData.hba1c}
                onChange={(e) => update('hba1c', Number(e.target.value))} />
            </Field>
            <Field label="Fasting Blood Sugar" hint="mg/dL">
              <input type="number" className="w-full" step={1} min={40} max={600}
                value={patientData.fbs}
                onChange={(e) => update('fbs', Number(e.target.value))} />
            </Field>
          </div>
        </fieldset>

        {/* Vitals */}
        <fieldset className="card">
          <legend className="section-label mb-4 text-accent">Vitals</legend>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Systolic BP" hint="mmHg">
              <input type="number" className="w-full" step={1} min={60} max={250}
                value={patientData.sbp}
                onChange={(e) => update('sbp', Number(e.target.value))} />
            </Field>
            <Field label="Diastolic BP" hint="mmHg">
              <input type="number" className="w-full" step={1} min={40} max={160}
                value={patientData.dbp}
                onChange={(e) => update('dbp', Number(e.target.value))} />
            </Field>
          </div>
        </fieldset>

        {/* Error */}
        {error && (
          <div className="p-3 rounded-lg bg-risk-high/10 border border-risk-high/30 text-risk-high text-[13px]">
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={loading}
          className={[
            'w-full py-3.5 rounded-full font-semibold text-[14px] transition-all duration-200',
            loading
              ? 'bg-accent/40 text-black/60 cursor-not-allowed'
              : 'bg-accent text-[#000000] hover:bg-accent-hover glow-accent hover:scale-[1.01]',
          ].join(' ')}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running ML Prediction…
            </span>
          ) : (
            'Run ML Prediction'
          )}
        </button>
      </form>
    </section>
  )
}

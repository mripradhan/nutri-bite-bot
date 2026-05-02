'use client'

import React, { useMemo } from 'react'
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
} from 'recharts'
import { useApp } from '@/context/AppContext'
import type { RiskLabel, TargetName } from '@/types'

// ─── Helpers ──────────────────────────────────────────────────────────────────
const RISK_COLORS: Record<RiskLabel, string> = {
  low: '#34d399',
  moderate: '#fbbf24',
  high: '#f87171',
}

const RISK_BG: Record<RiskLabel, string> = {
  low: 'rgba(52,211,153,0.08)',
  moderate: 'rgba(251,191,36,0.08)',
  high: 'rgba(248,113,113,0.08)',
}

const TARGET_LABELS: Record<TargetName, string> = {
  sodium_sensitivity: 'Sodium',
  potassium_sensitivity: 'Potassium',
  protein_restriction: 'Protein',
  carb_sensitivity: 'Carbs',
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function RiskBadge({ label }: { label: RiskLabel }) {
  return (
    <span
      className={[
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold',
        `badge-${label}`,
      ].join(' ')}
    >
      {capitalize(label)}
    </span>
  )
}

// ─── Custom Radar dot ─────────────────────────────────────────────────────────
function GlowDot({ cx = 0, cy = 0, color = '#34d59a' }: { cx?: number; cy?: number; color?: string; [key: string]: unknown }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={5} fill={color} opacity={0.9} />
      <circle cx={cx} cy={cy} r={9} fill="none" stroke={color} strokeWidth={1} opacity={0.4} />
    </g>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function RiskDashboard() {
  const { predictionResult, patientData } = useApp()

  const radarData = useMemo(() => {
    if (!predictionResult) return []
    return (Object.entries(predictionResult.risk_levels) as [TargetName, (typeof predictionResult.risk_levels)[TargetName]][]).map(
      ([target, info]) => {
        const p = info.probabilities
        const score = p.low * 0 + p.moderate * 1 + p.high * 2
        return {
          subject: TARGET_LABELS[target],
          score: Math.round(score * 100) / 100,
          fullMark: 2,
        }
      },
    )
  }, [predictionResult])

  const maxScore = useMemo(() => {
    if (!radarData.length) return 0
    return Math.max(...radarData.map(d => d.score))
  }, [radarData])

  const radarColor = maxScore >= 1.5 ? '#f87171' : maxScore >= 0.5 ? '#fbbf24' : '#34d59a'

  if (!predictionResult) return null

  const { risk_levels, patient_summary } = predictionResult
  const riskEntries = Object.entries(risk_levels) as [TargetName, (typeof risk_levels)[TargetName]][]

  return (
    <section id="risk-analysis" className="px-6 md:px-8 py-10 space-y-8 animate-fade-in">
      <div>
        <div className="section-label mb-2">Deep Learning Results</div>
        <h2 className="text-[20px] font-bold text-text-primary">Risk Analysis</h2>
      </div>

      {/* Patient summary strip */}
      <div className="card">
        <div className="section-label mb-3">Patient Summary</div>
        <div className="flex flex-wrap gap-2">
          <span className="chip">
            Age {patient_summary.age} · {patient_summary.sex}
          </span>
          {Object.entries(patient_summary.conditions)
            .filter(([, v]) => v)
            .map(([cond]) => (
              <span key={cond} className="chip chip-accent">
                {cond}
              </span>
            ))}
          {Object.entries(patient_summary.key_labs).map(([k, v]) => (
            <span key={k} className="chip">
              <span className="text-text-muted mr-1">{k}:</span>
              <span className="text-text-primary font-medium">{String(v)}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Main grid: radar + risk cards */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Radar chart */}
        <div className="card flex flex-col">
          <div className="section-label mb-4">Nutrient Risk Radar</div>
          <div className="flex-1 min-h-[300px] radar-glow">
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                <defs>
                  <radialGradient id="radarFill" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor={radarColor} stopOpacity={0.30} />
                    <stop offset="100%" stopColor={radarColor} stopOpacity={0.04} />
                  </radialGradient>
                  <filter id="radarGlow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                    <feMerge>
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <PolarGrid
                  gridType="polygon"
                  stroke="#303236"
                  strokeWidth={1}
                />
                <PolarAngleAxis
                  dataKey="subject"
                  tick={{ fill: '#8a8f98', fontSize: 12, fontFamily: 'Inter' }}
                  tickLine={false}
                />
                <Radar
                  name="Severity Score"
                  dataKey="score"
                  stroke={radarColor}
                  strokeWidth={2}
                  fill="url(#radarFill)"
                  filter="url(#radarGlow)"
                  dot={<GlowDot color={radarColor} />}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center justify-between text-[11px] text-text-muted pt-2 border-t border-[#303236]">
            <span>Score 0 = Low risk</span>
            <span className="text-accent">Score 2 = High risk</span>
          </div>
        </div>

        {/* Risk cards grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {riskEntries.map(([target, info]) => {
          const score =
            info.probabilities.low * 0 +
            info.probabilities.moderate * 1 +
            info.probabilities.high * 2
          return (
            <div
              key={target}
              className="relative overflow-hidden rounded-[4px] border p-4 flex flex-col gap-3 transition-all duration-200 hover:scale-[1.01]"
              style={{
                backgroundColor: RISK_BG[info.label],
                borderColor: RISK_COLORS[info.label] + '40',
              }}
            >
              {/* Left accent bar */}
              <div
                className="absolute left-0 top-0 bottom-0 w-[3px]"
                style={{ backgroundColor: RISK_COLORS[info.label] }}
              />

              <div className="flex items-start justify-between gap-2 pl-2">
                <div>
                  <div className="text-[11px] text-text-muted mb-0.5">
                    {TARGET_LABELS[target]}
                  </div>
                  <div className="text-[15px] font-bold text-text-primary leading-tight">
                    {info.display_name}
                  </div>
                </div>
                <RiskBadge label={info.label} />
              </div>

              <div className="pl-2 space-y-2">
                {/* Confidence bar */}
                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-text-muted">Confidence</span>
                    <span className="font-semibold" style={{ color: RISK_COLORS[info.label] }}>
                      {info.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-[#303236] overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${info.confidence}%`,
                        backgroundColor: RISK_COLORS[info.label],
                      }}
                    />
                  </div>
                </div>

                {/* Severity score */}
                <div className="flex justify-between text-[11px]">
                  <span className="text-text-muted">Severity Score</span>
                  <span className="font-mono font-semibold text-text-primary">
                    {score.toFixed(3)}
                  </span>
                </div>

                {/* Clinical note */}
                <p className="text-[11px] text-text-secondary leading-relaxed">
                  {info.clinical_note}
                </p>
              </div>
            </div>
          )
        })}
      </div>
      </div>
    </section>
  )
}

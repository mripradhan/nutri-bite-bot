'use client'

import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { useApp } from '@/context/AppContext'
import type { TargetName } from '@/types'

const TARGET_LABELS: Record<TargetName, string> = {
  sodium_sensitivity: 'Sodium Sensitivity',
  potassium_sensitivity: 'Potassium Sensitivity',
  protein_restriction: 'Protein Restriction',
  carb_sensitivity: 'Carb Sensitivity',
}

const CLASS_COLORS = {
  Low: '#34d399',
  Moderate: '#fbbf24',
  High: '#f87171',
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ value: number; name: string }>
  label?: string
}

function CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-bg-card border border-[#303236] rounded-lg px-3 py-2 text-[12px] shadow-lg">
      <p className="font-semibold text-text-primary mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: CLASS_COLORS[p.name as keyof typeof CLASS_COLORS] ?? '#e8ebf0' }}>
          {p.value.toFixed(1)}%
        </p>
      ))}
    </div>
  )
}

function ProbabilityChart({
  targetName,
  probabilities,
}: {
  targetName: TargetName
  probabilities: { low: number; moderate: number; high: number }
}) {
  const data = [
    { class: 'Low', value: probabilities.low, color: '#34d399' },
    { class: 'Moderate', value: probabilities.moderate, color: '#fbbf24' },
    { class: 'High', value: probabilities.high, color: '#f87171' },
  ]

  return (
    <div className="card">
      <div className="text-[12px] font-semibold text-text-secondary mb-3">
        {TARGET_LABELS[targetName]}
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 40, bottom: 0, left: 0 }}
          barCategoryGap="25%"
        >
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: '#4b5563', fontSize: 10, fontFamily: 'Inter' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="class"
            tick={{ fill: '#8a8f98', fontSize: 11, fontFamily: 'Inter' }}
            tickLine={false}
            axisLine={false}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.color} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Value labels */}
      <div className="flex gap-3 mt-2 flex-wrap">
        {data.map(({ class: cls, value, color }) => (
          <div key={cls} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-[11px] text-text-muted">{cls}</span>
            <span className="text-[11px] font-semibold" style={{ color }}>
              {value.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ProbabilityCharts() {
  const { predictionResult } = useApp()

  if (!predictionResult) return null

  const { risk_levels } = predictionResult
  const targets = Object.keys(risk_levels) as TargetName[]

  return (
    <section className="px-6 md:px-8 pb-10 animate-fade-in">
      <div className="section-label mb-2">Detailed Output</div>
      <h3 className="text-[17px] font-bold text-text-primary mb-5">
        Class Probability Distribution
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {targets.map((target) => (
          <ProbabilityChart
            key={target}
            targetName={target}
            probabilities={risk_levels[target].probabilities}
          />
        ))}
      </div>
    </section>
  )
}

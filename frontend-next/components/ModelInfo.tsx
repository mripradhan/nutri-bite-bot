'use client'

import React, { useState, useEffect } from 'react'
import { apiModelInfo } from '@/lib/api'
import type { ModelInfo as ModelInfoType, TargetName } from '@/types'

const fmt = (v: number | null | undefined) => (typeof v === 'number' ? v.toFixed(3) : 'n/a')

export default function ModelInfo() {
  const [info, setInfo] = useState<ModelInfoType | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open || info) return
    setLoading(true)
    setError(null)
    apiModelInfo()
      .then(setInfo)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load'))
      .finally(() => setLoading(false))
  }, [open, info])

  return (
    <section id="model-info" className="px-6 md:px-8 py-10">
      <div className="section-label mb-2">Architecture</div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between p-4 rounded-[4px] bg-bg-card border border-[#303236] hover:border-accent/30 transition-all group"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[4px] bg-accent/10 flex items-center justify-center">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34d59a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div className="text-left">
            <div className="text-[14px] font-semibold text-text-primary">Model Information</div>
            <div className="text-[12px] text-text-muted">TabNet architecture, held-out performance, feature names</div>
          </div>
        </div>
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#8a8f98"
          strokeWidth="2"
          strokeLinecap="round"
          className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {open && (
        <div className="mt-3 card animate-slide-up space-y-6">
          {loading && (
            <div className="flex items-center gap-3 text-text-secondary text-[13px]">
              <svg className="animate-spin w-4 h-4 text-accent" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading model metadata…
            </div>
          )}

          {error && (
            <div className="p-3 rounded-[4px] bg-risk-high/10 border border-risk-high/30 text-risk-high text-[13px]">
              {error}
            </div>
          )}

          {info && (
            <>
              {/* Accuracy metrics */}
              <div>
                <div className="section-label mb-3">Performance Metrics</div>
                <p className="text-[11px] text-text-muted mb-3">
                  Held-out real MIMIC-IV patients (retrospective). Performance is modest in
                  absolute terms: better than a simple guideline-rule baseline, but not shown
                  to be clinically adequate. Targets are outcome-risk proxies, not measured
                  dietary responses.
                </p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    {
                      label: 'Mean macro-F1',
                      value: fmt(info.accuracy_metrics.mean_f1_macro),
                    },
                    {
                      label: 'Mean macro-AUROC',
                      value: fmt(info.accuracy_metrics.mean_auroc_ovr),
                    },
                    {
                      label: 'Mean balanced accuracy',
                      value: fmt(info.accuracy_metrics.mean_balanced_accuracy),
                    },
                  ].map(({ label, value }) => (
                    <div
                      key={label}
                      className="text-center p-3 rounded-[4px] bg-[#0d0d0d] border border-[#303236]"
                    >
                      <div className="text-[20px] font-extrabold text-accent leading-none">
                        {value}
                      </div>
                      <div className="text-[10px] text-text-muted mt-1">{label}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Backend info */}
              <div>
                <div className="section-label mb-3">Backend</div>
                <div className="flex flex-wrap gap-2">
                  <span className="chip chip-accent">{info.model_backend}</span>
                  {info.preprocessing.map((p) => (
                    <span key={p} className="chip">{p}</span>
                  ))}
                  <span className="chip">{info.feature_count} features</span>
                </div>
              </div>

              {/* Models */}
              <div>
                <div className="section-label mb-3">Outcome-risk proxy targets</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {(Object.entries(info.models) as [TargetName, (typeof info.models)[TargetName]][]).map(
                    ([key, model]) => (
                      <div
                        key={key}
                        className="p-3 rounded-[4px] bg-[#0d0d0d] border border-[#303236]"
                      >
                        <div className="text-[13px] font-semibold text-text-primary mb-1">
                          {model.display_name}
                        </div>
                        <div className="flex gap-1 mb-2">
                          {model.classes.map((cls) => (
                            <span
                              key={cls}
                              className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium badge-${cls === 'low' ? 'low' : cls === 'moderate' ? 'moderate' : 'high'}`}
                            >
                              {cls}
                            </span>
                          ))}
                        </div>
                        <div className="text-[10px] text-text-muted">{model.type}</div>
                      </div>
                    ),
                  )}
                </div>
              </div>

              {/* Feature names */}
              <div>
                <div className="section-label mb-3">Feature Names ({info.feature_count})</div>
                <div className="flex flex-wrap gap-1.5">
                  {info.feature_names.map((f) => (
                    <span key={f} className="chip text-[11px] font-mono">
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}

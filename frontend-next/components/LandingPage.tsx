'use client'

import React, { useEffect, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { TextEffect } from '@/components/ui/text-effect'

const ShaderAnimation = dynamic(
  () => import('@/components/ui/shader-lines').then((mod) => ({ default: mod.ShaderAnimation })),
  { ssr: false, loading: () => <div className="w-full h-full bg-black" /> }
)

const STATS = [
  { value: '99.91%', label: 'Mean Accuracy' },
  { value: 'κ 0.9983', label: "Cohen's Kappa" },
  { value: '55k+', label: 'MIMIC-IV Episodes' },
  { value: '14', label: 'EHR Features' },
  { value: 'IFCT 2017', label: 'Food Database' },
]

const TECH_CHIPS = ['TabNet Classifier', 'KDIGO 2024', 'ADA 2024', 'AHA / ACC', 'MIMIC-IV EHR']

function IconArrowRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M12 5l7 7-7 7" />
    </svg>
  )
}

function IconGithub({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
    </svg>
  )
}

function IconStar() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 .587l3.668 7.568 8.332 1.151-6.064 5.828 1.48 8.279L12 19.771l-7.416 3.642 1.48-8.279L0 9.306l8.332-1.151z" />
    </svg>
  )
}

function IconActivity() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  )
}

type LandingPageProps = {
  onEnter: () => void
}

export default function LandingPage({ onEnter }: LandingPageProps) {
  const stableOnEnter = useCallback(onEnter, [onEnter])

  useEffect(() => {
    let triggered = false
    const trigger = () => {
      if (triggered) return
      triggered = true
      stableOnEnter()
    }

    const handleWheel = (e: WheelEvent) => { if (e.deltaY > 40) trigger() }
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown' || e.key === ' ' || e.key === 'Enter') trigger()
    }
    let touchStartY = 0
    const handleTouchStart = (e: TouchEvent) => { touchStartY = e.touches[0].clientY }
    const handleTouchEnd = (e: TouchEvent) => {
      if (touchStartY - e.changedTouches[0].clientY > 60) trigger()
    }

    window.addEventListener('wheel', handleWheel, { passive: true })
    window.addEventListener('keydown', handleKey)
    window.addEventListener('touchstart', handleTouchStart, { passive: true })
    window.addEventListener('touchend', handleTouchEnd, { passive: true })

    return () => {
      window.removeEventListener('wheel', handleWheel)
      window.removeEventListener('keydown', handleKey)
      window.removeEventListener('touchstart', handleTouchStart)
      window.removeEventListener('touchend', handleTouchEnd)
    }
  }, [stableOnEnter])

  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden bg-black w-full select-none">
      {/* ── Background shader ─────────────────────────────────────────── */}
      <div className="absolute inset-0 z-0">
        <ShaderAnimation />
      </div>

      {/* Semitransparent black overlay to dim the shader */}
      <div className="absolute inset-0 z-[0] bg-black/50 pointer-events-none" />

      {/* Gradient overlays */}
      <div className="absolute inset-0 z-[1] bg-gradient-to-t from-black/80 via-black/20 to-black/50 pointer-events-none" />
      <div className="absolute inset-0 z-[1] bg-gradient-to-r from-black/40 via-transparent to-black/40 pointer-events-none" />
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 h-80 w-80 rounded-full bg-[#34d59a]/12 blur-[140px] z-[1] pointer-events-none" />

      {/* ── Navbar ────────────────────────────────────────────────────── */}
      <nav className="relative z-20 w-full">
        <div className="absolute inset-0 bg-black/30 backdrop-blur-md border-b border-[#34d59a]/10" />
        <div className="relative max-w-7xl mx-auto px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-[4px] bg-[#34d59a] flex items-center justify-center flex-shrink-0 shadow-[0_0_20px_rgba(52,213,154,0.4)] text-black">
                <IconActivity />
              </div>
              <span className="text-[15px] font-semibold text-white tracking-tight">
                NutriBiteBot
              </span>
            </div>

            {/* Right actions */}
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/mripradhan/nutri-bite-bot"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:flex items-center gap-2.5 px-5 py-2.5 rounded-full border border-white/20 bg-white/5 backdrop-blur-sm text-white hover:bg-white/10 hover:border-white/30 transition-all duration-200 text-[13px] font-semibold shadow-lg"
              >
                <IconGithub size={18} />
                <span>Star on GitHub</span>
                <IconStar />
              </a>
              <button
                onClick={onEnter}
                className="flex items-center gap-2 px-5 py-2 rounded-full bg-[#34d59a] text-black font-semibold text-[13px] hover:bg-[#3de8aa] transition-colors shadow-[0_0_16px_rgba(52,213,154,0.35)]"
              >
                Begin Analysis
                <IconArrowRight />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* ── Hero content ──────────────────────────────────────────────── */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center text-center px-6 pt-12 pb-24">
        <div className="max-w-3xl mx-auto">

          {/* Research badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#0d0d0d]/70 backdrop-blur-sm border border-[#34d59a]/25 mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-[#34d59a] animate-pulse" />
            <span
              className="text-[11px] font-semibold tracking-[0.18em] uppercase text-[#34d59a]"
              style={{ fontFamily: "'Geist Mono', ui-monospace, monospace" }}
            >
              Research · RVCE Bengaluru · 2024
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-[42px] sm:text-[58px] md:text-[68px] font-extrabold text-white leading-[1.04] tracking-tight mb-6">
            <TextEffect
              per="word"
              preset="slide"
              delay={0.1}
              as="span"
              className="block"
            >
              Clinically-safe dietary portions,
            </TextEffect>
            <TextEffect
              per="word"
              preset="slide"
              delay={0.45}
              as="span"
              className="block text-[#34d59a] [text-shadow:0_0_48px_rgba(52,213,154,0.4),0_0_96px_rgba(52,213,154,0.18)]"
            >
              powered by ML.
            </TextEffect>
          </h1>

          {/* Subheading */}
          <div className="mb-10">
            <TextEffect
              per="word"
              preset="fade"
              delay={0.8}
              as="p"
              className="text-[16px] sm:text-[18px] text-[#8a8f98] leading-relaxed max-w-2xl mx-auto"
            >
              NutriBiteBot analyses 14 lab values from your EHR and computes exact gram limits for every ingredient — keeping every meal within clinically validated bounds for HTN, T2DM, and CKD.
            </TextEffect>
          </div>

          {/* CTA buttons */}
          <div className="flex flex-col sm:flex-row gap-3 justify-center mb-14">
            <button
              onClick={onEnter}
              className="inline-flex items-center justify-center gap-2.5 px-8 py-3.5 rounded-full bg-[#34d59a] text-black font-semibold text-[15px] hover:bg-[#3de8aa] transition-all duration-200 shadow-[0_0_24px_rgba(52,213,154,0.40)] hover:shadow-[0_0_32px_rgba(52,213,154,0.55)]"
            >
              Begin Analysis
              <IconArrowRight />
            </button>
            <a
              href="https://github.com/mripradhan/nutri-bite-bot"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-3 px-8 py-3.5 rounded-full border border-white/20 bg-white/5 backdrop-blur-sm text-white hover:bg-white/10 hover:border-white/30 transition-all duration-200 text-[15px] font-semibold shadow-lg"
            >
              <IconGithub size={20} />
              Star on GitHub
              <IconStar />
            </a>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-4 sm:gap-6 max-w-2xl mx-auto">
            {STATS.map(({ value, label }) => (
              <div key={label} className="flex flex-col items-center gap-1.5">
                <div
                  className="text-[18px] sm:text-[22px] font-bold text-[#34d59a] leading-none"
                  style={{ fontFamily: "'Geist Mono', ui-monospace, monospace" }}
                >
                  {value}
                </div>
                <div className="text-[10px] text-[#4b5563] tracking-wide text-center leading-snug">
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Tech chips strip ──────────────────────────────────────────── */}
      <div className="relative z-10 border-t border-[#1a1a1a] bg-black/60 backdrop-blur-sm px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center gap-2 justify-center">
          {TECH_CHIPS.map((chip) => (
            <span
              key={chip}
              className="inline-flex items-center px-3 py-1 rounded-full text-[10px] font-medium border border-[#303236] bg-[#0d0d0d]/80 text-[#4b5563] tracking-wide"
            >
              {chip}
            </span>
          ))}
        </div>
      </div>

      {/* ── Scroll hint ───────────────────────────────────────────────── */}
      <div className="absolute bottom-14 sm:bottom-16 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2 pointer-events-none">
        <span
          className="text-[10px] tracking-[0.2em] uppercase text-[#303236]"
          style={{ fontFamily: "'Geist Mono', ui-monospace, monospace" }}
        >
          scroll to enter
        </span>
        <div className="w-px h-8 bg-gradient-to-b from-[#303236] to-transparent" />
      </div>
    </section>
  )
}

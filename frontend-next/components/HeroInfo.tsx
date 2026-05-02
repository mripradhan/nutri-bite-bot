'use client'

import React from 'react'

const STATS = [
  { value: '99.91%', label: 'Mean Accuracy' },
  { value: 'κ 0.9983', label: "Cohen's Kappa" },
  { value: '55k+', label: 'MIMIC-IV Episodes' },
  { value: '14', label: 'EHR Features' },
  { value: 'IFCT 2017', label: 'Food Database' },
]

const TECH_CHIPS = [
  'TabNet Classifier',
  'KDIGO 2024',
  'ADA 2024',
  'AHA/ACC',
  'MIMIC-IV EHR',
  '4 Nutrient Targets',
]

function AbstractScanlines() {
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      xmlns="http://www.w3.org/2000/svg"
      preserveAspectRatio="none"
    >
      <defs>
        <pattern id="scan-h" x="0" y="0" width="100%" height="20" patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="100%" y2="0" stroke="#34d59a" strokeWidth="1" strokeOpacity="0.035" />
        </pattern>
        <pattern id="scan-dots" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="1" fill="#34d59a" fillOpacity="0.07" />
        </pattern>
        <radialGradient id="hero-glow-center" cx="50%" cy="35%" r="55%">
          <stop offset="0%" stopColor="#34d59a" stopOpacity="0.13" />
          <stop offset="60%" stopColor="#34d59a" stopOpacity="0.03" />
          <stop offset="100%" stopColor="#34d59a" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="scan-fade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#000000" stopOpacity="0" />
          <stop offset="40%" stopColor="#000000" stopOpacity="0" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0.8" />
        </linearGradient>
      </defs>
      {/* Horizontal scan lines */}
      <rect width="100%" height="100%" fill="url(#scan-h)" />
      {/* Dot grid */}
      <rect width="100%" height="100%" fill="url(#scan-dots)" />
      {/* Central radial glow */}
      <rect width="100%" height="100%" fill="url(#hero-glow-center)" />
      {/* Diagonal accent slash lines */}
      <line x1="0" y1="30%" x2="20%" y2="0" stroke="#34d59a" strokeWidth="0.5" strokeOpacity="0.12" />
      <line x1="0" y1="70%" x2="35%" y2="0" stroke="#34d59a" strokeWidth="0.5" strokeOpacity="0.06" />
      <line x1="100%" y1="20%" x2="75%" y2="100%" stroke="#34d59a" strokeWidth="0.5" strokeOpacity="0.09" />
      {/* Bottom fade */}
      <rect width="100%" height="100%" fill="url(#scan-fade)" />
    </svg>
  )
}

export default function HeroInfo() {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <section
      id="overview"
      className="relative min-h-[580px] flex flex-col justify-between overflow-hidden bg-[#000000]"
    >
      <AbstractScanlines />

      {/* Main content */}
      <div className="relative z-10 px-8 pt-16 pb-10 flex-1 flex flex-col justify-center">
        {/* Section label */}
        <div className="flex items-center gap-3 mb-8">
          <span className="inline-block w-6 h-px bg-accent opacity-70" />
          <span
            className="text-accent text-[10px] font-semibold tracking-[0.22em] uppercase"
            style={{ fontFamily: "'Geist Mono', ui-monospace, monospace" }}
          >
            Research · RVCE Bengaluru
          </span>
          <span className="inline-block w-6 h-px bg-accent opacity-70" />
        </div>

        {/* Headline */}
        <h1 className="text-[36px] sm:text-[50px] md:text-[60px] font-extrabold text-text-primary leading-[1.05] tracking-tight max-w-2xl mb-5">
          Clinically-safe{' '}
          <span className="text-accent glow-text-accent">
            dietary portions
          </span>
          ,<br className="hidden sm:block" /> powered by ML.
        </h1>

        {/* Subheading */}
        <p className="text-[15px] text-text-secondary leading-relaxed max-w-xl mb-10">
          NutriBiteBot analyses 14 lab values from your EHR and computes exact
          gram limits for every ingredient in your fridge — keeping every meal
          within clinically validated bounds for HTN, T2DM, and CKD.
        </p>

        {/* CTA row */}
        <div className="flex flex-wrap gap-3 mb-14">
          <button
            type="button"
            onClick={() => scrollTo('patient-data')}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-accent text-[#000000] font-semibold text-[14px] hover:bg-accent-hover transition-colors glow-accent"
          >
            Begin Analysis
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
          <a
            href="https://github.com/mripradhan/nutri-bite-bot"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-full border border-[#303236] text-text-secondary font-medium text-[14px] hover:border-accent hover:text-accent transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
            </svg>
            GitHub
          </a>
        </div>

        {/* Stats row */}
        <div className="flex flex-wrap gap-x-8 gap-y-5">
          {STATS.map(({ value, label }) => (
            <div key={label}>
              <div
                className="text-[20px] font-bold text-accent leading-none"
                style={{ fontFamily: "'Geist Mono', ui-monospace, monospace" }}
              >
                {value}
              </div>
              <div className="text-[11px] text-text-muted mt-1.5 tracking-wide">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom chip strip */}
      <div className="relative z-10 px-8 py-4 border-t border-[#303236] bg-[#000000]">
        <div className="flex flex-wrap gap-2">
          {TECH_CHIPS.map((chip) => (
            <span key={chip} className="chip text-[10px]">{chip}</span>
          ))}
        </div>
      </div>
    </section>
  )
}

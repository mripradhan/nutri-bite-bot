'use client'

import React, { useState, useEffect, useCallback } from 'react'

// ─── Nav items ────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  {
    id: 'overview',
    label: 'Overview',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
  },
  {
    id: 'patient-data',
    label: 'Patient Data',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    ),
  },
  {
    id: 'risk-analysis',
    label: 'Risk Analysis',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  },
  {
    id: 'fridge-scanner',
    label: 'Fridge Scanner',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="5" y="2" width="14" height="20" rx="2" />
        <line x1="5" y1="10" x2="19" y2="10" />
        <line x1="9" y1="6" x2="9" y2="8" />
        <line x1="9" y1="14" x2="9" y2="18" />
      </svg>
    ),
  },
  {
    id: 'portion-analysis',
    label: 'Portion Analysis',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 3v18h18" />
        <rect x="7" y="13" width="3" height="5" />
        <rect x="12" y="9" width="3" height="9" />
        <rect x="17" y="6" width="3" height="12" />
      </svg>
    ),
  },
  {
    id: 'recipe-generator',
    label: 'Recipe Generator',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a5 5 0 0 1 5 5c0 2.5-1.5 4.5-3 5.5V20a2 2 0 0 1-4 0v-7.5C8.5 11.5 7 9.5 7 7a5 5 0 0 1 5-5z" />
      </svg>
    ),
  },
  {
    id: 'model-info',
    label: 'Model Info',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    ),
  },
]

// ─── Component ────────────────────────────────────────────────────────────────
type SidebarProps = {
  onGoHome?: () => void
}

export default function Sidebar({ onGoHome }: SidebarProps) {
  const [activeSection, setActiveSection] = useState('overview')
  const [mobileOpen, setMobileOpen] = useState(false)

  // Intersection Observer to track which section is in view
  useEffect(() => {
    const sections = NAV_ITEMS.map((item) =>
      document.getElementById(item.id),
    ).filter(Boolean) as HTMLElement[]

    if (sections.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        // Find the entry with highest intersection ratio that is intersecting
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (visible.length > 0) {
          setActiveSection(visible[0].target.id)
        }
      },
      {
        rootMargin: '-20% 0px -60% 0px',
        threshold: [0, 0.1, 0.25, 0.5],
      },
    )

    sections.forEach((s) => observer.observe(s))
    return () => observer.disconnect()
  }, [])

  const scrollTo = useCallback((id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    setMobileOpen(false)
  }, [])

  // ── Sidebar inner content ──────────────────────────────────────────────────
  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-[#303236]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[4px] bg-accent flex items-center justify-center flex-shrink-0 glow-accent">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#000000" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a5 5 0 0 1 5 5c0 2.5-1.5 4.5-3 5.5V20a2 2 0 0 1-4 0v-7.5C8.5 11.5 7 9.5 7 7a5 5 0 0 1 5-5z" />
            </svg>
          </div>
          <div>
            <div className="text-[13px] font-700 text-text-primary leading-tight font-semibold">
              NutriBiteBot
            </div>
            <div className="text-[10px] text-text-secondary">
              Clinical Deep Learning
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 overflow-y-auto">
        {/* Back to home button */}
        {onGoHome && (
          <button
            onClick={onGoHome}
            className="w-full flex items-center gap-3 px-3 py-2.5 mb-3 rounded-[4px] text-left text-text-secondary hover:text-accent hover:bg-accent-muted transition-all duration-150 border border-transparent hover:border-[#34d59a]/20"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
            <span className="text-[13px] font-medium">Back to Home</span>
          </button>
        )}
        <div className="section-label px-2 mb-3">Navigation</div>
        <ul className="space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const isActive = activeSection === item.id
            return (
              <li key={item.id}>
                <button
                  onClick={() => scrollTo(item.id)}
                  className={[
                    'w-full flex items-center gap-3 px-3 py-2.5 rounded-[4px] text-left transition-all duration-150',
                    isActive
                      ? 'bg-accent-muted text-accent'
                      : 'text-text-secondary hover:text-text-primary hover:bg-[#151617]',
                  ].join(' ')}
                >
                  <span
                    className={[
                      'transition-colors duration-150',
                      isActive ? 'text-accent' : 'text-text-muted',
                    ].join(' ')}
                  >
                    {item.icon}
                  </span>
                  <span className="text-[13px] font-medium">{item.label}</span>
                  {isActive && (
                    <span className="ml-auto w-1 h-1 rounded-full bg-accent" />
                  )}
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-[#303236]">
        <a
          href="https://github.com/pradhanmrida/nutri-bite-bot"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 px-3 py-2.5 rounded-[6px] border border-[#303236] bg-[#0d0d0d] hover:border-[#34d59a]/30 hover:bg-accent-muted transition-all duration-200 group"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-text-secondary group-hover:text-white transition-colors flex-shrink-0">
            <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
          </svg>
          <div className="flex flex-col">
            <span className="text-[13px] font-semibold text-text-primary group-hover:text-white transition-colors">Star on GitHub</span>
            <span className="text-[10px] text-text-muted">View source code</span>
          </div>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-auto text-text-muted group-hover:text-accent transition-colors">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </a>
        <div className="mt-3 text-[10px] text-text-muted leading-relaxed">
          RVCE · Bengaluru
          <br />
          TabNet + IFCT 2017
        </div>
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex flex-col w-[280px] flex-shrink-0 bg-bg-sidebar border-r border-[#303236] h-screen sticky top-0 overflow-hidden">
        <SidebarContent />
      </aside>

      {/* Mobile hamburger button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2.5 rounded-[4px] bg-bg-card border border-[#303236] text-text-secondary hover:text-text-primary transition-colors"
        aria-label="Open menu"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile sidebar drawer */}
      <aside
        className={[
          'lg:hidden fixed top-0 left-0 z-50 h-full w-[280px] bg-bg-sidebar border-r border-[#303236] transform transition-transform duration-200',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
        ].join(' ')}
      >
        <div className="absolute top-3 right-3">
          <button
            onClick={() => setMobileOpen(false)}
            className="p-1.5 rounded-[4px] text-text-secondary hover:text-text-primary transition-colors"
            aria-label="Close menu"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <SidebarContent />
      </aside>
    </>
  )
}

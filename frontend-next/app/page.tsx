'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import dynamic from 'next/dynamic'

import Sidebar from '@/components/Sidebar'
import HeroInfo from '@/components/HeroInfo'
import PatientForm from '@/components/PatientForm'
import RiskDashboard from '@/components/RiskDashboard'
import ProbabilityCharts from '@/components/ProbabilityCharts'
import FridgeScanner from '@/components/FridgeScanner'
import IngredientSection from '@/components/IngredientSection'
import RecipeGenerator from '@/components/RecipeGenerator'
import ModelInfo from '@/components/ModelInfo'
import WakeUpBanner from '@/components/WakeUpBanner'

const LandingPage = dynamic(() => import('@/components/LandingPage'), {
  ssr: false,
  loading: () => <div className="fixed inset-0 bg-black z-[100]" />,
})

export default function HomePage() {
  const [showApp, setShowApp] = useState(false)
  const enterApp = useCallback(() => setShowApp(true), [])
  const goHome = useCallback(() => setShowApp(false), [])

  return (
    <div className="bg-black min-h-screen">
      {/* ── App content (always mounted, hidden until landing exits) ───── */}
      <motion.div
        className="flex min-h-screen bg-bg-primary"
        initial={false}
        animate={{ opacity: showApp ? 1 : 0, y: showApp ? 0 : 20 }}
        transition={{ duration: 0.75, ease: [0.4, 0, 0.2, 1], delay: showApp ? 0.35 : 0 }}
        style={{ pointerEvents: showApp ? 'auto' : 'none' }}
      >
        <WakeUpBanner />
        <Sidebar onGoHome={goHome} />

        <main className="flex-1 min-w-0 overflow-x-hidden">
          <div className="border-b border-[#303236]">
            <HeroInfo />
          </div>
          <div className="border-b border-[#303236]">
            <PatientForm />
          </div>
          <div className="border-b border-[#303236]">
            <RiskDashboard />
            <ProbabilityCharts />
          </div>
          <div className="border-b border-[#303236]">
            <FridgeScanner />
          </div>
          <div className="border-b border-[#303236]">
            <IngredientSection />
          </div>
          <div className="border-b border-[#303236]">
            <RecipeGenerator />
          </div>
          <ModelInfo />

          <footer className="px-8 py-6 border-t border-[#303236] text-center">
            <p className="text-[11px] text-text-muted">
              NutriBiteBot · RV College of Engineering, Bengaluru ·{' '}
              <span className="text-accent">TabNet</span> + IFCT 2017 · KDIGO 2024 · ADA 2024 · AHA/ACC
            </p>
          </footer>
        </main>
      </motion.div>

      {/* ── Landing overlay (exits via AnimatePresence) ─────────────────── */}
      <AnimatePresence>
        {!showApp && (
          <motion.div
            key="landing"
            className="fixed inset-0 z-[100]"
            exit={{
              opacity: 0,
              y: -32,
              transition: { duration: 0.85, ease: [0.4, 0, 0.2, 1] },
            }}
          >
            <LandingPage onEnter={enterApp} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

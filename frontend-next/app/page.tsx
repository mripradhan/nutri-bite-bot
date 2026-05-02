'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import dynamic from 'next/dynamic'

import Sidebar from '@/components/Sidebar'
import HeroInfo from '@/components/HeroInfo'
import PatientForm from '@/components/PatientForm'
import RiskDashboard from '@/components/RiskDashboard'
import FridgeScanner from '@/components/FridgeScanner'
import IngredientSection from '@/components/IngredientSection'
import RecipeGenerator from '@/components/RecipeGenerator'
import ModelInfo from '@/components/ModelInfo'
import WakeUpBanner from '@/components/WakeUpBanner'

const LineWaves = dynamic(
  () => import('@/components/ui/LineWaves').then((mod) => mod.default),
  { ssr: false, loading: () => <div className="w-full h-full bg-black" /> }
)

const LandingPage = dynamic(() => import('@/components/LandingPage'), {
  ssr: false,
  loading: () => <div className="fixed inset-0 bg-black z-[100]" />,
})

export default function HomePage() {
  const [showApp, setShowApp] = useState(false)
  const enterApp = useCallback(() => setShowApp(true), [])
  const goHome = useCallback(() => setShowApp(false), [])

  return (
    <div className="bg-black min-h-screen relative overflow-hidden">
      {/* ── Background shader for entire app ──────────────────────────── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <LineWaves
          speed={0.3}
          innerLineCount={32}
          outerLineCount={36}
          warpIntensity={1.0}
          rotation={-45}
          edgeFadeWidth={0.0}
          colorCycleSpeed={1.0}
          brightness={0.2}
          color1="#34d59a"
          color2="#151617"
          color3="#303236"
          enableMouseInteraction={true}
          mouseInfluence={2.0}
        />
        <div 
          className="absolute inset-0 transition-all duration-700 ease-in-out pointer-events-none"
          style={{
            backgroundColor: showApp ? 'rgba(0,0,0,0.85)' : 'rgba(0,0,0,0.80)',
            backdropFilter: showApp ? 'blur(8px)' : 'none'
          }}
        />
      </div>

      {/* ── App content (always mounted, hidden until landing exits) ───── */}
      <motion.div
        className="flex h-screen relative z-10 text-white"
        initial={false}
        animate={{ opacity: showApp ? 1 : 0, y: showApp ? 0 : 20 }}
        transition={{ duration: 0.75, ease: [0.4, 0, 0.2, 1], delay: showApp ? 0.35 : 0 }}
        style={{ pointerEvents: showApp ? 'auto' : 'none' }}
      >
        <WakeUpBanner />
        <Sidebar onGoHome={goHome} />

        <main className="flex-1 min-w-0 h-screen overflow-y-auto overflow-x-hidden relative">
          <div className="border-b border-white/5 bg-black/20">
            <HeroInfo />
          </div>
          <div className="border-b border-white/5 bg-black/20">
            <PatientForm />
          </div>
          <div className="border-b border-white/5 bg-black/20">
            <RiskDashboard />
          </div>
          <div className="border-b border-white/5 bg-black/20">
            <FridgeScanner />
          </div>
          <div className="border-b border-white/5 bg-black/20">
            <IngredientSection />
          </div>
          <div className="border-b border-white/5 bg-black/20">
            <RecipeGenerator />
          </div>
          <div className="bg-black/20">
            <ModelInfo />
          </div>

          <footer className="px-8 py-6 border-t border-white/5 bg-black/40 text-center">
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

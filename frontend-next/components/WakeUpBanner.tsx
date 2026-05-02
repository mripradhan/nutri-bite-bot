'use client'

import React, { useEffect, useState } from 'react'
import { subscribeToWakeUp } from '@/lib/api'

export default function WakeUpBanner() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const unsub = subscribeToWakeUp((show) => setVisible(show))
    return unsub
  }, [])

  if (!visible) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-[100] flex items-center justify-center px-4 py-2.5 bg-[#0a0a0a] border-b border-[#303236] animate-fade-in">
      <div className="flex items-center gap-2.5 text-[13px]">
        <svg className="animate-spin w-3.5 h-3.5 text-accent flex-shrink-0" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span className="text-accent font-medium">Backend waking up from sleep</span>
        <span className="text-text-secondary">
          — first request may take ~30s on Render&apos;s free tier.
        </span>
      </div>
    </div>
  )
}

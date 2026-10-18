'use client'

import React from 'react'

/** Persistent research-prototype notice. Wording mirrors the manuscript's validation scope. */
export default function DisclaimerBanner() {
  return (
    <div
      role="note"
      className="sticky top-0 z-30 px-6 md:px-8 py-2 bg-[#1a1608]/95 border-b border-risk-moderate/30 backdrop-blur-sm"
    >
      <p className="text-[11px] leading-relaxed text-risk-moderate">
        <span className="font-semibold">Research prototype, not for clinical use.</span>{' '}
        Risk tiers predict retrospective outcome proxies, and gram quantities are computed
        guideline-constrained ceilings. They have not been validated by dietitians or shown to
        be safe for an individual patient. Review by a qualified clinician or dietitian is required.
      </p>
    </div>
  )
}

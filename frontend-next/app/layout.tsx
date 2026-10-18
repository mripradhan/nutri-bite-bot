import type { Metadata, Viewport } from 'next'
import './globals.css'
import { AppProvider } from '@/context/AppContext'
import { Manrope } from "next/font/google";
import { cn } from "@/lib/utils";

const manrope = Manrope({subsets:['latin'],variable:'--font-sans'});

export const metadata: Metadata = {
  title: 'NutriBiteBot — Dietary Portion Research Prototype',
  description:
    'A research prototype: a TabNet-based framework that predicts outcome-risk proxies and computes guideline-constrained dietary portion ceilings for HTN, T2DM, and CKD. Not validated for clinical use.',
  keywords: [
    'NutriBiteBot',
    'clinical nutrition',
    'TabNet',
    'MIMIC-IV',
    'dietary recommendation',
    'CKD',
    'diabetes',
    'hypertension',
  ],
  authors: [
    { name: 'Mrida Pradhan' },
    { name: 'Gayathri Sunil Nambiar' },
    { name: 'Mohana' },
    { name: 'B. G. Sudarshan' },
  ],
  openGraph: {
    title: 'NutriBiteBot — Dietary Portion Research Prototype',
    description: 'Research prototype: guideline-constrained portion ceilings from TabNet outcome-risk prediction and the IFCT 2017 database. Not for clinical use.',
    type: 'website',
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#000000',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className={cn("font-sans", manrope.variable)}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Geist+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-bg-primary text-text-primary font-sans antialiased">
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  )
}

'use client'

import React, { useCallback, useRef, useState } from 'react'
import { useApp } from '@/context/AppContext'
import { apiDetect } from '@/lib/api'

export default function FridgeScanner() {
  const { detectedIngredients, setDetectedIngredients, addDetectedToIngredients } = useApp()
  const [isDragging, setIsDragging] = useState(false)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file.')
        return
      }

      // Preview
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target?.result as string)
      reader.readAsDataURL(file)

      setLoading(true)
      setError(null)
      setDetectedIngredients([])

      try {
        const result = await apiDetect(file)
        setDetectedIngredients(result.mapped_ifct.length > 0 ? result.mapped_ifct : result.detected_raw)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Detection failed')
      } finally {
        setLoading(false)
      }
    },
    [setDetectedIngredients],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) processFile(file)
    },
    [processFile],
  )

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }

  return (
    <section id="fridge-scanner" className="px-6 md:px-8 py-10">
      <div className="section-label mb-2">Computer Vision</div>
      <h2 className="text-[20px] font-bold text-text-primary mb-1">Fridge Scanner</h2>
      <p className="text-text-secondary text-[13px] mb-6">
        Upload a photo of your fridge or pantry. The CV model will detect ingredients and map
        them to the IFCT 2017 nutritional database.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={[
            'relative flex flex-col items-center justify-center rounded-[4px] border-2 border-dashed cursor-pointer transition-all duration-200 min-h-[240px] overflow-hidden',
            isDragging
              ? 'border-accent bg-accent/10 scale-[1.01]'
              : 'border-[#303236] bg-bg-card hover:border-accent/40 hover:bg-[#151617]',
          ].join(' ')}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onFileChange}
          />

          {imagePreview ? (
            <div className="relative w-full h-full min-h-[240px]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imagePreview}
                alt="Uploaded fridge scan"
                className="w-full h-full object-cover opacity-60 absolute inset-0"
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-bg-primary/80 rounded-[4px] px-4 py-2 text-[12px] text-text-secondary backdrop-blur-sm">
                  Click to replace image
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 p-8 text-center">
              <div className="w-14 h-14 rounded-[4px] border border-[#303236] flex items-center justify-center bg-[#0d0d0d]">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8a8f98" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <div>
                <p className="text-[13px] font-medium text-text-primary">
                  Drop an image here
                </p>
                <p className="text-[12px] text-text-muted mt-1">or click to browse</p>
              </div>
              <p className="text-[11px] text-text-muted">JPG, PNG, WEBP supported</p>
            </div>
          )}
        </div>

        {/* Results */}
        <div className="card flex flex-col gap-4">
          <div className="section-label">Detection Results</div>

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 py-8">
              <svg className="animate-spin w-8 h-8 text-accent" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <p className="text-[13px] text-text-secondary">Detecting ingredients…</p>
            </div>
          )}

          {error && !loading && (
            <div className="p-3 rounded-[4px] bg-risk-high/10 border border-risk-high/30 text-risk-high text-[13px]">
              {error}
            </div>
          )}

          {!loading && !error && detectedIngredients.length === 0 && (
            <div className="flex-1 flex items-center justify-center py-8">
              <p className="text-[13px] text-text-muted text-center">
                No ingredients detected yet.
                <br />
                Upload an image to begin.
              </p>
            </div>
          )}

          {!loading && detectedIngredients.length > 0 && (
            <>
              <div>
                <p className="text-[12px] text-text-secondary mb-3">
                  <span className="font-semibold text-accent">{detectedIngredients.length}</span>{' '}
                  ingredient{detectedIngredients.length !== 1 ? 's' : ''} detected
                </p>
                <div className="flex flex-wrap gap-2">
                  {detectedIngredients.map((ing) => (
                    <span
                      key={ing}
                      className="chip chip-accent text-[12px]"
                    >
                      {ing}
                    </span>
                  ))}
                </div>
              </div>

              <button
                onClick={addDetectedToIngredients}
                className="mt-auto py-2.5 px-4 rounded-full bg-accent text-[#000000] text-[13px] font-semibold hover:bg-accent-hover transition-colors"
              >
                Auto-add to Portion Analysis
              </button>
            </>
          )}
        </div>
      </div>
    </section>
  )
}

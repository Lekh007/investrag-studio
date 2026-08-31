import { useEffect, useMemo, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import type { ChunkRecord, PageGeometry } from './api'
import { fitScale, overlayRectsForChunk } from './bbox'

/**
 * PDF renderer choice (design §26, "frontend PDF component selection").
 *
 * `react-pdf` 10.5.0 (MIT) over `pdfjs-dist` 5.4.296 (Apache-2.0) directly. Both are
 * permissively licensed and both ship as plain JS with no native build step and no
 * remote-code-execution equivalent to `trust_remote_code`. react-pdf is a thin React
 * wrapper *around* pdfjs-dist, so the heavy dependency is the same either way; what it
 * buys is the part that is genuinely fiddly by hand — canvas lifecycle across page
 * changes, the text layer, cancellation of in-flight render tasks on unmount, and a
 * document cache. Measured cost on this build: the renderer is code-split into a
 * 420 kB / 124 kB-gzip chunk plus the 1,046 kB worker, both fetched only when a
 * paginated original is first opened, leaving the initial bundle at 632 kB / 186 kB
 * gzip. The worker is served from our own `node_modules` through Vite, never a CDN,
 * so the localhost-only trust boundary in design §18 still holds.
 */
pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()

export type PdfViewerProps = {
  url: string
  page: number
  pages: PageGeometry[]
  /** Chunks whose bounding boxes are drawn over the rendered page. */
  highlighted: ChunkRecord[]
  onPageChange: (page: number) => void
}

export default function PdfViewer({ url, page, pages, highlighted, onPageChange }: PdfViewerProps) {
  const container = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [pageCount, setPageCount] = useState(pages.length)
  const [error, setError] = useState('')

  useEffect(() => {
    const element = container.current
    if (!element) return
    const observer = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    observer.observe(element)
    setWidth(element.clientWidth)
    return () => observer.disconnect()
  }, [])

  const geometry = pages.find((item) => item.page === page)
  const scale = useMemo(() => (geometry ? fitScale(geometry, width) : 1), [geometry, width])
  const rects = useMemo(
    () => highlighted.flatMap((chunk) => overlayRectsForChunk(chunk, page, scale)),
    [highlighted, page, scale],
  )

  return (
    <div className="pdf-viewer" ref={container} data-testid="pdf-viewer">
      <div className="pdf-toolbar">
        <button
          className="secondary-button"
          data-testid="pdf-prev"
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
        >
          ‹ Prev
        </button>
        <span className="pdf-page-indicator" data-testid="pdf-page-indicator">
          Page {page} of {pageCount || pages.length || '?'}
        </span>
        <button
          className="secondary-button"
          data-testid="pdf-next"
          disabled={pageCount > 0 && page >= pageCount}
          onClick={() => onPageChange(page + 1)}
        >
          Next ›
        </button>
        <span className="muted">
          {rects.length} bounding box{rects.length === 1 ? '' : 'es'} on this page
        </span>
      </div>
      {error ? (
        <div className="callout callout-bad" data-testid="pdf-error">
          {error}
        </div>
      ) : null}
      <div className="pdf-stage">
        <Document
          file={url}
          onLoadSuccess={(document) => {
            setPageCount(document.numPages)
            setError('')
          }}
          onLoadError={(reason: Error) => setError(`The original document could not be rendered: ${reason.message}`)}
          loading={<div className="pdf-placeholder">Rendering the original page…</div>}
          error={<div className="pdf-placeholder">The original document could not be rendered.</div>}
        >
          <div className="pdf-page-frame">
            <Page
              pageNumber={page}
              scale={scale}
              renderAnnotationLayer={false}
              renderTextLayer
              loading={<div className="pdf-placeholder">Rendering page {page}…</div>}
            />
            <div className="bbox-layer" data-testid="bbox-layer">
              {rects.map((rect) => (
                <span
                  key={rect.key}
                  className="bbox"
                  data-testid="bbox"
                  data-element-id={rect.elementId}
                  title={rect.elementId ?? 'extracted element'}
                  style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
                />
              ))}
            </div>
          </div>
        </Document>
      </div>
    </div>
  )
}

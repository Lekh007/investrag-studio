import type { ChunkRecord, PageGeometry, SourceLocation } from './api'

/**
 * Bounding-box overlay geometry (design §15.2 "inspect … bounding boxes").
 *
 * The backend emits `CanonicalElement.bbox` as `[x0, y0, x1, y1]` in **PDF points**
 * with a **TOPLEFT** origin, whichever parser produced it: the PyMuPDF path already
 * used that convention, and the Docling path converts its BOTTOMLEFT coordinates into
 * it at parse time. `GET /source-versions/{id}/pages` reports the page box in the same
 * units, and a `pdf.js` viewport at scale S is exactly `points × S` pixels — so the
 * mapping is a single multiply, with no per-parser special-casing anywhere in the UI.
 */

export type OverlayRect = {
  key: string
  left: number
  top: number
  width: number
  height: number
  elementId?: string
}

export type PageBox = Pick<PageGeometry, 'width' | 'height'>

/** Convert one point-space bbox to a pixel-space rect on a page rendered at `scale`. */
export function rectFromBbox(
  bbox: [number, number, number, number],
  scale: number,
  key: string,
  elementId?: string,
): OverlayRect {
  const [x0, y0, x1, y1] = bbox
  return {
    key,
    elementId,
    left: Math.min(x0, x1) * scale,
    top: Math.min(y0, y1) * scale,
    width: Math.abs(x1 - x0) * scale,
    height: Math.abs(y1 - y0) * scale,
  }
}

/**
 * Every location a chunk actually occupies, not only its first element's.
 *
 * A chunk normally spans several layout blocks; highlighting `metadata.bbox` alone
 * would frame the first paragraph of the evidence and silently drop the rest of it.
 * `metadata.bbox` is still used as the fallback for chunks whose per-element
 * provenance predates that being carried through (or whose profile does not emit it).
 */
export function chunkLocations(chunk: ChunkRecord): SourceLocation[] {
  const locations = chunk.metadata.source_locations ?? []
  const withBoxes = locations.filter((location) => Array.isArray(location.bbox))
  if (withBoxes.length) return withBoxes
  if (Array.isArray(chunk.metadata.bbox) && typeof chunk.metadata.page === 'number') {
    return [{ page: chunk.metadata.page, bbox: chunk.metadata.bbox }]
  }
  return []
}

/** The page a chunk should open at: its first located element's page. */
export function chunkPage(chunk: ChunkRecord): number | null {
  const located = chunkLocations(chunk).find((location) => typeof location.page === 'number')
  if (located?.page) return located.page
  return typeof chunk.metadata.page === 'number' ? chunk.metadata.page : null
}

/** Overlay rects for one chunk on one page, in pixel space. */
export function overlayRectsForChunk(chunk: ChunkRecord, page: number, scale: number): OverlayRect[] {
  return chunkLocations(chunk)
    .filter((location) => location.page === page && Array.isArray(location.bbox))
    .map((location, index) =>
      rectFromBbox(location.bbox as [number, number, number, number], scale, `${chunk.chunk_id}:${index}`, location.element_id),
    )
}

/**
 * The render scale that fits a page into `availableWidth` pixels, clamped so a very
 * narrow container cannot produce an unreadable page or a division by zero.
 */
export function fitScale(page: PageBox, availableWidth: number, min = 0.2, max = 3): number {
  if (!page.width || availableWidth <= 0) return 1
  return Math.min(max, Math.max(min, availableWidth / page.width))
}

/** True when a bbox is inside its page box, allowing one point of rounding slack. */
export function isWithinPage(bbox: [number, number, number, number], page: PageBox): boolean {
  const [x0, y0, x1, y1] = bbox
  return x0 >= -1 && y0 >= -1 && x1 <= page.width + 1 && y1 <= page.height + 1 && x1 > x0 && y1 > y0
}

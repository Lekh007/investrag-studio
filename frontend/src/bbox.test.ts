import { describe, expect, it } from 'vitest'
import type { ChunkRecord } from './api'
import { chunkLocations, chunkPage, fitScale, isWithinPage, overlayRectsForChunk, rectFromBbox } from './bbox'

const chunk = (metadata: ChunkRecord['metadata']): ChunkRecord => ({
  chunk_id: 'chunk_a',
  source_id: 'version_1',
  profile: 'structure-aware',
  text: 'Revenue grew 12%.',
  element_ids: ['el_1'],
  metadata,
})

// A US-Letter page: 612 × 792 points.
const letter = { width: 612, height: 792 }

describe('bbox → overlay geometry', () => {
  it('maps points to pixels through the render scale', () => {
    expect(rectFromBbox([72, 96, 300, 120], 2, 'k')).toEqual({
      key: 'k',
      elementId: undefined,
      left: 144,
      top: 192,
      width: 456,
      height: 48,
    })
  })

  it('normalises a box whose corners arrive in the other order', () => {
    // Neither parser should emit this, but an overlay must never render a negative
    // width if one ever does — a silently invisible highlight is worse than a wrong one.
    const rect = rectFromBbox([300, 120, 72, 96], 1, 'k')
    expect(rect).toMatchObject({ left: 72, top: 96, width: 228, height: 24 })
  })

  it('uses every element location a chunk spans, not only the first', () => {
    const record = chunk({
      page: 1,
      bbox: [72, 96, 300, 120],
      source_locations: [
        { element_id: 'el_1', page: 1, bbox: [72, 96, 300, 120] },
        { element_id: 'el_2', page: 1, bbox: [72, 130, 480, 160] },
      ],
    })
    expect(overlayRectsForChunk(record, 1, 1)).toHaveLength(2)
    expect(overlayRectsForChunk(record, 1, 1)[1]).toMatchObject({ elementId: 'el_2', top: 130 })
  })

  it('falls back to the chunk-level bbox when per-element boxes are absent', () => {
    const record = chunk({ page: 3, bbox: [10, 20, 30, 40], source_locations: [{ element_id: 'el_1', page: 3 }] })
    expect(chunkLocations(record)).toEqual([{ page: 3, bbox: [10, 20, 30, 40] }])
    expect(chunkPage(record)).toBe(3)
  })

  it('returns no rects for a chunk with no geometry at all', () => {
    const record = chunk({ sheet: 'Summary', cell_range: 'A1:B4' })
    expect(chunkLocations(record)).toEqual([])
    expect(overlayRectsForChunk(record, 1, 1)).toEqual([])
    expect(chunkPage(record)).toBeNull()
  })

  it('draws only the boxes belonging to the page being rendered', () => {
    const record = chunk({
      source_locations: [
        { element_id: 'el_1', page: 1, bbox: [72, 96, 300, 120] },
        { element_id: 'el_2', page: 2, bbox: [72, 96, 300, 120] },
      ],
    })
    expect(overlayRectsForChunk(record, 2, 1).map((rect) => rect.elementId)).toEqual(['el_2'])
  })

  it('opens a chunk at its first located page', () => {
    const record = chunk({ source_locations: [{ element_id: 'el_1', page: 4, bbox: [1, 2, 3, 4] }] })
    expect(chunkPage(record)).toBe(4)
  })

  it('fits a page to the container width and clamps degenerate inputs', () => {
    expect(fitScale(letter, 918)).toBeCloseTo(1.5)
    expect(fitScale(letter, 0)).toBe(1)
    expect(fitScale({ width: 0, height: 0 }, 500)).toBe(1)
    expect(fitScale(letter, 100_000)).toBe(3)
  })

  it('validates that a box lies inside its page box', () => {
    expect(isWithinPage([72, 96, 300, 120], letter)).toBe(true)
    expect(isWithinPage([72, 96, 700, 120], letter)).toBe(false)
    expect(isWithinPage([72, 96, 72, 120], letter)).toBe(false)
  })
})

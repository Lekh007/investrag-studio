import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

describe('InvestRAG API client', () => {
  afterEach(() => vi.restoreAllMocks())

  it('posts a query to the local evidence endpoint', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ answer: 'grounded', citations: [] }), { status: 200 }),
    )

    await api.query('What changed?', 'hybrid')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/v1/queries',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('surfaces FastAPI detail messages without raw JSON noise', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'only the portable experiment track is implemented' }), { status: 400 }),
    )

    await expect(api.query('What changed?', 'hybrid')).rejects.toThrow(
      'only the portable experiment track is implemented',
    )
  })

  it('builds artifact links with PDF page fragments', () => {
    expect(api.artifactContentUrl('version one', 3)).toBe(
      'http://127.0.0.1:8000/api/v1/artifacts/version%20one/content#page=3',
    )
  })
})

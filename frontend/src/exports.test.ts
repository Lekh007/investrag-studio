import { describe, expect, it } from 'vitest'
import { classifyQuestion, compareMetrics, reportFilename } from './exports'

const result = (over: Partial<Parameters<typeof classifyQuestion>[0]> = {}) => ({
  success_at_k: 1,
  recall_at_k: 1,
  reciprocal_rank: 1,
  relevant_chunk_ids: ['chunk_a'],
  predicted_abstention: false,
  abstention_correct: null,
  ...over,
})

describe('report filenames', () => {
  it('uses the conventional extension per format', () => {
    expect(reportFilename('exp_1', 'markdown')).toBe('exp_1.md')
    expect(reportFilename('exp_1', 'csv')).toBe('exp_1.csv')
    expect(reportFilename('exp_1', 'json')).toBe('exp_1.json')
  })
})

describe('per-question failure causes', () => {
  it('calls a correct abstention correct, not a zero score', () => {
    // The structural trap this project already hit once: an unanswerable question has
    // no relevant chunks, so every retrieval metric is 0 by construction. That is a
    // pass, and the drill-down must say so.
    const verdict = classifyQuestion(result({ relevant_chunk_ids: [], success_at_k: 0, recall_at_k: 0, reciprocal_rank: 0, predicted_abstention: true, abstention_correct: true }))
    expect(verdict.kind).toBe('ok')
    expect(verdict.explanation).toContain('correctly abstained')
  })

  it('flags an unanswerable question that returned evidence', () => {
    expect(
      classifyQuestion(result({ relevant_chunk_ids: [], predicted_abstention: false, abstention_correct: false })).kind,
    ).toBe('missed-abstention')
  })

  it('flags an answerable question the evidence gate refused', () => {
    expect(classifyQuestion(result({ predicted_abstention: true })).kind).toBe('wrong-abstention')
  })

  it('separates a retrieval miss from a ranking problem', () => {
    expect(classifyQuestion(result({ success_at_k: 0, recall_at_k: 0, reciprocal_rank: 0 })).kind).toBe('miss')
    const lowRank = classifyQuestion(result({ reciprocal_rank: 0.25 }))
    expect(lowRank.kind).toBe('low-rank')
    expect(lowRank.explanation).toContain('rank 4')
  })

  it('flags partial recall even when the top hit was right', () => {
    const partial = classifyQuestion(result({ recall_at_k: 0.5, relevant_chunk_ids: ['a', 'b'] }))
    expect(partial.kind).toBe('low-rank')
    expect(partial.explanation).toContain('50%')
  })

  it('passes a fully-retrieved, top-ranked question', () => {
    expect(classifyQuestion(result()).kind).toBe('ok')
  })
})

describe('manifest comparison', () => {
  it('pairs metrics by name and reports the delta', () => {
    const rows = compareMetrics(
      [{ name: 'faiss/ndcg@10', value: 0.8 }, { name: 'faiss/success@10', value: 1 }],
      [{ name: 'faiss/ndcg@10', value: 0.9 }, { name: 'qdrant/ndcg@10', value: 0.7 }],
    )
    expect(rows.find((row) => row.name === 'faiss/ndcg@10')).toMatchObject({ left: 0.8, right: 0.9 })
    expect(rows.find((row) => row.name === 'faiss/ndcg@10')!.delta).toBeCloseTo(0.1)
  })

  it('leaves the delta null when a metric exists on only one side', () => {
    const rows = compareMetrics([{ name: 'faiss/ndcg@10', value: 0.8 }], [])
    expect(rows[0]).toEqual({ name: 'faiss/ndcg@10', left: 0.8, right: null, delta: null })
  })
})

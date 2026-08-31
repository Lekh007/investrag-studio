import { describe, expect, it } from 'vitest'
import type { QueryResponse, QueryTrace } from './api'
import { type LabRun, candidateStages, compareRuns, rankMovement, survivalSeries } from './compare'

const candidate = (id: string, rank: number) => ({ chunk_id: id, source_id: 'version_1', rank, score: 1 / rank })

const trace = (stages: QueryTrace['stages']): QueryTrace =>
  ({ profile: 'hybrid', vector_store: 'faiss', stages }) as QueryTrace

const run = (key: string, stages: QueryTrace['stages']): LabRun => ({
  key,
  vectorStore: key,
  profile: 'hybrid',
  status: 'ok',
  response: { trace: trace(stages) } as QueryResponse,
})

describe('stage ordering', () => {
  it('orders stages by the pipeline, not by the order the backend happened to append them', () => {
    const ordered = candidateStages(
      trace([
        { stage: 'final', candidates: [candidate('a', 1)] },
        { stage: 'dense', candidates: [candidate('a', 1)] },
        { stage: 'rrf', candidates: [candidate('a', 1)] },
      ]),
    )
    expect(ordered.map((stage) => stage.stage)).toEqual(['dense', 'rrf', 'final'])
  })

  it('skips provenance-only stages that carry no candidate list', () => {
    // `multi-query-generation` reports which model produced the query variants; it has
    // no `candidates` key at all, and once crashed the backend for exactly that reason.
    const stages = candidateStages(
      trace([
        { stage: 'multi-query-generation', generation_mode: 'model' },
        { stage: 'dense', candidates: [candidate('a', 1)] },
      ]),
    )
    expect(stages.map((stage) => stage.stage)).toEqual(['dense'])
  })
})

describe('rank movement within one trace', () => {
  const moved = trace([
    { stage: 'dense', candidates: [candidate('a', 1), candidate('b', 2), candidate('c', 3)] },
    { stage: 'rerank', candidates: [candidate('c', 1), candidate('a', 2), candidate('b', 3)] },
    { stage: 'final', candidates: [candidate('c', 1), candidate('a', 2)] },
  ])

  it('reports a positive movement for a chunk the reranker promoted', () => {
    const rows = rankMovement(moved)
    const c = rows.find((row) => row.chunkId === 'c')!
    expect(c.ranks).toEqual({ dense: 3, rerank: 1, final: 1 })
    expect(c.movement).toBe(2)
  })

  it('reports a negative movement for a chunk that fell', () => {
    expect(rankMovement(moved).find((row) => row.chunkId === 'a')!.movement).toBe(-1)
  })

  it('distinguishes "dropped by a stage" from "ranked last"', () => {
    const b = rankMovement(moved).find((row) => row.chunkId === 'b')!
    expect(b.ranks.final).toBeNull()
    expect(b.finalRank).toBeNull()
  })

  it('sorts surviving chunks by their final rank, dropped ones last', () => {
    expect(rankMovement(moved).map((row) => row.chunkId)).toEqual(['c', 'a', 'b'])
  })

  it('reports no movement for a chunk seen at only one stage', () => {
    const rows = rankMovement(trace([{ stage: 'dense', candidates: [candidate('a', 1)] }]))
    expect(rows[0].movement).toBeNull()
  })
})

describe('cross-store comparison', () => {
  const runs = [
    run('faiss', [{ stage: 'final', candidates: [candidate('a', 1), candidate('b', 2)] }]),
    run('qdrant', [{ stage: 'final', candidates: [candidate('a', 1), candidate('c', 2)] }]),
    run('pgvector', [{ stage: 'final', candidates: [candidate('a', 2), candidate('b', 1)] }]),
  ]

  it('ranks chunks by how many stores agree on them', () => {
    const rows = compareRuns(runs)
    expect(rows[0].chunkId).toBe('a')
    expect(rows[0].agreement).toBe(3)
    expect(rows[0].perRun).toEqual({ faiss: 1, qdrant: 1, pgvector: 2 })
  })

  it('records a null, not a zero, where a store never surfaced the chunk', () => {
    const c = compareRuns(runs).find((row) => row.chunkId === 'c')!
    expect(c.perRun).toEqual({ faiss: null, qdrant: 2, pgvector: null })
    expect(c.agreement).toBe(1)
  })

  it('ignores runs that errored so a dead store cannot look like agreement', () => {
    const rows = compareRuns([...runs, { key: 'weaviate', vectorStore: 'weaviate', profile: 'hybrid', status: 'error', error: 'down' }])
    expect(Object.keys(rows[0].perRun)).toEqual(['faiss', 'qdrant', 'pgvector'])
  })

  it('is empty when nothing has completed yet', () => {
    expect(compareRuns([{ key: 'faiss', vectorStore: 'faiss', profile: 'hybrid', status: 'running' }])).toEqual([])
  })
})

describe('survival series', () => {
  it('produces one point per stage with a value per completed run', () => {
    const series = survivalSeries([
      run('faiss', [
        { stage: 'dense', candidates: [candidate('a', 1), candidate('b', 2)] },
        { stage: 'final', candidates: [candidate('a', 1)] },
      ]),
      run('qdrant', [{ stage: 'dense', candidates: [candidate('a', 1)] }]),
    ])
    expect(series).toEqual([
      { stage: 'Dense', faiss: 2, qdrant: 1 },
      { stage: 'Final', faiss: 1 },
    ])
  })
})

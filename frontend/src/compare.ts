import type { QueryResponse, QueryTrace, TraceStage } from './api'

/**
 * Retrieval Lab comparison logic (design §15.4: "run the same query against selected
 * databases or profiles", "compare dense, lexical, fused and reranked stages",
 * "inspect rank movement").
 *
 * Kept as pure functions over real `QueryTrace` values so the arithmetic that decides
 * "this chunk moved up four places between fusion and reranking" is unit-testable
 * without a browser, a backend, or a chart.
 */

export type LabRun = {
  key: string
  vectorStore: string
  profile: string
  status: 'running' | 'ok' | 'error'
  response?: QueryResponse
  error?: string
  /** Wall-clock time of the round trip, which includes transport, unlike `trace.total_ms`. */
  wallMs?: number
}

export const STAGE_ORDER = ['dense', 'bm25', 'rrf', 'multi-query-fusion', 'rerank', 'final', 'evidence-gate'] as const

/** Human labels for the raw stage names the backend emits. */
export const STAGE_LABELS: Record<string, string> = {
  dense: 'Dense',
  bm25: 'Lexical (BM25)',
  rrf: 'Fused (RRF)',
  'multi-query-fusion': 'Fused (multi-query)',
  rerank: 'Reranked',
  final: 'Final',
  'evidence-gate': 'Evidence gate',
  'multi-query-generation': 'Query expansion',
}

export const stageLabel = (stage: string): string => STAGE_LABELS[stage] ?? stage

/** Stages that actually carry candidates, in pipeline order, for one trace. */
export function candidateStages(trace: QueryTrace): TraceStage[] {
  const withCandidates = trace.stages.filter((stage) => Array.isArray(stage.candidates))
  const rank = (stage: TraceStage) => {
    const index = STAGE_ORDER.indexOf(stage.stage as (typeof STAGE_ORDER)[number])
    return index === -1 ? STAGE_ORDER.length : index
  }
  return [...withCandidates].sort((left, right) => rank(left) - rank(right))
}

export type RankRow = {
  chunkId: string
  sourceId: string
  /** Rank per stage; `null` where the stage dropped (or never saw) the chunk. */
  ranks: Record<string, number | null>
  /** Rank change from the first stage that held this chunk to the last one. Positive = moved up. */
  movement: number | null
  finalRank: number | null
}

/**
 * Rank movement for every chunk that appeared at any stage of one trace.
 *
 * A chunk absent from a stage gets `null` rather than a sentinel number: "dropped by
 * the evidence gate" and "ranked last" are different facts and must not render the
 * same way.
 */
export function rankMovement(trace: QueryTrace): RankRow[] {
  const stages = candidateStages(trace)
  const rows = new Map<string, RankRow>()
  for (const stage of stages) {
    for (const candidate of stage.candidates ?? []) {
      const row = rows.get(candidate.chunk_id) ?? {
        chunkId: candidate.chunk_id,
        sourceId: candidate.source_id,
        ranks: Object.fromEntries(stages.map((item) => [item.stage, null])) as Record<string, number | null>,
        movement: null,
        finalRank: null,
      }
      row.ranks[stage.stage] = candidate.rank
      rows.set(candidate.chunk_id, row)
    }
  }
  const lastStage = stages.at(-1)?.stage
  for (const row of rows.values()) {
    const present = stages.map((stage) => row.ranks[stage.stage]).filter((rank): rank is number => rank !== null)
    row.movement = present.length > 1 ? present[0] - present[present.length - 1] : null
    row.finalRank = lastStage ? row.ranks[lastStage] : null
  }
  return [...rows.values()].sort((left, right) => {
    if (left.finalRank !== null && right.finalRank !== null) return left.finalRank - right.finalRank
    if (left.finalRank !== null) return -1
    if (right.finalRank !== null) return 1
    return left.chunkId.localeCompare(right.chunkId)
  })
}

export type StoreComparisonRow = {
  chunkId: string
  sourceId: string
  /** Final rank in each run key; `null` when that store never surfaced the chunk. */
  perRun: Record<string, number | null>
  /** How many runs surfaced it at all — the agreement signal. */
  agreement: number
}

/**
 * Cross-store comparison: which chunks each database surfaced for the same query, and
 * how far the rankings agree. This is the answer to "does the store choice change what
 * the model is allowed to see?", which is the point of running the lab at all.
 */
export function compareRuns(runs: LabRun[]): StoreComparisonRow[] {
  const completed = runs.filter((run) => run.response)
  const rows = new Map<string, StoreComparisonRow>()
  for (const run of completed) {
    const finalStage = candidateStages(run.response!.trace).at(-1)
    for (const candidate of finalStage?.candidates ?? []) {
      const row = rows.get(candidate.chunk_id) ?? {
        chunkId: candidate.chunk_id,
        sourceId: candidate.source_id,
        perRun: Object.fromEntries(completed.map((item) => [item.key, null])) as Record<string, number | null>,
        agreement: 0,
      }
      row.perRun[run.key] = candidate.rank
      rows.set(candidate.chunk_id, row)
    }
  }
  for (const row of rows.values()) {
    row.agreement = Object.values(row.perRun).filter((rank) => rank !== null).length
  }
  return [...rows.values()].sort((left, right) => {
    if (right.agreement !== left.agreement) return right.agreement - left.agreement
    return bestRank(left) - bestRank(right)
  })
}

const bestRank = (row: StoreComparisonRow): number =>
  Math.min(...Object.values(row.perRun).map((rank) => rank ?? Number.POSITIVE_INFINITY))

/** Recharts series: candidate survival by stage, one line per run. */
export function survivalSeries(runs: LabRun[]): Array<Record<string, string | number>> {
  const stages = new Set<string>()
  for (const run of runs) {
    if (!run.response) continue
    for (const stage of candidateStages(run.response.trace)) stages.add(stage.stage)
  }
  const ordered = [...stages].sort(
    (left, right) =>
      (STAGE_ORDER.indexOf(left as (typeof STAGE_ORDER)[number]) + 1 || 99) -
      (STAGE_ORDER.indexOf(right as (typeof STAGE_ORDER)[number]) + 1 || 99),
  )
  return ordered.map((stage) => {
    const point: Record<string, string | number> = { stage: stageLabel(stage) }
    for (const run of runs) {
      if (!run.response) continue
      const match = run.response.trace.stages.find((item) => item.stage === stage)
      if (match?.candidates) point[run.key] = match.candidates.length
    }
    return point
  })
}

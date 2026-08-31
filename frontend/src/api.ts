export type SourceVersion = {
  logical_source_id: string
  version_id: string
  source_id: string
  name: string
  media_type: string
  parser: string
  status: 'ready' | 'partial' | 'failed'
  quality_score: number
  element_count: number
  chunk_count: number
  warnings: string[]
  queryable: boolean
  requires_review: boolean
}

/** One element's own location inside a chunk. `bbox` is `[x0, y0, x1, y1]` in PDF
 * points with a TOPLEFT origin, matching `GET /source-versions/{id}/pages`. */
export type SourceLocation = {
  element_id?: string
  page?: number
  slide?: number
  sheet?: string
  cell_range?: string
  json_path?: string
  bbox?: [number, number, number, number]
}

export type ChunkRecord = {
  chunk_id: string
  source_id: string
  profile: string
  text: string
  parent_id?: string | null
  element_ids: string[]
  metadata: Record<string, unknown> & { source_locations?: SourceLocation[]; bbox?: [number, number, number, number]; page?: number }
}

export type SourceDetail = { source: SourceVersion; chunks: ChunkRecord[] }

export type PageGeometry = { page: number; width: number; height: number; rotation: number }

export type StoreCapability = {
  name: string
  implemented: boolean
  available?: boolean
  installed?: boolean
  description: string
  reason?: string
}

export type CollectionRow = {
  name: string
  reachable: boolean
  count: number
  expected_count: number
  consistent_with_catalog: boolean | null
  reason?: string
}

export type Health = {
  status: string
  service: string
  embedding_model: string
  embedding_fallback: boolean
  embedding_state: 'configured' | 'ready' | 'fallback'
  indexed_chunks: number
  available_vector_stores: StoreCapability[]
  resource_profile: string
}

export type Citation = {
  chunk_id: string
  source_id: string
  label: string
  source_name: string
  excerpt: string
  page?: number | null
  slide?: number | null
  sheet?: string | null
  cell_range?: string | null
  json_path?: string | null
}

export type TraceStage = {
  stage: string
  candidates?: Array<{ chunk_id: string; source_id: string; rank: number; score: number }>
  [key: string]: unknown
}

export type QueryTrace = {
  trace_id: string
  profile: string
  vector_store: string
  original_query: string
  rewritten_query: string
  filters: Record<string, unknown>
  dense_candidates: number
  lexical_candidates: number
  fused_candidates: number
  final_context_chunks: number
  retrieval_ms: number
  generation_ms: number
  total_ms: number
  model: string
  generation_mode: 'model' | 'extractive-fallback' | 'abstained'
  generation_fallback_reason?: string | null
  citation_repair_attempted?: boolean
  citation_repair_succeeded?: boolean
  embedding_model: string
  embedding_fallback: boolean
  stages: TraceStage[]
}

export type QueryResponse = {
  answer: string
  citations: Citation[]
  insufficient_evidence: boolean
  answer_withheld?: boolean
  conflicting_evidence?: boolean
  validator_messages: string[]
  trace: QueryTrace
}

export type GoldenQuestion = {
  question_id: string
  question: string
  relevant_chunk_ids: string[]
  relevant_element_ids: string[]
  category: string
  answerable: boolean
}

export type GoldenDataset = {
  dataset_id: string
  name: string
  description: string
  questions: GoldenQuestion[]
  corpus_source_ids: string[]
}

export type MetricResult = { name: string; value: number; unit: string }

export type ExperimentQuestionResult = {
  question_id: string
  vector_store: string
  profile: string
  retrieved_chunk_ids: string[]
  relevant_chunk_ids: string[]
  success_at_k: number
  recall_at_k: number
  reciprocal_rank: number
  ndcg_at_k: number
  retrieval_ms: number
  predicted_abstention: boolean
  abstention_correct: boolean | null
}

export type ExperimentRecord = {
  experiment_id: string
  manifest: { dataset_id: string; track: string; vector_stores: string[]; profiles: string[]; top_k: number; repetitions?: number }
  status: 'completed' | 'partial' | 'failed'
  corpus_chunk_count: number
  results: ExperimentQuestionResult[]
  metrics: MetricResult[]
  warnings: string[]
  reproducibility: Record<string, unknown>
}

export type JobStage = {
  name: string
  status: 'running' | 'completed' | 'failed' | 'skipped'
  detail: string
  started_at: string
  elapsed_ms: number
}

export type IngestionJob = {
  job_id: string
  kind: 'ingestion' | 'url-ingestion' | 'reprocess'
  label: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  stages: JobStage[]
  source_id: string | null
  source: SourceVersion | null
  error: string | null
  resumable: boolean
  parser: string | null
  chunk_profile: string | null
  resumed_from: string | null
  reprocess_of: string | null
}

export type NamedOption = { name: string; description: string; version?: string }

export const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const body = await response.text()
    let message = body
    try {
      const parsed = JSON.parse(body) as { detail?: string }
      message = parsed.detail ?? body
    } catch {
      // Preserve non-JSON backend or proxy errors.
    }
    throw new Error(message || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

async function requestText(path: string): Promise<string> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) throw new Error(await response.text())
  return response.text()
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  health: () => request<Health>('/health'),
  collections: () => request<CollectionRow[]>('/collections'),
  vectorStores: () => request<StoreCapability[]>('/vector-stores'),
  parserModes: () => request<NamedOption[]>('/parser-modes'),
  chunkProfiles: () => request<NamedOption[]>('/chunk-profiles'),

  sources: () => request<SourceVersion[]>('/sources'),
  sourceDetail: (sourceId: string) => request<SourceDetail>(`/sources/${encodeURIComponent(sourceId)}`),
  sourcePages: (sourceId: string) => request<PageGeometry[]>(`/source-versions/${encodeURIComponent(sourceId)}/pages`),
  artifactContentUrl: (sourceId: string, page?: number | null) =>
    `${API_BASE}/artifacts/${encodeURIComponent(sourceId)}/content${page ? `#page=${page}` : ''}`,

  /** Design §16: ingestion returns a job id; follow it with `ingestionEventsUrl`. */
  ingest: (file: File, options: { parser?: string; chunkProfile?: string } = {}) => {
    const body = new FormData()
    body.append('file', file)
    if (options.parser) body.append('parser', options.parser)
    if (options.chunkProfile) body.append('chunk_profile', options.chunkProfile)
    return request<IngestionJob>('/ingestions', { method: 'POST', body })
  },
  ingestUrl: (url: string) => request<IngestionJob>('/ingestions/url', json({ url })),
  jobs: () => request<IngestionJob[]>('/ingestions'),
  job: (jobId: string) => request<IngestionJob>(`/ingestions/${encodeURIComponent(jobId)}`),
  ingestionEventsUrl: (jobId: string) => `${API_BASE}/ingestions/${encodeURIComponent(jobId)}/events`,
  resumeJob: (jobId: string) => request<IngestionJob>(`/ingestions/${encodeURIComponent(jobId)}/resume`, { method: 'POST' }),
  reprocess: (sourceId: string, body: { parser?: string; chunk_profile?: string }) =>
    request<IngestionJob>(`/source-versions/${encodeURIComponent(sourceId)}/reprocess`, json(body)),
  review: (sourceId: string, decision: 'approve' | 'reject', note = '') =>
    request<SourceVersion>(`/source-versions/${encodeURIComponent(sourceId)}/review`, json({ decision, note })),

  query: (question: string, profile: string, vectorStore = 'faiss', topK = 6) =>
    request<QueryResponse>('/queries', json({ question, profile, vector_store: vectorStore, top_k: topK })),
  traces: () => request<QueryTrace[]>('/queries/traces'),

  goldenDatasets: () => request<GoldenDataset[]>('/golden-datasets'),
  saveGoldenDataset: (dataset: GoldenDataset) => request<GoldenDataset>('/golden-datasets', json(dataset)),
  experiments: () => request<ExperimentRecord[]>('/experiments'),
  runExperiment: (datasetId: string, manifest: Partial<ExperimentRecord['manifest']> = {}) =>
    request<ExperimentRecord>(
      '/experiments',
      json({
        manifest: {
          dataset_id: datasetId,
          vector_stores: ['faiss', 'chroma', 'qdrant'],
          profiles: ['hybrid'],
          top_k: 10,
          track: 'portable',
          ...manifest,
        },
      }),
    ),
  experimentReport: (experimentId: string, format: 'json' | 'markdown' | 'csv') =>
    requestText(`/experiments/${encodeURIComponent(experimentId)}/report?format=${format}`),
}

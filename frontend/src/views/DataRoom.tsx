import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  type ChunkRecord,
  type IngestionJob,
  type NamedOption,
  type PageGeometry,
  type SourceDetail,
  type SourceVersion,
  api,
} from '../api'
import { chunkPage } from '../bbox'
import { followIngestion } from '../sse'
import { Badge, Callout, EmptyState, StateBanner, formatMetadataValue } from '../ui'

/**
 * pdf.js is ~1 MB of the production bundle on its own — more than the rest of the
 * application put together. Loading it only when a paginated original is actually
 * opened keeps the initial payload to the shell, which matters because four of the
 * five views never render a PDF at all.
 */
const PdfViewer = lazy(() => import('../PdfViewer'))

type Props = {
  sources: SourceVersion[]
  parserModes: NamedOption[]
  chunkProfiles: NamedOption[]
  onSourcesChanged: () => void
  /** Chunk to open on mount — set when a citation is followed from the Research Workspace. */
  focus?: { sourceId: string; chunkId?: string; page?: number } | null
  onFocusConsumed?: () => void
}

const stageStatusIcon: Record<string, string> = {
  running: '◐',
  completed: '✓',
  failed: '✕',
  skipped: '–',
}

export function DataRoom({ sources, parserModes, chunkProfiles, onSourcesChanged, focus, onFocusConsumed }: Props) {
  const [url, setUrl] = useState('')
  const [parser, setParser] = useState('')
  const [chunkProfile, setChunkProfile] = useState('')
  const [job, setJob] = useState<IngestionJob | null>(null)
  const [jobs, setJobs] = useState<IngestionJob[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const [selected, setSelected] = useState<SourceVersion | null>(null)
  const [detail, setDetail] = useState<SourceDetail | null>(null)
  const [pages, setPages] = useState<PageGeometry[]>([])
  const [activeChunk, setActiveChunk] = useState<ChunkRecord | null>(null)
  const [page, setPage] = useState(1)
  const [detailError, setDetailError] = useState('')
  const [reviewNote, setReviewNote] = useState('')
  const stopStream = useRef<(() => void) | null>(null)

  const refreshJobs = useCallback(() => {
    api.jobs().then(setJobs).catch(() => undefined)
  }, [])

  useEffect(() => {
    refreshJobs()
    return () => stopStream.current?.()
  }, [refreshJobs])

  /**
   * Follow a submitted job's server-sent events (design §8 step 14). The terminal
   * frame is what refreshes the source registry, so the table never shows a version
   * whose publication stage has not actually finished.
   */
  const follow = useCallback(
    (submitted: IngestionJob) => {
      setJob(submitted)
      stopStream.current?.()
      stopStream.current = followIngestion(
        api.ingestionEventsUrl(submitted.job_id),
        (event) => {
          setJob(event.job)
          if (event.type === 'completed' || event.type === 'failed') {
            setBusy(false)
            onSourcesChanged()
            refreshJobs()
          }
        },
        (reason) => {
          setBusy(false)
          setError(`Lost the ingestion progress stream: ${reason.message}. The job itself is still running — reopen the Data Room to see its result.`)
        },
      )
    },
    [onSourcesChanged, refreshJobs],
  )

  const submit = (start: () => Promise<IngestionJob>) => {
    setBusy(true)
    setError('')
    start()
      .then(follow)
      .catch((reason: unknown) => {
        setBusy(false)
        setError(reason instanceof Error ? reason.message : 'Ingestion could not be started.')
      })
  }

  const inspect = useCallback((source: SourceVersion, openChunkId?: string) => {
    setSelected(source)
    setDetail(null)
    setPages([])
    setActiveChunk(null)
    setDetailError('')
    setReviewNote('')
    api
      .sourceDetail(source.source_id)
      .then((loaded) => {
        setDetail(loaded)
        const opened = openChunkId ? loaded.chunks.find((chunk) => chunk.chunk_id === openChunkId) : undefined
        if (opened) {
          setActiveChunk(opened)
          setPage(chunkPage(opened) ?? 1)
        } else {
          setPage(1)
        }
      })
      .catch((reason: unknown) => setDetailError(reason instanceof Error ? reason.message : 'Source preview could not be loaded.'))
    api.sourcePages(source.source_id).then(setPages).catch(() => setPages([]))
  }, [])

  // A citation opened from the Research Workspace lands here with the exact chunk.
  useEffect(() => {
    if (!focus) return
    const source = sources.find((item) => item.source_id === focus.sourceId)
    if (source) {
      inspect(source, focus.chunkId)
      onFocusConsumed?.()
    }
  }, [focus, sources, inspect, onFocusConsumed])

  const openChunk = (chunk: ChunkRecord) => {
    setActiveChunk(chunk)
    const target = chunkPage(chunk)
    if (target) setPage(target)
  }

  const reprocess = () => {
    if (!selected) return
    const body: { parser?: string; chunk_profile?: string } = {}
    if (parser) body.parser = parser
    if (chunkProfile) body.chunk_profile = chunkProfile
    if (!body.parser && !body.chunk_profile) {
      setError('Choose a different parser mode or chunk profile before reprocessing.')
      return
    }
    submit(() => api.reprocess(selected.source_id, body))
  }

  const review = (decision: 'approve' | 'reject') => {
    if (!selected) return
    api
      .review(selected.source_id, decision, reviewNote)
      .then((updated) => {
        setSelected(updated)
        onSourcesChanged()
      })
      .catch((reason: unknown) => setDetailError(reason instanceof Error ? reason.message : 'Review decision failed.'))
  }

  const isPdf = Boolean(selected?.media_type.endsWith('pdf')) && pages.length > 0
  const highlighted = useMemo(() => (activeChunk ? [activeChunk] : []), [activeChunk])
  const resumable = jobs.filter((item) => item.resumable)

  return (
    <>
      <section className="page-intro">
        <div>
          <span className="eyebrow">Ingestion / provenance</span>
          <h1>Data Room</h1>
          <p>Bring in mixed-format research material and compare the original page against what the parser actually recovered.</p>
        </div>
        <div className="form-row">
          <select value={parser} onChange={(event) => setParser(event.target.value)} data-testid="parser-select" aria-label="Parser mode">
            <option value="">Parser: default routing</option>
            {parserModes.map((mode) => (
              <option key={mode.name} value={mode.name}>
                Parser: {mode.name}
              </option>
            ))}
          </select>
          <select value={chunkProfile} onChange={(event) => setChunkProfile(event.target.value)} data-testid="chunk-profile-select" aria-label="Chunk profile">
            <option value="">Chunks: default (structure-aware)</option>
            {chunkProfiles.map((profile) => (
              <option key={profile.name} value={profile.name}>
                Chunks: {profile.name}
              </option>
            ))}
          </select>
          <label className="primary-button">
            {busy ? 'Processing…' : '+ Import source'}
            <input
              type="file"
              hidden
              data-testid="file-input"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) submit(() => api.ingest(file, { parser: parser || undefined, chunkProfile: chunkProfile || undefined }))
                event.target.value = ''
              }}
              disabled={busy}
            />
          </label>
          <input
            className="url-input"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://…"
            data-testid="url-input"
            disabled={busy}
          />
          <button
            className="secondary-button"
            data-testid="url-submit"
            disabled={busy || !url}
            onClick={() => {
              submit(() => api.ingestUrl(url))
              setUrl('')
            }}
          >
            Import URL
          </button>
        </div>
      </section>

      {error ? (
        <StateBanner kind="error" title="Ingestion could not start">
          {error}
        </StateBanner>
      ) : null}

      {job ? <JobProgress job={job} onResume={(id) => api.resumeJob(id).then(follow).catch(() => undefined)} /> : null}

      {resumable.length > 0 ? (
        <StateBanner
          kind="partial"
          title={`${resumable.length} ingestion job${resumable.length === 1 ? '' : 's'} did not finish cleanly`}
          actions={
            <button
              className="secondary-button"
              data-testid="resume-latest"
              onClick={() => api.resumeJob(resumable[0].job_id).then(follow).catch((reason: unknown) => setError(String(reason)))}
            >
              Resume “{resumable[0].label}”
            </button>
          }
        >
          Completed stages are retained; resuming re-runs the pipeline from the staged artifact rather than re-uploading it.
        </StateBanner>
      ) : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Source registry</span>
            <h2>Immutable source versions</h2>
          </div>
          <span className="muted">{sources.length} artifacts</span>
        </div>
        {sources.length === 0 ? (
          <EmptyState title="Drop a source to begin" icon="+">
            PDF, DOCX, PPTX, XLSX, CSV, HTML, JSON, Markdown, TXT, EML and Outlook MSG are supported.
          </EmptyState>
        ) : (
          <div className="table-wrap">
            <table data-testid="source-table">
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Parser</th>
                  <th>Quality</th>
                  <th>Elements</th>
                  <th>Chunks</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {sources.map((source) => (
                  <tr
                    key={source.version_id || source.source_id}
                    className="clickable-row"
                    data-testid="source-row"
                    data-source-id={source.source_id}
                    data-status={source.status}
                    onClick={() => inspect(source)}
                  >
                    <td>
                      <strong>{source.name}</strong>
                      <span className="table-sub">{source.version_id || source.source_id}</span>
                    </td>
                    <td>
                      <code>{source.parser}</code>
                    </td>
                    <td>
                      <div className="quality">
                        <span style={{ width: `${Math.round(source.quality_score * 100)}%` }} />
                        <b>{Math.round(source.quality_score * 100)}%</b>
                      </div>
                    </td>
                    <td>{source.element_count}</td>
                    <td>{source.chunk_count}</td>
                    <td>
                      {/* `failed` outranks `requires_review`: the ingestion error
                          handler sets both, but a failed parse recovered nothing to
                          review — `POST /review` refuses it — so labelling it
                          "review required" would offer an action that does not exist. */}
                      <Badge tone={source.queryable ? 'good' : source.status === 'failed' ? 'bad' : 'warn'}>
                        {source.status === 'failed'
                          ? 'failed'
                          : source.queryable
                            ? 'queryable'
                            : source.requires_review
                              ? 'review required'
                              : source.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selected ? (
        <section className="panel source-detail" data-testid="source-detail">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Original page vs extracted structure</span>
              <h2>{selected.name}</h2>
            </div>
            <div className="form-row">
              <button className="secondary-button" data-testid="reprocess" onClick={reprocess}>
                Reprocess with selected parser / profile
              </button>
              <a className="secondary-button" href={api.artifactContentUrl(selected.source_id, page)} target="_blank" rel="noreferrer">
                Open raw artifact
              </a>
              <button
                className="secondary-button"
                onClick={() => {
                  setSelected(null)
                  setDetail(null)
                }}
              >
                Close
              </button>
            </div>
          </div>

          {selected.status === 'failed' ? (
            <StateBanner kind="error" title="This version failed to parse">
              {selected.warnings.join(' · ')}
            </StateBanner>
          ) : null}

          {selected.requires_review ? (
            <StateBanner
              kind="review"
              title="Low-confidence extraction — review required before it becomes evidence"
              actions={
                <>
                  <input
                    className="url-input"
                    placeholder="Reviewer note"
                    value={reviewNote}
                    data-testid="review-note"
                    onChange={(event) => setReviewNote(event.target.value)}
                  />
                  <button className="primary-button" data-testid="review-approve" onClick={() => review('approve')}>
                    Accept as evidence
                  </button>
                  <button className="secondary-button" data-testid="review-reject" onClick={() => review('reject')}>
                    Withhold
                  </button>
                </>
              }
            >
              {selected.warnings.join(' · ') || 'This version is inspectable but is withheld from retrieval until an operator accepts it.'}
            </StateBanner>
          ) : null}

          {!selected.requires_review && selected.status === 'partial' ? (
            <StateBanner kind="partial" title="Partial success — some content was recovered">
              {selected.warnings.join(' · ')}
            </StateBanner>
          ) : null}

          {detailError ? <Callout tone="bad">{detailError}</Callout> : null}

          <div className="dataroom-split">
            <div className="dataroom-original">
              <h3 className="panel-subheading">Original</h3>
              {isPdf ? (
                <Suspense fallback={<StateBanner kind="progress" title="Loading the PDF renderer…" />}>
                  <PdfViewer
                    url={api.artifactContentUrl(selected.source_id)}
                    page={page}
                    pages={pages}
                    highlighted={highlighted}
                    onPageChange={setPage}
                  />
                </Suspense>
              ) : (
                <EmptyState title="No page image for this format" icon="▤">
                  {selected.media_type} has no paginated original to render. The extracted structure on the right is the
                  authoritative view, and the raw artifact is one click away.
                </EmptyState>
              )}
            </div>
            <div className="dataroom-extracted">
              <h3 className="panel-subheading">
                Extracted structure
                {detail ? <span className="muted"> · {detail.chunks.length} chunks</span> : null}
              </h3>
              {!detail ? (
                <StateBanner kind="progress" title="Loading the extracted structure…" />
              ) : detail.chunks.length === 0 ? (
                <EmptyState title="The parser recovered no content">Nothing was extracted from this artifact.</EmptyState>
              ) : (
                <div className="chunk-list" data-testid="chunk-list">
                  {detail.chunks.map((chunk) => {
                    const target = chunkPage(chunk)
                    return (
                      <article
                        key={chunk.chunk_id}
                        className={activeChunk?.chunk_id === chunk.chunk_id ? 'chunk-card chunk-card-active' : 'chunk-card'}
                        data-testid="chunk-card"
                        data-chunk-id={chunk.chunk_id}
                        onClick={() => openChunk(chunk)}
                      >
                        <div className="chunk-card-head">
                          <code>{chunk.chunk_id}</code>
                          <Badge tone="neutral">{chunk.profile}</Badge>
                          {target ? <span className="table-sub">page {target}</span> : null}
                        </div>
                        <p>{chunk.text}</p>
                        <span className="table-sub">
                          {Object.entries(chunk.metadata)
                            .filter(([key]) => key !== 'source_locations')
                            .map(([key, value]) => `${key}: ${formatMetadataValue(value)}`)
                            .join(' · ') || 'no location metadata'}
                        </span>
                      </article>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </section>
      ) : null}
    </>
  )
}

function JobProgress({ job, onResume }: { job: IngestionJob; onResume: (jobId: string) => void }) {
  const kind = job.status === 'failed' ? 'error' : job.status === 'completed' ? 'partial' : 'progress'
  return (
    <section className="panel job-panel" data-testid="job-panel" data-job-status={job.status}>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Ingestion progress · server-sent events</span>
          <h2>{job.label}</h2>
        </div>
        <Badge tone={job.status === 'failed' ? 'bad' : job.status === 'completed' ? 'good' : 'neutral'} testId="job-status">
          {job.status}
        </Badge>
      </div>
      <ol className="stage-list" data-testid="stage-list">
        {job.stages.map((stage) => (
          <li key={stage.name} className={`stage stage-${stage.status}`} data-testid="stage" data-stage={stage.name} data-status={stage.status}>
            <span className="stage-icon">{stageStatusIcon[stage.status]}</span>
            <div>
              <strong>{stage.name}</strong>
              <span className="table-sub">{stage.detail || '—'}</span>
            </div>
            <span className="row-state">{stage.status === 'running' ? 'running' : `${Math.round(stage.elapsed_ms)} ms`}</span>
          </li>
        ))}
        {job.stages.length === 0 ? <li className="stage stage-running">Waiting for the first stage…</li> : null}
      </ol>
      {job.status === 'failed' ? (
        <StateBanner
          kind="error"
          title="Ingestion failed — completed stages are retained"
          actions={
            job.resumable ? (
              <button className="secondary-button" data-testid="job-resume" onClick={() => onResume(job.job_id)}>
                Resume
              </button>
            ) : undefined
          }
        >
          {job.error}
        </StateBanner>
      ) : null}
      {job.status === 'running' || job.status === 'queued' ? (
        <StateBanner kind={kind as 'progress'} title="Streaming stage transitions from the ingestion pipeline" />
      ) : null}
    </section>
  )
}

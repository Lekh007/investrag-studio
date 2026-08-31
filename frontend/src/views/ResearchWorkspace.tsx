import { useState } from 'react'
import { type CollectionRow, type Health, type QueryResponse, api } from '../api'
import { generationBadge, generationProvenance, generationSummary } from '../generation'
import { Badge, EmptyState, StateBanner, storeAvailable } from '../ui'

type Props = {
  stores: Health['available_vector_stores']
  collections: CollectionRow[]
  onOpenCitation: (focus: { sourceId: string; chunkId: string; page?: number }) => void
}

export function ResearchWorkspace({ stores, collections, onOpenCitation }: Props) {
  const [question, setQuestion] = useState('What does the indexed corpus say about revenue growth and risk?')
  const [profile, setProfile] = useState('hybrid')
  const [vectorStore, setVectorStore] = useState('faiss')
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const health = new Map(collections.map((row) => [row.name, row]))
  const availableStores = stores.filter(storeAvailable)
  const disabled = stores.filter((store) => !storeAvailable(store))

  const ask = () => {
    setBusy(true)
    setError('')
    api
      .query(question, profile, vectorStore)
      .then(setResult)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : 'Query failed.'))
      .finally(() => setBusy(false))
  }

  const badge = result ? generationBadge(result) : null

  return (
    <>
      <section className="page-intro">
        <div>
          <span className="eyebrow">Research / cited answer</span>
          <h1>Research Workspace</h1>
          <p>Ask a question, inspect citation-label validation, and open the exact page the evidence came from.</p>
        </div>
        <Badge tone="good">evidence required</Badge>
      </section>

      {disabled.length ? (
        <StateBanner kind="unavailable" title={`${disabled.length} database${disabled.length === 1 ? '' : 's'} unavailable`}>
          {disabled.map((store) => `${store.name}: ${health.get(store.name)?.reason ?? store.description}`).join(' · ')}
        </StateBanner>
      ) : null}

      <section className="research-grid">
        <div className="panel query-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Question</span>
              <h2>Ask the corpus</h2>
            </div>
            <div className="form-row">
              <select value={profile} onChange={(event) => setProfile(event.target.value)} data-testid="research-profile">
                <option value="hybrid">Hybrid + RRF</option>
                <option value="dense">Dense similarity</option>
                <option value="mmr">MMR diversity</option>
                <option value="parent">Parent-aware expansion</option>
                <option value="multi-query">Multi-query expansion</option>
              </select>
              <select value={vectorStore} onChange={(event) => setVectorStore(event.target.value)} data-testid="research-store">
                {availableStores.map((store) => (
                  <option key={store.name} value={store.name}>
                    {store.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} data-testid="research-question" />
          <button className="primary-button submit-button" data-testid="research-run" disabled={busy} onClick={ask}>
            {busy ? 'Retrieving evidence…' : 'Run cited query  →'}
          </button>
          {error ? <StateBanner kind="error" title="Query failed">{error}</StateBanner> : null}
          <div className="query-hints">
            <span>Try asking:</span>
            <button onClick={() => setQuestion('Which documents contain numerical evidence of changing margins?')}>numerical evidence</button>
            <button data-testid="hint-abstention" onClick={() => setQuestion('What is the current price of gold in Zurich?')}>
              abstention behaviour
            </button>
          </div>
        </div>

        <div className="panel answer-panel" data-testid="answer-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Answer / citation-label check</span>
              <h2>{result?.insufficient_evidence ? 'Insufficient evidence' : result ? 'Cited response' : 'Waiting for a question'}</h2>
            </div>
            {badge ? <Badge tone={badge.tone} testId="answer-badge">{badge.label}</Badge> : null}
          </div>

          {busy ? <StateBanner kind="progress" title="Retrieving and grounding an answer…" /> : null}

          {result ? (
            <>
              {result.insufficient_evidence ? (
                <StateBanner kind="abstained" title="Abstained — no indexed evidence matched this question">
                  The evidence gate found nothing that supports an answer, so no model call was made. This is a designed
                  outcome, not a failure.
                </StateBanner>
              ) : null}

              {result.answer_withheld ? (
                <StateBanner kind="verifier-defect" title="Verifier defect — the answer was withheld">
                  The generated response did not resolve to the supplied citation labels, so it is not shown. The retrieved
                  evidence is listed below and is still inspectable.
                </StateBanner>
              ) : null}

              {result.conflicting_evidence ? (
                <StateBanner kind="conflict" title="Conflicting evidence in the citation set">
                  Two cited passages disagree on a numeric value for the same subject. Both are shown; neither is silently
                  preferred.
                </StateBanner>
              ) : null}

              {result.trace.generation_mode === 'extractive-fallback' ? (
                <StateBanner kind="degraded" title="Degraded model — this is retrieved text, not a generated answer">
                  {result.trace.generation_fallback_reason ?? 'the local generation model was unavailable'}
                </StateBanner>
              ) : null}

              <div className="answer-copy" data-testid="answer-copy">
                {result.answer}
              </div>

              {result.validator_messages.length ? (
                <ul className="validator-list" data-testid="validator-messages">
                  {result.validator_messages.map((message) => (
                    <li key={message}>{message}</li>
                  ))}
                </ul>
              ) : null}

              <div className="citation-list" data-testid="citation-list">
                {result.citations.map((citation) => (
                  <button
                    className="citation-card"
                    key={citation.label}
                    data-testid="citation-card"
                    data-chunk-id={citation.chunk_id}
                    onClick={() =>
                      onOpenCitation({ sourceId: citation.source_id, chunkId: citation.chunk_id, page: citation.page ?? undefined })
                    }
                  >
                    <span className="citation-label">{citation.label}</span>
                    <div>
                      <strong>{citation.source_name}</strong>
                      <span className="table-sub">
                        {citation.page
                          ? `Page ${citation.page}`
                          : citation.slide
                            ? `Slide ${citation.slide}`
                            : citation.sheet
                              ? `${citation.sheet} · ${citation.cell_range ?? 'range'}`
                              : (citation.json_path ?? 'source excerpt')}{' '}
                        · open in the Data Room with its bounding box highlighted
                      </span>
                      <p>{citation.excerpt}</p>
                    </div>
                  </button>
                ))}
              </div>

              <div className="trace-strip">
                <span>retrieval {result.trace.retrieval_ms} ms</span>
                <span>generation {result.trace.generation_ms} ms</span>
                <span>{generationSummary(result.trace)}</span>
                <span>{result.trace.final_context_chunks} context chunks</span>
                <span>{generationProvenance(result.trace)}</span>
              </div>
            </>
          ) : busy ? null : (
            <EmptyState title="Your answer will appear here">
              Citation cards open the Data Room at the exact page, with the extracted element's bounding box drawn over
              the original.
            </EmptyState>
          )}
        </div>
      </section>
    </>
  )
}

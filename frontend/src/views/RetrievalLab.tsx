import { useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { type CollectionRow, type Health, api } from '../api'
import { type LabRun, candidateStages, compareRuns, rankMovement, stageLabel, survivalSeries } from '../compare'
import { Badge, EmptyState, StateBanner, storeAvailable } from '../ui'

const PROFILES = ['hybrid', 'dense', 'mmr', 'parent', 'multi-query'] as const
const SERIES_COLOURS = ['#e37c45', '#274c77', '#3a966e', '#8a5fbf', '#b3453b', '#2f7f92']

type Props = {
  stores: Health['available_vector_stores']
  collections: CollectionRow[]
}

/**
 * Design §15.4: run the same query against several databases or profiles at once and
 * compare the stages side by side. Fan-out happens in the browser against the real
 * `POST /queries` endpoint, one request per selection, so every column is a genuine
 * round trip through that store's own adapter — not one query replayed.
 */
export function RetrievalLab({ stores, collections }: Props) {
  const [question, setQuestion] = useState('What does the indexed corpus say about revenue growth and risk?')
  const [selectedStores, setSelectedStores] = useState<string[]>(['faiss'])
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>(['hybrid'])
  const [runs, setRuns] = useState<LabRun[]>([])
  const [busy, setBusy] = useState(false)
  const [focus, setFocus] = useState<string | null>(null)

  const health = new Map(collections.map((row) => [row.name, row]))
  const available = stores.filter(storeAvailable)
  const disabled = stores.filter((store) => !storeAvailable(store))

  const toggle = (list: string[], value: string, setter: (next: string[]) => void) =>
    setter(list.includes(value) ? list.filter((item) => item !== value) : [...list, value])

  const run = () => {
    const combinations = selectedStores.flatMap((store) => selectedProfiles.map((profile) => ({ store, profile })))
    if (!combinations.length) return
    setBusy(true)
    setFocus(null)
    const pending: LabRun[] = combinations.map(({ store, profile }) => ({
      key: `${store}·${profile}`,
      vectorStore: store,
      profile,
      status: 'running',
    }))
    setRuns(pending)
    void Promise.all(
      pending.map(async (item): Promise<LabRun> => {
        const started = performance.now()
        try {
          const response = await api.query(question, item.profile, item.vectorStore)
          return { ...item, status: 'ok', response, wallMs: performance.now() - started }
        } catch (reason: unknown) {
          return {
            ...item,
            status: 'error',
            error: reason instanceof Error ? reason.message : 'query failed',
            wallMs: performance.now() - started,
          }
        }
      }),
    ).then((settled) => {
      setRuns(settled)
      setBusy(false)
      setFocus(settled.find((item) => item.status === 'ok')?.key ?? null)
    })
  }

  const completed = runs.filter((item) => item.status === 'ok')
  const comparison = compareRuns(runs)
  const focused = completed.find((item) => item.key === focus) ?? completed[0]

  return (
    <>
      <section className="page-intro">
        <div>
          <span className="eyebrow">Diagnostics / ranking trace</span>
          <h1>Retrieval Lab</h1>
          <p>Run one question against several databases and profiles at once, then compare dense, lexical, fused and reranked stages side by side.</p>
        </div>
        <Badge>{completed.length ? `${completed.length} measured runs` : 'no query yet'}</Badge>
      </section>

      {disabled.length ? (
        <StateBanner kind="unavailable" title={`${disabled.length} database${disabled.length === 1 ? ' is' : 's are'} unavailable and cannot be selected`}>
          {disabled
            .map((store) => `${store.name}: ${health.get(store.name)?.reason ?? store.description ?? 'no health detail reported'}`)
            .join(' · ')}
        </StateBanner>
      ) : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Comparison setup</span>
            <h2>Select databases and profiles</h2>
          </div>
          <span className="muted">{selectedStores.length * selectedProfiles.length} runs per query</span>
        </div>
        <div className="chip-row" data-testid="store-chips">
          {stores.map((store) => {
            const ok = storeAvailable(store)
            const reason = health.get(store.name)?.reason
            return (
              <button
                key={store.name}
                className={selectedStores.includes(store.name) ? 'chip chip-on' : 'chip'}
                data-testid="store-chip"
                data-store={store.name}
                data-disabled={!ok}
                disabled={!ok}
                title={ok ? store.description : `unavailable — ${reason ?? store.description}`}
                onClick={() => toggle(selectedStores, store.name, setSelectedStores)}
              >
                {store.name}
                {ok ? null : <span className="chip-reason"> · unavailable</span>}
              </button>
            )
          })}
        </div>
        <div className="chip-row" data-testid="profile-chips">
          {PROFILES.map((profile) => (
            <button
              key={profile}
              className={selectedProfiles.includes(profile) ? 'chip chip-on' : 'chip'}
              data-testid="profile-chip"
              data-profile={profile}
              onClick={() => toggle(selectedProfiles, profile, setSelectedProfiles)}
            >
              {profile}
            </button>
          ))}
        </div>
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} data-testid="lab-question" rows={3} />
        <button className="primary-button submit-button" data-testid="lab-run" disabled={busy || !selectedStores.length} onClick={run}>
          {busy ? 'Running across selected databases…' : `Compare across ${available.length ? selectedStores.length : 0} database(s) →`}
        </button>
      </section>

      {busy ? <StateBanner kind="progress" title="Querying each selected database…" /> : null}

      {!runs.length && !busy ? (
        <EmptyState title="No measured retrieval comparison yet">
          Pick two or more databases and run a question. This view intentionally shows no illustrative performance numbers.
        </EmptyState>
      ) : null}

      {completed.length ? (
        <>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Measured pipeline</span>
                <h2>Candidate survival by stage</h2>
              </div>
              <span className="muted">one line per database · profile</span>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={survivalSeries(completed)} margin={{ top: 12, right: 16, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6eaf0" vertical={false} />
                <XAxis dataKey="stage" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {completed.map((item, index) => (
                  <Line
                    key={item.key}
                    type="monotone"
                    dataKey={item.key}
                    stroke={SERIES_COLOURS[index % SERIES_COLOURS.length]}
                    strokeWidth={2.5}
                    dot={{ r: 4 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </section>

          <section className="panel" data-testid="store-comparison">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Side-by-side</span>
                <h2>Which chunk each database surfaced</h2>
              </div>
              <span className="muted">final rank per run · “—” means that store never surfaced it</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Chunk</th>
                    {completed.map((item) => (
                      <th key={item.key}>{item.key}</th>
                    ))}
                    <th>Agreement</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.map((row) => (
                    <tr key={row.chunkId} data-testid="comparison-row">
                      <td>
                        <code>{row.chunkId}</code>
                      </td>
                      {completed.map((item) => (
                        <td key={item.key} data-testid="comparison-rank">
                          {row.perRun[item.key] === null || row.perRun[item.key] === undefined ? (
                            <span className="muted">—</span>
                          ) : (
                            <b>#{row.perRun[item.key]}</b>
                          )}
                        </td>
                      ))}
                      <td>
                        <Badge tone={row.agreement === completed.length ? 'good' : 'warn'}>
                          {row.agreement}/{completed.length}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="run-latency-row">
              {runs.map((item) => (
                <span key={item.key} data-testid="run-summary" data-status={item.status}>
                  <b>{item.key}</b>{' '}
                  {item.status === 'error'
                    ? `failed — ${item.error}`
                    : `${Math.round(item.wallMs ?? 0)} ms round trip · ${item.response?.trace.retrieval_ms ?? 0} ms retrieval`}
                </span>
              ))}
            </div>
          </section>

          <section className="panel" data-testid="rank-movement">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Rank movement</span>
                <h2>Stage-by-stage, within one run</h2>
              </div>
              <select value={focus ?? ''} onChange={(event) => setFocus(event.target.value)} data-testid="focus-run">
                {completed.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.key}
                  </option>
                ))}
              </select>
            </div>
            {focused?.response ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Chunk</th>
                      {candidateStages(focused.response.trace).map((stage) => (
                        <th key={stage.stage}>{stageLabel(stage.stage)}</th>
                      ))}
                      <th>Movement</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rankMovement(focused.response.trace).map((row) => (
                      <tr key={row.chunkId} data-testid="movement-row">
                        <td>
                          <code>{row.chunkId}</code>
                        </td>
                        {candidateStages(focused.response!.trace).map((stage) => (
                          <td key={stage.stage}>
                            {row.ranks[stage.stage] === null ? <span className="muted">dropped</span> : <b>#{row.ranks[stage.stage]}</b>}
                          </td>
                        ))}
                        <td>
                          {row.movement === null ? (
                            <span className="muted">—</span>
                          ) : (
                            <Badge tone={row.movement > 0 ? 'good' : row.movement < 0 ? 'warn' : 'neutral'}>
                              {row.movement > 0 ? `▲ ${row.movement}` : row.movement < 0 ? `▼ ${Math.abs(row.movement)}` : '='}
                            </Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            {focused?.response ? (
              <div className="trace-strip">
                <span>rewritten: {focused.response.trace.rewritten_query}</span>
                <span>filters: {JSON.stringify(focused.response.trace.filters)}</span>
                <span>trace id: {focused.response.trace.trace_id}</span>
              </div>
            ) : null}
          </section>
        </>
      ) : null}

      {runs.some((item) => item.status === 'error') ? (
        <StateBanner kind="error" title="Some databases failed for this query">
          {runs
            .filter((item) => item.status === 'error')
            .map((item) => `${item.key}: ${item.error}`)
            .join(' · ')}
        </StateBanner>
      ) : null}
    </>
  )
}

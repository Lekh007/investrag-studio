import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { type ExperimentRecord, type GoldenDataset, type Health, api } from '../api'
import { type ExportFormat, classifyQuestion, compareMetrics, exportExperimentReport } from '../exports'
import { Badge, Callout, EmptyState, Metric, StateBanner, storeAvailable } from '../ui'

type Props = {
  datasets: GoldenDataset[]
  experiments: ExperimentRecord[]
  stores: Health['available_vector_stores']
  onChanged: () => void
}

const FAILURE_TONE = {
  ok: 'good',
  miss: 'bad',
  'low-rank': 'warn',
  'wrong-abstention': 'bad',
  'missed-abstention': 'bad',
} as const

export function EvaluationStudio({ datasets, experiments, stores, onChanged }: Props) {
  const [selectedDataset, setSelectedDataset] = useState('')
  const [datasetText, setDatasetText] = useState('{"dataset_id":"my-review-set","name":"My reviewed questions","questions":[]}')
  const [datasetError, setDatasetError] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [runStores, setRunStores] = useState<string[]>(['faiss', 'chroma', 'qdrant'])
  const [running, setRunning] = useState<AbortController | null>(null)
  const [selectedExperiment, setSelectedExperiment] = useState('')
  const [compareWith, setCompareWith] = useState('')

  const latest = experiments.find((item) => item.experiment_id === selectedExperiment) ?? experiments[0]
  const other = experiments.find((item) => item.experiment_id === compareWith)
  const available = stores.filter(storeAvailable).map((store) => store.name)

  const metric = (suffix: string) => {
    const values = latest?.metrics.filter((item) => item.name.includes(suffix)).map((item) => item.value) ?? []
    return values.length ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3) : '—'
  }

  const saveDataset = () => {
    try {
      const parsed = JSON.parse(datasetText) as GoldenDataset
      setBusy(true)
      api
        .saveGoldenDataset(parsed)
        .then(() => {
          setSelectedDataset(parsed.dataset_id)
          setDatasetError('')
          onChanged()
        })
        .catch((reason: unknown) => setDatasetError(reason instanceof Error ? reason.message : 'Dataset could not be saved.'))
        .finally(() => setBusy(false))
    } catch (reason: unknown) {
      setDatasetError(reason instanceof Error ? `Invalid dataset JSON: ${reason.message}` : 'Invalid dataset JSON.')
    }
  }

  /**
   * `POST /experiments` runs synchronously and returns the finished record. Cancel
   * therefore abandons *this client's* wait, not the server-side run — the banner says
   * exactly that rather than implying a kill the backend does not offer. Resume is the
   * honest counterpart: re-running the same manifest is idempotent by construction
   * (design §9's immutable experiment inputs), so it is offered as a re-launch.
   */
  const runBenchmark = () => {
    const datasetId = selectedDataset || datasets[0]?.dataset_id
    if (!datasetId) return
    const controller = new AbortController()
    setRunning(controller)
    setError('')
    api
      .runExperiment(datasetId, { vector_stores: runStores })
      .then((record) => {
        if (controller.signal.aborted) return
        setSelectedExperiment(record.experiment_id)
        onChanged()
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setError(reason instanceof Error ? reason.message : 'Experiment failed.')
      })
      .finally(() => setRunning((current) => (current === controller ? null : current)))
  }

  const exportReport = (format: ExportFormat) => {
    if (!latest) return
    exportExperimentReport(latest.experiment_id, format).catch((reason: unknown) =>
      setError(reason instanceof Error ? reason.message : 'Export failed.'),
    )
  }

  const chartData =
    latest?.metrics
      .filter((item) => item.name.includes('/ndcg@'))
      .map((item) => ({ name: item.name.split('/')[0], ndcg: item.value })) ?? []

  return (
    <>
      <section className="page-intro">
        <div>
          <span className="eyebrow">Evaluation / reproducibility</span>
          <h1>Evaluation Studio</h1>
          <p>Run deterministic retrieval benchmarks against operator-reviewed judgments, drill into every question, and export the evidence.</p>
        </div>
        <Badge>{latest ? latest.status : 'no run yet'}</Badge>
      </section>

      {error ? <StateBanner kind="error" title="Benchmark error">{error}</StateBanner> : null}
      {datasetError ? <Callout tone="bad">{datasetError}</Callout> : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Golden dataset</span>
            <h2>Review before you measure</h2>
          </div>
          <span className="muted">operator-supplied labels</span>
        </div>
        <p className="muted">
          Question records carry the relevant chunk or element IDs. These judgments are the auditable ground truth for the portable comparison.
        </p>
        <div className="form-row">
          {datasets.length > 0 ? (
            <select
              value={selectedDataset || datasets[0].dataset_id}
              onChange={(event) => setSelectedDataset(event.target.value)}
              data-testid="dataset-select"
            >
              {datasets.map((dataset) => (
                <option key={dataset.dataset_id} value={dataset.dataset_id}>
                  {dataset.name} · {dataset.questions.length} questions
                </option>
              ))}
            </select>
          ) : null}
          <div className="chip-row">
            {stores.map((store) => (
              <button
                key={store.name}
                className={runStores.includes(store.name) ? 'chip chip-on' : 'chip'}
                disabled={!available.includes(store.name)}
                data-testid="eval-store-chip"
                data-store={store.name}
                onClick={() =>
                  setRunStores((current) =>
                    current.includes(store.name) ? current.filter((item) => item !== store.name) : [...current, store.name],
                  )
                }
              >
                {store.name}
              </button>
            ))}
          </div>
          {running ? (
            <button
              className="secondary-button"
              data-testid="cancel-benchmark"
              onClick={() => {
                running.abort()
                setRunning(null)
              }}
            >
              Cancel
            </button>
          ) : (
            <button
              className="primary-button"
              data-testid="run-benchmark"
              disabled={busy || !(selectedDataset || datasets[0]?.dataset_id) || !runStores.length}
              onClick={runBenchmark}
            >
              Run portable benchmark →
            </button>
          )}
        </div>
        {running ? (
          <StateBanner kind="progress" title="Benchmark running">
            The run holds one HTTP request; cancelling stops this client waiting for it and leaves the server-side run to
            finish. Re-launching the same manifest is idempotent, so a cancelled run can simply be launched again.
          </StateBanner>
        ) : null}
        <details>
          <summary>Create or replace a dataset JSON</summary>
          <textarea value={datasetText} onChange={(event) => setDatasetText(event.target.value)} rows={7} data-testid="dataset-json" />
          <button className="secondary-button" onClick={saveDataset} data-testid="save-dataset">
            Save reviewed dataset
          </button>
        </details>
      </section>

      {!latest ? (
        <EmptyState title="No measured experiment yet">
          Save an operator-reviewed dataset, then run the benchmark. Empty or illustrative charts are intentionally omitted.
        </EmptyState>
      ) : (
        <>
          <section className="metric-grid">
            <Metric label="Success@k" value={metric('/success@')} detail={`${latest.experiment_id} · mean across groups`} />
            <Metric label="nDCG@k" value={metric('/ndcg@')} detail="mean across measured store/profile groups" />
            <Metric label="Abstention accuracy" value={metric('/abstention-accuracy')} detail="answerable + unanswerable groups" />
            <Metric label="p95 retrieval" value={`${metric('/retrieval-latency-p95-ms')} ms`} detail="mean of per-group p95 values" />
          </section>

          {latest.warnings.length ? (
            <StateBanner kind="partial" title="Partial success — the run completed with warnings">
              {latest.warnings.join(' · ')}
            </StateBanner>
          ) : null}

          <section className="split-grid">
            <div className="panel panel-tall">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">Measured leaderboard</span>
                  <h2>nDCG by store</h2>
                </div>
                <div className="form-row">
                  {(['csv', 'json', 'markdown'] as ExportFormat[]).map((format) => (
                    <button key={format} className="secondary-button" data-testid={`export-${format}`} onClick={() => exportReport(format)}>
                      Export {format.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
              {chartData.length ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={chartData} margin={{ top: 12, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e6eaf0" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="ndcg" radius={[5, 5, 0, 0]} fill="#274c77" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState title="No nDCG metric in this record" />
              )}
            </div>

            <div className="panel panel-tall">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">Manifest comparison</span>
                  <h2>Against another run</h2>
                </div>
                <select value={compareWith} onChange={(event) => setCompareWith(event.target.value)} data-testid="compare-select">
                  <option value="">Select a run…</option>
                  {experiments
                    .filter((item) => item.experiment_id !== latest.experiment_id)
                    .map((item) => (
                      <option key={item.experiment_id} value={item.experiment_id}>
                        {item.experiment_id}
                      </option>
                    ))}
                </select>
              </div>
              {other ? (
                <div className="table-wrap">
                  <table data-testid="manifest-comparison">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>{latest.experiment_id.slice(0, 10)}</th>
                        <th>{other.experiment_id.slice(0, 10)}</th>
                        <th>Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareMetrics(latest.metrics, other.metrics).map((row) => (
                        <tr key={row.name}>
                          <td>
                            <code>{row.name}</code>
                          </td>
                          <td>{row.left?.toFixed(4) ?? '—'}</td>
                          <td>{row.right?.toFixed(4) ?? '—'}</td>
                          <td>
                            {row.delta === null ? (
                              <span className="muted">—</span>
                            ) : (
                              <Badge tone={row.delta > 0 ? 'good' : row.delta < 0 ? 'warn' : 'neutral'}>{row.delta.toFixed(4)}</Badge>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState title="Pick a second run to diff its manifest metrics" icon="⇄" />
              )}
            </div>
          </section>

          <section className="panel" data-testid="question-drilldown">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Per-question drill-down</span>
                <h2>Failure causes</h2>
              </div>
              <span className="muted">{latest.results.length} question/store runs</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Question</th>
                    <th>Store · profile</th>
                    <th>Success</th>
                    <th>Recall</th>
                    <th>RR</th>
                    <th>nDCG</th>
                    <th>Cause</th>
                  </tr>
                </thead>
                <tbody>
                  {latest.results.map((result, index) => {
                    const verdict = classifyQuestion(result)
                    return (
                      <tr key={`${result.question_id}-${result.vector_store}-${index}`} data-testid="drilldown-row" data-cause={verdict.kind}>
                        <td>
                          <strong>{result.question_id}</strong>
                          <span className="table-sub">{result.retrieval_ms.toFixed(1)} ms</span>
                        </td>
                        <td>
                          <code>
                            {result.vector_store} · {result.profile}
                          </code>
                        </td>
                        <td>{result.success_at_k.toFixed(2)}</td>
                        <td>{result.recall_at_k.toFixed(2)}</td>
                        <td>{result.reciprocal_rank.toFixed(2)}</td>
                        <td>{result.ndcg_at_k.toFixed(3)}</td>
                        <td>
                          <Badge tone={FAILURE_TONE[verdict.kind]}>{verdict.kind}</Badge>
                          <span className="table-sub">{verdict.explanation}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <details>
              <summary>Reproducibility manifest</summary>
              <pre>{JSON.stringify(latest.reproducibility, null, 2)}</pre>
            </details>
          </section>
        </>
      )}
    </>
  )
}

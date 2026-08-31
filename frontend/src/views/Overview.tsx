import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { CollectionRow, ExperimentRecord, Health, SourceVersion } from '../api'
import { Badge, EmptyState, Metric, StateBanner, storeAvailable } from '../ui'

type Props = {
  health: Health | null
  sources: SourceVersion[]
  experiments: ExperimentRecord[]
  collections: CollectionRow[]
}

export function Overview({ health, sources, experiments, collections }: Props) {
  const ready = sources.filter((source) => source.status === 'ready').length
  const review = sources.filter((source) => source.requires_review).length
  const latest = experiments[0]
  const ndcgMetrics = latest?.metrics.filter((metric) => metric.name.includes('/ndcg@')) ?? []
  const benchmarkData = ndcgMetrics.map((metric) => ({ name: metric.name.split('/')[0].toUpperCase(), ndcg: metric.value }))
  const ndcgLabel = ndcgMetrics[0]?.name.split('/').at(-1)?.replace('ndcg', 'nDCG') ?? 'nDCG@k'
  const healthByName = new Map(collections.map((row) => [row.name, row]))
  const unavailable = (health?.available_vector_stores ?? []).filter((store) => !storeAvailable(store))

  return (
    <>
      <section className="page-intro">
        <div>
          <span className="eyebrow">Portfolio intelligence / local control plane</span>
          <h1>Know what the model knows.</h1>
          <p>Trace every answer from source artifact to parser, chunk, vector ranking and cited evidence.</p>
        </div>
        <Badge tone="good">All data stays on this laptop</Badge>
      </section>

      {review > 0 ? (
        <StateBanner kind="review" title={`${review} source version${review === 1 ? '' : 's'} require review`}>
          Low-confidence extractions stay out of retrieval until an operator accepts them in the Data Room.
        </StateBanner>
      ) : null}

      <section className="metric-grid">
        <Metric label="Indexed chunks" value={String(health?.indexed_chunks ?? 0)} detail="FAISS active collection" />
        <Metric label="Source versions" value={String(sources.length)} detail={`${ready} ready · ${sources.length - ready} need review`} />
        <Metric
          label="Embedding model"
          value={health?.embedding_model ?? 'checking'}
          detail={
            health?.embedding_state === 'configured'
              ? 'loads on first embedding request'
              : health?.embedding_fallback
                ? 'offline deterministic fallback'
                : 'active local dense model'
          }
        />
        <Metric label="Active profile" value="Hybrid" detail="dense + BM25 + RRF" />
        <Metric
          label="Resource profile"
          value={health?.resource_profile ?? 'checking'}
          detail={health?.resource_profile === 'interactive' ? 'generation model resident' : 'generation model unloaded'}
        />
      </section>

      <section className="split-grid">
        <div className="panel panel-tall">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Runtime health</span>
              <h2>Services at a glance</h2>
            </div>
            <Badge tone={unavailable.length ? 'warn' : 'good'}>{unavailable.length ? `${unavailable.length} unavailable` : 'all reachable'}</Badge>
          </div>
          <div className="health-list" data-testid="health-list">
            {(health?.available_vector_stores ?? []).map((store) => {
              const available = storeAvailable(store)
              const live = healthByName.get(store.name)
              return (
                <div className="health-row" key={store.name} data-testid="health-row" data-store={store.name} data-available={available}>
                  <span className={`health-icon ${available ? 'health-on' : 'health-off'}`}>{available ? '✓' : '·'}</span>
                  <div>
                    <strong>{store.name}</strong>
                    <span className="muted" data-testid="health-reason">
                      {available ? store.description : (live?.reason ?? store.description ?? 'no health detail reported')}
                    </span>
                  </div>
                  <span className="row-state">{available ? 'ready' : store.implemented ? 'unavailable' : 'planned'}</span>
                </div>
              )
            })}
          </div>
        </div>
        <div className="panel panel-tall">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Evaluation preview</span>
              <h2>Portable leaderboard</h2>
            </div>
            <Badge>{latest ? latest.status : 'no run yet'}</Badge>
          </div>
          {benchmarkData.length ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={benchmarkData} margin={{ top: 12, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6eaf0" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="ndcg" name={ndcgLabel} radius={[5, 5, 0, 0]} fill="#274c77" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No measured benchmark yet">
              Create a reviewed golden dataset and run the portable track from Evaluation Studio.
            </EmptyState>
          )}
          <p className="chart-note">Only measured experiment results appear here; no fixture values are presented as performance claims.</p>
        </div>
      </section>
    </>
  )
}

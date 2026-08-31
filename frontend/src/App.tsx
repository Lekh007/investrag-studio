import { useCallback, useEffect, useState } from 'react'
import {
  type CollectionRow,
  type ExperimentRecord,
  type GoldenDataset,
  type Health,
  type NamedOption,
  type SourceVersion,
  api,
} from './api'
import { StateBanner } from './ui'
import { DataRoom } from './views/DataRoom'
import { EvaluationStudio } from './views/EvaluationStudio'
import { Overview } from './views/Overview'
import { ResearchWorkspace } from './views/ResearchWorkspace'
import { RetrievalLab } from './views/RetrievalLab'

type Tab = 'overview' | 'data-room' | 'research' | 'retrieval' | 'evaluation'

const tabs: { id: Tab; label: string; eyebrow: string; icon: string }[] = [
  { id: 'overview', label: 'Overview', eyebrow: 'Control plane', icon: '◈' },
  { id: 'data-room', label: 'Data Room', eyebrow: 'Ingestion', icon: '▦' },
  { id: 'research', label: 'Research Workspace', eyebrow: 'Cited answers', icon: '⌕' },
  { id: 'retrieval', label: 'Retrieval Lab', eyebrow: 'Rank diagnostics', icon: '↗' },
  { id: 'evaluation', label: 'Evaluation Studio', eyebrow: 'Evidence quality', icon: '▥' },
]

function Header({ health, onRefresh }: { health: Health | null; onRefresh: () => void }) {
  const embeddingStatus = !health
    ? 'checking embeddings'
    : health.embedding_state === 'configured'
      ? `${health.embedding_model} configured`
      : health.embedding_fallback
        ? `${health.embedding_model} fallback`
        : `${health.embedding_model} ready`
  return (
    <header className="topbar">
      <div className="brand-lockup">
        <div className="brand-mark">IR</div>
        <div>
          <div className="brand-name">
            InvestRAG <span>Studio</span>
          </div>
          <div className="brand-subtitle">Evidence-first investment intelligence</div>
        </div>
      </div>
      <div className="topbar-actions">
        <div className="system-status">
          <span className="status-dot" /> local runtime <span className="muted">·</span> {embeddingStatus}
        </div>
        <button className="icon-button" onClick={onRefresh} aria-label="Refresh system status">
          ↻
        </button>
      </div>
    </header>
  )
}

export function App() {
  const [tab, setTab] = useState<Tab>('overview')
  const [health, setHealth] = useState<Health | null>(null)
  const [collections, setCollections] = useState<CollectionRow[]>([])
  const [sources, setSources] = useState<SourceVersion[]>([])
  const [datasets, setDatasets] = useState<GoldenDataset[]>([])
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([])
  const [parserModes, setParserModes] = useState<NamedOption[]>([])
  const [chunkProfiles, setChunkProfiles] = useState<NamedOption[]>([])
  const [error, setError] = useState('')
  const [focus, setFocus] = useState<{ sourceId: string; chunkId?: string; page?: number } | null>(null)

  const refresh = useCallback(() => {
    Promise.all([api.health(), api.sources(), api.goldenDatasets(), api.experiments()])
      .then(([nextHealth, nextSources, nextDatasets, nextExperiments]) => {
        setHealth(nextHealth)
        setSources(nextSources)
        setDatasets(nextDatasets)
        setExperiments(nextExperiments)
        setError('')
      })
      .catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : 'API is not reachable. Start the backend on port 8000.'),
      )
    // §15.6: a store's *reason* for being unavailable comes from /collections, which
    // opens each store for real. It must never block the rest of the shell loading.
    api.collections().then(setCollections).catch(() => setCollections([]))
    api.parserModes().then(setParserModes).catch(() => setParserModes([]))
    api.chunkProfiles().then(setChunkProfiles).catch(() => setChunkProfiles([]))
  }, [])

  useEffect(refresh, [refresh])

  const openCitation = useCallback((next: { sourceId: string; chunkId: string; page?: number }) => {
    setFocus(next)
    setTab('data-room')
  }, [])

  const content =
    tab === 'overview' ? (
      <Overview health={health} sources={sources} experiments={experiments} collections={collections} />
    ) : tab === 'data-room' ? (
      <DataRoom
        sources={sources}
        parserModes={parserModes}
        chunkProfiles={chunkProfiles}
        onSourcesChanged={refresh}
        focus={focus}
        onFocusConsumed={() => setFocus(null)}
      />
    ) : tab === 'research' ? (
      <ResearchWorkspace
        stores={health?.available_vector_stores ?? []}
        collections={collections}
        onOpenCitation={openCitation}
      />
    ) : tab === 'retrieval' ? (
      <RetrievalLab stores={health?.available_vector_stores ?? []} collections={collections} />
    ) : (
      <EvaluationStudio
        datasets={datasets}
        experiments={experiments}
        stores={health?.available_vector_stores ?? []}
        onChanged={refresh}
      />
    )

  return (
    <div className="app-shell">
      <Header health={health} onRefresh={refresh} />
      <div className="app-body">
        <aside className="sidebar">
          <div className="sidebar-label">Workspace</div>
          <nav>
            {tabs.map((item) => (
              <button
                key={item.id}
                className={tab === item.id ? 'nav-item active' : 'nav-item'}
                data-testid={`nav-${item.id}`}
                onClick={() => {
                  setTab(item.id)
                  setError('')
                }}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>
                  <b>{item.label}</b>
                  <small>{item.eyebrow}</small>
                </span>
              </button>
            ))}
          </nav>
          <div className="sidebar-footer">
            <div className="profile-dot">LK</div>
            <div>
              <strong>Local operator</strong>
              <span>localhost only</span>
            </div>
          </div>
        </aside>
        <main className="main-content">
          {error ? (
            <StateBanner kind="error" title="The InvestRAG API is not reachable">
              {error}
            </StateBanner>
          ) : null}
          {content}
        </main>
      </div>
    </div>
  )
}

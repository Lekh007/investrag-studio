import type { ReactNode } from 'react'

export type Tone = 'good' | 'warn' | 'bad' | 'neutral'

export function Badge({ children, tone = 'neutral', testId }: { children: ReactNode; tone?: Tone; testId?: string }) {
  return (
    <span className={`badge badge-${tone}`} data-testid={testId}>
      {children}
    </span>
  )
}

export function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="metric-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <span className="muted">{detail}</span>
    </div>
  )
}

/**
 * Design §15.6 requires progress, empty, error, partial-success and degraded-model to
 * be *designed* states rather than incidental blanks. They share one component so a
 * new view cannot accidentally ship an undesigned one.
 */
export type StateKind =
  | 'progress'
  | 'empty'
  | 'error'
  | 'partial'
  /** A *database* is unavailable — design §15.6's first bullet. */
  | 'unavailable'
  /** The *generation model* is unavailable and the answer is retrieved text. */
  | 'degraded'
  | 'review'
  | 'abstained'
  | 'verifier-defect'
  | 'conflict'

const STATE_TONE: Record<StateKind, Tone> = {
  progress: 'neutral',
  empty: 'neutral',
  error: 'bad',
  partial: 'warn',
  unavailable: 'warn',
  degraded: 'warn',
  review: 'warn',
  abstained: 'warn',
  'verifier-defect': 'bad',
  conflict: 'warn',
}

const STATE_ICON: Record<StateKind, string> = {
  progress: '◐',
  empty: '+',
  error: '!',
  partial: '◑',
  unavailable: '⊘',
  degraded: '▽',
  review: '⚑',
  abstained: '∅',
  'verifier-defect': '⚠',
  conflict: '⇄',
}

export function StateBanner({
  kind,
  title,
  children,
  actions,
}: {
  kind: StateKind
  title: string
  children?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className={`state-banner state-${kind}`} data-testid={`state-${kind}`} data-state={kind} role={kind === 'error' ? 'alert' : 'status'}>
      <span className="state-icon">{STATE_ICON[kind]}</span>
      <div className="state-body">
        <strong>{title}</strong>
        {children ? <span className="state-detail">{children}</span> : null}
      </div>
      {actions ? <div className="state-actions">{actions}</div> : null}
    </div>
  )
}

export function EmptyState({ title, children, icon = '⌁' }: { title: string; children?: ReactNode; icon?: string }) {
  return (
    <div className="empty-state" data-testid="state-empty" data-state="empty">
      <div className="empty-icon">{icon}</div>
      <strong>{title}</strong>
      <span>{children}</span>
    </div>
  )
}

export function Callout({ tone = 'warn', children }: { tone?: Tone; children: ReactNode }) {
  return <div className={`callout callout-${tone}`}>{children}</div>
}

export function stateTone(kind: StateKind): Tone {
  return STATE_TONE[kind]
}

export function formatMetadataValue(value: unknown): string {
  return typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)
}

/** A store is selectable only when it reports itself genuinely available. */
export function storeAvailable(store: { available?: boolean; implemented: boolean }): boolean {
  return store.available ?? store.implemented
}

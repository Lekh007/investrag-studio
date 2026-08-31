import type { IngestionJob, JobStage } from './api'

/**
 * Ingestion progress arrives as server-sent events (design §8 step 14).
 *
 * `EventSource` is the obvious client, and it is the wrong one here: it cannot be
 * given headers, it reconnects on its own schedule, and — decisively — it gives no
 * way to stop following a stream the moment the job reaches a terminal state, so a
 * finished ingestion would keep an open connection per upload. `fetch` + a reader is
 * a few more lines and stays under our control, including cancellation on unmount.
 */

export type IngestionEvent =
  | { type: 'snapshot'; job: IngestionJob }
  | { type: 'stage'; job: IngestionJob; stage: JobStage }
  | { type: 'completed'; job: IngestionJob }
  | { type: 'failed'; job: IngestionJob }

/**
 * Split a raw SSE byte stream into events.
 *
 * Chunk boundaries fall wherever the network puts them, so a frame routinely arrives
 * split across two reads. The parser therefore keeps a buffer and only emits complete
 * `\n\n`-terminated frames. `:` comment lines (our keep-alives) carry no event.
 */
export class SseParser {
  private buffer = ''

  push(chunk: string): IngestionEvent[] {
    this.buffer += chunk.replace(/\r\n/g, '\n')
    const events: IngestionEvent[] = []
    let boundary = this.buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const frame = this.buffer.slice(0, boundary)
      this.buffer = this.buffer.slice(boundary + 2)
      const parsed = parseFrame(frame)
      if (parsed) events.push(parsed)
      boundary = this.buffer.indexOf('\n\n')
    }
    return events
  }
}

function parseFrame(frame: string): IngestionEvent | null {
  let event = 'message'
  const data: string[] = []
  for (const line of frame.split('\n')) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).trim())
  }
  if (!data.length) return null
  let payload: { job: IngestionJob; stage?: JobStage }
  try {
    payload = JSON.parse(data.join('\n')) as { job: IngestionJob; stage?: JobStage }
  } catch {
    return null
  }
  if (event === 'stage' && payload.stage) return { type: 'stage', job: payload.job, stage: payload.stage }
  if (event === 'snapshot' || event === 'completed' || event === 'failed') {
    return { type: event, job: payload.job }
  }
  return null
}

export const isTerminal = (event: IngestionEvent): boolean =>
  event.type === 'completed' || event.type === 'failed'

/**
 * Follow one ingestion job's event stream. Returns an abort function so a component
 * that unmounts mid-ingestion does not leak the connection.
 */
export function followIngestion(
  url: string,
  onEvent: (event: IngestionEvent) => void,
  onError?: (reason: Error) => void,
): () => void {
  const controller = new AbortController()
  void (async () => {
    try {
      const response = await fetch(url, { signal: controller.signal, headers: { Accept: 'text/event-stream' } })
      if (!response.ok || !response.body) throw new Error(`stream failed (${response.status})`)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      const parser = new SseParser()
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        for (const event of parser.push(decoder.decode(value, { stream: true }))) {
          onEvent(event)
          if (isTerminal(event)) {
            controller.abort()
            return
          }
        }
      }
    } catch (reason) {
      if (controller.signal.aborted) return
      onError?.(reason instanceof Error ? reason : new Error('ingestion stream failed'))
    }
  })()
  return () => controller.abort()
}

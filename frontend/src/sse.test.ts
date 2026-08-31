import { describe, expect, it } from 'vitest'
import { SseParser, isTerminal } from './sse'

const job = (over: Record<string, unknown> = {}) => ({ job_id: 'job_1', status: 'running', stages: [], ...over })

const frame = (event: string, payload: unknown) => `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`

describe('SSE frame parsing', () => {
  it('emits nothing until a frame is complete', () => {
    const parser = new SseParser()
    expect(parser.push('event: stage\ndata: {"job":')).toEqual([])
  })

  it('reassembles a frame split across network chunks', () => {
    const parser = new SseParser()
    const whole = frame('stage', { job: job(), stage: { name: 'parse', status: 'running' } })
    const cut = Math.floor(whole.length / 2)

    expect(parser.push(whole.slice(0, cut))).toEqual([])
    const events = parser.push(whole.slice(cut))

    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'stage', stage: { name: 'parse', status: 'running' } })
  })

  it('emits several frames delivered in one chunk, in order', () => {
    const parser = new SseParser()
    const events = parser.push(
      frame('snapshot', { job: job() }) +
        frame('stage', { job: job(), stage: { name: 'chunk', status: 'completed' } }) +
        frame('completed', { job: job({ status: 'completed' }) }),
    )

    expect(events.map((event) => event.type)).toEqual(['snapshot', 'stage', 'completed'])
  })

  it('ignores keep-alive comments without breaking the stream', () => {
    const parser = new SseParser()
    const events = parser.push(': keep-alive\n\n' + frame('completed', { job: job({ status: 'completed' }) }))

    expect(events.map((event) => event.type)).toEqual(['completed'])
  })

  it('tolerates CRLF line endings', () => {
    const parser = new SseParser()
    const events = parser.push('event: snapshot\r\ndata: {"job":{"job_id":"job_1"}}\r\n\r\n')

    expect(events).toHaveLength(1)
  })

  it('drops an unparseable frame rather than throwing mid-ingestion', () => {
    const parser = new SseParser()
    expect(parser.push('event: stage\ndata: {not json}\n\n')).toEqual([])
    expect(parser.push(frame('completed', { job: job() }))).toHaveLength(1)
  })

  it('treats a stage frame without a stage payload as unusable', () => {
    const parser = new SseParser()
    expect(parser.push(frame('stage', { job: job() }))).toEqual([])
  })

  it('recognises exactly the two terminal event types', () => {
    expect(isTerminal({ type: 'completed', job: job() as never })).toBe(true)
    expect(isTerminal({ type: 'failed', job: job() as never })).toBe(true)
    expect(isTerminal({ type: 'snapshot', job: job() as never })).toBe(false)
  })
})

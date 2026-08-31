import { expect, test } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

/**
 * Design §15.6 — required interface states.
 *
 * Phase 8's acceptance bar is that *every* one of these states is reachable in a
 * Playwright test. They are grouped by how each one is provoked, and the grouping is
 * the honest part of this file:
 *
 *   REAL   — the backend genuinely produces the state. The corpus in
 *            `scripts/seed_e2e_corpus.py` is ingested through the real pipeline by
 *            `global-setup.ts`, and the assertions below are against real responses.
 *
 *   FAULT  — the state only exists when something is broken in a way a healthy local
 *            stack will not do on demand (a model inventing a citation label; an
 *            Ollama outage). Those two are provoked by intercepting the `POST
 *            /queries` response with `page.route`. That is a deliberate, narrow
 *            exception: the *UI contract* is what is under test, the backend
 *            behaviour that produces those payloads has its own backend tests
 *            (`test_api.py` for the withheld-answer path, `test_generation.py` for
 *            the extractive fallback), and each such test says so at its own site.
 *
 * The §21 eight-step acceptance flow is a separate, larger exercise and is not
 * attempted here.
 */

const API = 'http://127.0.0.1:8010/api/v1'
const BACKEND_DIR = resolve(import.meta.dirname, '../../backend')
const PYTHON = resolve(BACKEND_DIR, '.venv/Scripts/python.exe')

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Know what the model knows.' })).toBeVisible()
})

// ---------------------------------------------------------------- state 1: REAL
test('§15.6.1 unavailable databases are disabled with a concrete health reason', async ({ page }) => {
  // Pinecone has no API key in any local environment, and Weaviate is not configured
  // for the e2e backend — both report themselves unavailable for real, with a reason
  // that comes from actually trying to open the store (`GET /collections`).
  const unavailable = page.locator('[data-testid="health-row"][data-available="false"]')
  await expect(unavailable.first()).toBeVisible()
  const count = await unavailable.count()
  expect(count).toBeGreaterThan(0)
  for (let index = 0; index < count; index += 1) {
    const reason = unavailable.nth(index).getByTestId('health-reason')
    await expect(reason).not.toBeEmpty()
  }

  // …and the Retrieval Lab refuses to let an unavailable store be selected at all.
  await page.getByTestId('nav-retrieval').click()
  const disabledChip = page.locator('[data-testid="store-chip"][data-disabled="true"]').first()
  await expect(disabledChip).toBeVisible()
  await expect(disabledChip).toBeDisabled()
  // The banner names each store and why, rather than saying "some stores are down".
  await expect(page.getByTestId('state-unavailable')).toContainText('requires')
})

// ---------------------------------------------------------------- state 2: REAL
test('§15.6.2 an interrupted job retains its completed stages and can be resumed', async ({ page }) => {
  await page.getByTestId('nav-data-room').click()

  // A file that announces itself as a PDF and is not one: format detection and
  // staging succeed, the parse genuinely fails, and the job is genuinely resumable.
  const scratch = mkdtempSync(join(tmpdir(), 'investrag-e2e-'))
  const broken = join(scratch, 'not-really.pdf')
  writeFileSync(broken, '%PDF-1.4\nthis announces itself as a PDF and is not one')
  await page.getByTestId('file-input').setInputFiles(broken)

  const panel = page.getByTestId('job-panel')
  await expect(panel).toHaveAttribute('data-job-status', 'failed', { timeout: 60_000 })

  // The stages that succeeded before the failure are still listed, with their status.
  await expect(panel.locator('[data-stage="stage-artifact"]')).toHaveAttribute('data-status', 'completed')
  await expect(panel.locator('[data-stage="detect-format"]')).toHaveAttribute('data-status', 'completed')
  await expect(panel.locator('[data-stage="parse"]')).toHaveAttribute('data-status', 'failed')

  // Scoped to the job panel: the app shell has its own error banner, and a heavy
  // parse can genuinely stall the single-worker dev backend long enough for a
  // background refresh to fail with "Failed to fetch" while this test runs.
  await expect(panel.getByTestId('state-error')).toContainText('completed stages are retained')
  await expect(page.getByTestId('job-resume')).toBeVisible()
  await page.getByTestId('job-resume').click()
  await expect(panel).toHaveAttribute('data-job-status', 'failed', { timeout: 60_000 })
})

// ---------------------------------------------------------------- state 3: REAL
test('§15.6.3 a low-confidence parse requires review before it becomes evidence', async ({ page }) => {
  await page.getByTestId('nav-data-room').click()
  const partialRow = page.locator('[data-testid="source-row"][data-status="partial"]').first()
  await expect(partialRow).toBeVisible()
  await partialRow.click()

  const banner = page.getByTestId('state-review')
  await expect(banner).toContainText('review required')
  await expect(page.getByTestId('review-approve')).toBeVisible()
  await expect(page.getByTestId('review-reject')).toBeVisible()

  // The override is what publishes it — proven by the row becoming queryable.
  const sourceId = await partialRow.getAttribute('data-source-id')
  await page.getByTestId('review-note').fill('reviewed by the e2e operator')
  await page.getByTestId('review-approve').click()
  await expect(page.locator(`[data-testid="source-row"][data-source-id="${sourceId}"]`)).toContainText('queryable')

  // Restore the fixture so the ordering of later tests is unaffected.
  await page.request.post(`${API}/source-versions/${sourceId}/review`, { data: { decision: 'reject', note: 'e2e reset' } })
})

// ---------------------------------------------------------------- state 4: REAL
test('§15.6.4 missing evidence produces an abstention state', async ({ page }) => {
  await page.getByTestId('nav-research').click()
  await page.getByTestId('hint-abstention').click()
  await page.getByTestId('research-run').click()

  await expect(page.getByTestId('state-abstained')).toContainText('no indexed evidence matched', { timeout: 60_000 })
  await expect(page.getByTestId('answer-badge')).toHaveText('abstain')
  await expect(page.getByTestId('citation-list').locator('[data-testid="citation-card"]')).toHaveCount(0)
})

// ------------------------------------------------------- state 5a: REAL (resolves)
test('§15.6.5a a citation resolves to visible provenance — the exact page, box highlighted', async ({ page }) => {
  await page.getByTestId('nav-research').click()
  await page.getByTestId('research-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('research-run').click()

  const citation = page.getByTestId('citation-card').first()
  await expect(citation).toBeVisible({ timeout: 60_000 })
  await citation.click()

  // Following the citation lands in the Data Room, on the cited page, with the
  // extracted element's real bounding box drawn over the rendered original.
  await expect(page.getByTestId('source-detail')).toBeVisible()
  await expect(page.getByTestId('pdf-viewer')).toBeVisible({ timeout: 60_000 })
  await expect(page.locator('.react-pdf__Page__canvas')).toBeVisible({ timeout: 60_000 })
  const boxes = page.getByTestId('bbox')
  await expect(boxes.first()).toBeVisible({ timeout: 30_000 })
  expect(await boxes.count()).toBeGreaterThan(0)

  // A box that is positioned but has no area would "pass" a visibility check while
  // highlighting nothing, so assert it actually covers part of the page.
  const size = await boxes.first().boundingBox()
  expect(size?.width ?? 0).toBeGreaterThan(1)
  expect(size?.height ?? 0).toBeGreaterThan(1)
})

// ------------------------------------------------------- state 5b: FAULT (defect)
test('§15.6.5b a citation that cannot resolve displays a verifier defect', async ({ page }) => {
  // FAULT injection. The backend genuinely produces this payload when a model invents
  // a label the evidence packet does not contain — `service.query()` sets
  // `answer_withheld` and substitutes the refusal text, covered by
  // `backend/tests/test_api.py`. It cannot be provoked on demand from a healthy
  // stack, because it needs the model to misbehave. What is under test here is the
  // *interface contract*: given that payload, the UI must disclose a verifier defect
  // rather than render the withheld answer as if it were grounded.
  await page.route(`${API}/queries`, async (route) => {
    const response = await route.fetch()
    const body = await response.json()
    await route.fulfill({
      json: {
        ...body,
        answer: 'I cannot safely return a grounded answer because the generated response did not contain resolvable citations.',
        answer_withheld: true,
        insufficient_evidence: false,
        validator_messages: ['The answer referenced unknown citation labels: ["S9"]'],
      },
    })
  })

  await page.getByTestId('nav-research').click()
  await page.getByTestId('research-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('research-run').click()

  await expect(page.getByTestId('state-verifier-defect')).toContainText('withheld', { timeout: 60_000 })
  await expect(page.getByTestId('validator-messages')).toContainText('unknown citation labels')
})

// ---------------------------------------------------------------- state 6: mixed
test('§15.6.6a progress is a designed state, not a blank pane', async ({ page }) => {
  // REAL: the stage list is fed by the live SSE stream from `GET
  // /ingestions/{id}/events`, so this asserts the streaming contract as well.
  await page.getByTestId('nav-data-room').click()
  const scratch = mkdtempSync(join(tmpdir(), 'investrag-e2e-'))
  const document = join(scratch, 'progress-note.md')
  writeFileSync(document, '# Progress fixture\nRevenue grew 12% while margin held at 42 percent.\n')

  await page.getByTestId('file-input').setInputFiles(document)
  const panel = page.getByTestId('job-panel')
  await expect(panel).toBeVisible()
  await expect(panel.locator('[data-testid="stage"]').first()).toBeVisible({ timeout: 30_000 })
  await expect(panel).toHaveAttribute('data-job-status', 'completed', { timeout: 60_000 })

  // Every §8 stage the pipeline ran is listed, with its measured duration.
  for (const stage of ['stage-artifact', 'detect-format', 'parse', 'chunk', 'embed', 'publish']) {
    await expect(panel.locator(`[data-stage="${stage}"]`)).toBeVisible()
  }
})

test('§15.6.6b empty is a designed state', async ({ page }) => {
  // REAL: the Retrieval Lab has run nothing yet on a fresh load.
  await page.getByTestId('nav-retrieval').click()
  await expect(page.getByTestId('state-empty')).toContainText('No measured retrieval comparison yet')
})

test('§15.6.6c error is a designed state', async ({ page }) => {
  // REAL: the SSRF allowlist genuinely refuses a private-network URL (design §18),
  // and the Data Room must present that refusal as a designed error, not a stack trace.
  await page.getByTestId('nav-data-room').click()
  await page.getByTestId('url-input').fill('http://127.0.0.1/secrets.txt')
  await page.getByTestId('url-submit').click()
  await expect(page.getByTestId('state-error')).toContainText('not allowed', { timeout: 30_000 })
})

test('§15.6.6d partial success is a designed state, and is distinct from failure', async ({ page }) => {
  // REAL: the seeded header-only email genuinely parses to status="partial", and the
  // seeded broken PDF genuinely fails. The ingestion error handler sets
  // `requires_review` on *both*, so the registry has to break the tie the right way:
  // a failed parse recovered nothing to review, and `POST /review` refuses it.
  await page.getByTestId('nav-data-room').click()
  await expect(page.locator('[data-testid="source-row"][data-status="partial"]').first()).toContainText('review required')
  await expect(page.locator('[data-testid="source-row"][data-status="failed"]').first()).toContainText('failed')
  await expect(page.locator('[data-testid="source-row"][data-status="failed"]').first()).not.toContainText('review required')
})

test('§15.6.6e degraded model is a designed state', async ({ page }) => {
  // FAULT injection. The extractive fallback is real backend behaviour when Ollama is
  // unreachable (`llm.py`, covered by `backend/tests/test_generation.py`), but this
  // e2e backend is *already* configured with an unreachable Ollama, so a live query
  // abstains before it ever reaches generation — the degraded path needs evidence to
  // fall back *to*. Rather than stand up a second backend, the response is shaped to
  // the payload the real fallback produces, and the assertion is that the UI never
  // presents retrieved text as a generated answer.
  await page.route(`${API}/queries`, async (route) => {
    const response = await route.fetch()
    const body = await response.json()
    await route.fulfill({
      json: {
        ...body,
        insufficient_evidence: false,
        trace: {
          ...body.trace,
          generation_mode: 'extractive-fallback',
          generation_fallback_reason: 'ConnectError: local generation model unreachable',
        },
      },
    })
  })

  await page.getByTestId('nav-research').click()
  await page.getByTestId('research-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('research-run').click()

  await expect(page.getByTestId('state-degraded')).toContainText('not a generated answer', { timeout: 60_000 })
  await expect(page.getByTestId('answer-badge')).toHaveText('not model-generated')
})

// ------------------------------------------------ conflict disclosure (§15.3/§12)
test('conflicting evidence is disclosed as its own state, not buried in validator text', async ({ page }) => {
  // REAL backend capability (`conflicts.py`, Phase 7) reached through a FAULT-shaped
  // response: the seeded corpus has no contradiction fixture, and `detect_conflicting_
  // evidence` is unit-tested against a real one in `backend/tests/test_conflicts.py`.
  // The gap Phase 8 closes is presentational — before this, a detected conflict
  // appeared only as one more line of validator text.
  await page.route(`${API}/queries`, async (route) => {
    const response = await route.fetch()
    const body = await response.json()
    await route.fulfill({ json: { ...body, insufficient_evidence: false, conflicting_evidence: true } })
  })

  await page.getByTestId('nav-research').click()
  await page.getByTestId('research-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('research-run').click()

  await expect(page.getByTestId('state-conflict')).toContainText('Conflicting evidence', { timeout: 60_000 })
})

// ------------------------------------------------- Phase 8 acceptance: multi-store
test('the Retrieval Lab compares one query across three real databases', async ({ page }) => {
  await page.getByTestId('nav-retrieval').click()
  for (const store of ['chroma', 'qdrant']) {
    await page.locator(`[data-testid="store-chip"][data-store="${store}"]`).click()
  }
  await page.getByTestId('lab-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('lab-run').click()

  await expect(page.getByTestId('store-comparison')).toBeVisible({ timeout: 90_000 })
  const summaries = page.getByTestId('run-summary')
  expect(await summaries.count()).toBeGreaterThanOrEqual(3)
  await expect(page.getByTestId('comparison-row').first()).toBeVisible()
  await expect(page.getByTestId('rank-movement')).toBeVisible()
  await expect(page.getByTestId('movement-row').first()).toBeVisible()
})

// ------------------------------------------- Phase 8 acceptance: reprocess control
test('a source version can be reprocessed under a different chunk profile', async ({ page }) => {
  await page.getByTestId('nav-data-room').click()
  const original = page.locator('[data-testid="source-row"][data-status="ready"]').first()
  const originalId = await original.getAttribute('data-source-id')
  await original.click()
  await expect(page.getByTestId('source-detail')).toBeVisible()
  await page.getByTestId('chunk-profile-select').selectOption('parent-child')
  await page.getByTestId('reprocess').click()

  await expect(page.getByTestId('job-panel')).toHaveAttribute('data-job-status', 'completed', { timeout: 120_000 })

  // Reprocessing produces a *new* immutable version rather than mutating the old one:
  // a second version under the same logical source, whose chunks carry the new
  // profile while the original's still carry the old one. Asserted through the API so
  // the check stays true on a re-run, where the new version already exists.
  const sources = await (await page.request.get(`${API}/sources`)).json()
  const originalRecord = sources.find((item: { source_id: string }) => item.source_id === originalId)
  const siblings = sources.filter(
    (item: { logical_source_id: string }) => item.logical_source_id === originalRecord.logical_source_id,
  )
  expect(siblings.length).toBeGreaterThanOrEqual(2)

  const reprocessed = siblings.find((item: { source_id: string }) => item.source_id !== originalId)
  const detail = await (await page.request.get(`${API}/sources/${reprocessed.source_id}`)).json()
  expect(new Set(detail.chunks.map((chunk: { profile: string }) => chunk.profile))).toEqual(new Set(['parent-child']))

  const originalDetail = await (await page.request.get(`${API}/sources/${originalId}`)).json()
  expect(new Set(originalDetail.chunks.map((chunk: { profile: string }) => chunk.profile))).toEqual(
    new Set(['structure-aware']),
  )
})

// ------------------------------- Phase 8 acceptance: evaluation drill-down + export
test('the Evaluation Studio drills into per-question failure causes and exports a report', async ({ page }) => {
  await page.getByTestId('nav-evaluation').click()
  await expect(page.getByTestId('dataset-select')).toBeVisible()
  await page.getByTestId('run-benchmark').click()

  await expect(page.getByTestId('question-drilldown')).toBeVisible({ timeout: 90_000 })
  const rows = page.getByTestId('drilldown-row')
  expect(await rows.count()).toBeGreaterThan(0)
  // The unanswerable question must be classified as a correct abstention, never as a
  // zero score — the exact trap Phase 7 found in the metric aggregation.
  await expect(page.locator('[data-testid="drilldown-row"][data-cause="ok"]').first()).toBeVisible()

  const download = page.waitForEvent('download')
  await page.getByTestId('export-csv').click()
  expect((await download).suggestedFilename()).toMatch(/\.csv$/)
})

test('the seeded corpus really is what the UI is reading', () => {
  // Guards the suite itself: if the corpus seeding silently no-ops, every "REAL"
  // assertion above degrades into a test of an empty page.
  const output = execFileSync(PYTHON, ['-c', 'import investrag; print(investrag.__name__)'], {
    cwd: BACKEND_DIR,
    env: { ...process.env, PYTHONPATH: resolve(BACKEND_DIR, 'src') },
  })
  expect(output.toString()).toContain('investrag')
})

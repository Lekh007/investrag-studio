import { expect, test } from '@playwright/test'
import { mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

/**
 * Capture the Phase 8 evidence screenshots against the real running stack.
 *
 * This is a capture harness, not a second test suite — the assertions here exist only
 * so a screenshot can never be taken of a pane that failed to load. Everything it
 * photographs is asserted properly in `states.spec.ts`.
 *
 * Run with:  npx playwright test e2e/evidence.spec.ts
 */

const OUT = resolve(import.meta.dirname, '../../../docs/evidence/phase-8')
mkdirSync(OUT, { recursive: true })

const shot = (name: string) => join(OUT, `${name}.png`)

test.use({ viewport: { width: 1600, height: 1100 } })

test('capture: ingestion stage streaming on a large PDF', async ({ page }) => {
  // A real 60-page Docling parse runs for minutes on this machine — which is exactly
  // why it is the right document for demonstrating streamed stage transitions, and
  // why this capture needs far more than the suite's default 90 s budget.
  test.setTimeout(900_000)
  await page.goto('/')
  await page.getByTestId('nav-data-room').click()

  // A 60-page PDF: big enough that the parse stage is genuinely on screen for
  // seconds, so the screenshot shows a run in flight rather than a finished list.
  const scratch = join(tmpdir(), `investrag-evidence-${Date.now()}`)
  mkdirSync(scratch, { recursive: true })
  const pdf = join(scratch, 'large-operating-review.pdf')
  const { execFileSync } = await import('node:child_process')
  const backendDir = resolve(import.meta.dirname, '../../backend')
  execFileSync(
    resolve(backendDir, '.venv/Scripts/python.exe'),
    [
      '-c',
      [
        'import pymupdf, sys',
        'doc = pymupdf.open()',
        'for i in range(1, 61):',
        '    p = doc.new_page()',
        '    p.insert_text((72, 90), f"Section {i}: Operating review", fontsize=16)',
        '    p.insert_text((72, 130), f"Revenue in period {i} grew 12% year over year to $430 million.", fontsize=11)',
        '    p.insert_text((72, 160), f"Operating margin held at 42 percent in period {i}.", fontsize=11)',
        '    top = 200.0',
        '    for r in range(8):',
        '        p.draw_rect(pymupdf.Rect(72, top, 480, top + 22))',
        '        p.insert_text((78, top + 15), f"Loan {i}-{r}  Balance $1{r}0,000  Status Current", fontsize=9)',
        '        top += 22',
        'doc.save(sys.argv[1])',
      ].join('\n'),
      pdf,
    ],
    { cwd: backendDir },
  )

  await page.getByTestId('file-input').setInputFiles(pdf)
  const panel = page.getByTestId('job-panel')
  await expect(panel).toBeVisible()
  // Wait until the pipeline is demonstrably mid-run: the parse stage exists.
  await expect(panel.locator('[data-stage="parse"]')).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: shot('01-ingestion-stages-streaming'), fullPage: false })

  await expect(panel).toHaveAttribute('data-job-status', 'completed', { timeout: 180_000 })
  await page.screenshot({ path: shot('02-ingestion-stages-complete'), fullPage: false })
  writeFileSync(join(OUT, 'large-pdf-path.txt'), `${pdf}\n60 pages\n`)
})

test('capture: citation opens the exact page with its bounding box highlighted', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('nav-research').click()
  await page.getByTestId('research-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('research-run').click()

  const citation = page.getByTestId('citation-card').first()
  await expect(citation).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: shot('03-research-cited-answer') })

  await citation.click()
  await expect(page.locator('.react-pdf__Page__canvas')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('bbox').first()).toBeVisible({ timeout: 30_000 })
  await page.screenshot({ path: shot('04-dataroom-bbox-overlay') })

  // A tight crop of the page frame, so the overlay alignment is actually legible.
  const frame = page.locator('.pdf-page-frame')
  await frame.screenshot({ path: shot('05-bbox-overlay-closeup') })
})

test('capture: Retrieval Lab multi-store comparison', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('nav-retrieval').click()
  for (const store of ['chroma', 'qdrant']) {
    await page.locator(`[data-testid="store-chip"][data-store="${store}"]`).click()
  }
  await page.getByTestId('lab-question').fill('How much did revenue grow year over year?')
  await page.getByTestId('lab-run').click()
  await expect(page.getByTestId('store-comparison')).toBeVisible({ timeout: 90_000 })
  await expect(page.getByTestId('comparison-row').first()).toBeVisible()
  await page.screenshot({ path: shot('06-retrieval-lab-multi-store'), fullPage: true })
})

test('capture: §15.6 states', async ({ page }) => {
  // Unavailable databases with concrete reasons (Overview health list).
  await page.goto('/')
  await expect(page.locator('[data-testid="health-row"][data-available="false"]').first()).toBeVisible()
  await page.screenshot({ path: shot('07-state-unavailable-databases') })

  // Review-required low-confidence parse.
  await page.getByTestId('nav-data-room').click()
  await page.locator('[data-testid="source-row"][data-status="partial"]').first().click()
  await expect(page.getByTestId('state-review')).toBeVisible()
  await page.screenshot({ path: shot('08-state-review-required') })

  // Abstention.
  await page.getByTestId('nav-research').click()
  await page.getByTestId('hint-abstention').click()
  await page.getByTestId('research-run').click()
  await expect(page.getByTestId('state-abstained')).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: shot('09-state-abstained') })

  // Failed + resumable ingestion job.
  await page.getByTestId('nav-data-room').click()
  const scratch = join(tmpdir(), `investrag-evidence-fail-${Date.now()}`)
  mkdirSync(scratch, { recursive: true })
  const broken = join(scratch, 'evidence-broken.pdf')
  writeFileSync(broken, '%PDF-1.4\nnot a real PDF, captured for the failed-job screenshot')
  await page.getByTestId('file-input').setInputFiles(broken)
  await expect(page.getByTestId('job-panel')).toHaveAttribute('data-job-status', 'failed', { timeout: 60_000 })
  await page.screenshot({ path: shot('10-state-failed-resumable-job') })
})

test('capture: Evaluation Studio drill-down', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('nav-evaluation').click()
  await page.getByTestId('run-benchmark').click()
  await expect(page.getByTestId('question-drilldown')).toBeVisible({ timeout: 90_000 })
  await page.screenshot({ path: shot('11-evaluation-drilldown'), fullPage: true })
})

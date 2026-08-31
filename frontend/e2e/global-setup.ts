import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'

/**
 * Seed the real corpus the §15.6 suite runs against, then start the real backend
 * over it.
 *
 * The backend is started here rather than through Playwright's `webServer` for a
 * reason that cost a debugging session: `webServer` entries start *before*
 * `globalSetup`, so the API process had the SQLite catalog and the FAISS index open
 * while a second process re-seeded the same directory. Two writers over one data
 * directory produced exactly the corruption `GET /collections` is designed to catch —
 * a FAISS index holding 4 vectors against a catalog of 77 chunks, reported live as
 * `consistent_with_catalog: false`. Seeding in a process that has fully exited before
 * the API process starts makes the ordering single-writer by construction.
 */

const BACKEND_DIR = resolve(import.meta.dirname, '../../backend')
// `global-setup.ts` lives at frontend/e2e; two parent steps reach the
// repository root, so the e2e catalog/index stays inside this standalone repo.
const DATA_DIR = resolve(import.meta.dirname, '../../.data/investrag-e2e')
const PYTHON = resolve(BACKEND_DIR, '.venv/Scripts/python.exe')
const BACKEND_PORT = 8010

const backendEnv = {
  ...process.env,
  INVESTRAG_DATA_DIR: DATA_DIR,
  INVESTRAG_EMBEDDING_MODE: 'hash',
  INVESTRAG_CORS_ORIGINS: 'http://127.0.0.1:5175,http://localhost:5175',
  // The suite must be reproducible on a machine with no GPU and no Ollama running.
  INVESTRAG_OLLAMA_BASE_URL: 'http://127.0.0.1:1',
  PYTHONPATH: resolve(BACKEND_DIR, 'src'),
}

async function waitForBackend(timeoutMs = 120_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${BACKEND_PORT}/health/live`)
      if (response.ok) return
    } catch {
      // not listening yet
    }
    await sleep(300)
  }
  throw new Error(`the e2e backend did not come up on port ${BACKEND_PORT}`)
}

export default async function globalSetup(): Promise<() => Promise<void>> {
  if (existsSync(DATA_DIR)) rmSync(DATA_DIR, { recursive: true, force: true })

  // stdio: 'inherit' is fragile here: when this Node process's own stdout is itself
  // redirected (CI, or any non-interactive/piped invocation), the inherited handle
  // can break partway through the child's output, crashing seeding non-deterministically
  // right as its progress bar flushes. Capture via pipes instead and only surface output
  // on failure, so seeding succeeds identically under an interactive terminal or CI.
  try {
    execFileSync(PYTHON, ['scripts/seed_e2e_corpus.py', '--data-dir', DATA_DIR], {
      cwd: BACKEND_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: backendEnv,
      // Docling/RapidOCR can emit a large diagnostic stream on first load;
      // Node's 1 MiB default would terminate a successful seed with ENOBUFS.
      maxBuffer: 16 * 1024 * 1024,
    })
  } catch (error) {
    const err = error as { stdout?: Buffer; stderr?: Buffer; message: string }
    process.stderr.write(err.stdout?.toString() ?? '')
    process.stderr.write(err.stderr?.toString() ?? '')
    throw new Error(`corpus seeding failed: ${err.message}`)
  }

  const backend: ChildProcess = spawn(
    PYTHON,
    ['-m', 'uvicorn', 'investrag.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)],
    { cwd: BACKEND_DIR, env: backendEnv, stdio: 'ignore' },
  )
  await waitForBackend()

  return async () => {
    backend.kill()
    await sleep(500)
  }
}

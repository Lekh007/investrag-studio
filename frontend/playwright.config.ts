import { defineConfig, devices } from '@playwright/test'
import { resolve } from 'node:path'

/**
 * End-to-end configuration for the design §15.6 required-interface-state suite.
 *
 * Both servers are real: a genuine uvicorn process serving the real FastAPI app over
 * an isolated data directory, and the real Vite dev server pointed at it. Nothing in
 * the stack is stubbed — the §15.6 states are produced by the backend actually
 * behaving that way, apart from the two faults documented in `states.spec.ts` that
 * cannot be provoked on demand from a healthy local stack.
 *
 * The ports are deliberately different from the development ones (8010/5175 rather
 * than 8000/5174) so an e2e run can never talk to, or clobber, a Data Room the
 * developer has open.
 */

const BACKEND_PORT = 8010
const FRONTEND_PORT = 5175
const BACKEND_DIR = resolve(import.meta.dirname, '../backend')
const DATA_DIR = resolve(import.meta.dirname, '../../.data/investrag-e2e')
const PYTHON = resolve(BACKEND_DIR, '.venv/Scripts/python.exe')

export default defineConfig({
  testDir: './e2e',
  globalSetup: './e2e/global-setup.ts',
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  outputDir: './e2e/.artifacts',
  use: {
    baseURL: `http://127.0.0.1:${FRONTEND_PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // Only the frontend is a `webServer`. The backend is owned by `global-setup.ts`,
  // because `webServer` entries start *before* `globalSetup` and would then hold the
  // data directory open while the corpus is seeded into it — see that file for the
  // two-writer corruption that caused.
  webServer: {
    command: `npx vite --port ${FRONTEND_PORT} --host 127.0.0.1 --strictPort`,
    cwd: import.meta.dirname,
    url: `http://127.0.0.1:${FRONTEND_PORT}`,
    env: { VITE_API_BASE: `http://127.0.0.1:${BACKEND_PORT}/api/v1` },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})

export { BACKEND_DIR, BACKEND_PORT, DATA_DIR, FRONTEND_PORT, PYTHON }

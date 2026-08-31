/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: { port: 5174, host: '127.0.0.1' },
  test: {
    // Vitest and Playwright both claim `*.spec.ts`. The e2e specs import
    // `@playwright/test`, which has no meaning inside a Vitest worker, so they must
    // stay out of the unit run — `npm test` covers `src/`, `npm run test:e2e` covers
    // `e2e/`, and neither silently swallows the other's files.
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
  },
})

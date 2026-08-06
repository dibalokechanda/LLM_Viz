import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Honour PORT when the harness assigns one, otherwise use Vite's default.
const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
const port = Number(env?.PORT) || 5173

export default defineConfig({
  base: '/LLM_Viz/',
  plugins: [react()],
  server: {
    port,
    // The content editor's backend. Dev only — the built site inlines
    // content/*.md at build time and never calls this.
    proxy: { '/api': 'http://127.0.0.1:8788' },
  },
})

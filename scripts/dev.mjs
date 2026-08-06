/**
 * Start the Vite app and, only when needed, its local content editor server.
 *
 * An editor server may already be running from another terminal. In that case
 * we deliberately reuse it: Vite's /api proxy continues to target port 8788,
 * and this launcher owns only the Vite process it starts.
 */
import { spawn } from 'node:child_process'
import { createConnection } from 'node:net'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = Number(process.env.CONTENT_SERVER_PORT) || 8788
const vite = join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js')
const contentServer = join(ROOT, 'server', 'content-server.mjs')

function portIsInUse(port) {
  return new Promise((resolve) => {
    const socket = createConnection({ host: '127.0.0.1', port })
    const finish = (inUse) => {
      socket.removeAllListeners()
      socket.destroy()
      resolve(inUse)
    }
    socket.once('connect', () => finish(true))
    socket.once('error', () => finish(false))
  })
}

const reusingContentServer = await portIsInUse(PORT)
let content = null

if (reusingContentServer) {
  console.log(`[content] reusing the server already listening on http://127.0.0.1:${PORT}`)
} else {
  content = spawn(process.execPath, [contentServer], { cwd: ROOT, stdio: 'inherit' })
}

const web = spawn(process.execPath, [vite], { cwd: ROOT, stdio: 'inherit' })

let stopping = false
function stopChild(child) {
  if (child && !child.killed) child.kill('SIGTERM')
}

function stop(code = 0) {
  if (stopping) return
  stopping = true
  stopChild(web)
  // Do not stop a content server that existed before this command.
  stopChild(content)
  process.exitCode = code
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => stop())
}

web.on('exit', (code) => stop(code ?? 0))
web.on('error', (error) => {
  console.error('[web] failed to start:', error)
  stop(1)
})

if (content) {
  content.on('exit', (code) => {
    if (!stopping && code !== 0) {
      console.error(`[content] server exited with code ${code}; the web app will keep running.`)
    }
  })
  content.on('error', (error) => console.error('[content] failed to start:', error))
}

/**
 * Local backend for the content editor. Its only job is to read and write the
 * files in `content/`, so edits made in the running app land in the same
 * Markdown you would otherwise edit by hand — one source of truth, git-visible,
 * no browser storage holding a second copy.
 *
 * Vite proxies /api/* here in dev. Deliberately dev-only and deliberately
 * local: it binds to loopback, serves nothing but these twelve files, and the
 * built site never talks to it (content is inlined at build time).
 */

import { createServer } from 'node:http'
import { readFile, writeFile, readdir, mkdir, copyFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname, join, basename } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const CONTENT_DIR = join(ROOT, 'content')
/* Images live under `public/` so Vite serves them in dev and copies them into
   the bundle on build; a folder inside `content/` has no URL. Content files
   reference them as `assets/<name>` and the renderer resolves the prefix. */
const ASSET_DIR = join(ROOT, 'public', 'content-assets')
const BACKUP_DIR = join(ROOT, '.content-backups')
const PORT = Number(process.env.CONTENT_SERVER_PORT) || 8788

/** Images the editor is willing to store, by magic bytes rather than by claim. */
const IMAGE_KINDS = [
  { ext: 'png', sig: [0x89, 0x50, 0x4e, 0x47] },
  { ext: 'jpg', sig: [0xff, 0xd8, 0xff] },
  { ext: 'gif', sig: [0x47, 0x49, 0x46, 0x38] },
  { ext: 'webp', sig: [0x52, 0x49, 0x46, 0x46] },
]

const sniffImage = (buf) =>
  IMAGE_KINDS.find((k) => k.sig.every((b, i) => buf[i] === b))?.ext ?? null

/** Only ever touch `<name>.md` directly inside content/ — no paths, no traversal. */
const safeName = (n) =>
  typeof n === 'string' && /^[a-z0-9-]+\.md$/.test(n) && n === basename(n)

function sendJson(res, status, obj) {
  res.writeHead(status, {
    'content-type': 'application/json',
    'cache-control': 'no-store',
  })
  res.end(JSON.stringify(obj))
}

async function readBodyBuffer(req, limit = 4_000_000) {
  const chunks = []
  let size = 0
  for await (const c of req) {
    size += c.length
    if (size > limit) throw new Error('body too large')
    chunks.push(c)
  }
  return Buffer.concat(chunks)
}

const readBody = async (req, limit) => (await readBodyBuffer(req, limit)).toString('utf8')

/**
 * Pull the single file part out of a multipart/form-data body.
 *
 * Hand-rolled rather than pulling in a parser: this endpoint accepts exactly
 * one image from a local editor, so the general cases a library exists for
 * (streaming, many parts, nested boundaries) are cases this will never see.
 * Works on the raw buffer because image bytes must not pass through a string.
 */
function extractMultipartFile(buf, contentType) {
  const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType)
  if (!m) return null
  const boundary = Buffer.from(`--${(m[1] ?? m[2]).trim()}`)

  let start = buf.indexOf(boundary)
  while (start !== -1) {
    const headerStart = start + boundary.length
    const headerEnd = buf.indexOf('\r\n\r\n', headerStart)
    if (headerEnd === -1) return null

    const headers = buf.subarray(headerStart, headerEnd).toString('utf8')
    const next = buf.indexOf(boundary, headerEnd)
    if (next === -1) return null

    if (/filename="/.test(headers)) {
      const fn = /filename="([^"]*)"/.exec(headers)
      // Trailing CRLF before the next boundary is delimiter, not payload.
      const data = buf.subarray(headerEnd + 4, next - 2)
      return { filename: fn?.[1] ?? '', data }
    }
    start = next
  }
  return null
}

/**
 * Keep the previous revision before every write. The editor is editing the
 * only copy of a lot of hand-written prose; a bad save should be recoverable
 * without needing the file to have been committed first.
 */
async function backup(name, text) {
  await mkdir(BACKUP_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  await writeFile(join(BACKUP_DIR, `${name}.${stamp}.bak`), text, 'utf8')
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost')

    if (req.method === 'GET' && url.pathname === '/api/content') {
      const files = (await readdir(CONTENT_DIR)).filter((f) => f.endsWith('.md')).sort()
      return sendJson(res, 200, { ok: true, dir: CONTENT_DIR, files })
    }

    const fileMatch = url.pathname.match(/^\/api\/content\/([^/]+)$/)
    if (fileMatch) {
      const name = decodeURIComponent(fileMatch[1])
      if (!safeName(name)) return sendJson(res, 400, { error: 'bad file name' })
      const path = join(CONTENT_DIR, name)

      if (req.method === 'GET') {
        if (!existsSync(path)) return sendJson(res, 404, { error: 'no such content file' })
        return sendJson(res, 200, { ok: true, name, text: await readFile(path, 'utf8') })
      }

      if (req.method === 'PUT') {
        const body = JSON.parse(await readBody(req))
        if (typeof body?.text !== 'string' || !body.text.trim()) {
          return sendJson(res, 400, { error: 'missing text' })
        }
        // Refuse anything that would not parse back. The editor validates too,
        // but this is the last gate before overwriting hand-written prose.
        if (!/^---\r?\n[\s\S]*?\r?\n---/.test(body.text)) {
          return sendJson(res, 400, { error: 'content must start with YAML front matter' })
        }
        if (existsSync(path)) await backup(name, await readFile(path, 'utf8'))
        await writeFile(path, body.text, 'utf8')
        console.log(`[content] wrote ${name} (${body.text.length} bytes)`)
        return sendJson(res, 200, { ok: true, name })
      }
    }

    /*
     * Image upload. Images live in `content/assets/` beside the prose that
     * uses them and are referenced relatively, so they travel with the
     * repository and stay reviewable — base64 inlined into the Markdown would
     * make the file unreadable and every edit an unreadable diff.
     */
    if (req.method === 'POST' && url.pathname === '/api/asset') {
      const raw = await readBodyBuffer(req, 12_000_000)
      const file = extractMultipartFile(raw, req.headers['content-type'] ?? '')
      if (!file) return sendJson(res, 400, { error: 'no file in request' })

      const ext = sniffImage(file.data)
      if (!ext) return sendJson(res, 400, { error: 'not a PNG, JPEG, GIF or WebP' })

      // Name from the original where it is safe, plus a short suffix so two
      // pastes of "screenshot.png" cannot clobber each other.
      const stem = (file.filename || 'image')
        .replace(/\.[^.]*$/, '')
        .replace(/[^a-zA-Z0-9-_]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 40) || 'image'
      const name = `${stem}-${Date.now().toString(36)}.${ext}`

      await mkdir(ASSET_DIR, { recursive: true })
      await writeFile(join(ASSET_DIR, name), file.data)
      console.log(`[content] stored assets/${name} (${file.data.length} bytes)`)
      return sendJson(res, 200, { ok: true, path: `assets/${name}` })
    }

    sendJson(res, 404, { error: 'not found' })
  } catch (e) {
    sendJson(res, 500, { error: e instanceof Error ? e.message : String(e) })
  }
})

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    // A launcher can reuse the already-running local server instead.
    console.warn(`[content] port ${PORT} already in use — leaving the running instance alone.`)
  } else {
    console.error('[content] server error:', err)
  }
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[content] editing ${CONTENT_DIR}`)
  console.log(`[content] listening on http://127.0.0.1:${PORT}`)
})

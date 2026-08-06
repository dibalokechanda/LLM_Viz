// Named imports: js-yaml ships ESM without a default export, so `import yaml
// from 'js-yaml'` resolves to undefined under Node's ESM loader.
import { load as yamlLoad, dump as yamlDump } from 'js-yaml'
import type { Block, Variant } from '../data/types'

/**
 * The content file format.
 *
 * One `.md` per block, split the way the writing splits:
 *
 *   ---                     YAML front matter — everything structured.
 *   id, label, ordinal…     Block metadata.
 *   lineage: [...]          The arrows, with their labels.
 *   variants: [...]         Per variant: metadata, maths, code, figures,
 *                           trade-offs, cost rows. Everything except prose.
 *   ---
 *
 *   ## role                 Markdown body — the long prose, and only that.
 *   …paragraphs…            What this position in the stack is for.
 *
 *   ## <variant-id>         One section per variant: its `detail` paragraphs.
 *   …paragraphs…
 *
 *   ### fixes               Optional: the one sentence on what it repairs.
 *   …one paragraph…
 *
 * Prose lives in the body because that is what gets rewritten most and what
 * benefits from being real Markdown — blank-line paragraphs, `**bold**`, no
 * quoting rules. Structure lives in front matter because it is data, and YAML
 * block scalars (`|`) hold TeX and source code verbatim with no escaping.
 *
 * The split is also what makes the round trip safe: `serialize` reproduces this
 * exact shape, so the in-app editor can write a file back without disturbing
 * anything a human wrote by hand.
 */

/**
 * A variant as it appears in front matter — everything but the prose.
 *
 * `fixes` is one short sentence, so it may be written either way: as a
 * `### fixes` subsection next to the prose, or inline in front matter. The
 * body wins when both are present.
 */
type VariantMeta = Omit<Variant, 'detail'>

interface BlockMeta extends Omit<Block, 'role' | 'variants'> {
  variants: VariantMeta[]
}

export interface ParsedFile {
  block: Block
  /** Non-fatal problems, surfaced by the validator and the editor. */
  warnings: string[]
}

const FM = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/

/** Split "## heading" sections at a given level, preserving order. */
function sections(body: string, level: 2 | 3): { name: string; text: string }[] {
  const marker = '#'.repeat(level)
  const re = new RegExp(`^${marker} +(.+)$`, 'gm')
  const out: { name: string; text: string }[] = []
  const hits: { name: string; at: number; end: number }[] = []
  let m: RegExpExecArray | null
  while ((m = re.exec(body))) hits.push({ name: m[1].trim(), at: m.index, end: re.lastIndex })
  hits.forEach((h, i) => {
    const stop = i + 1 < hits.length ? hits[i + 1].at : body.length
    out.push({ name: h.name, text: body.slice(h.end, stop).trim() })
  })
  return out
}

/** Markdown paragraphs — blank-line separated, which is how `detail` is typed. */
function paragraphs(text: string): string[] {
  return text
    .split(/\r?\n[ \t]*\r?\n/)
    .map((p) => p.trim())
    .filter(Boolean)
}

/**
 * Parse one content file into the Block the app renders.
 *
 * Throws only on genuinely unusable input (no front matter, unparseable YAML,
 * missing id). Everything softer — a prose section with no matching variant, a
 * variant with no prose — comes back as a warning so one bad edit shows up as a
 * message rather than a blank screen.
 */
export function parseBlockFile(source: string, filename = '<content>'): ParsedFile {
  const warnings: string[] = []

  const fm = FM.exec(source)
  if (!fm) throw new Error(`${filename}: missing YAML front matter (the --- block at the top)`)

  let meta: BlockMeta
  try {
    meta = yamlLoad(fm[1]) as BlockMeta
  } catch (e) {
    throw new Error(`${filename}: front matter is not valid YAML — ${e instanceof Error ? e.message : e}`)
  }
  if (!meta || typeof meta !== 'object') throw new Error(`${filename}: front matter is empty`)
  if (!meta.id) throw new Error(`${filename}: front matter has no \`id\``)

  const body = source.slice(fm[0].length)
  const secs = sections(body, 2)

  const roleSec = secs.find((s) => s.name.toLowerCase() === 'role')
  if (!roleSec) warnings.push(`${filename}: no "## role" section — the block has no overview prose`)

  const proseById = new Map<string, { detail: string[]; fixes?: string }>()
  for (const s of secs) {
    if (s.name.toLowerCase() === 'role') continue
    // A variant section may carry a "### fixes" subsection; everything above it
    // is the detail prose.
    const subs = sections(s.text, 3)
    const fixesSub = subs.find((x) => x.name.toLowerCase() === 'fixes')
    const head = subs.length ? s.text.slice(0, s.text.indexOf(`### ${subs[0].name}`)) : s.text
    proseById.set(s.name, {
      detail: paragraphs(head),
      fixes: fixesSub ? paragraphs(fixesSub.text).join(' ') : undefined,
    })
  }

  const variants: Variant[] = (meta.variants ?? []).map((v) => {
    const prose = proseById.get(v.id)
    if (!prose) warnings.push(`${filename}: variant "${v.id}" has no "## ${v.id}" prose section`)
    proseById.delete(v.id)
    return { ...v, detail: prose?.detail ?? [], fixes: prose?.fixes ?? v.fixes }
  })

  for (const orphan of proseById.keys()) {
    warnings.push(`${filename}: prose section "## ${orphan}" matches no variant id`)
  }

  const block: Block = {
    ...(meta as unknown as Block),
    role: roleSec ? paragraphs(roleSec.text) : [],
    variants,
  }
  return { block, warnings }
}

/* ────────────────────────────── writing ────────────────────────────── */

/**
 * Render a Block back to its content file.
 *
 * Kept byte-stable for unchanged content so the editor saving one field does
 * not rewrite the whole file and bury the real edit in a noisy diff.
 */
export function serializeBlockFile(block: Block): string {
  const { role, variants, ...rest } = block

  const meta: BlockMeta = {
    ...(rest as unknown as Omit<Block, 'role' | 'variants'>),
    variants: variants.map(({ detail: _d, fixes: _f, ...v }) => v),
  }

  const front = yamlDump(meta, {
    lineWidth: 100,
    noRefs: true,
    sortKeys: false,
  })

  const parts = [`---\n${front}---\n`]
  parts.push(`\n## role\n\n${role.join('\n\n')}\n`)
  for (const v of variants) {
    parts.push(`\n## ${v.id}\n\n${v.detail.join('\n\n')}\n`)
    if (v.fixes) parts.push(`\n### fixes\n\n${v.fixes}\n`)
  }
  return parts.join('')
}

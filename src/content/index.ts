import type { Block, BlockId, Variant } from '../data/types'
import { parseBlockFile } from './parse'
import { checkTemplate } from './expr'
import { REFERENCE_DIMS } from '../ModelContext'

/**
 * Loads the thirteen content files and exposes them as the app's block data.
 *
 * This is the module the rest of the app imports; it replaced a set of
 * hand-written TypeScript modules so that every word, equation, code sample,
 * figure and cost formula lives in `content/*.md` and can be edited without
 * touching code — by hand, or through the in-app editor.
 *
 * Vite inlines the files at build time, so the published site stays a static
 * bundle with no loader and no fetch. In dev, editing a file hot-reloads it.
 */

/** Forward-pass order. The files themselves carry no ordering. */
const ORDER: BlockId[] = [
  'tokenizer',
  'embedding',
  'positional',
  'norm',
  'mixer',
  'qkv',
  'pattern',
  'scores',
  'kvcache',
  'ffn',
  'residual',
  'lmhead',
  'sampling',
]

const files = import.meta.glob('../../content/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

export interface ContentIssue {
  file: string
  message: string
}

/** Everything the parser and the cost templates complained about, for the editor. */
export const contentIssues: ContentIssue[] = []

const loaded = new Map<BlockId, Block>()

/** Only files named for a block are content; README.md is documentation. */
const isBlockFile = (name: string) => ORDER.includes(name.replace(/\.md$/, '') as BlockId)

for (const [path, source] of Object.entries(files)) {
  const name = path.split('/').pop() ?? path
  if (!isBlockFile(name)) continue
  try {
    const { block, warnings } = parseBlockFile(source, name)
    for (const w of warnings) contentIssues.push({ file: name, message: w })

    // Cost templates are the one part that can fail at render time rather than
    // parse time, so probe them once here against a reference model. A typo
    // then shows up as a startup issue instead of a stray "?" nobody notices.
    for (const v of block.variants) {
      for (const row of v.cost ?? []) {
        for (const field of [row.value, row.note].filter(Boolean) as string[]) {
          for (const err of checkTemplate(field, REFERENCE_DIMS)) {
            contentIssues.push({ file: name, message: `${block.id}:${v.id} — ${err}` })
          }
        }
      }
    }
    loaded.set(block.id, block)
  } catch (e) {
    contentIssues.push({ file: name, message: e instanceof Error ? e.message : String(e) })
  }
}

/** The fixed cohort, in forward-pass order. This array is the stack diagram. */
export const blocks: Block[] = ORDER.map((id) => loaded.get(id)).filter(
  (b): b is Block => Boolean(b),
)

for (const id of ORDER) {
  if (!loaded.has(id)) contentIssues.push({ file: `${id}.md`, message: `missing content file` })
}

export const blockById = new Map<BlockId, Block>(blocks.map((b) => [b.id, b]))

/** `blockId:variantId` → variant, for cross-block references. */
export const variantByKey = new Map<string, Variant>()
for (const b of blocks) {
  for (const v of b.variants) variantByKey.set(`${b.id}:${v.id}`, v)
}

export function variantOf(blockId: BlockId, variantId?: string): Variant | undefined {
  const b = blockById.get(blockId)
  if (!b) return undefined
  return b.variants.find((v) => v.id === variantId) ?? b.variants.find((v) => v.id === b.defaultVariant)
}

/** Default path — what the stack shows before any model is loaded. */
export const defaultPath: Partial<Record<BlockId, string>> = Object.fromEntries(
  blocks.map((b) => [b.id, b.defaultVariant]),
)

/** Lineage edges naming a variant that does not exist would silently vanish. */
if (import.meta.env?.DEV) {
  for (const b of blocks) {
    const ids = new Set(b.variants.map((v) => v.id))
    for (const l of b.lineage ?? []) {
      if (!ids.has(l.from)) contentIssues.push({ file: `${b.id}.md`, message: `lineage: unknown from "${l.from}"` })
      if (!ids.has(l.to)) contentIssues.push({ file: `${b.id}.md`, message: `lineage: unknown to "${l.to}"` })
    }
    if (!ids.has(b.defaultVariant)) {
      contentIssues.push({ file: `${b.id}.md`, message: `unknown defaultVariant "${b.defaultVariant}"` })
    }
  }
  if (contentIssues.length) {
    console.warn(
      `[content] ${contentIssues.length} issue(s):\n` +
        contentIssues.map((i) => `  ${i.file}: ${i.message}`).join('\n'),
    )
  }
}

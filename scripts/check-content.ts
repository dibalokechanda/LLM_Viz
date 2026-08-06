/**
 * Validate `content/*.md` without starting the app.
 *
 * Run with `npm run check`. Exists because the content files are now the thing
 * people edit, and a typo in one of them should produce a precise message on
 * the command line rather than a blank panel in the browser.
 */
import { readdir, readFile } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { BlockId, Concept, Dims } from '../src/data/types'
import { parseBlockFile } from '../src/content/parse'
import { checkTemplate } from '../src/content/expr'
import { rawImplementationSnippet, transformersSnippet } from '../src/content/transformersSnippets'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

const EXPECTED: BlockId[] = [
  'tokenizer', 'embedding', 'positional', 'norm', 'mixer', 'qkv', 'pattern',
  'scores', 'kvcache', 'ffn', 'residual', 'lmhead', 'sampling',
]

/** A model shape to probe cost templates against. Mirrors REFERENCE_DIMS. */
const PROBE: Dims = {
  name: 'reference 8B',
  nLayer: 32, dModel: 4096, nHead: 32, nKvHead: 8, dHead: 128,
  dFF: 14336, vocab: 128256, ctx: 131072, ropeTheta: 500000, kvBytes: 2,
}
/** A second shape with the MoE and window fields set, so their branches run too. */
const PROBE_MOE: Dims = { ...PROBE, nExperts: 256, nActive: 8, nShared: 1, window: 4096 }

const problems: string[] = []
const note = (f: string, m: string) => problems.push(`${f}: ${m}`)

const dir = join(root, 'content')
// Only files named for a block are content; README.md is documentation.
const files = (await readdir(dir)).filter((f) =>
  EXPECTED.includes(f.replace(/\.md$/, '') as BlockId),
)

let variants = 0
let costRows = 0
const seen = new Set<string>()
const CONCEPT_KINDS = new Set<Concept['kind']>(['idea', 'formula', 'method', 'metric', 'pitfall', 'tradeoff'])

function checkConcepts(cards: unknown, file: string, variantId: string, path = 'concepts', ids = new Set<string>()) {
  if (cards === undefined) return
  if (!Array.isArray(cards)) {
    note(file, `${variantId} ${path}: must be a list of card objects`)
    return
  }
  for (const [index, rawCard] of cards.entries()) {
    const at = `${variantId} ${path}[${index}]`
    if (!rawCard || typeof rawCard !== 'object' || Array.isArray(rawCard)) {
      note(file, `${at}: must be a card object`)
      continue
    }
    const card = rawCard as Partial<Concept>
    if (!card.id?.trim()) note(file, `${at}: missing id`)
    else if (ids.has(card.id)) note(file, `${at}: duplicate card id "${card.id}"`)
    else ids.add(card.id)
    if (!card.label?.trim()) note(file, `${at}: missing title`)
    if (!card.summary?.trim()) note(file, `${at}: missing card summary`)
    if (!CONCEPT_KINDS.has(card.kind)) note(file, `${at}: unknown kind "${card.kind}"`)
    checkConcepts(card.children, file, variantId, `${path}[${index}].children`, ids)
  }
}

for (const file of files.sort()) {
  const source = await readFile(join(dir, file), 'utf8')
  let parsed
  try {
    parsed = parseBlockFile(source, file)
  } catch (e) {
    note(file, e instanceof Error ? e.message : String(e))
    continue
  }
  const { block, warnings } = parsed
  warnings.forEach((w) => problems.push(w))
  seen.add(block.id)

  const ids = new Set(block.variants.map((v) => v.id))
  if (ids.size !== block.variants.length) note(file, 'duplicate variant ids')
  if (!ids.has(block.defaultVariant)) note(file, `defaultVariant "${block.defaultVariant}" is not a variant`)

  for (const l of block.lineage ?? []) {
    if (!ids.has(l.from)) note(file, `lineage from "${l.from}" — no such variant`)
    if (!ids.has(l.to)) note(file, `lineage to "${l.to}" — no such variant`)
    if (!l.label) note(file, `lineage ${l.from}→${l.to} has no label`)
  }

  for (const v of block.variants) {
    variants++
    if (!v.detail?.length) note(file, `${v.id}: no prose — add a "## ${v.id}" section`)
    if (!v.label || !v.full) note(file, `${v.id}: missing label or full name`)
    if (!v.concepts?.length) note(file, `${v.id}: missing authored concept map`)
    const conceptIds = new Set((v.concepts ?? []).map((concept) => concept.id))
    if (conceptIds.size !== (v.concepts ?? []).length) note(file, `${v.id}: duplicate concept-map root ids`)
    for (const [kind, snippet] of [
      ['raw implementation', rawImplementationSnippet(block, v)],
      ['Transformers contrast', transformersSnippet(block, v)],
    ] as const) {
      if (!snippet.source) note(file, `${v.id} ${kind}: missing Raschka reference link`)
      else if (!snippet.source.url.startsWith('https://github.com/rasbt/LLMs-from-scratch')) {
        note(file, `${v.id} ${kind}: reference must point to rasbt/LLMs-from-scratch`)
      }
    }
    checkConcepts(v.concepts, file, v.id)
    for (const row of v.cost ?? []) {
      costRows++
      for (const field of [row.value, row.note].filter(Boolean) as string[]) {
        for (const probe of [PROBE, PROBE_MOE]) {
          checkTemplate(field, probe).forEach((e) => note(file, `${v.id} cost "${row.label}": ${e}`))
        }
      }
    }
  }
}

for (const id of EXPECTED) if (!seen.has(id)) note(`${id}.md`, 'missing content file')

if (problems.length) {
  console.error(`\n✗ ${problems.length} problem(s):\n`)
  problems.forEach((p) => console.error(`  ${p}`))
  process.exit(1)
}
console.log(`✓ ${files.length} blocks · ${variants} variants · ${costRows} cost rows — all valid`)

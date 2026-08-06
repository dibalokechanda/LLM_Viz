/**
 * One-shot migration: the TypeScript block modules → `content/<block>.md`.
 *
 * Run with `npx tsx scripts/migrate-content.ts`. Everything transfers
 * mechanically except `cost`, which was written as JavaScript closures and
 * cannot be read back out of a compiled function. Those 23 entries are
 * hand-converted below into the template syntax (see src/content/expr.ts) and
 * checked against the originals by `scripts/check-content.ts`.
 */
import { writeFile, mkdir } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Block, CostRow } from '../src/data/types'
import { blocks } from '../src/data/blocks/index'
import { serializeBlockFile } from '../src/content/parse'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

/** `nExperts` and friends are absent on dense models; the scope maps them to 0. */
const E = (fallback: number) => `(nExperts == 0 ? ${fallback} : nExperts)`
const K = (fallback: number) => `(nActive == 0 ? ${fallback} : nActive)`
const S = (fallback: number) => `(nShared == 0 ? ${fallback} : nShared)`
const W = (fallback: number) => `(window == 0 ? ${fallback} : window)`
/** Bytes held by a KV cache with `heads` key/value heads at full context. */
const kv = (heads: string) => `2 * nLayer * ${heads} * dHead * ctx * kvBytes`

const COST: Record<string, CostRow[]> = {
  'embedding:untied': [
    { label: 'Input embedding', value: '{fixed(vocab * dModel / 1e6, 0)} M' },
    { label: 'Output head', value: '{fixed(vocab * dModel / 1e6, 0)} M' },
    { label: 'Combined', value: '{fixed(2 * vocab * dModel / 1e6, 0)} M', key: true },
  ],
  'embedding:tied': [
    { label: 'Shared matrix', value: '{fixed(vocab * dModel / 1e6, 0)} M' },
    { label: 'Saved against untied', value: '{fixed(vocab * dModel / 1e6, 0)} M', key: true },
    {
      label: 'Share of a model this size',
      value: '{fixed(100 * vocab * dModel / (12 * nLayer * dModel * dModel + vocab * dModel), 1)}%',
      note: 'rough — trunk parameters estimated at 12·L·d²',
    },
  ],
  'ffn:swiglu': [
    { label: 'Params per layer', value: '{si(3 * dModel * dFF)}', note: 'three matrices, d_model × d_ff each' },
    { label: 'Params across all layers', value: '{si(3 * dModel * dFF * nLayer)}', key: true },
    {
      label: 'Expansion ratio',
      value: '{fixed(dFF / dModel, 2)}×',
      note: 'the 8/3 ≈ 2.67 heuristic, or wider if the model spends extra here',
    },
  ],
  'ffn:topk-moe': [
    { label: 'Experts', value: `{${K(2)}} of {${E(8)}} active per token` },
    { label: 'Total FFN params', value: `{si(3 * dModel * dFF * ${E(8)} * nLayer)}`, note: 'what the checkpoint weighs' },
    {
      label: 'Active FFN params',
      value: `{si(3 * dModel * dFF * ${K(2)} * nLayer)}`,
      note: 'what each token pays for',
      key: true,
    },
    { label: 'Sparsity ratio', value: `{fixed(${E(8)} / ${K(2)}, 1)}× more knowledge than compute` },
  ],
  'ffn:shared-expert': [
    { label: 'Experts', value: `{${K(8)}} routed of {${E(256)}}, plus {${S(1)}} shared` },
    { label: 'Total FFN params', value: `{si(3 * dModel * dFF * (${E(256)} + ${S(1)}) * nLayer)}` },
    { label: 'Active FFN params', value: `{si(3 * dModel * dFF * (${K(8)} + ${S(1)}) * nLayer)}`, key: true },
    {
      label: 'Routing combinations',
      value: `{${E(256)}} choose {${K(8)}}`,
      note: 'against 8-choose-2 = 28 for a Mixtral-style layer',
    },
  ],
  'kvcache:full': [
    { label: 'Per token per layer', value: '{fixed(2 * nHead * dHead * kvBytes / 1024, 1)} KB' },
    { label: 'One sequence at full context', value: `{bytes(${kv('nHead')})}`, key: true },
    {
      label: 'Batch of 32',
      value: `{bytes((${kv('nHead')}) * 32)}`,
      note: 'this is what decides how many users fit on a GPU',
    },
  ],
  'kvcache:grouped': [
    { label: 'KV heads stored', value: '{nKvHead} of {nHead}' },
    { label: 'One sequence at full context', value: `{bytes(${kv('nKvHead')})}`, key: true },
    { label: 'Saved against full', value: `{bytes(${kv('(nHead - nKvHead)')})}` },
  ],
  'kvcache:sliding': [
    { label: 'Window', value: `{num(${W(4096)})} tokens` },
    {
      label: 'Cache ceiling',
      value: `{bytes(2 * nLayer * nKvHead * dHead * ${W(4096)} * kvBytes)}`,
      note: 'constant, whatever the conversation length',
      key: true,
    },
  ],
  'kvcache:quantized': [
    { label: 'Current precision', value: '{kvBytes * 8}-bit' },
    {
      label: 'At int8',
      value: `{bytes(2 * nLayer * nKvHead * dHead * ctx * 1)}`,
      note: 'usually under 1% quality cost',
      key: true,
    },
    {
      label: 'At int4',
      value: `{bytes(2 * nLayer * nKvHead * dHead * ctx * 0.5)}`,
      note: 'measurable degradation on long-context retrieval',
    },
  ],
  'kvcache:mla-latent': [
    { label: 'Cached per token', value: '512 latent + 64 RoPE' },
    { label: 'At full context', value: '{bytes((512 + 64) * nLayer * kvBytes * ctx)}', key: true },
    {
      label: "vs. this model's grouped cache",
      value: `{fixed((${kv('nKvHead')}) / ((512 + 64) * nLayer * kvBytes * ctx), 1)}× smaller`,
    },
  ],
  'kvcache:cross-layer': [
    { label: 'Layers', value: '{nLayer}' },
    {
      label: 'Sharing every 4 layers',
      value: `{bytes((${kv('nKvHead')}) / 4)}`,
      note: 'the depth multiplier drops from n_layer to n_layer/4',
      key: true,
    },
  ],
  'lmhead:untied-head': [
    { label: 'Head parameters', value: '{fixed(vocab * dModel / 1e6, 0)} M' },
    {
      label: 'FLOPs per generated token',
      value: '{fixed(2 * vocab * dModel / 1e9, 2)} GFLOP',
      note: 'for one token — compare against the whole trunk below',
      key: true,
    },
    {
      label: 'Trunk FLOPs per token',
      value: '{fixed(2 * 12 * nLayer * dModel * dModel / 1e9, 2)} GFLOP',
      note: 'approximate — 12·L·d² is the usual rule of thumb',
    },
  ],
  'lmhead:tied-head': [
    { label: 'Added parameters', value: '0', note: 'shares the embedding matrix' },
    { label: 'Saved', value: '{fixed(vocab * dModel / 1e6, 0)} M', key: true },
  ],
  'norm:rmsnorm': [
    { label: 'Params per norm', value: '{num(dModel)}', note: 'one gain vector; LayerNorm would need twice this' },
    { label: 'Norms in the model', value: '{num(nLayer * 2 + 1)}', note: 'two per layer under pre-norm, plus a final one' },
    { label: 'Total norm params', value: '{num(dModel * (nLayer * 2 + 1))}', key: true },
  ],
  'pattern:causal': [
    { label: 'Pairs at full context', value: '{sci(ctx * (ctx + 1) / 2, 2)}', note: 'per head, per layer' },
    {
      label: 'Score matrix memory',
      value: '{fixed(ctx * ctx * nHead * 2 / 1073741824, 1)} GB',
      note: 'if materialised — which is exactly why FlashAttention does not',
      key: true,
    },
  ],
  'pattern:window': [
    { label: 'Window', value: '{window == 0 ? "not set" : num(window) + " tokens"}' },
    {
      label: 'Nominal receptive field',
      value: '{window == 0 ? "—" : num(nLayer * (window - 1) + 1) + " tokens"}',
      note: 'through layer stacking — reliable retrieval is far shorter',
    },
    {
      label: 'KV cache ceiling',
      value: '{window == 0 ? "—" : bytes(2 * nLayer * nKvHead * dHead * min(window, ctx) * kvBytes)}',
      note: 'constant — the cache stops growing once the window fills',
      key: true,
    },
  ],
  'positional:rope': [
    { label: 'rope_theta', value: '{ropeTheta == 0 ? "not set" : num(ropeTheta)}', note: 'higher stretches all wavelengths' },
    {
      label: 'Slowest wavelength',
      value: '{ropeTheta == 0 ? "—" : num(round(2 * pi * ropeTheta)) + " tokens"}',
      note: 'if this is below the context length, the slowest pair wraps and position becomes ambiguous',
      key: true,
    },
    { label: 'Trained context', value: '{num(ctx)}' },
    { label: 'Added parameters', value: '0' },
  ],
  'qkv:mha': [
    { label: 'KV heads', value: '{nHead} of {nHead}', note: 'one per query head' },
    {
      label: 'KV cache at full context',
      value: `{bytes(${kv('nHead')})}`,
      note: '{nLayer} layers × {nHead} heads × {dHead} dims × {num(ctx)} tokens × 2 (K,V)',
      key: true,
    },
    { label: 'Q/K/V projection params', value: '{si(3 * dModel * nHead * dHead * nLayer)}' },
  ],
  'qkv:mqa': [
    { label: 'KV heads', value: '1 of {nHead}', note: '{nHead}× fewer than MHA' },
    {
      label: 'KV cache at full context',
      value: `{bytes(${kv('1')})}`,
      note: `down from {bytes(${kv('nHead')})} under MHA`,
      key: true,
    },
    { label: 'Params saved vs MHA', value: '{si(2 * dModel * (nHead - 1) * dHead * nLayer)}' },
  ],
  'qkv:gqa': [
    {
      label: 'Groups',
      value: '{nKvHead} KV heads for {nHead} query heads',
      note: '{fixed(nHead / max(nKvHead, 1), 0)} query heads share each KV head',
    },
    {
      label: 'KV cache at full context',
      value: `{bytes(${kv('nKvHead')})}`,
      note: `{fixed(nHead / max(nKvHead, 1), 0)}× smaller than MHA's {bytes(${kv('nHead')})}`,
      key: true,
    },
    {
      label: 'Cache per token',
      value: '{fixed(2 * nLayer * nKvHead * dHead * kvBytes / 1024, 1)} KB',
      note: 'multiply by context length and batch size',
    },
  ],
  'qkv:mla': [
    { label: 'Cached per token', value: '512 latent + 64 RoPE dims', note: 'independent of head count' },
    {
      label: 'KV cache at full context',
      value: '{bytes((512 + 64) * nLayer * kvBytes * ctx)}',
      note: `against {bytes(${kv('nKvHead')})} for this model's GQA shape`,
      key: true,
    },
    {
      label: 'vs. MHA',
      value: `{fixed((${kv('nHead')}) / ((512 + 64) * nLayer * kvBytes * ctx), 1)}× smaller`,
    },
  ],
  'residual:pre-ln': [
    { label: 'Norms per layer', value: '2', note: 'one before attention, one before the FFN' },
    { label: 'Total including the final norm', value: '{nLayer * 2 + 1}', key: true },
  ],
  'tokenizer:byte-bpe': [
    { label: 'Vocabulary', value: '{num(vocab)}', note: '256 byte base plus learned merges' },
    {
      label: 'Embedding parameters',
      value: '{fixed(vocab * dModel / 1e6, 0)} M',
      note: 'vocab × d_model — often 10%+ of a small model',
      key: true,
    },
  ],
}

async function main() {
  const dir = join(root, 'content')
  await mkdir(dir, { recursive: true })

  let converted = 0
  let missing: string[] = []

  for (const block of blocks) {
    const withCosts: Block = {
      ...block,
      variants: block.variants.map((v) => {
        const key = `${block.id}:${v.id}`
        const hadCost = typeof (v as { cost?: unknown }).cost === 'function' || Array.isArray(v.cost)
        const spec = COST[key]
        if (hadCost && !spec) missing.push(key)
        if (spec) converted++
        return { ...v, cost: spec }
      }),
    }
    const file = join(dir, `${block.id}.md`)
    await writeFile(file, serializeBlockFile(withCosts), 'utf8')
    console.log(`wrote content/${block.id}.md  (${block.variants.length} variants)`)
  }

  console.log(`\n${converted} cost tables converted to templates.`)
  if (missing.length) {
    console.error(`MISSING cost conversions for: ${missing.join(', ')}`)
    process.exitCode = 1
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})

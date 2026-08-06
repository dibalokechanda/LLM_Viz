/**
 * Turning a Hugging Face `config.json` into a path through the design space.
 *
 * This is the load-bearing idea of the app. HF Viewer reads a config and draws
 * that model's module graph; we read the same config and answer a different
 * question — *which variant did this model pick at each of the thirteen
 * positions, and what did that cost it?*
 *
 * Everything here is best-effort and honest about failure. `config.json` is not
 * a specification; it is whatever the `transformers` port of a model happened
 * to need. Two models with identical architectures disagree on key names, and
 * some choices (the sampler, the tokenizer algorithm) are not in this file at
 * all. Where we cannot tell, we say `null` and the UI reports the block as
 * unresolved rather than quietly showing a default and calling it fact.
 */
import type { BlockId, Dims, ModelSpec } from '../data/types'

export type RawConfig = Record<string, unknown>

const num = (c: RawConfig, ...keys: string[]): number | undefined => {
  for (const k of keys) {
    const v = c[k]
    if (typeof v === 'number' && Number.isFinite(v)) return v
  }
  return undefined
}
const str = (c: RawConfig, ...keys: string[]): string | undefined => {
  for (const k of keys) {
    const v = c[k]
    if (typeof v === 'string') return v
  }
  return undefined
}
const bool = (c: RawConfig, ...keys: string[]): boolean | undefined => {
  for (const k of keys) {
    const v = c[k]
    if (typeof v === 'boolean') return v
  }
  return undefined
}
const obj = (c: RawConfig, ...keys: string[]): RawConfig | undefined => {
  for (const k of keys) {
    const v = c[k]
    if (v && typeof v === 'object' && !Array.isArray(v)) return v as RawConfig
  }
  return undefined
}

/**
 * Multimodal and hybrid configs nest the language model one level down, under
 * `text_config` or `llm_config`. Descend before reading anything, or an
 * 8B language model reports the vision tower's four layers.
 */
export function textConfig(c: RawConfig): RawConfig {
  const inner = obj(c, 'text_config', 'llm_config', 'language_config')
  // Only descend if the inner object actually looks like a transformer config;
  // some repos carry a stub `text_config` with nothing but a model_type.
  if (inner && num(inner, 'num_hidden_layers', 'n_layer', 'num_layers')) {
    return { ...c, ...inner }
  }
  return c
}

/* ═══════════════════════════ dimensions ═══════════════════════════ */

export function deriveDims(raw: RawConfig, name: string): Dims {
  const c = textConfig(raw)

  const nLayer = num(c, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers') ?? 0
  const dModel = num(c, 'hidden_size', 'n_embd', 'd_model', 'dim') ?? 0
  const nHead = num(c, 'num_attention_heads', 'n_head', 'num_heads') ?? 0
  // The single most informative number in the file: when it is absent the model
  // predates GQA and every head carries its own KV, so it defaults to nHead.
  const nKvHead = num(c, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads') ?? nHead
  const dHead = num(c, 'head_dim', 'attention_head_dim') ?? (nHead ? Math.floor(dModel / nHead) : 0)
  const dFF = num(c, 'intermediate_size', 'ffn_dim', 'n_inner', 'moe_intermediate_size') ?? dModel * 4
  const vocab = num(c, 'vocab_size') ?? 0
  const ctx = num(c, 'max_position_embeddings', 'n_positions', 'max_seq_len') ?? 0

  const nExperts = num(c, 'num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts')
  const nActive = num(c, 'num_experts_per_tok', 'num_selected_experts', 'moe_top_k', 'top_k')
  const nShared = num(c, 'n_shared_experts', 'num_shared_experts', 'shared_expert_intermediate_size')
    ? num(c, 'n_shared_experts', 'num_shared_experts') ?? 1
    : undefined

  // Qwen and others ship a `sliding_window` value alongside an explicit
  // `use_sliding_window: false`. Reading the number without checking the flag
  // reports windowed attention for a model that runs fully causal.
  const windowOff = bool(c, 'use_sliding_window') === false
  const window = windowOff ? undefined : num(c, 'sliding_window', 'attention_window_size')
  const dtype = str(c, 'torch_dtype', 'dtype') ?? 'bfloat16'

  return {
    name,
    nLayer,
    dModel,
    nHead,
    nKvHead,
    dHead,
    dFF,
    vocab,
    ctx,
    ropeTheta: num(c, 'rope_theta', 'rotary_emb_base'),
    nExperts,
    nActive,
    nShared,
    window: window && window > 0 ? window : undefined,
    kvBytes: /fp8|float8/i.test(dtype) ? 1 : /32/.test(dtype) ? 4 : 2,
  }
}

/* ═══════════════════════════ the path ═══════════════════════════ */

/**
 * One detector per block. Returning `null` means "this file does not say",
 * which is a real and common answer — not a failure to handle.
 */
type Detector = (c: RawConfig, d: Dims) => string | null

/** Mamba trunks do not have attention internals. Hybrids such as Jamba and
 * Zamba do, so they must remain visible to the downstream detectors. */
const modelType = (c: RawConfig) => (str(c, 'model_type') ?? '').toLowerCase()
const isPureSsm = (c: RawConfig) => {
  const type = modelType(c)
  return type.includes('mamba') && !type.includes('jamba') && !type.includes('zamba')
}

const detectors: Record<BlockId, Detector> = {
  /* config.json does not name the tokenizer algorithm; that lives in
     tokenizer.json. Inferring from model_type is a guess dressed as a fact, so
     we decline. */
  tokenizer: () => null,

  embedding: (c) =>
    bool(c, 'tie_word_embeddings', 'tie_weights') === true ? 'tied' : 'untied',

  positional: (c, d) => {
    const pet = str(c, 'position_embedding_type')
    if (pet === 'alibi' || bool(c, 'alibi', 'use_alibi')) return 'alibi'
    if (pet === 'absolute') return 'learned'
    const scaling = obj(c, 'rope_scaling')
    if (scaling) {
      const t = (str(scaling, 'rope_type', 'type') ?? '').toLowerCase()
      if (t.includes('yarn')) return 'yarn'
      if (t.includes('llama3')) return 'llama3-rope'
      if (t.includes('dynamic')) return 'dynamic-ntk'
      if (t.includes('linear')) return 'linear-interp'
      return 'rope'
    }
    if (d.ropeTheta || num(c, 'rotary_dim', 'partial_rotary_factor')) return 'rope'
    if (num(c, 'n_positions') && !d.ropeTheta) return 'learned'
    return null
  },

  norm: (c) => {
    if (num(c, 'rms_norm_eps') !== undefined) return 'rmsnorm'
    if (num(c, 'layer_norm_eps', 'layer_norm_epsilon') !== undefined) return 'layernorm'
    return null
  },

  mixer: (c, d) => {
    const type = modelType(c)
    if (type.includes('jamba') || type.includes('zamba')) return 'hybrid-ssm'
    if (type.includes('mamba2') || type.includes('mamba_2')) return 'mamba2-ssd'
    if (type.includes('mamba')) return 'mamba-s6'
    if (type.includes('s4') || type.includes('ssm')) return 's4'
    return d.nHead > 0 ? 'attention' : null
  },

  qkv: (c, d) => {
    if (isPureSsm(c)) return null
    // DeepSeek's MLA is unmistakable: it is the only scheme that stores a
    // low-rank latent instead of K and V, and it advertises the rank.
    if (num(c, 'kv_lora_rank') !== undefined) return 'mla'
    if (!d.nHead || !d.nKvHead) return null
    if (d.nKvHead === 1) return 'mqa'
    if (d.nKvHead === d.nHead) return 'mha'
    return 'gqa'
  },

  pattern: (c, d) => {
    if (isPureSsm(c)) return null
    const layerTypes = c['layer_types']
    // Gemma-style interleaving: some layers windowed, some global. This is the
    // case the one-variant-per-block model genuinely strains against, so it
    // gets its own variant rather than being rounded to one side.
    if (Array.isArray(layerTypes) && new Set(layerTypes).size > 1) return 'interleaved'
    if (d.window) return 'window'
    if (bool(c, 'is_decoder') === false) return 'bidirectional'
    return 'causal'
  },

  scores: (c) => {
    if (isPureSsm(c)) return null
    if (num(c, 'attn_logit_softcapping') !== undefined) return 'softcap'
    if (num(c, 'query_pre_attn_scalar') !== undefined) return 'custom-scale'
    return 'softmax'
  },

  kvcache: (c, d) => {
    if (isPureSsm(c)) return null
    if (num(c, 'kv_lora_rank') !== undefined) return 'mla-latent'
    if (d.window) return 'sliding'
    if (d.nKvHead && d.nHead && d.nKvHead < d.nHead) return 'grouped'
    return 'full'
  },

  ffn: (c, d) => {
    if (isPureSsm(c)) return null
    if (d.nExperts && d.nExperts > 1) {
      if (d.nShared) return 'shared-expert'
      if (d.nActive === 1) return 'switch'
      return 'topk-moe'
    }
    const act = (str(c, 'hidden_act', 'activation_function', 'hidden_activation') ?? '').toLowerCase()
    if (act.includes('silu') || act.includes('swish')) return 'swiglu'
    if (act.includes('gelu')) return 'gelu-mlp'
    if (act.includes('relu')) return 'relu-mlp'
    return null
  },

  residual: (c) => {
    if (modelType(c).includes('mhc') || (num(c, 'num_residual_streams', 'n_residual_streams') ?? 1) > 1)
      return 'mhc'
    if (bool(c, 'parallel_attn', 'parallel_residual', 'use_parallel_residual') === true)
      return 'parallel'
    if (bool(c, 'use_post_layernorm', 'post_layer_norm') === true) return 'post-ln'
    // Every decoder-only model since GPT-2 is pre-norm; saying so is safe in a
    // way that guessing the tokenizer is not.
    return 'pre-ln'
  },

  lmhead: (c) => (bool(c, 'tie_word_embeddings') === true ? 'tied-head' : 'untied-head'),

  /* Sampling is a runtime argument, not an architectural fact. generation_config
     carries a default, but it is a suggestion the caller overrides. */
  sampling: () => null,
}

export function derivePath(raw: RawConfig, dims: Dims): Pick<ModelSpec, 'path' | 'unresolved'> {
  const c = textConfig(raw)
  const path: ModelSpec['path'] = {}
  const unresolved: BlockId[] = []

  for (const [id, detect] of Object.entries(detectors) as [BlockId, Detector][]) {
    const hit = detect(c, dims)
    if (hit) path[id] = hit
    else unresolved.push(id)
  }
  return { path, unresolved }
}

/* ═══════════════════════════ fetching ═══════════════════════════ */

/** Accepts a bare `owner/model`, a full HF URL, or a URL with extra path. */
export function parseModelId(input: string): string | null {
  const s = input.trim().replace(/\/+$/, '')
  if (!s) return null
  const url = s.match(/huggingface\.co\/(.+)$/)
  const core = (url ? url[1] : s).split(/[?#]/)[0]
  // Strip repo sub-paths like /tree/main or /blob/main/config.json.
  const parts = core.split('/').filter(Boolean)
  const cut = parts.findIndex((p) => ['tree', 'blob', 'resolve', 'raw'].includes(p))
  const seg = cut > 0 ? parts.slice(0, cut) : parts
  if (seg.length < 2) return null
  return `${seg[0]}/${seg[1]}`
}

export class HFError extends Error {
  constructor(
    message: string,
    readonly hint?: string,
  ) {
    super(message)
  }
}

/**
 * Fetches straight from the public resolve endpoint. No token, no backend —
 * which keeps the app deployable to GitHub Pages, at the cost of only ever
 * seeing public models.
 */
export async function fetchConfig(id: string): Promise<RawConfig> {
  const url = `https://huggingface.co/${id}/resolve/main/config.json`
  let res: Response
  try {
    res = await fetch(url, { headers: { Accept: 'application/json' } })
  } catch {
    throw new HFError(
      'Could not reach huggingface.co',
      'The request was blocked before it got a response — usually offline, or a network that blocks the host.',
    )
  }
  if (res.status === 401 || res.status === 403) {
    throw new HFError(
      `${id} is gated or private`,
      'This app talks to Hugging Face anonymously, so it can only read public repos. Llama and Gemma repos need you to accept their licence while signed in, which an anonymous request cannot do.',
    )
  }
  if (res.status === 404) {
    throw new HFError(`No config.json at ${id}`, 'Check the owner/model spelling, or try a preset.')
  }
  if (!res.ok) throw new HFError(`Hugging Face returned ${res.status}`)

  try {
    return (await res.json()) as RawConfig
  } catch {
    throw new HFError('config.json was not valid JSON')
  }
}

export async function loadModel(input: string): Promise<ModelSpec> {
  const id = parseModelId(input)
  if (!id) throw new HFError('Enter a model as owner/name', 'For example: Qwen/Qwen3-8B')

  const raw = await fetchConfig(id)
  const dims = deriveDims(raw, id.split('/')[1] ?? id)
  if (!dims.nLayer || !dims.dModel) {
    throw new HFError(
      `${id} does not look like a transformer`,
      'No num_hidden_layers or hidden_size in the config — this may be an embedding model, an adapter, or a non-text architecture.',
    )
  }

  const { path, unresolved } = derivePath(raw, dims)
  const c = textConfig(raw)
  return {
    id,
    label: id.split('/')[1] ?? id,
    family: str(c, 'model_type') ?? 'unknown',
    dims,
    path,
    source: 'hf',
    unresolved,
    raw,
  }
}

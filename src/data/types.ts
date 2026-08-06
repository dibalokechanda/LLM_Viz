/**
 * The forward pass is described once, here, and everything else is derived
 * from it: the stack diagram, the lineage maps, the instantiated block panel,
 * and the model path inferred from a Hugging Face config.
 *
 * The central claim of this app is that **a model is a path through a fixed
 * design space**. There are thirteen positions in the forward pass. Every model
 * ever shipped picks one variant at each position. HF Viewer draws you one
 * model's module graph; this draws the space those graphs are drawn from, and
 * then shows you where a given model sits in it.
 */
import type { IconName } from '../components/Icon'

export type { IconName }

/* ═══════════════════════════ the fixed cohort ═══════════════════════════ */

/**
 * The thirteen positions. These never change between models — that is the whole
 * point. A model that "has no positional encoding" has still made a choice at
 * the `positional` block; the choice is NoPE.
 */
export type BlockId =
  | 'tokenizer'
  | 'embedding'
  | 'positional'
  | 'norm'
  | 'mixer'
  | 'qkv'
  | 'pattern'
  | 'scores'
  | 'kvcache'
  | 'ffn'
  | 'residual'
  | 'lmhead'
  | 'sampling'

/**
 * Coarse grouping, used for the lane headers and the colour family. `layer` is
 * the repeating trunk — everything in it runs `n_layer` times.
 */
export type Slot = 'input' | 'layer' | 'output'

export interface Block {
  id: BlockId
  label: string
  /** Position in the forward pass, e.g. "5". Purely for orientation. */
  ordinal: string
  icon: IconName
  slot: Slot
  /** One line, always visible on the card. */
  tagline: string
  /**
   * What this *position* is responsible for, independent of which variant
   * fills it. This is the text that stays constant across every model, and it
   * is what makes the cohort meaningful rather than arbitrary.
   */
  role: string[]
  /** Tensor shapes in and out, written over the symbols in `Dims`. */
  io: { in: string; out: string }
  /** The design space at this position. Nodes of the lineage map. */
  variants: Variant[]
  /** The progression among them. Edges of the lineage map. A DAG, not a tree. */
  lineage: Lineage[]
  /** Selected when no model is loaded. */
  defaultVariant: string
  /**
   * Some blocks resist the one-variant-per-position framing — a model can use
   * two positional schemes on different layers, for instance. Where that is
   * true, this names the escape hatch so the UI can say so rather than lying.
   */
  caveat?: string
}

/* ═══════════════════════════ the design space ═══════════════════════════ */

/**
 * How a variant sits in its block's history. Drives the node's shape and
 * weight in the lineage map — a `legacy` node is drawn faded because you only
 * need it to read old code, a `frontier` node is drawn open because the
 * verdict is not in yet.
 */
export type VariantRole =
  /** The first thing anyone did at this position. Roots of the DAG. */
  | 'origin'
  /** Strictly better than its parent on the axis the parent was weak on. */
  | 'refinement'
  /** A different bet, not a strict improvement. */
  | 'branch'
  /** Combines two previously separate lines. */
  | 'synthesis'
  /** Superseded. Kept because you will meet it in papers and old checkpoints. */
  | 'legacy'
  /** Recent enough that adoption is still an open question. */
  | 'frontier'

export interface Variant {
  id: string
  /** Short, for the lineage node and the block card. */
  label: string
  /** Spelled out, for the panel heading. */
  full: string
  /** Publication year. Drives the x-axis of the lineage map. */
  year: number
  role: VariantRole
  /** One line, shown on the lineage node and on the block card when selected. */
  tagline: string
  /**
   * The single most important sentence: what was wrong with what came before.
   * Rendered directly under the heading, because this is the thing that makes
   * the progression a progression rather than a list.
   */
  fixes?: string
  paper?: { title: string; url: string; authors?: string }
  /** Panel body. Each string is a markdown paragraph. */
  detail: string[]
  math?: MathBlock[]
  figures?: Figure[]
  code?: CodeBlock[]
  example?: Example
  distinctions?: Distinction[]
  /** Real models known to use this. Rendered as chips. */
  usedBy?: string[]
  /**
   * Live numbers for the loaded model. This is where the HF config earns its
   * keep: the same variant reads very differently at 8B and at 671B, and a
   * table of parameter counts and KV-cache bytes says so concretely.
   *
   * Declarative rather than a function so it can live in the content file with
   * everything else — `value` is a template over the model's dims. See
   * `content/expr.ts` for the syntax.
   */
  cost?: CostRow[]
  /** Deeper sub-structure, for variants that need their own map. */
  concepts?: Concept[]
  stack?: StackItem[]
}

/**
 * An edge in a block's lineage DAG. The label is the reason the arrow exists,
 * and it is not optional — an unlabelled arrow in a progression graph is just
 * decoration.
 */
export interface Lineage {
  from: string
  to: string
  /** Why. Drawn on the edge, e.g. "share one KV head across a group". */
  label: string
  kind: LineageKind
}

export type LineageKind =
  /** Straightforward descendant: same idea, developed further. */
  | 'derives'
  /** Exists specifically to repair a named failure of the source. */
  | 'fixes'
  /** Pulls an idea in from another line. Drawn dashed. */
  | 'combines'
  /** Throws the source out rather than building on it. */
  | 'replaces'
  /** Weak influence, not a direct descendant. Drawn dotted. */
  | 'inspires'

/**
 * One row of the live cost readout.
 *
 * `value` and `note` are templates evaluated against the loaded model's `Dims`:
 * text outside braces is literal, and each `{...}` is arithmetic over the
 * model's dimensions — `{bytes(2 * nLayer * nKvHead * dHead * ctx * kvBytes)}`.
 */
export interface CostRow {
  label: string
  value: string
  /** Optional second line explaining where the number came from. */
  note?: string
  /** Draw this row emphasised — the number that matters most. */
  key?: boolean
}

/* ═══════════════════════════ the model ═══════════════════════════ */

/**
 * Everything the app needs from a model's `config.json`, normalised across the
 * naming inconsistencies between architectures. Deriving this is the entire
 * job of `lib/hf.ts`.
 */
export interface Dims {
  /** Display name. */
  name: string
  /** `num_hidden_layers`. */
  nLayer: number
  /** `hidden_size`. The residual stream width. */
  dModel: number
  /** `num_attention_heads`. */
  nHead: number
  /** `num_key_value_heads`. Equal to nHead for MHA, 1 for MQA, else GQA. */
  nKvHead: number
  /** `head_dim`, or dModel / nHead when the config omits it. */
  dHead: number
  /** `intermediate_size`. Per-expert for MoE. */
  dFF: number
  /** `vocab_size`. */
  vocab: number
  /** `max_position_embeddings`. */
  ctx: number
  /** RoPE base frequency, `rope_theta`. */
  ropeTheta?: number
  /** Total experts, when the FFN is an MoE. */
  nExperts?: number
  /** Experts activated per token. */
  nActive?: number
  /** Always-on experts, DeepSeek style. */
  nShared?: number
  /** Sliding-window width, when attention is windowed. */
  window?: number
  /** Bytes per stored element in the KV cache. */
  kvBytes?: number
}

/**
 * A model, expressed the way this app thinks about models: a set of dimensions
 * plus one chosen variant per block.
 */
export interface ModelSpec {
  /** Hugging Face id, e.g. `meta-llama/Llama-3.1-8B`. */
  id: string
  label: string
  family: string
  dims: Dims
  /** The path through the design space. */
  path: Partial<Record<BlockId, string>>
  /** Where this came from. Drives the provenance badge. */
  source: 'preset' | 'hf' | 'manual'
  /**
   * Blocks whose variant could not be determined from the config alone —
   * `config.json` says nothing about the sampler, and tokenizer detection
   * needs a second file. Named rather than silently defaulted.
   */
  unresolved?: BlockId[]
  /** Raw config, for the inspector. */
  raw?: Record<string, unknown>
}

/* ═══════════════════════════ content primitives ═══════════════════════════ */
/* These carry over from RAG_Viz unchanged — the rendering components are the
   same, and the teaching vocabulary (worked maths, trade-offs, "commonly
   conflated") transfers directly. */

export interface Example {
  beforeLabel?: string
  before: string
  afterLabel?: string
  after: string
  /** Rendered in a monospace block rather than prose. */
  mono?: boolean
}

/** A displayed equation, with the symbols spelled out underneath. */
export interface MathBlock {
  title?: string
  /** KaTeX source, rendered in display mode. */
  tex: string
  /** `symbol` → what it means. Rendered as a legend below the equation. */
  where?: { sym: string; means: string }[]
  /** A worked numeric substitution, rendered as a second equation. */
  worked?: { tex: string; caption?: string }[]
  note?: string
}

/** A "these two are commonly conflated" callout. */
export interface Distinction {
  title: string
  body: string
}

export interface StackItem {
  name: string
  what: string
  url?: string
}

export interface CodeBlock {
  /** Framework label shown in the header, e.g. "PyTorch". */
  title?: string
  language: 'python' | 'bash' | 'json' | 'text'
  code: string
  note?: string
  /**
   * A compact provenance link for a teaching snippet. This is deliberately
   * separate from `note`: readers can inspect the full reference without
   * crowding the explanation beneath the code.
   */
  source?: { label: string; url: string }
}

/**
 * A node in a variant's concept sub-map. Used where a single variant has
 * enough internal structure to need its own breakdown.
 */
export interface Concept {
  id: string
  label: string
  kind: 'idea' | 'formula' | 'method' | 'metric' | 'pitfall' | 'tradeoff'
  summary: string
  detail?: string[]
  math?: MathBlock[]
  figures?: Figure[]
  code?: CodeBlock[]
  example?: Example
  children?: Concept[]
  stack?: StackItem[]
}

/* ═══════════════════════════ figures ═══════════════════════════ */

/**
 * Small inline diagrams, declared as data so the content files stay readable.
 * The first six kinds carry over from RAG_Viz; `heatmap`, `routing` and
 * `tensor` are new, and exist because attention masks, expert routing and
 * shape bookkeeping are the three things about transformer internals that
 * prose consistently fails to convey.
 */
export type Figure =
  | {
      kind: 'bars'
      title?: string
      caption?: string
      categories: string[]
      series: { label: string; values: number[] }[]
      cutoff?: { after: number; label: string }
      highlight?: number[]
      yMax?: number
      showValues?: boolean
    }
  | {
      kind: 'curve'
      title?: string
      caption?: string
      xLabel: string
      yLabel: string
      lines: { label?: string; points: [number, number][]; dashed?: boolean }[]
      marks?: { x: number; y: number; label: string }[]
      xTicks?: { at: number; label: string }[]
      yTicks?: { at: number; label: string }[]
    }
  /** Horizontal tracks split into spans. */
  | {
      kind: 'segments'
      title?: string
      caption?: string
      total: number
      rows: {
        label: string
        spans: { from: number; to: number; label?: string; ghost?: boolean }[]
      }[]
    }
  | {
      kind: 'ranked'
      title?: string
      caption?: string
      grades: number[]
      maxGrade?: number
      markFirstRelevant?: boolean
    }
  /** Rows of labelled boxes with optional arrows between them. */
  | {
      kind: 'blocks'
      title?: string
      caption?: string
      rows: {
        label?: string
        boxes: { text: string; span?: number; filled?: boolean; dashed?: boolean }[]
        arrow?: string
      }[]
    }
  | {
      kind: 'network'
      title?: string
      caption?: string
      nodes: { id: string; x: number; y: number; isEntry?: boolean; isTarget?: boolean; label?: string }[]
      links: { source: string; target: string }[]
      path?: string[]
      annotations?: { x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end' }[]
      steps?: string[]
    }
  | {
      kind: 'layered'
      title?: string
      caption?: string
      layers: number[]
      nodes: { id: string; x: number; y: number; maxLayer: number; isEntry?: boolean; isTarget?: boolean }[]
      links: { source: string; target: string; layer: number }[]
      path?: { node: string; layer: number }[]
      layerLabels?: { layer: number; text: string }[]
      annotations?: { x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end' }[]
      steps?: string[]
    }
  /**
   * An n×n attention matrix. `mask` decides which cells are visible, and it is
   * a named pattern rather than a blob of data so that causal, sliding-window,
   * sink and dilated masks all come from one line of content. Optional
   * `weights` shade the visible cells to show where attention actually lands.
   */
  | {
      kind: 'heatmap'
      title?: string
      caption?: string
      /** Sequence length. Kept small (8–16) so individual cells stay legible. */
      n: number
      mask: 'full' | 'causal' | 'window' | 'sink' | 'dilated' | 'block'
      /** Window width, for `window`, `dilated` and `block`. */
      w?: number
      /** Number of always-visible prefix tokens, for `sink`. */
      sinks?: number
      /** Axis tick labels. Defaults to token indices. */
      tokens?: string[]
      /** Highlight one query row, to read off "what token i can see". */
      focusRow?: number
      steps?: string[]
    }
  /**
   * MoE routing: tokens on the left, experts on the right, edges for the
   * top-k assignment. Shows load imbalance, which is the entire reason
   * auxiliary losses exist.
   */
  | {
      kind: 'routing'
      title?: string
      caption?: string
      tokens: string[]
      experts: string[]
      /** Per token, the expert indices it routes to, with gate weights. */
      routes: { token: number; expert: number; weight: number }[]
      /** Experts that run for every token regardless of the router. */
      shared?: number[]
      /** Draw the per-expert load histogram underneath. */
      showLoad?: boolean
      steps?: string[]
    }
  /**
   * Token embeddings, in three linked panels: the raw signed vector, a 2D
   * projection of the space, and cosine similarity to every other token.
   * Interactive and self-contained — the vectors are synthetic, generated with
   * cluster structure and a deliberate semantic axis so the classic analogy
   * resolves. See the component for why that is the honest choice here.
   */
  | {
      kind: 'embedding'
      title?: string
      caption?: string
      steps?: string[]
      /** Draw the parallel offset arrows on load. Defaults to true. */
      analogy?: boolean
    }
  /**
   * Shape bookkeeping. A row of labelled tensors with the operation between
   * them — the thing every transformer explanation assumes you are tracking
   * in your head, drawn explicitly instead.
   */
  | {
      kind: 'tensor'
      title?: string
      caption?: string
      steps?: string[]
      chain: {
        label: string
        /** e.g. ['B', 'T', 'd_model'] */
        shape: string[]
        /** Operation applied to get here, drawn on the incoming arrow. */
        via?: string
        /** Draw emphasised — the tensor the surrounding prose is about. */
        focus?: boolean
      }[]
    }

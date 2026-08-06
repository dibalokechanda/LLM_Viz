import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
  type NodeMouseHandler,
} from '@xyflow/react'
import type { Block, Variant, VariantRole } from '../data/types'
import { useModel } from '../ModelContext'
import EnsureMeasured from './EnsureMeasured'
import VariantDetail from './VariantDetail'
import ConceptMap from './ConceptMap'
import Icon from './Icon'

/**
 * The lineage map: a block's variants laid out as a dated progression.
 *
 * A radial mind-map was the obvious thing to reach for, and it is wrong here.
 * Radial layouts imply a tree — one parent per node, siblings unrelated — and
 * these graphs are emphatically not trees. GQA has two parents. MLA draws from
 * MQA and GQA both. Encoding that as a DAG on a time axis is the whole point:
 * you should be able to read left to right and see *when* each idea arrived,
 * and follow the arrows to see *what it was reacting to*.
 */

/* Ranks run up the page and siblings spread sideways, matching the stack's
   bottom-to-top reading. `RANK_GAP` is generous because every edge carries a
   label, and a cramped vertical gap turns those labels into a stack of text. */
/*
 * These are handed to React Flow as each node's declared width/height rather
 * than left for it to measure — see the note in StackView.tsx. The rank packing
 * and the edge/box collision test both work against these exact boxes, so the
 * CSS fills them rather than defining them.
 */
const NODE_W = 214
const NODE_H = 136
const RANK_GAP = 132
const SIB_GAP = 58
/** How far a detour route runs vertically before turning out to its corridor. */
const DETOUR_STUB = 38

const ROLE_ORDER: Record<VariantRole, number> = {
  origin: 0,
  legacy: 1,
  refinement: 2,
  branch: 3,
  synthesis: 4,
  frontier: 5,
}

const ROLE_LABEL: Record<VariantRole, string> = {
  origin: 'origin',
  refinement: 'refinement',
  branch: 'branch',
  synthesis: 'synthesis',
  legacy: 'legacy',
  frontier: 'frontier',
}

interface VNodeData extends Record<string, unknown> {
  variant: Variant
  selected: boolean
  /** This is what the loaded model actually uses. */
  isModel: boolean
}

function VariantNode({ data }: NodeProps<Node<VNodeData, 'variant'>>) {
  const { variant, selected, isModel } = data
  const conceptCount = variant.concepts?.length ?? 0
  return (
    <div
      className={`ln-node role-${variant.role}${selected ? ' is-selected' : ''}${isModel ? ' is-model' : ''}`}
    >
      {/* Descendants sit above their parents, so influence arrives at the
          bottom and leaves from the top — same reading as the stack. */}
      <Handle type="target" position={Position.Bottom} />
      <div className="ln-node-head">
        <span className="ln-node-label">{variant.label}</span>
        <span className="ln-node-meta">
          <span className="ln-node-year">{variant.year}</span>
          {conceptCount > 0 && (
            <span className="ln-concept-dot" title={`${conceptCount} concept card${conceptCount === 1 ? '' : 's'}`}>
              <Icon name="bulb" size={12} />
            </span>
          )}
        </span>
      </div>
      <div className="ln-node-tagline">{variant.tagline}</div>
      <span className={`ln-role role-${variant.role}`}>{ROLE_LABEL[variant.role]}</span>
      {isModel && <span className="ln-model-tick">this model</span>}
      <Handle type="source" position={Position.Top} />
    </div>
  )
}

/**
 * Edges carry a label saying *why* the arrow exists, and several edges often
 * leave one node at once — MHA fans out to both MQA and GQA, Softmax to three
 * successors. React Flow parks every label at its path's midpoint, so those
 * siblings land at identical heights and overprint each other.
 *
 * This edge shifts the label along the path by a per-edge offset computed in
 * `layout`, which is the whole reason it exists.
 */
function LineageEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  data,
  markerEnd,
}: EdgeProps) {
  const bow = (data?.bow as number) ?? 0
  const corridor = data?.corridorX as number | undefined

  let path: string
  let labelX: number
  let labelY: number

  if (corridor !== undefined) {
    /*
     * The escape hatch for a long skip through a crowded map. A bow cannot
     * help here: the curve has to start and end in its own lane, so on the way
     * out it sweeps through whatever sits beside the source — RoPE reaching
     * llama3-rope crosses NoPE low down, well before the curve's apex clears
     * the graph. So route it orthogonally instead: a short stub out of the
     * source, a vertical run in a corridor outside every card, and a stub back
     * in at the target.
     */
    const y1 = sourceY - DETOUR_STUB
    const y2 = targetY + DETOUR_STUB
    const out = corridor > sourceX ? 1 : -1
    const back = corridor > targetX ? -1 : 1
    const r = 16
    path = [
      `M ${sourceX} ${sourceY}`,
      `L ${sourceX} ${y1 + r}`,
      `Q ${sourceX} ${y1} ${sourceX + out * r} ${y1}`,
      `L ${corridor - out * r} ${y1}`,
      `Q ${corridor} ${y1} ${corridor} ${y1 - r}`,
      `L ${corridor} ${y2 + r}`,
      `Q ${corridor} ${y2} ${corridor + back * r} ${y2}`,
      `L ${targetX - back * r} ${y2}`,
      `Q ${targetX} ${y2} ${targetX} ${y2 - r}`,
      `L ${targetX} ${targetY}`,
    ].join(' ')
    labelX = corridor
    labelY = (y1 + y2) / 2
  } else if (bow) {
    /*
     * An edge that skips a rank — sinusoidal → relative jumps straight past
     * learned-absolute — runs dead through the card in between, and drops its
     * label on top of it. Bowing it sideways fixes both problems and is the
     * more truthful drawing: the influence bypasses that node, it does not
     * pass through it.
     */
    const dy = sourceY - targetY
    path = `M ${sourceX} ${sourceY} C ${sourceX + bow} ${sourceY - dy * 0.3}, ${targetX + bow} ${targetY + dy * 0.3}, ${targetX} ${targetY}`
    labelX = (sourceX + targetX) / 2 + bow * 0.75
    labelY = (sourceY + targetY) / 2
  } else {
    const [p, lx, ly] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 14,
    })
    path = p
    labelX = lx
    labelY = ly + ((data?.labelShift as number) ?? 0)
  }

  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} />
      {label && (
        <EdgeLabelRenderer>
          <div
            className="ln-edge-label"
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

const nodeTypes = { variant: VariantNode }
const edgeTypes = { lineage: LineageEdge }

/**
 * Lays the DAG out with time running *up* the page and siblings spread
 * sideways, so the oldest idea sits at the bottom and the newest at the top —
 * the same reading direction as the stack.
 *
 * Years are collapsed to ranks rather than used directly: the real dates
 * cluster (2023 alone holds four positional-encoding entries) and a literal
 * time axis would overlap them into illegibility. Rank preserves the ordering,
 * which is the part that carries meaning.
 */
function layout(block: Block, modelVariant?: string, selectedId?: string | null) {
  const sorted = [...block.variants].sort(
    (a, b) => a.year - b.year || ROLE_ORDER[a.role] - ROLE_ORDER[b.role],
  )

  // Distinct years become ranks — but a rank is a floor, not a fixed slot.
  // GEGLU and SwiGLU are both 2020 and one derives from the other; placing them
  // on the same rank would draw that arrow sideways or backwards. So a node
  // sits at least one rank above every parent, which keeps the graph strictly
  // upward while still reading chronologically wherever it can.
  const years = [...new Set(sorted.map((v) => v.year))].sort((a, b) => a - b)
  const yearCol = new Map(years.map((y, i) => [y, i]))

  const parentsOf = new Map<string, string[]>()
  for (const l of block.lineage) {
    parentsOf.set(l.to, [...(parentsOf.get(l.to) ?? []), l.from])
  }

  // Longest path from any root, floored at the node's year column. Recursive
  // rather than a single pass over `sorted`, because a child can precede its
  // parent in year order — SwiGLU and GEGLU are both 2020 — and a single pass
  // would then read the parent's column before it was assigned.
  const yearOf = new Map(sorted.map((v) => [v.id, v.year]))
  const col = new Map<string, number>()
  const visiting = new Set<string>()

  const depthOf = (id: string): number => {
    const cached = col.get(id)
    if (cached !== undefined) return cached
    // Content bug guard: a cycle in the lineage would otherwise hang the tab.
    if (visiting.has(id)) return yearCol.get(yearOf.get(id)!) ?? 0
    visiting.add(id)
    const floor = yearCol.get(yearOf.get(id)!) ?? 0
    const d = Math.max(floor, ...(parentsOf.get(id) ?? []).map((p) => depthOf(p) + 1))
    visiting.delete(id)
    col.set(id, d)
    return d
  }
  for (const v of sorted) depthOf(v.id)

  // Within a rank, spread sideways. Nudge each node toward the mean lane of its
  // parents so edges stay short and the progression reads as a flow rather than
  // a grid.
  const laneOf = new Map<string, number>()

  for (const v of sorted) {
    const c = col.get(v.id)!
    const ps = (parentsOf.get(v.id) ?? [])
      .map((p) => laneOf.get(p))
      .filter((r): r is number => r !== undefined)
    const want = ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : 0
    // Take the wanted lane if free, otherwise the next free lane to its right.
    const used = new Set(
      sorted.filter((o) => laneOf.has(o.id) && col.get(o.id) === c).map((o) => laneOf.get(o.id)!),
    )
    let r = Math.round(want)
    while (used.has(r)) r++
    laneOf.set(v.id, r)
  }

  const minLane = Math.min(...laneOf.values())
  const maxRank = Math.max(...col.values())

  const nodes: Node<VNodeData, 'variant'>[] = sorted.map((v) => ({
    id: v.id,
    type: 'variant',
    // Declared, not measured — see the note on NODE_W/NODE_H.
    width: NODE_W,
    height: NODE_H,
    position: {
      x: (laneOf.get(v.id)! - minLane) * (NODE_W + SIB_GAP),
      // Invert the rank so rank 0 lands at the bottom of the canvas.
      y: (maxRank - col.get(v.id)!) * (NODE_H + RANK_GAP),
    },
    data: { variant: v, selected: selectedId === v.id, isModel: modelVariant === v.id },
  }))

  /* ── Edge routing ──────────────────────────────────────────────────────
   *
   * Guessing which edges collide does not work: a skip over a rank, a long
   * diagonal across lanes, and a sibling fan can all end up drawn through a
   * card. So test it. Every candidate route is sampled and checked against the
   * real node boxes, and an edge that hits one is bowed sideways until it
   * clears — which is also the more truthful drawing, since the influence
   * bypasses that node rather than passing through it.
   *
   * This depends on NODE_H matching the card's fixed CSS height exactly.
   */
  const boxes = nodes.map((n) => ({ id: n.id, x: n.position.x, y: n.position.y }))
  const MARGIN = 12

  /** How many sampled points of a route fall inside a card that is not an endpoint. */
  const countHits = (pts: [number, number][], a: string, b: string) => {
    let n = 0
    for (const r of boxes) {
      if (r.id === a || r.id === b) continue
      for (const [px, py] of pts) {
        if (
          px > r.x - MARGIN &&
          px < r.x + NODE_W + MARGIN &&
          py > r.y - MARGIN &&
          py < r.y + NODE_H + MARGIN
        ) {
          n++
        }
      }
    }
    return n
  }

  /*
   * Sample density is load-bearing, not a tuning detail. A long bowed edge can
   * be 700px of arc; at 25 samples the gaps are ~28px and the test steps clean
   * over a card corner, reporting a route as clear when it visibly clips one.
   * Sample counts here are sized so the spacing stays well under a card corner.
   */
  const lerp = (ax: number, ay: number, bx: number, by: number, n = 24) =>
    Array.from(
      { length: n + 1 },
      (_, i) => [ax + ((bx - ax) * i) / n, ay + ((by - ay) * i) / n] as [number, number],
    )

  /** Samples of React Flow's smoothstep route: up, across at mid-height, up. */
  const stepPts = (sx: number, sy: number, tx: number, ty: number) => {
    const mid = (sy + ty) / 2
    return [...lerp(sx, sy, sx, mid), ...lerp(sx, mid, tx, mid), ...lerp(tx, mid, tx, ty)]
  }

  /** Samples of the orthogonal detour drawn by LineageEdge for the same endpoints. */
  const detourPts = (sx: number, sy: number, tx: number, ty: number, cx: number) => {
    const y1 = sy - DETOUR_STUB
    const y2 = ty + DETOUR_STUB
    return [
      ...lerp(sx, sy, sx, y1, 6),
      ...lerp(sx, y1, cx, y1, 40),
      ...lerp(cx, y1, cx, y2, 60),
      ...lerp(cx, y2, tx, y2, 40),
      ...lerp(tx, y2, tx, ty, 6),
    ]
  }

  /** Samples of the bowed cubic drawn by LineageEdge for the same endpoints. */
  const bowPts = (sx: number, sy: number, tx: number, ty: number, bow: number) => {
    const dy = sy - ty
    const c1x = sx + bow
    const c1y = sy - dy * 0.3
    const c2x = tx + bow
    const c2y = ty + dy * 0.3
    // Density scaled to the arc: enough samples that no card corner fits
    // between two of them. See the note on `lerp`.
    const steps = Math.max(40, Math.ceil((Math.abs(dy) + Math.abs(bow) * 2) / 8))
    return Array.from({ length: steps + 1 }, (_, i) => {
      const t = i / steps
      const m = 1 - t
      return [
        m ** 3 * sx + 3 * m * m * t * c1x + 3 * m * t * t * c2x + t ** 3 * tx,
        m ** 3 * sy + 3 * m * m * t * c1y + 3 * m * t * t * c2y + t ** 3 * ty,
      ] as [number, number]
    })
  }

  // Stagger labels among edges that leave the same node, so siblings do not
  // print on top of one another at the shared midpoint height.
  const outCount = new Map<string, number>()
  for (const l of block.lineage) outCount.set(l.from, (outCount.get(l.from) ?? 0) + 1)
  const seen = new Map<string, number>()

  const pos = new Map(nodes.map((n) => [n.id, n.position]))

  const edges: Edge[] = block.lineage.map((l) => {
    const i = seen.get(l.from) ?? 0
    seen.set(l.from, i + 1)
    const n = outCount.get(l.from)!

    const from = pos.get(l.from)!
    const to = pos.get(l.to)!
    // Handles: source on the card's top edge, target on its bottom edge.
    const sx = from.x + NODE_W / 2
    const sy = from.y
    const tx = to.x + NODE_W / 2
    const ty = to.y + NODE_H

    let bow = 0
    let corridorX: number | undefined
    if (countHits(stepPts(sx, sy, tx, ty), l.from, l.to) > 0) {
      /*
       * Try both sides at widening offsets and keep the best route, scored by
       * how much of it still lands on a card and then by how little it detours.
       * Picking the *best* rather than the first-that-clears matters in a dense
       * map: where every candidate clips something, "least bad" is a real
       * answer, whereas falling back to the widest arbitrarily is not.
       */
      const mags = [NODE_W / 2 + 40, NODE_W + 70, NODE_W * 1.6 + 100, NODE_W * 2.2 + 130]
      const first = i % 2 === 0 ? -1 : 1
      const trials = mags.flatMap((m) => [first * m, -first * m])

      let best = trials[0]
      let bestScore = Infinity
      for (const b of trials) {
        const score = countHits(bowPts(sx, sy, tx, ty, b), l.from, l.to) * 1000 + Math.abs(b)
        if (score < bestScore) {
          bestScore = score
          best = b
        }
        if (score < 1000) break // clear route, and trials only widen from here
      }

      if (bestScore >= 1000) {
        /*
         * Nothing in the trial set is clear. That happens on a long skip through
         * a crowded map — RoPE reaches llama3-rope across four ranks, with three
         * cards in its own lane, one lane to the left and one to the right.
         *
         * No bow can fix it. The curve has to begin and end in its own lane, so
         * on the way out it crosses whatever sits beside the source: widening
         * the bow until the *apex* clears the graph still leaves the ascent
         * cutting through NoPE, which sits low and to the right of RoPE. The
         * only shape that works is an explicit detour — out, up a corridor
         * beyond every card, and back in.
         */
        const minX = Math.min(...boxes.map((r) => r.x))
        const maxX = Math.max(...boxes.map((r) => r.x + NODE_W))
        const clear = NODE_W / 2 + 46
        const candidates = [maxX + clear, minX - clear].sort(
          (p, q) => Math.abs(p - sx) - Math.abs(q - sx),
        )
        // Still test it: the two horizontal stubs can cross a card even though
        // the vertical run cannot.
        corridorX =
          candidates.find((cx) => countHits(detourPts(sx, sy, tx, ty, cx), l.from, l.to) === 0) ??
          candidates[0]
        bow = 0
      } else {
        bow = best
      }
    }

    return {
      id: `${l.from}->${l.to}`,
      source: l.from,
      target: l.to,
      label: l.label,
      type: 'lineage',
      className: `ln-edge kind-${l.kind}`,
      data: {
        labelShift: n > 1 && !bow && corridorX === undefined ? (i - (n - 1) / 2) * 30 : 0,
        bow,
        corridorX,
      },
      markerEnd: { type: MarkerType.ArrowClosed, width: 13, height: 13 },
    }
  })

  return { nodes, edges }
}

function Inner({ block, onClose }: { block: Block; onClose: () => void }) {
  const { path, model, setVariant } = useModel()
  const current = path[block.id]
  const modelVariant = model?.path[block.id]
  const [selectedId, setSelectedId] = useState<string | null>(current ?? block.defaultVariant)
  const [conceptMapFor, setConceptMapFor] = useState<Variant | null>(null)

  // When a concept map is open it owns Escape. Closing both overlays with one
  // key press makes it too easy to lose the lineage context the user came from.
  useEffect(() => {
    if (conceptMapFor) return
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [conceptMapFor, onClose])

  const { nodes, edges } = useMemo(
    () => layout(block, modelVariant, selectedId),
    [block, modelVariant, selectedId],
  )
  const nodeIds = useMemo(() => nodes.map((n) => n.id), [nodes])

  const selected = block.variants.find((v) => v.id === selectedId) ?? null

  const onNodeClick: NodeMouseHandler = useCallback((_e, node) => setSelectedId(node.id), [])

  return (
    <div className="ln-overlay" role="dialog" aria-modal="true" aria-label={`${block.label} lineage`}>
      <header className="ln-head">
        <div className="ln-head-text">
          <span className="ln-eyebrow">
            <Icon name={block.icon} size={16} /> Block {block.ordinal} · the design space
          </span>
          <h2>{block.label}</h2>
          <p>{block.tagline}</p>
        </div>
        <button className="ln-close" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </header>

      <div className="ln-body">
        <div className="ln-canvas">
          <ReactFlow
            id="lineage"
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodeClick={onNodeClick}
            fitView
            fitViewOptions={{ padding: 0.14, minZoom: 0.3 }}
            minZoom={0.25}
            maxZoom={1.5}
            nodesDraggable={false}
            nodesConnectable={false}
            proOptions={{ hideAttribution: true }}
          >
            <EnsureMeasured ids={nodeIds} />
            <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="var(--bg-grid)" />
            <Controls showInteractive={false} position="bottom-left" />
          </ReactFlow>

          {/* Always on screen, not a toggle. Two of the role hues sit under
              3:1 on the light surface, so the legend plus the printed role on
              each card are what keep the encoding readable. */}
          <div className="ln-legend">
            <div className="ln-legend-head">Role</div>
            <span><i className="ln-swatch origin" /> origin</span>
            <span><i className="ln-swatch refinement" /> refinement</span>
            <span><i className="ln-swatch branch" /> branch</span>
            <span><i className="ln-swatch synthesis" /> synthesis</span>
            <span><i className="ln-swatch frontier" /> frontier</span>
            <span><i className="ln-swatch legacy" /> legacy</span>
            {model && (
              <>
                <div className="ln-legend-head">Path</div>
                <span><i className="ln-swatch model" /> {model.label} uses this</span>
              </>
            )}
            <div className="ln-legend-head">Relationship</div>
            <span><i className="ln-key derives" /> derives from</span>
            <span><i className="ln-key fixes" /> fixes a failure</span>
            <span><i className="ln-key combines" /> combines lines</span>
            <span><i className="ln-key replaces" /> replaces</span>
            <span><i className="ln-key inspires" /> influences</span>
          </div>
        </div>

        <aside className="ln-panel">
          {selected ? (
            <>
              <div className="ln-panel-actions">
                {current === selected.id ? (
                  <span className="ln-active-tag">Rendered in the stack</span>
                ) : (
                  <button
                    className="ln-use"
                    onClick={() => {
                      setVariant(block.id, selected.id)
                      onClose()
                    }}
                  >
                    Render this block as {selected.label} →
                  </button>
                )}
                {modelVariant === selected.id && (
                  <span className="ln-model-tag">{model?.label} uses this</span>
                )}
                {selected.concepts && selected.concepts.length > 0 && (
                  <button className="ln-concepts" onClick={() => setConceptMapFor(selected)}>
                    <Icon name="bulb" size={15} />
                    Explore concept map
                    <span>{selected.concepts.length}</span>
                  </button>
                )}
              </div>
              <VariantDetail block={block} variant={selected} />
            </>
          ) : (
            <p className="ln-empty">Pick a node to read what it does and what it cost.</p>
          )}
        </aside>
      </div>
      {conceptMapFor && (
        <ConceptMap block={block} variant={conceptMapFor} onClose={() => setConceptMapFor(null)} />
      )}
    </div>
  )
}

export default function LineageMap({ block, onClose }: { block: Block; onClose: () => void }) {
  // No provider — see the note in App.tsx. The <ReactFlow> inside makes its own.
  return <Inner block={block} onClose={onClose} />
}

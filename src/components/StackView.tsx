import { useCallback, useMemo } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeMouseHandler,
} from '@xyflow/react'
import type { Block, BlockId, Slot, Variant } from '../data/types'
import { blocks } from '../content'
import { useModel } from '../ModelContext'
import EnsureMeasured from './EnsureMeasured'
import Icon from './Icon'

/**
 * The stack, running bottom-to-top.
 *
 * Direction carries meaning here. Text enters at the bottom and a token leaves
 * at the top, so the diagram reads the way the residual stream actually runs —
 * upward, through a trunk that repeats. A left-to-right layout would have
 * implied a pipeline of separate stages handing work along, which is the wrong
 * mental model: these are transformations accumulating into one stream.
 *
 * Mechanically that means every node sources from its Top handle and targets on
 * its Bottom handle, and y decreases as the forward pass advances.
 */

/*
 * Sizes live here and are handed to React Flow on every node, rather than being
 * left for it to measure off the DOM. Two reasons, one of them load-bearing:
 * the layout packs ranks on these exact numbers, and React Flow's measurement
 * pass is a race that intermittently never completes — when it stalls, nodes
 * paint fine but `handleBounds` is never captured and every edge is silently
 * dropped, with nothing in the console. Declaring the geometry removes the race.
 * The CSS fills these boxes rather than defining them.
 */
const CARD_W = 380
const CARD_H = 122
const LANE_W = 214
const GAP = 26
const GROUP_GAP = 52
const COL_X = 0
const LANE_X = -250

/** Bottom to top: input at the bottom of the canvas, output at the top. */
const SLOT_ORDER: Slot[] = ['input', 'layer', 'output']
const ATTENTION_INTERNALS = new Set<BlockId>(['qkv', 'pattern', 'scores', 'kvcache', 'ffn'])

const SLOT_META: Record<Slot, { title: string; sub: string }> = {
  input: { title: 'Input', sub: 'Text becomes vectors. Runs once.' },
  layer: { title: 'The trunk', sub: 'Everything here repeats every layer.' },
  output: { title: 'Output', sub: 'Vectors become a token. Once per step.' },
}

interface BNodeData extends Record<string, unknown> {
  block: Block
  variant: Variant | undefined
  selected: boolean
  diverged: boolean
  unresolved: boolean
  bypassed: boolean
}

interface LaneData extends Record<string, unknown> {
  title: string
  sub: string
  slot: Slot
  count: number
}

function BlockNode({ data }: NodeProps<Node<BNodeData, 'block'>>) {
  const { block, variant, selected, diverged, unresolved, bypassed } = data
  return (
    <div
      className={`bk ${block.slot}${selected ? ' is-selected' : ''}${unresolved && !bypassed ? ' is-unresolved' : ''}${bypassed ? ' is-bypassed' : ''}`}
    >
      {/* Flow runs upward: work arrives at the bottom, leaves from the top. */}
      <Handle type="target" position={Position.Bottom} />

      <div className="bk-head">
        <span className="bk-icon">
          <Icon name={block.icon} size={18} />
        </span>
        <span className="bk-title">{block.label}</span>
        <span className="bk-ord">§{block.ordinal}</span>
        {diverged && (
          <span className="bk-flag diverged" title="Moved off the model's choice">
            edited
          </span>
        )}
        {bypassed && !diverged && (
          <span className="bk-flag bypassed" title="This pure SSM model does not run this attention component">
            SSM bypass
          </span>
        )}
        {unresolved && !diverged && !bypassed && (
          <span className="bk-flag unresolved" title="config.json does not say">
            not in config
          </span>
        )}
      </div>

      <div className="bk-variant">
        <span className="bk-variant-label">{bypassed ? 'Not used' : variant?.label ?? '—'}</span>
        <span className="bk-variant-tagline">{bypassed ? 'The selected state-space mixer replaces this attention component.' : variant?.tagline ?? block.tagline}</span>
      </div>

      <div className="bk-foot">
        <span className="bk-count">{block.variants.length} variants</span>
        <span className="bk-explore">open lineage →</span>
      </div>

      <Handle type="source" position={Position.Top} />
    </div>
  )
}

function LaneNode({ data }: NodeProps<Node<LaneData, 'lane'>>) {
  return (
    <div className={`bk-lane ${data.slot}`}>
      <span className="bk-lane-rule" />
      <h2>{data.title}</h2>
      <p>{data.sub}</p>
      <span className="bk-lane-count">{data.count} blocks</span>
    </div>
  )
}

const nodeTypes = { block: BlockNode, lane: LaneNode }

export default function StackView({
  selectedId,
  onSelect,
  onExplore,
}: {
  selectedId: BlockId | null
  onSelect: (id: BlockId | null) => void
  onExplore: (id: BlockId) => void
}) {
  const { path, dims, diverged, unresolved } = useModel()

  const { nodes, edges, total } = useMemo(() => {
    const bySlot: Record<Slot, Block[]> = { input: [], layer: [], output: [] }
    for (const b of blocks) bySlot[b.slot].push(b)

    const ns: Node<BNodeData | LaneData>[] = []
    const es: Edge[] = []
    const pureSsm = path.mixer === 'mamba-s6' || path.mixer === 'mamba2-ssd'

    // Total height first, so y can be measured downward from the top while the
    // reading order runs upward from the bottom.
    const total =
      blocks.length * CARD_H + (blocks.length - 1) * GAP + (SLOT_ORDER.length - 1) * (GROUP_GAP - GAP)

    let y = total - CARD_H // start at the bottom-most card
    const ordered: BlockId[] = []

    for (const slot of SLOT_ORDER) {
      const list = bySlot[slot]
      const groupBottom = y + CARD_H

      for (const b of list) {
        ns.push({
          id: b.id,
          type: 'block',
          position: { x: COL_X, y },
          width: CARD_W,
          height: CARD_H,
          data: {
            block: b,
            variant: b.variants.find((v) => v.id === path[b.id]),
            selected: selectedId === b.id,
            diverged: diverged.has(b.id),
            unresolved: unresolved.has(b.id),
            bypassed: pureSsm && ATTENTION_INTERNALS.has(b.id),
          },
        } as Node<BNodeData, 'block'>)
        ordered.push(b.id)
        y -= CARD_H + GAP
      }

      const groupTop = y + CARD_H + GAP
      const meta = SLOT_META[slot]
      ns.push({
        id: `lane-${slot}`,
        type: 'lane',
        position: { x: LANE_X, y: (groupTop + groupBottom) / 2 - 52 },
        width: LANE_W,
        height: 104,
        data: {
          title: slot === 'layer' ? `${meta.title} · ×${dims.nLayer}` : meta.title,
          sub: meta.sub,
          slot,
          count: list.length,
        },
        draggable: false,
        selectable: false,
      } as Node<LaneData, 'lane'>)

      y -= GROUP_GAP - GAP
    }

    // One edge per consecutive pair, pointing the way the data moves: up.
    for (let i = 0; i < ordered.length - 1; i++) {
      const from = ordered[i]
      const to = ordered[i + 1]
      const crossesGroup =
        blocks.find((b) => b.id === from)!.slot !== blocks.find((b) => b.id === to)!.slot
      es.push({
        id: `${from}->${to}`,
        source: from,
        target: to,
        type: 'smoothstep',
        className: `bk-edge${crossesGroup ? ' join' : ''}`,
        label: crossesGroup && to === 'lmhead' ? `after ${dims.nLayer} layers` : undefined,
        markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
      })
    }

    return { nodes: ns, edges: es, total }
  }, [path, selectedId, diverged, unresolved, dims.nLayer])

  const nodeIds = useMemo(() => nodes.map((n) => n.id), [nodes])

  const onNodeClick: NodeMouseHandler = useCallback(
    (e, node) => {
      if (node.type === 'lane') return
      const id = node.id as BlockId
      const el = e.target as HTMLElement
      if (el.closest('.bk-explore') || el.closest('.bk-count')) onExplore(id)
      else onSelect(selectedId === id ? null : id)
    },
    [onExplore, onSelect, selectedId],
  )

  return (
    <div className="stack-wrap">
      <ReactFlow
        id="stack"
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={() => onSelect(null)}
        /*
         * `fitView` is required, not cosmetic: it is what drives React Flow's
         * initial measurement pass. Without it the nodes still paint but their
         * handle bounds are never captured, and every edge silently fails to
         * render — the flow looks like a column of disconnected cards.
         */
        fitView
        /*
         * The fit it produces is not the one we want, though. Thirteen stacked
         * cards are ~1750px tall, so fitting all of them lands near 0.25 zoom
         * and every label goes unreadable. Since the diagram reads bottom-to-top,
         * re-frame on init to where the forward pass starts and let the user
         * travel up. The fit control still shows the whole shape on demand.
         */
        onInit={(inst) => {
          // Defer past init: calling fitBounds synchronously inside onInit runs
          // before React Flow has measured its nodes, and re-entering the
          // viewport machinery there leaves the measurement pass unfinished —
          // which drops every edge. A frame later it is safe.
          const compact = window.matchMedia('(max-width: 860px)').matches
          requestAnimationFrame(() =>
            inst.fitBounds(
              {
                x: LANE_X,
                // On a phone, framing five cards makes every card illegible.
                // Start on the first two stages instead; the whole diagram is
                // still one pinch or fit-control away.
                y: total - (compact ? 350 : 760),
                width: 380 - LANE_X,
                height: compact ? 350 : 760,
              },
              { padding: compact ? 0.02 : 0.05 },
            ),
          )
        }}
        minZoom={0.18}
        maxZoom={1.4}
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <EnsureMeasured ids={nodeIds} />
        <Background variant={BackgroundVariant.Dots} gap={26} size={1} color="var(--bg-grid)" />
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
    </div>
  )
}

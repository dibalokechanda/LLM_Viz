import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeMouseHandler,
  type NodeProps,
} from '@xyflow/react'
import type { Block, Concept, StackItem, Variant } from '../data/types'
import MathBlockView from './Math'
import FigureView from './Figure'
import CodeBlockView from './Code'
import Prose from './Prose'
import Icon, { type IconName } from './Icon'
import EnsureMeasured from './EnsureMeasured'

/* These dimensions are declared on the React Flow nodes as well as in CSS.
   That keeps the radial layout and edge measurements deterministic. */
const CARD_W = 264
const CARD_H = 112
const ROOT_W = 300
const ROOT_H = 132
const ARC_GAP = 30
const PITCH = CARD_W + ARC_GAP
const R1_MIN = 400
const R2_MIN = 330
const SLOT_USE = 0.8

const KIND_ICON: Record<Concept['kind'], IconName> = {
  idea: 'bulb',
  formula: 'fx',
  method: 'steps',
  metric: 'chart',
  pitfall: 'warning',
  tradeoff: 'scale',
}

interface ConceptNodeData extends Record<string, unknown> {
  concept: Concept
  expanded: boolean
  selected: boolean
  hasChildren: boolean
}

interface RootNodeData extends Record<string, unknown> {
  block: Block
  variant: Variant
}

function RootNode({ data }: NodeProps<Node<RootNodeData, 'root'>>) {
  return (
    <div className="cm-root">
      <Handle type="source" position={Position.Top} style={{ opacity: 0 }} />
      <div className="cm-root-eyebrow">Block {data.block.ordinal} · {data.block.slot}</div>
      <div className="cm-root-title">{data.variant.label}</div>
      <div className="cm-root-sub">{data.variant.tagline}</div>
    </div>
  )
}

function ConceptNode({ data }: NodeProps<Node<ConceptNodeData, 'concept'>>) {
  const { concept, expanded, selected, hasChildren } = data
  return (
    <div className={`cm-node k-${concept.kind}${selected ? ' is-selected' : ''}${expanded ? ' is-expanded' : ''}`}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
      <div className="cm-node-head">
        <span className="cm-glyph"><Icon name={KIND_ICON[concept.kind]} size={16} /></span>
        <span className="cm-node-title">{concept.label}</span>
        {hasChildren && <span className="cm-expand">{expanded ? '−' : '+'}</span>}
      </div>
      <div className="cm-node-sum">{concept.summary}</div>
    </div>
  )
}

const nodeTypes = { root: RootNode, concept: ConceptNode }

type CmNode = Node<RootNodeData, 'root'> | Node<ConceptNodeData, 'concept'>

function buildConceptGraph(
  block: Block,
  variant: Variant,
  expanded: Set<string>,
  selectedId: string | null,
): { nodes: CmNode[]; edges: Edge[] } {
  const nodes: CmNode[] = [
    {
      id: `root-${block.id}-${variant.id}`,
      type: 'root',
      position: { x: -ROOT_W / 2, y: -ROOT_H / 2 },
      width: ROOT_W,
      height: ROOT_H,
      data: { block, variant },
    },
  ]
  const edges: Edge[] = []
  const roots = variant.concepts ?? []
  const n = roots.length
  if (!n) return { nodes, edges }

  // Grow the first ring until every card has a full card-width of arc to itself.
  const r1 = Math.max(R1_MIN, (n * PITCH) / (2 * Math.PI))
  const slot = ((2 * Math.PI) / n) * SLOT_USE
  const rootId = `root-${block.id}-${variant.id}`

  roots.forEach((concept, index) => {
    const angle = (index / n) * Math.PI * 2 - Math.PI / 2
    nodes.push({
      id: concept.id,
      type: 'concept',
      position: { x: Math.cos(angle) * r1 - CARD_W / 2, y: Math.sin(angle) * r1 - CARD_H / 2 },
      width: CARD_W,
      height: CARD_H,
      data: {
        concept,
        expanded: expanded.has(concept.id),
        selected: selectedId === concept.id,
        hasChildren: Boolean(concept.children?.length),
      },
    })
    edges.push({ id: `${rootId}->${concept.id}`, source: rootId, target: concept.id, type: 'straight', className: 'cm-edge' })

    if (!expanded.has(concept.id) || !concept.children?.length) return

    const childCount = concept.children.length
    const childRadius = Math.max(r1 + R2_MIN, ((childCount - 1) * PITCH) / slot)
    const separation = PITCH / childRadius
    concept.children.forEach((child, childIndex) => {
      const childAngle = angle + (childIndex - (childCount - 1) / 2) * separation
      nodes.push({
        id: child.id,
        type: 'concept',
        position: {
          x: Math.cos(childAngle) * childRadius - CARD_W / 2,
          y: Math.sin(childAngle) * childRadius - CARD_H / 2,
        },
        width: CARD_W,
        height: CARD_H,
        data: {
          concept: child,
          expanded: false,
          selected: selectedId === child.id,
          hasChildren: false,
        },
      })
      edges.push({
        id: `${concept.id}->${child.id}`,
        source: concept.id,
        target: child.id,
        type: 'straight',
        className: 'cm-edge cm-edge-child',
      })
    })
  })

  return { nodes, edges }
}

function ExampleBlock({ concept }: { concept: Concept }) {
  const example = concept.example
  if (!example) return null
  return (
    <div className="example cm-example">
      <div className="example-half">
        {example.beforeLabel && <div className="example-label">{example.beforeLabel}</div>}
        <div className={`example-text${example.mono ? ' mono' : ''}`}>{example.before}</div>
      </div>
      <div className="arrow-sep">↓</div>
      <div className="example-half">
        {example.afterLabel && <div className="example-label">{example.afterLabel}</div>}
        <div className={`example-text${example.mono ? ' mono' : ''}`}>{example.after}</div>
      </div>
    </div>
  )
}

function StackList({ items }: { items: StackItem[] }) {
  return (
    <div className="stack-list cm-stack-list">
      <div className="sub-label">Tech stack</div>
      {items.map((item) =>
        item.url ? (
          <a key={item.name} className="stack-item" href={item.url} target="_blank" rel="noopener noreferrer">
            <b>{item.name}</b><span>{item.what}</span><span className="stack-arrow">↗</span>
          </a>
        ) : (
          <div key={item.name} className="stack-item">
            <b>{item.name}</b><span>{item.what}</span>
          </div>
        ),
      )}
    </div>
  )
}

function ConceptDetail({ concept }: { concept: Concept | null }) {
  if (!concept) {
    return (
      <aside className="cm-detail cm-detail-empty">
        <p>Click a card to read it. Cards marked <b>+</b> expand into sub-cards.</p>
      </aside>
    )
  }

  return (
    <aside className="cm-detail">
      <div className={`cm-detail-kind k-${concept.kind}`}><Icon name={KIND_ICON[concept.kind]} size={15} />{concept.kind}</div>
      <h3>{concept.label}</h3>
      <div className="cm-detail-sum">{concept.summary}</div>
      {concept.detail && concept.detail.length > 0 && <Prose>{concept.detail.join('\n\n')}</Prose>}
      {concept.stack && concept.stack.length > 0 && <StackList items={concept.stack} />}
      {concept.figures?.map((figure, index) => <FigureView figure={figure} key={index} />)}
      {concept.math?.map((math, index) => <MathBlockView block={math} key={index} />)}
      {concept.code?.map((code, index) => <CodeBlockView block={code} key={index} />)}
      <ExampleBlock concept={concept} />
    </aside>
  )
}

function flattenConcepts(concepts?: Concept[]) {
  const flat = new Map<string, Concept>()
  const walk = (cards?: Concept[]) => cards?.forEach((card) => {
    flat.set(card.id, card)
    walk(card.children)
  })
  walk(concepts)
  return flat
}

function Inner({ block, variant, onClose }: { block: Block; variant: Variant; onClose: () => void }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const flatConcepts = useMemo(() => flattenConcepts(variant.concepts), [variant.concepts])
  const { nodes, edges } = useMemo(
    () => buildConceptGraph(block, variant, expanded, selectedId),
    [block, variant, expanded, selectedId],
  )
  const nodeIds = useMemo(() => nodes.map((node) => node.id), [nodes])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const onNodeClick: NodeMouseHandler = useCallback((_event, node) => {
    if (node.type === 'root') {
      setSelectedId(null)
      return
    }
    setSelectedId(node.id)
    const concept = flatConcepts.get(node.id)
    if (concept?.children?.length) {
      setExpanded((previous) => {
        const next = new Set(previous)
        if (next.has(concept.id)) next.delete(concept.id)
        else next.add(concept.id)
        return next
      })
    }
  }, [flatConcepts])

  return (
    <div className="cm-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label={`${variant.full} concept map`}>
      <div className="cm-shell" onClick={(event) => event.stopPropagation()}>
        <header className="cm-bar">
          <span className="cm-bar-title">Concept map · {variant.label}</span>
          <span className="cm-bar-hint">Click to read · + to expand</span>
          <button className="cm-close" onClick={onClose} aria-label="Close concept map">✕</button>
        </header>
        <div className="cm-body">
          <div className="cm-canvas">
            <ReactFlow
              id="concept-map"
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodeClick={onNodeClick}
              fitView
              fitViewOptions={{ padding: 0.22 }}
              minZoom={0.2}
              maxZoom={1.5}
              nodesDraggable={false}
              nodesConnectable={false}
              proOptions={{ hideAttribution: true }}
            >
              <EnsureMeasured ids={nodeIds} />
              <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--bg-grid)" />
            </ReactFlow>
          </div>
          <ConceptDetail concept={selectedId ? (flatConcepts.get(selectedId) ?? null) : null} />
        </div>
      </div>
    </div>
  )
}

export default function ConceptMap({ block, variant, onClose }: { block: Block; variant: Variant; onClose: () => void }) {
  return <Inner block={block} variant={variant} onClose={onClose} />
}

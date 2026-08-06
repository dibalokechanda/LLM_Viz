import { useEffect, useMemo, useState } from 'react'
import type { Figure } from '../data/types'

type NetworkFigure = Extract<Figure, { kind: 'network' }>

function neighboursOf(f: NetworkFigure, id: string) {
  return f.links
    .filter((link) => link.source === id || link.target === id)
    .map((link) => link.source === id ? link.target : link.source)
    .filter((value, index, all) => all.indexOf(value) === index)
}

/** Use a supplied teaching path when present; otherwise reveal the same greedy
 * walk the old autoplay animation performed, one deliberate step at a time. */
function greedyPath(f: NetworkFigure, entryId: string, targetId: string) {
  if (f.path?.length) return f.path
  const entry = f.nodes.find((node) => node.id === entryId)
  const target = f.nodes.find((node) => node.id === targetId)
  if (!entry || !target) return [entryId]

  const xSpan = Math.max(...f.nodes.map((node) => node.x)) - Math.min(...f.nodes.map((node) => node.x))
  const ySpan = Math.max(...f.nodes.map((node) => node.y)) - Math.min(...f.nodes.map((node) => node.y))
  const maxDistance = Math.hypot(xSpan, ySpan) || 1
  const score = (id: string) => {
    const node = f.nodes.find((item) => item.id === id)!
    return 1 - Math.hypot(node.x - target.x, node.y - target.y) / maxDistance
  }

  const path = [entry.id]
  const seen = new Set(path)
  let current = entry.id
  while (current !== target.id) {
    const next = neighboursOf(f, current)
      .filter((id) => !seen.has(id))
      .sort((a, b) => score(b) - score(a))[0]
    if (!next || score(next) <= score(current)) break
    path.push(next)
    seen.add(next)
    current = next
  }
  return path
}

const sameEdge = (a: string, b: string, source: string, target: string) =>
  (a === source && b === target) || (a === target && b === source)

export default function FigureD3Network({ f }: { f: NetworkFigure }) {
  const entry = f.nodes.find((node) => node.isEntry)
  const target = f.nodes.find((node) => node.isTarget)
  const path = useMemo(() => entry && target ? greedyPath(f, entry.id, target.id) : [], [f, entry, target])
  const [step, setStep] = useState(0)
  const [inspectedId, setInspectedId] = useState<string | null>(null)

  useEffect(() => {
    setStep(0)
    setInspectedId(null)
  }, [f])

  if (!entry || !target || path.length === 0) return null

  const safeStep = Math.min(step, path.length - 1)
  const currentId = path[safeStep]
  const inspected = f.nodes.find((node) => node.id === inspectedId) ?? f.nodes.find((node) => node.id === currentId)!
  const nextId = path[safeStep + 1]
  const xSpan = Math.max(...f.nodes.map((node) => node.x)) - Math.min(...f.nodes.map((node) => node.x))
  const ySpan = Math.max(...f.nodes.map((node) => node.y)) - Math.min(...f.nodes.map((node) => node.y))
  const maxDistance = Math.hypot(xSpan, ySpan) || 1
  const scores = neighboursOf(f, inspected.id)
    .map((id) => {
      const node = f.nodes.find((item) => item.id === id)!
      return { id, similarity: 1 - Math.hypot(node.x - target.x, node.y - target.y) / maxDistance }
    })
    .sort((a, b) => b.similarity - a.similarity)
  const trail = path.slice(0, safeStep + 1)

  return (
    <div className="net-fig fig-interactive">
      <div className="fig-controls" aria-label="Search path controls">
        <button type="button" className="fig-control" onClick={() => { setStep((value) => Math.max(0, value - 1)); setInspectedId(null) }} disabled={safeStep === 0}>Back</button>
        <span className="fig-progress">step {safeStep + 1} / {path.length}</span>
        <button type="button" className="fig-control" onClick={() => { setStep((value) => Math.min(path.length - 1, value + 1)); setInspectedId(null) }} disabled={safeStep === path.length - 1}>Next hop</button>
      </div>

      <svg viewBox="0 0 400 240" width="100%" className="fig-svg" role="img" aria-label={`Interactive graph search. Current node ${currentId}.`}>
        <title>Interactive graph search</title>
        {f.links.map((link, index) => {
          const source = f.nodes.find((node) => node.id === link.source)
          const targetNode = f.nodes.find((node) => node.id === link.target)
          if (!source || !targetNode) return null
          const traced = trail.slice(1).some((node, i) => sameEdge(trail[i], node, link.source, link.target))
          const candidate = sameEdge(inspected.id, nextId ?? '', link.source, link.target)
          return (
            <line
              key={`${link.source}-${link.target}-${index}`}
              className="fig-network-edge"
              x1={source.x}
              y1={source.y}
              x2={targetNode.x}
              y2={targetNode.y}
              stroke={traced ? 'var(--model)' : candidate ? 'var(--seq-400)' : 'var(--border-heavy)'}
              strokeWidth={traced ? 2.6 : candidate ? 1.8 : 1.2}
              strokeOpacity={traced ? 1 : candidate ? 0.8 : 0.55}
            />
          )
        })}

        {f.nodes.map((node) => {
          const isCurrent = node.id === currentId
          const isInspected = node.id === inspected.id
          const isVisited = trail.includes(node.id)
          const isCandidate = scores.some((item) => item.id === node.id)
          const fill = node.isTarget
            ? 'var(--role-frontier)'
            : isCurrent
              ? 'var(--model)'
              : isVisited
                ? 'var(--model-wash)'
                : isCandidate
                  ? 'var(--surface-hover)'
                  : 'var(--surface)'
          return (
            <g key={node.id}>
              {isCurrent && <circle className="fig-network-halo" cx={node.x} cy={node.y} r="12" fill="none" stroke="var(--model)" />}
              <circle
                className="fig-mark fig-network-node"
                cx={node.x}
                cy={node.y}
                r={node.isTarget ? 5.5 : isCurrent ? 7.5 : 6}
                fill={fill}
                stroke={isInspected ? 'var(--model)' : node.isTarget ? 'var(--role-frontier)' : 'var(--border-strong)'}
                strokeWidth={isInspected ? 2 : 1.3}
                tabIndex={0}
                role="button"
                aria-label={`Inspect node ${node.label ?? node.id}`}
                onPointerEnter={() => setInspectedId(node.id)}
                onFocus={() => setInspectedId(node.id)}
                onClick={() => setInspectedId(node.id)}
              />
              <text x={node.x} y={node.y - 12} fontSize="10" fontFamily="var(--mono)" fill={isCurrent ? 'var(--model)' : 'var(--text-dim)'} textAnchor="middle">
                {node.label ?? node.id}
              </text>
            </g>
          )
        })}

        {f.annotations?.map((annotation, index) => (
          <text key={index} x={annotation.x} y={annotation.y} fontSize="10" fontFamily="var(--mono)" fill="var(--text-faint)" textAnchor={annotation.anchor ?? 'start'}>
            {annotation.text}
          </text>
        ))}
      </svg>

      <FigureReadout label={inspected.label ?? inspected.id} values={scores} chosen={nextId} />
    </div>
  )
}

function FigureReadout({ label, values, chosen }: { label: string; values: { id: string; similarity: number }[]; chosen?: string }) {
  return (
    <div className="net-readout" aria-live="polite">
      <div className="fig-detail"><span className="fig-detail-label">inspecting {label}</span>{values.length ? `${values.length} connected neighbour${values.length === 1 ? '' : 's'}` : 'no connected neighbours'}</div>
      {values.length > 0 && (
        <table className="net-table">
          <thead><tr><th>Neighbour</th><th>Similarity to query</th></tr></thead>
          <tbody>
            {values.map((value) => (
              <tr key={value.id} className={value.id === chosen ? 'is-chosen' : ''}>
                <td>{value.id}</td>
                <td><div className="net-bar" style={{ width: `${Math.max(2, value.similarity * 100)}%` }} /><div className="net-val">{(value.similarity * 100).toFixed(1)}%</div></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

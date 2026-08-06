import { useEffect, useState } from 'react'
import type { Figure } from '../data/types'

type LayeredFigure = Extract<Figure, { kind: 'layered' }>

const getCy = (layer: number) => 210 - layer * 70

export default function FigureD3Layered({ f }: { f: LayeredFigure }) {
  const path = f.path ?? []
  const [step, setStep] = useState(0)

  useEffect(() => setStep(0), [f])
  if (path.length === 0) return null

  const safeStep = Math.min(step, path.length - 1)
  const current = path[safeStep]
  const traversed = path.slice(0, safeStep + 1)
  const maxLayer = Math.max(...f.layers)
  const currentNode = f.nodes.find((node) => node.id === current.node)

  const isTracedLink = (layer: number, source: string, target: string) =>
    traversed.slice(1).some((item, index) => {
      const previous = traversed[index]
      return previous.layer === item.layer && item.layer === layer && (
        (previous.node === source && item.node === target) || (previous.node === target && item.node === source)
      )
    })

  return (
    <div className="layered-fig fig-interactive">
      <div className="fig-controls" aria-label="Layered path controls">
        <button type="button" className="fig-control" onClick={() => setStep((value) => Math.max(0, value - 1))} disabled={safeStep === 0}>Back</button>
        <span className="fig-progress">hop {safeStep + 1} / {path.length}</span>
        <button type="button" className="fig-control" onClick={() => setStep((value) => Math.min(path.length - 1, value + 1))} disabled={safeStep === path.length - 1}>Next hop</button>
      </div>

      <svg viewBox="0 0 470 270" width="100%" className="fig-svg" role="img" aria-label={`Interactive layered path. Current node ${current.node} at layer ${current.layer}.`}>
        <title>Interactive layered search path</title>
        {f.layers.map((layer) => {
          const y = getCy(layer)
          const description = f.layerLabels?.find((item) => item.layer === layer)?.text
          return (
            <g key={`plane-${layer}`}>
              <polygon points={`20,${y + 15} 380,${y + 15} 360,${y - 25} 40,${y - 25}`} fill="var(--surface-hover)" opacity="0.5" />
              <text x="10" y={y + 4} fontSize="12" fontWeight="600" fontFamily="var(--mono)" fill="var(--text-dim)">L{layer}</text>
              {description && (
                <text x="388" y={y - 3} fontSize="9.5" fontFamily="var(--mono)" fill="var(--text-faint)">
                  {description.split('\n').map((line, index) => <tspan key={index} x="388" dy={index === 0 ? 0 : 12}>{line}</tspan>)}
                </text>
              )}
            </g>
          )
        })}

        {f.links.map((link) => {
          const source = f.nodes.find((node) => node.id === link.source)
          const target = f.nodes.find((node) => node.id === link.target)
          if (!source || !target) return null
          const traced = isTracedLink(link.layer, link.source, link.target)
          return (
            <line
              key={`link-${link.layer}-${link.source}-${link.target}`}
              className="fig-layer-edge"
              x1={source.x}
              y1={getCy(link.layer)}
              x2={target.x}
              y2={getCy(link.layer)}
              stroke={traced ? 'var(--model)' : 'var(--border-heavy)'}
              strokeWidth={traced ? 2.5 : 1.2}
              strokeOpacity={traced ? 1 : 0.55}
            />
          )
        })}

        {traversed.slice(1).map((item, index) => {
          const previous = traversed[index]
          if (previous.layer === item.layer) return null
          const source = f.nodes.find((node) => node.id === previous.node)
          const target = f.nodes.find((node) => node.id === item.node)
          if (!source || !target) return null
          return (
            <path
              key={`drop-${index}`}
              className="fig-layer-drop"
              d={`M ${source.x} ${getCy(previous.layer)} C ${source.x} ${(getCy(previous.layer) + getCy(item.layer)) / 2}, ${target.x} ${(getCy(previous.layer) + getCy(item.layer)) / 2}, ${target.x} ${getCy(item.layer)}`}
              fill="none"
              stroke="var(--model)"
              strokeWidth="2.5"
              strokeDasharray="4 3"
            />
          )
        })}

        {f.layers.map((layer) => (
          <g key={`nodes-${layer}`}>
            {f.nodes.filter((node) => node.maxLayer >= layer).map((node) => {
              const pathIndex = path.findIndex((item) => item.node === node.id && item.layer === layer)
              const visited = pathIndex >= 0 && pathIndex <= safeStep
              const active = node.id === current.node && layer === current.layer
              const target = node.isTarget
              return (
                <g key={`node-${layer}-${node.id}`}>
                  {active && <circle className="fig-network-halo" cx={node.x} cy={getCy(layer)} r="10" fill="none" stroke="var(--model)" />}
                  <circle
                    className="fig-mark fig-layer-node"
                    cx={node.x}
                    cy={getCy(layer)}
                    r={target ? 4.5 : active ? 6.5 : 5}
                    fill={target ? 'var(--role-frontier)' : active ? 'var(--model)' : visited ? 'var(--model-wash)' : 'var(--surface)'}
                    stroke={active || visited ? 'var(--model)' : target ? 'var(--role-frontier)' : 'var(--border-strong)'}
                    strokeWidth={active ? 2 : 1.2}
                    tabIndex={pathIndex >= 0 ? 0 : -1}
                    role={pathIndex >= 0 ? 'button' : undefined}
                    aria-label={pathIndex >= 0 ? `Show ${node.id} at layer ${layer}` : undefined}
                    onPointerEnter={pathIndex >= 0 ? () => setStep(pathIndex) : undefined}
                    onFocus={pathIndex >= 0 ? () => setStep(pathIndex) : undefined}
                    onClick={pathIndex >= 0 ? () => setStep(pathIndex) : undefined}
                  />
                  {node.isEntry && layer === maxLayer && <text x={node.x} y={getCy(layer) - 11} fontSize="10" fontFamily="var(--mono)" fill="var(--text-faint)" textAnchor="middle">entry</text>}
                  {target && layer === 0 && <text x={node.x} y={getCy(layer) - 10} fontSize="10" fontFamily="var(--mono)" fill="var(--role-frontier)" textAnchor="middle">query</text>}
                </g>
              )
            })}
          </g>
        ))}

        {f.annotations?.map((annotation, index) => (
          <text key={`annotation-${index}`} x={annotation.x} y={annotation.y} fontSize="9.5" fontFamily="var(--mono)" fill="var(--text-faint)" textAnchor={annotation.anchor ?? 'start'}>{annotation.text}</text>
        ))}
      </svg>

      <div className="fig-detail" aria-live="polite"><span className="fig-detail-label">layer {current.layer}</span>{currentNode?.id ?? current.node} · click any highlighted node to inspect that hop</div>
    </div>
  )
}

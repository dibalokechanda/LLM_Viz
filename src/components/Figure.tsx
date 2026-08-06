import { useState } from 'react'
import type { Figure } from '../data/types'
import FigureD3Network from './FigureD3Network'
import FigureD3Layered from './FigureD3Layered'
import FigureEmbedding from './FigureEmbedding'

/**
 * Inline monochrome diagrams. Everything is plain SVG on a 0–W viewBox that
 * scales to the panel width; colour comes from CSS variables so the figures
 * stay in step with the theme.
 */

const W = 400
const INK = 'var(--ink)'
const LINE = 'var(--border-strong)'
const MUTE = 'var(--border-heavy)'
const LABEL = 'var(--text-faint)'
const MONO = 'var(--mono)'

function Frame({ h, label, children }: { h: number; label: string; children: React.ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${h}`} width="100%" role="img" aria-label={label} className="fig-svg">
      {children}
    </svg>
  )
}

function FigureDetail({ children }: { children: React.ReactNode }) {
  return <div className="fig-detail" aria-live="polite">{children}</div>
}

/* ────────────────────────────── bars ────────────────────────────── */

function Bars({ f }: { f: Extract<Figure, { kind: 'bars' }> }) {
  const [selected, setSelected] = useState(f.highlight?.[0] ?? 0)
  const panels = f.series.length
  const gap = 14
  const panelW = (W - gap * (panels - 1)) / panels
  const plotH = 96
  // Headroom for the panel title; text is drawn from its baseline, so a label
  // at y=9 actually starts above the viewBox and gets clipped.
  const top = panels > 1 ? 24 : 8
  const h = top + plotH + 30
  const yMax = f.yMax ?? Math.max(...f.series.flatMap((s) => s.values)) * 1.12

  const category = f.categories[selected] ?? `item ${selected + 1}`

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive bar chart. ${category} is selected.`}>
      {f.series.map((s, si) => {
        const x0 = si * (panelW + gap)
        const n = s.values.length
        const bw = Math.min(36, (panelW - 8) / n - 6)
        const step = (panelW - 8) / n
        return (
          <g key={si}>
            {panels > 1 && (
              <text x={x0 + panelW / 2} y={14} textAnchor="middle" fontSize="11.5" fontFamily={MONO} fill={INK}>
                {s.label}
              </text>
            )}
            {/* baseline */}
            <line x1={x0} y1={top + plotH} x2={x0 + panelW} y2={top + plotH} stroke={LINE} strokeWidth="1" />
            {s.values.map((v, i) => {
              const bh = Math.max(1, (v / yMax) * plotH)
              const bx = x0 + 4 + i * step + (step - bw) / 2
              const included = !f.highlight || f.highlight.includes(i)
              const active = i === selected
              const on = active || included
              return (
                <g key={i}>
                  <rect
                    className="fig-mark"
                    x={bx}
                    y={top + plotH - bh}
                    width={bw}
                    height={bh}
                    fill={active ? 'var(--seq-500)' : on ? 'var(--surface-hover)' : 'none'}
                    stroke={active ? 'var(--seq-600)' : on ? LINE : MUTE}
                    strokeWidth={active ? 1.5 : 1}
                    tabIndex={0}
                    role="button"
                    aria-label={`${f.categories[i] ?? i}: ${s.label} ${v.toFixed(v < 1 ? 3 : 1)}`}
                    onPointerEnter={() => setSelected(i)}
                    onFocus={() => setSelected(i)}
                    onClick={() => setSelected(i)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') setSelected(i)
                    }}
                  />
                  {f.showValues &&
                    (() => {
                      // A tall bar leaves no headroom for a label above it, and
                      // it would collide with the panel title. Put the value
                      // inside the bar instead once there is room for it.
                      const inside = bh > 26 && on
                      return (
                        <text
                          x={bx + bw / 2}
                          y={inside ? top + plotH - bh + 14 : top + plotH - bh - 5}
                          textAnchor="middle"
                          fontSize="10.5"
                          fontFamily={MONO}
                          fill={inside ? 'var(--text-onaccent)' : LABEL}
                        >
                          {v.toFixed(v < 1 ? 3 : 1)}
                        </text>
                      )
                    })()}
                  <text
                    x={bx + bw / 2}
                    y={top + plotH + 12}
                    textAnchor="middle"
                    fontSize="11"
                    fontFamily={MONO}
                    fill={LABEL}
                  >
                    {f.categories[i]}
                  </text>
                </g>
              )
            })}
            {f.cutoff && (
              <g>
                <line
                  x1={x0 + 4 + f.cutoff.after * step}
                  y1={top - 2}
                  x2={x0 + 4 + f.cutoff.after * step}
                  y2={top + plotH + 3}
                  stroke={INK}
                  strokeWidth="1"
                  strokeDasharray="3 2"
                />
                <text
                  x={x0 + 4 + f.cutoff.after * step + 4}
                  y={top + 6}
                  fontSize="10.5"
                  fontFamily={MONO}
                  fill={INK}
                >
                  {f.cutoff.label}
                </text>
              </g>
            )}
          </g>
        )
      })}
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">{category}</span>
      {f.series.map((series) => `${series.label}: ${series.values[selected]?.toFixed(series.values[selected] < 1 ? 3 : 1) ?? '—'}`).join(' · ')}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── curve ────────────────────────────── */

function Curve({ f }: { f: Extract<Figure, { kind: 'curve' }> }) {
  const [activeLine, setActiveLine] = useState(0)
  const padL = 44
  const padR = 8
  // Band above the plot for the y-axis label, so it clears the top tick — those
  // two were landing on each other whenever the label was wide.
  const padT = 28
  const plotH = 124
  // Room below the axis for the tick row plus the x-axis caption.
  const h = padT + plotH + 36
  const plotW = W - padL - padR

  const all = f.lines.flatMap((l) => l.points)
  const xs = all.map((p) => p[0])
  const ys = all.map((p) => p[1])
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const yMin = Math.min(0, ...ys)
  const yMax = Math.max(...ys) * 1.05

  const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin || 1)) * plotW
  const sy = (y: number) => padT + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH

  const active = f.lines[activeLine] ?? f.lines[0]

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive curve chart. ${active?.label ?? 'First series'} is selected.`}>
      {/* axes */}
      <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke={LINE} strokeWidth="1" />
      <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke={LINE} strokeWidth="1" />

      {f.yTicks?.map((t, i) => (
        <g key={i}>
          <line x1={padL - 3} y1={sy(t.at)} x2={padL} y2={sy(t.at)} stroke={LINE} />
          <text x={padL - 6} y={sy(t.at) + 3} textAnchor="end" fontSize="10.5" fontFamily={MONO} fill={LABEL}>
            {t.label}
          </text>
        </g>
      ))}
      {f.xTicks?.map((t, i) => {
        // Anchor the end ticks inward so they cannot hang off either edge, and
        // sit them low enough to clear the y-axis zero label.
        const tx = sx(t.at)
        const anchor = tx > W - padR - 14 ? 'end' : tx < padL + 14 ? 'start' : 'middle'
        return (
          <g key={i}>
            <line x1={tx} y1={padT + plotH} x2={tx} y2={padT + plotH + 3} stroke={LINE} />
            <text
              x={anchor === 'end' ? W : anchor === 'start' ? padL - 4 : tx}
              y={padT + plotH + 17}
              textAnchor={anchor}
              fontSize="10.5"
              fontFamily={MONO}
              fill={LABEL}
            >
              {t.label}
            </text>
          </g>
        )
      })}

      {f.lines.map((l, i) => (
        <polyline
          key={i}
          className="fig-curve-line"
          points={l.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')}
          fill="none"
          stroke={i === activeLine ? 'var(--model)' : MUTE}
          strokeWidth={i === activeLine ? 2.2 : 1.2}
          strokeDasharray={l.dashed ? '4 3' : undefined}
          tabIndex={0}
          role="button"
          aria-label={`Select ${l.label ?? `series ${i + 1}`}`}
          onPointerEnter={() => setActiveLine(i)}
          onFocus={() => setActiveLine(i)}
          onClick={() => setActiveLine(i)}
        />
      ))}

      {(() => {
        /*
         * Mark labels have to dodge two things: the plotted lines, and each
         * other. Rather than special-casing peaks and troughs, try a handful of
         * placements and take the first that is clear of both.
         */
        const placed: { x: number; y: number; w: number }[] = []

        // Densified points of every line, for hit-testing label boxes.
        const linePts: [number, number][] = []
        f.lines.forEach((l) => {
          for (let i = 0; i < l.points.length - 1; i++) {
            const [ax, ay] = l.points[i]
            const [bx, by] = l.points[i + 1]
            for (let k = 0; k <= 10; k++) {
              const t = k / 10
              linePts.push([sx(ax + (bx - ax) * t), sy(ay + (by - ay) * t)])
            }
          }
        })

        const hitsLine = (x: number, y: number, w: number) =>
          linePts.some((p) => p[0] >= x - 2 && p[0] <= x + w + 2 && p[1] >= y - 10 && p[1] <= y + 3)

        const hitsLabel = (x: number, y: number, w: number) =>
          placed.some((p) => Math.abs(p.y - y) < 13 && x < p.x + p.w + 4 && p.x < x + w + 4)

        return f.marks?.map((m, i) => {
        const px = sx(m.x)
        const py0 = sy(m.y)
        // Flip the label to the left near the right edge so it cannot run off
        // the figure.
        const flip = px > W - padR - 70
        const w = m.label.length * 6.3
        const x0 = flip ? px - 6 - w : px + 6

        // Above, below, then progressively further out on each side.
        const candidates = [py0 - 9, py0 + 18, py0 - 23, py0 + 32, py0 - 37, py0 + 46]
        let py = candidates[0]
        for (const c of candidates) {
          const clamped = Math.min(padT + plotH - 3, Math.max(padT + 9, c))
          if (!hitsLine(x0, clamped, w) && !hitsLabel(x0, clamped, w)) {
            py = clamped
            break
          }
          py = clamped
        }

        placed.push({ x: x0, y: py, w })
        return (
          <g key={i}>
            {/* The dot stays on the data point; only the label is nudged. */}
            <circle
              className="fig-mark"
              cx={px}
              cy={py0}
              r="3.3"
              fill="var(--model)"
              tabIndex={0}
              role="button"
              aria-label={`${m.label}: ${m.y} at ${m.x}`}
              onFocus={() => setActiveLine(0)}
              onClick={() => setActiveLine(0)}
            />
            <text
              x={flip ? px - 6 : px + 6}
              y={py}
              textAnchor={flip ? 'end' : 'start'}
              fontSize="10.5"
              fontFamily={MONO}
              fill={INK}
            >
              {m.label}
            </text>
          </g>
        )
        })
      })()}

      <text x={W - padR} y={h - 4} textAnchor="end" fontSize="10.5" fontFamily={MONO} fill={LABEL}>
        {f.xLabel}
      </text>
      <text x={0} y={12} fontSize="10.5" fontFamily={MONO} fill={LABEL}>
        {f.yLabel}
      </text>
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">{active?.label ?? `series ${activeLine + 1}`}</span>
      {active?.points.length ?? 0} samples · hover or focus a line to compare it
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── segments ────────────────────────────── */

function Segments({ f }: { f: Extract<Figure, { kind: 'segments' }> }) {
  const [selected, setSelected] = useState({ row: 0, span: 0 })
  const rowH = 34
  const gap = 20
  const labelW = 4
  /** The first row's label is drawn from its baseline, so it needs room above. */
  const TOP = 6
  const h = TOP + f.rows.length * (rowH + gap) + 6
  const trackW = W - labelW

  const selectedRow = f.rows[selected.row] ?? f.rows[0]
  const selectedSpan = selectedRow?.spans[selected.span] ?? selectedRow?.spans[0]

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive sequence spans. ${selectedRow?.label ?? 'A span'} is selected.`}>
      {f.rows.map((r, i) => {
        const y = TOP + i * (rowH + gap)
        return (
          <g key={i}>
            <text x={0} y={y + 8} fontSize="11" fontFamily={MONO} fill={LABEL}>
              {r.label}
            </text>
            {r.spans.map((s, j) => {
              const x = labelW + (s.from / f.total) * trackW
              const w = ((s.to - s.from) / f.total) * trackW
              const active = selected.row === i && selected.span === j
              return (
                <g key={j}>
                  <rect
                    className="fig-mark"
                    x={x + 0.5}
                    y={y + 13}
                    width={Math.max(2, w - 1)}
                    height={rowH - 12}
                    fill={active ? 'var(--model-wash)' : s.ghost ? 'none' : 'var(--surface-hover)'}
                    stroke={active ? 'var(--model)' : s.ghost ? MUTE : LINE}
                    strokeWidth={active ? 1.5 : 1}
                    strokeDasharray={s.ghost ? '3 2' : undefined}
                    rx="2"
                    tabIndex={0}
                    role="button"
                    aria-label={`${r.label ?? `row ${i + 1}`}: ${s.label ?? 'span'} from ${s.from} to ${s.to}`}
                    onPointerEnter={() => setSelected({ row: i, span: j })}
                    onFocus={() => setSelected({ row: i, span: j })}
                    onClick={() => setSelected({ row: i, span: j })}
                  />
                  {s.label && w > 26 && (
                    <text
                      x={x + w / 2}
                      y={y + 13 + (rowH - 12) / 2 + 3}
                      textAnchor="middle"
                      fontSize="10.5"
                      fontFamily={MONO}
                      fill={LABEL}
                    >
                      {s.label}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        )
      })}
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">{selectedRow?.label ?? 'span'}</span>
      {selectedSpan?.label ?? 'unlabelled span'} · positions {selectedSpan?.from}–{selectedSpan?.to}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── ranked ────────────────────────────── */

function Ranked({ f }: { f: Extract<Figure, { kind: 'ranked' }> }) {
  const [selected, setSelected] = useState(f.markFirstRelevant ? Math.max(0, f.grades.findIndex((g) => g > 0)) : 0)
  const n = f.grades.length
  const gap = 3
  const cw = (W - gap * (n - 1)) / n
  const ch = 40
  /** Band above the cells for the "first relevant" marker. */
  const TOP = 16
  const h = TOP + ch + 26
  const maxG = f.maxGrade ?? Math.max(...f.grades, 1)
  const firstRel = f.grades.findIndex((g) => g > 0)

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive relevance ranking. Rank ${selected + 1} is selected.`}>
      {f.grades.map((g, i) => {
        const x = i * (cw + gap)
        // Higher grades render darker; 0 stays an outline.
        const opacity = g === 0 ? 0 : 0.18 + 0.82 * (g / maxG)
        const active = i === selected
        return (
          <g key={i}>
            <rect
              className="fig-mark"
              x={x}
              y={TOP}
              width={cw}
              height={ch}
              rx="2"
              fill={active ? 'var(--model)' : INK}
              fillOpacity={active ? 0.9 : opacity}
              stroke={active ? 'var(--model)' : g === 0 ? MUTE : 'none'}
              strokeWidth={active ? 1.5 : 1}
              strokeDasharray={g === 0 && !active ? '3 2' : undefined}
              tabIndex={0}
              role="button"
              aria-label={`Rank ${i + 1}, relevance grade ${g}`}
              onPointerEnter={() => setSelected(i)}
              onFocus={() => setSelected(i)}
              onClick={() => setSelected(i)}
            />
            <text
              x={x + cw / 2}
              y={TOP + ch / 2 + 4}
              textAnchor="middle"
              fontSize="11.5"
              fontFamily={MONO}
              fill={active || g / maxG > 0.5 ? 'var(--text-onaccent)' : LABEL}
            >
              {g}
            </text>
            <text x={x + cw / 2} y={h - 4} textAnchor="middle" fontSize="10.5" fontFamily={MONO} fill={LABEL}>
              {i + 1}
            </text>
          </g>
        )
      })}
      {f.markFirstRelevant && firstRel >= 0 && (
        <g>
          <text
            // Centred on the cell, but clamped so a mark over cell 1 does not
            // hang off the left edge.
            x={Math.max(24, firstRel * (cw + gap) + cw / 2)}
            y={11}
            textAnchor="middle"
            fontSize="10.5"
            fontFamily={MONO}
            fill={INK}
          >
            ↓ first
          </text>
        </g>
      )}
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">rank {selected + 1}</span>
      relevance grade {f.grades[selected]} / {maxG}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── blocks ────────────────────────────── */

function Blocks({ f }: { f: Extract<Figure, { kind: 'blocks' }> }) {
  const [selected, setSelected] = useState({ row: 0, box: 0 })
  const rowH = 36
  const arrowH = 22
  const gap = 6
  /** Space above a row that carries a label, so the label is not clipped. */
  const LABEL_H = 15
  let h = 0
  f.rows.forEach((r) => {
    h += (r.label ? LABEL_H : 0) + rowH + gap + (r.arrow !== undefined ? arrowH : 0)
  })

  const selectedRow = f.rows[selected.row] ?? f.rows[0]
  const selectedBox = selectedRow?.boxes[selected.box] ?? selectedRow?.boxes[0]
  let y = 0
  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive block diagram. ${selectedBox?.text ?? 'A block'} is selected.`}>
      {f.rows.map((r, i) => {
        // Reserve the label band before drawing the boxes for this row.
        if (r.label) y += LABEL_H
        const rowY = y
        const totalSpan = r.boxes.reduce((s, b) => s + (b.span ?? 1), 0)
        const inner = W - gap * (r.boxes.length - 1)
        let x = 0
        const el = (
          <g key={i}>
            {r.label && (
              <text x={0} y={rowY - 5} fontSize="10.5" fontFamily={MONO} fill={LABEL}>
                {r.label}
              </text>
            )}
            {r.boxes.map((b, j) => {
              const bw = (inner * (b.span ?? 1)) / totalSpan
              const bx = x
              x += bw + gap
              const active = selected.row === i && selected.box === j
              return (
                <g key={j}>
                  <rect
                    className="fig-mark"
                    x={bx}
                    y={rowY}
                    width={bw}
                    height={rowH}
                    rx="2"
                    fill={active ? 'var(--model)' : b.filled ? INK : 'none'}
                    fillOpacity={active ? 0.9 : 1}
                    stroke={active ? 'var(--model)' : b.filled ? INK : b.dashed ? MUTE : LINE}
                    strokeWidth={active ? 1.6 : 1}
                    strokeDasharray={b.dashed ? '3 2' : undefined}
                    tabIndex={0}
                    role="button"
                    aria-label={`${r.label ?? `row ${i + 1}`}: ${b.text}`}
                    onPointerEnter={() => setSelected({ row: i, box: j })}
                    onFocus={() => setSelected({ row: i, box: j })}
                    onClick={() => setSelected({ row: i, box: j })}
                  />
                  <text
                    x={bx + bw / 2}
                    y={rowY + rowH / 2 + 3}
                    textAnchor="middle"
                    // Shrink to fit the box rather than spilling past its edges.
                    fontSize={Math.max(
                      7.5,
                      Math.min(11.5, ((bw - 12) / (b.text.length * 0.62)) | 0 || 7.5),
                    )}
                    fontFamily={MONO}
                    fill={active || b.filled ? 'var(--text-onaccent)' : 'var(--text-dim)'}
                  >
                    {b.text}
                  </text>
                </g>
              )
            })}
            {r.arrow !== undefined &&
              (() => {
                // The arrow sits on the left so its caption gets the full width;
                // centring it left long captions running off the right edge.
                const ax = 22
                const tipY = rowY + rowH + arrowH - 2
                const textX = ax + 14
                return (
                  <g>
                    <line x1={ax} y1={rowY + rowH + 3} x2={ax} y2={tipY - 1} stroke={MUTE} strokeWidth="1" />
                    <path
                      d={`M ${ax - 3} ${tipY - 4} L ${ax} ${tipY} L ${ax + 3} ${tipY - 4}`}
                      fill="none"
                      stroke={MUTE}
                      strokeWidth="1"
                    />
                    {r.arrow && (
                      <text
                        x={textX}
                        y={tipY - 1}
                        fontSize={Math.max(8, Math.min(10.5, (W - textX) / (r.arrow.length * 0.62)))}
                        fontFamily={MONO}
                        fill={LABEL}
                      >
                        {r.arrow}
                      </text>
                    )}
                  </g>
                )
              })()}
          </g>
        )
        y += rowH + gap + (r.arrow !== undefined ? arrowH : 0)
        return el
      })}
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">{selectedRow?.label ?? 'selected block'}</span>
      {selectedBox?.text}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── heatmap ────────────────────────────── */

/**
 * Which (query, key) pairs a mask admits. Expressed as a predicate rather than
 * stored data so that one line of content — `mask: 'window', w: 3` — produces
 * the whole matrix, and so the same function can be reused by the shape maths.
 */
function visible(f: Extract<Figure, { kind: 'heatmap' }>, q: number, k: number): boolean {
  const w = f.w ?? 3
  if (k > q && f.mask !== 'full') return false // causal is the floor for every LM mask
  switch (f.mask) {
    case 'full':
      return true
    case 'causal':
      return true
    case 'window':
      return q - k < w
    case 'sink':
      return q - k < w || k < (f.sinks ?? 1)
    case 'dilated':
      return (q - k) % w === 0 || q === k
    case 'block':
      return Math.floor(q / w) === Math.floor(k / w)
    default:
      return true
  }
}

function Heatmap({ f }: { f: Extract<Figure, { kind: 'heatmap' }> }) {
  const [selectedRow, setSelectedRow] = useState(f.focusRow ?? Math.max(0, f.n - 1))
  const n = f.n
  const pad = 34
  const cell = Math.min(26, (W - pad - 16) / n)
  const grid = cell * n
  const h = grid + pad + 26

  // Attention is row-stochastic, so shade each admitted cell by its share of
  // its own row. That makes "sinks absorb most of the mass" visible rather
  // than merely asserted.
  const rows = Array.from({ length: n }, (_, q) => {
    const open = Array.from({ length: n }, (_, k) => (visible(f, q, k) ? 1 : 0))
    const total = open.reduce((a: number, b) => a + b, 0) || 1
    return open.map((v) => v / total)
  })

  const visibleKeys = Array.from({ length: n }, (_, k) => k).filter((k) => visible(f, selectedRow, k))
  const queryLabel = f.tokens?.[selectedRow] ?? selectedRow

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive attention mask. Query ${queryLabel} can attend to ${visibleKeys.length} positions.`}>
      {Array.from({ length: n }, (_, q) =>
        Array.from({ length: n }, (_, k) => {
          const on = visible(f, q, k)
          const share = rows[q][k]
          const focused = selectedRow === q
          const selectedCell = focused && on
          return (
            <rect
              className="fig-mark"
              key={`${q}-${k}`}
              x={pad + k * cell}
              y={18 + q * cell}
              width={cell - 1.5}
              height={cell - 1.5}
              fill={selectedCell ? 'var(--seq-500)' : on ? 'var(--seq-300)' : 'none'}
              fillOpacity={selectedCell ? 0.28 + share * n * 0.26 : on ? 0.11 + share * n * 0.12 : 0}
              stroke={selectedCell ? 'var(--seq-600)' : on ? MUTE : LINE}
              strokeWidth={focused && on ? 1.5 : 0.6}
              strokeDasharray={on ? undefined : '2 2'}
              tabIndex={on ? 0 : -1}
              role={on ? 'button' : undefined}
              aria-label={on ? `Query ${f.tokens?.[q] ?? q} attends to key ${f.tokens?.[k] ?? k}` : undefined}
              onPointerEnter={() => setSelectedRow(q)}
              onFocus={() => setSelectedRow(q)}
              onClick={() => setSelectedRow(q)}
            />
          )
        }),
      )}

      {/* Axis labels: keys along the top, queries down the left. */}
      {Array.from({ length: n }, (_, i) => (
        <text
          key={`kt${i}`}
          x={pad + i * cell + (cell - 1.5) / 2}
          y={13}
          fontSize={8}
          fill={LABEL}
          fontFamily={MONO}
          textAnchor="middle"
        >
          {f.tokens?.[i] ?? i}
        </text>
      ))}
      {Array.from({ length: n }, (_, i) => (
        <text
          key={`qt${i}`}
          x={pad - 6}
          y={18 + i * cell + (cell - 1.5) / 2 + 3}
          fontSize={8}
          fill={selectedRow === i ? 'var(--model)' : LABEL}
          fontFamily={MONO}
          textAnchor="end"
        >
          {f.tokens?.[i] ?? i}
        </text>
      ))}

      <text x={pad} y={h - 8} fontSize={8.5} fill={LABEL}>
        key / attended-to →
      </text>
      <text
        x={10}
        y={18 + grid / 2}
        fontSize={8.5}
        fill={LABEL}
        textAnchor="middle"
        transform={`rotate(-90 10 ${18 + grid / 2})`}
      >
        query →
      </text>
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">query {queryLabel}</span>
      attends to {visibleKeys.length} key{visibleKeys.length === 1 ? '' : 's'}: {visibleKeys.map((k) => f.tokens?.[k] ?? k).join(', ')}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── routing ────────────────────────────── */

function Routing({ f }: { f: Extract<Figure, { kind: 'routing' }> }) {
  const [focus, setFocus] = useState<{ kind: 'token' | 'expert'; index: number }>({ kind: 'token', index: 0 })
  const tokens = f.tokens
  const experts = f.experts
  const rowH = 24
  const tx = 8
  const tw = 74
  const ex = W - 96
  const ew = 84
  const gridH = Math.max(tokens.length, experts.length) * rowH
  const loadH = f.showLoad ? 46 : 0
  const h = gridH + 34 + loadH

  const ty = (i: number) => 22 + i * rowH + (gridH - tokens.length * rowH) / 2
  const ey = (i: number) => 22 + i * rowH + (gridH - experts.length * rowH) / 2

  const load = experts.map((_, e) => f.routes.filter((r) => r.expert === e).length)
  const maxLoad = Math.max(1, ...load)
  const activeRoutes = f.routes.filter((r) => focus.kind === 'token' ? r.token === focus.index : r.expert === focus.index)
  const focusLabel = focus.kind === 'token' ? tokens[focus.index] : experts[focus.index]

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive expert routing diagram. ${focus.kind} ${focusLabel ?? ''} is selected.`}>
      <text x={tx} y={12} fontSize={8.5} fill={LABEL}>
        tokens
      </text>
      <text x={ex} y={12} fontSize={8.5} fill={LABEL}>
        experts
      </text>

      {/* Routing edges first, so the boxes sit on top of them. */}
      {f.routes.map((r, i) => {
        const y1 = ty(r.token) + rowH / 2 - 3
        const y2 = ey(r.expert) + rowH / 2 - 3
        const mx = (tx + tw + ex) / 2
        const active = activeRoutes.includes(r)
        return (
          <path
            className="fig-route"
            key={i}
            d={`M ${tx + tw} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${ex} ${y2}`}
            fill="none"
            stroke={active ? 'var(--model)' : MUTE}
            strokeOpacity={active ? 0.42 + r.weight * 0.58 : 0.24}
            strokeWidth={active ? 1 + r.weight * 3 : 0.8}
          />
        )
      })}

      {tokens.map((t, i) => (
        <g key={t + i}>
          <rect
            className="fig-mark"
            x={tx}
            y={ty(i)}
            width={tw}
            height={rowH - 6}
            fill={focus.kind === 'token' && focus.index === i ? 'var(--model-wash)' : 'none'}
            stroke={focus.kind === 'token' && focus.index === i ? 'var(--model)' : LINE}
            strokeWidth={focus.kind === 'token' && focus.index === i ? 1.5 : 1}
            tabIndex={0}
            role="button"
            aria-label={`Inspect routes for token ${t}`}
            onPointerEnter={() => setFocus({ kind: 'token', index: i })}
            onFocus={() => setFocus({ kind: 'token', index: i })}
            onClick={() => setFocus({ kind: 'token', index: i })}
          />
          <text
            x={tx + tw / 2}
            y={ty(i) + (rowH - 6) / 2 + 3}
            fontSize={9}
            fill={INK}
            fontFamily={MONO}
            textAnchor="middle"
          >
            {t}
          </text>
        </g>
      ))}

      {experts.map((e, i) => {
        const isShared = f.shared?.includes(i)
        const active = focus.kind === 'expert' && focus.index === i
        return (
          <g key={e + i}>
            <rect
              className="fig-mark"
              x={ex}
              y={ey(i)}
              width={ew}
              height={rowH - 6}
              fill={active ? 'var(--model-wash)' : isShared ? 'var(--role-refinement-wash)' : 'none'}
              fillOpacity={active ? 1 : isShared ? 1 : 0}
              stroke={active ? 'var(--model)' : isShared ? 'var(--role-refinement)' : LINE}
              strokeWidth={active ? 1.5 : 1}
              strokeDasharray={isShared && !active ? '3 2' : undefined}
              tabIndex={0}
              role="button"
              aria-label={`Inspect ${e}${isShared ? ', shared expert' : ''}`}
              onPointerEnter={() => setFocus({ kind: 'expert', index: i })}
              onFocus={() => setFocus({ kind: 'expert', index: i })}
              onClick={() => setFocus({ kind: 'expert', index: i })}
            />
            <text
              x={ex + ew / 2}
              y={ey(i) + (rowH - 6) / 2 + 3}
              fontSize={9}
              fill={INK}
              fontFamily={MONO}
              textAnchor="middle"
            >
              {e}
            </text>
          </g>
        )
      })}

      {/* Load histogram: the imbalance auxiliary losses exist to fight. */}
      {f.showLoad &&
        experts.map((_, i) => {
          const bw = (W - 20) / experts.length - 6
          const bh = (load[i] / maxLoad) * 26
          const bx = 10 + i * ((W - 20) / experts.length)
          return (
            <g key={`l${i}`}>
              <rect
                className="fig-mark"
                x={bx}
                y={gridH + 46 - bh}
                width={bw}
                height={bh}
                fill={focus.kind === 'expert' && focus.index === i ? 'var(--model)' : 'var(--seq-400)'}
                fillOpacity={focus.kind === 'expert' && focus.index === i ? 0.85 : 0.52}
                tabIndex={0}
                role="button"
                aria-label={`Inspect ${experts[i]}, ${load[i]} routed tokens`}
                onPointerEnter={() => setFocus({ kind: 'expert', index: i })}
                onFocus={() => setFocus({ kind: 'expert', index: i })}
                onClick={() => setFocus({ kind: 'expert', index: i })}
              />
              <text
                x={bx + bw / 2}
                y={gridH + 58}
                fontSize={8}
                fill={LABEL}
                fontFamily={MONO}
                textAnchor="middle"
              >
                {load[i]}
              </text>
            </g>
          )
        })}
      {f.showLoad && (
        <text x={10} y={gridH + 30} fontSize={8.5} fill={LABEL}>
          tokens routed per expert
        </text>
      )}
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">{focus.kind} {focusLabel}</span>
      {focus.kind === 'token'
        ? activeRoutes.map((r) => `${experts[r.expert]} ${r.weight.toFixed(2)}`).join(' · ')
        : `${activeRoutes.length} route${activeRoutes.length === 1 ? '' : 's'} · load ${load[focus.index] ?? 0}`}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── tensor ────────────────────────────── */

function TensorChain({ f }: { f: Extract<Figure, { kind: 'tensor' }> }) {
  const initial = f.chain.findIndex((step) => step.focus)
  const [selected, setSelected] = useState(initial >= 0 ? initial : 0)
  const rowH = 46
  const h = f.chain.length * rowH + 16
  const bx = 74
  const bw = W - bx - 12
  const current = f.chain[selected] ?? f.chain[0]

  return (
    <div className="fig-interactive">
    <Frame h={h} label={`Interactive tensor chain. ${current?.label ?? 'A tensor'} is selected.`}>
      {f.chain.map((step, i) => {
        const y = 8 + i * rowH
        const active = selected === i
        return (
          <g key={i}>
            {i > 0 && step.via && (
              <>
                <line x1={bx - 34} y1={y - 12} x2={bx - 34} y2={y + 4} stroke={active ? 'var(--model)' : MUTE} strokeWidth={active ? 1.8 : 1} />
                <path d={`M ${bx - 37} ${y + 1} l 3 4 l 3 -4`} fill="none" stroke={active ? 'var(--model)' : MUTE} strokeWidth={active ? 1.8 : 1} />
                <text x={bx - 28} y={y - 2} fontSize={8.5} fill={active ? 'var(--model)' : LABEL} fontFamily={MONO}>
                  {step.via}
                </text>
              </>
            )}
            <rect
              className="fig-mark"
              x={bx}
              y={y + 8}
              width={bw}
              height={26}
              fill={active ? 'var(--model-wash)' : step.focus ? INK : 'none'}
              fillOpacity={active ? 1 : step.focus ? 0.07 : 0}
              stroke={active ? 'var(--model)' : step.focus ? INK : LINE}
              strokeWidth={active ? 1.6 : step.focus ? 1.3 : 0.8}
              tabIndex={0}
              role="button"
              aria-label={`${step.label}: ${step.shape.join(' by ')}`}
              onPointerEnter={() => setSelected(i)}
              onFocus={() => setSelected(i)}
              onClick={() => setSelected(i)}
            />
            <text x={bx - 8} y={y + 25} fontSize={9} fill={active ? 'var(--model)' : LABEL} textAnchor="end">
              {step.label}
            </text>
            <text x={bx + 10} y={y + 25} fontSize={10} fill={active ? 'var(--model)' : INK} fontFamily={MONO}>
              [{step.shape.join(', ')}]
            </text>
          </g>
        )
      })}
    </Frame>
    <FigureDetail>
      <span className="fig-detail-label">{current?.label}</span>
      [{current?.shape.join(', ')}]{current?.via ? ` · after ${current.via}` : ''}
    </FigureDetail>
    </div>
  )
}

/* ────────────────────────────── entry ────────────────────────────── */

export default function FigureView({ figure }: { figure: Figure }) {
  const body =
    figure.kind === 'bars' ? (
      <Bars f={figure} />
    ) : figure.kind === 'curve' ? (
      <Curve f={figure} />
    ) : figure.kind === 'segments' ? (
      <Segments f={figure} />
    ) : figure.kind === 'ranked' ? (
      <Ranked f={figure} />
    ) : figure.kind === 'network' ? (
      <FigureD3Network f={figure} />
    ) : figure.kind === 'layered' ? (
      <FigureD3Layered f={figure} />
    ) : figure.kind === 'heatmap' ? (
      <Heatmap f={figure} />
    ) : figure.kind === 'routing' ? (
      <Routing f={figure} />
    ) : figure.kind === 'tensor' ? (
      <TensorChain f={figure} />
    ) : figure.kind === 'embedding' ? (
      <FigureEmbedding f={figure} />
    ) : (
      <Blocks f={figure} />
    )

  const steps =
    figure.kind === 'network' ||
    figure.kind === 'layered' ||
    figure.kind === 'heatmap' ||
    figure.kind === 'routing' ||
    figure.kind === 'tensor' ||
    figure.kind === 'embedding'
      ? figure.steps
      : undefined

  return (
    <figure className="fig">
      {figure.title && <figcaption className="fig-title">{figure.title}</figcaption>}
      {body}
      {steps && steps.length > 0 && (
        <ol className="fig-steps">
          {steps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      )}
      {figure.caption && <div className="fig-caption">{figure.caption}</div>}
    </figure>
  )
}

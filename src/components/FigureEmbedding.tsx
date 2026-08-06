import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { Figure } from '../data/types'

/**
 * What a token embedding actually *is*, in three linked views.
 *
 * The lookup table is the easy part to state and the hard part to feel: row `i`
 * of a matrix is a point in a space where direction carries meaning. Prose says
 * that and it lands as a slogan. Three coupled panels make it checkable —
 *
 *   1. the raw vector, as signed cells, so "a token is a list of numbers" is
 *      literal rather than metaphorical;
 *   2. the space, projected to 2D, where related tokens visibly cluster and the
 *      king − man + woman ≈ queen parallelogram closes;
 *   3. cosine similarity to every other token, which is the actual quantity the
 *      first two panels are gesturing at.
 *
 * The vectors are synthetic — generated from a seeded PRNG with a shared
 * per-cluster component plus per-token noise, then given a deliberate gender
 * axis so the analogy resolves. Real GloVe rows would need a megabyte of data
 * to make the same three points. The caption says so; this figure is here to
 * build intuition, not to report a measurement.
 */

const DIMS = 48

/** Mulberry32 — small, deterministic, good enough for illustration. */
function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Tok {
  word: string
  cluster: string
  /** Position in the 2D projection, already laid out to show the structure. */
  x: number
  y: number
  /** How far along the semantic axis this token sits (e.g. −1 male, +1 female). */
  axis: number
  vec: number[]
}

const CLUSTERS = ['royalty', 'people', 'animals', 'places'] as const

const SEED_TOKENS: Omit<Tok, 'vec'>[] = [
  { word: 'king', cluster: 'royalty', x: 0.18, y: 0.24, axis: -1 },
  { word: 'queen', cluster: 'royalty', x: 0.18, y: 0.62, axis: 1 },
  { word: 'prince', cluster: 'royalty', x: 0.3, y: 0.3, axis: -1 },
  { word: 'princess', cluster: 'royalty', x: 0.3, y: 0.68, axis: 1 },
  { word: 'man', cluster: 'people', x: 0.52, y: 0.22, axis: -1 },
  { word: 'woman', cluster: 'people', x: 0.52, y: 0.6, axis: 1 },
  { word: 'boy', cluster: 'people', x: 0.63, y: 0.28, axis: -1 },
  { word: 'girl', cluster: 'people', x: 0.63, y: 0.66, axis: 1 },
  { word: 'dog', cluster: 'animals', x: 0.82, y: 0.18, axis: 0 },
  { word: 'cat', cluster: 'animals', x: 0.9, y: 0.3, axis: 0 },
  { word: 'horse', cluster: 'animals', x: 0.85, y: 0.42, axis: 0 },
  { word: 'paris', cluster: 'places', x: 0.16, y: 0.9, axis: 0 },
  { word: 'london', cluster: 'places', x: 0.3, y: 0.94, axis: 0 },
  { word: 'tokyo', cluster: 'places', x: 0.45, y: 0.88, axis: 0 },
]

function buildTokens(): Tok[] {
  // One shared base vector per cluster, so same-cluster tokens genuinely
  // correlate — that correlation is what the strips and the bars are showing.
  const base: Record<string, number[]> = {}
  CLUSTERS.forEach((c, ci) => {
    const r = rng(1000 + ci * 77)
    base[c] = Array.from({ length: DIMS }, () => r() * 2 - 1)
  })
  // A single direction shared by every token, scaled by `axis`. This is the
  // vector that makes king − man + woman land on queen.
  const gr = rng(4242)
  const gender = Array.from({ length: DIMS }, () => gr() * 2 - 1)

  return SEED_TOKENS.map((t, i) => {
    const r = rng(7 + i * 131)
    const vec = Array.from({ length: DIMS }, (_, d) =>
      Math.max(-1, Math.min(1, base[t.cluster][d] * 0.72 + gender[d] * t.axis * 0.42 + (r() * 2 - 1) * 0.3)),
    )
    return { ...t, vec }
  })
}

function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1)
}

export default function FigureEmbedding({ f }: { f: Extract<Figure, { kind: 'embedding' }> }) {
  const tokens = useMemo(buildTokens, [])
  const [sel, setSel] = useState('king')
  const [showAnalogy, setShowAnalogy] = useState(f.analogy ?? true)

  const stripRef = useRef<SVGSVGElement>(null)
  const spaceRef = useRef<SVGSVGElement>(null)
  const barsRef = useRef<SVGSVGElement>(null)

  const current = tokens.find((t) => t.word === sel) ?? tokens[0]

  /* ── Panel 1: the raw vector ───────────────────────────────────────── */
  useEffect(() => {
    const svg = d3.select(stripRef.current)
    const motionMs = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 180
    const W = 420
    const cols = 24
    const rows = DIMS / cols
    const cell = W / cols
    const H = rows * cell + 2

    svg.attr('viewBox', `0 0 ${W} ${H}`)

    // Diverging, because these values are signed and zero is meaningful.
    const colour = d3
      .scaleDiverging<string>()
      .domain([-1, 0, 1])
      .interpolator(d3.interpolateRgbBasis(['#2a78d6', '#ebe9e4', '#d03b3b']))

    const cells = svg
      .selectAll<SVGRectElement, number>('rect.emb-cell')
      .data(current.vec, (_d, i) => i)

    cells
      .enter()
      .append('rect')
      .attr('class', 'emb-cell')
      .attr('x', (_d, i) => (i % cols) * cell)
      .attr('y', (_d, i) => Math.floor(i / cols) * cell)
      .attr('width', cell - 1.5)
      .attr('height', cell - 1.5)
      .attr('rx', 1.5)
      .attr('fill', '#ebe9e4')
      .merge(cells)
      .transition()
      .duration(motionMs)
      .attr('fill', (d) => colour(d))
  }, [current])

  /* ── Panel 2: the space ────────────────────────────────────────────── */
  useEffect(() => {
    const svg = d3.select(spaceRef.current)
    const motionMs = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 180
    const W = 420
    const H = 300
    const M = 18
    svg.attr('viewBox', `0 0 ${W} ${H}`)
    svg.selectAll('*').remove()

    const x = d3.scaleLinear().domain([0, 1]).range([M, W - M])
    const y = d3.scaleLinear().domain([0, 1]).range([M, H - M])

    const clusterColour = d3
      .scaleOrdinal<string, string>()
      .domain(CLUSTERS as unknown as string[])
      .range(['#4a3aa7', '#2a78d6', '#1baf7a', '#eb6834'])

    // Cluster hulls, drawn first so points sit on top.
    CLUSTERS.forEach((c) => {
      const pts = tokens.filter((t) => t.cluster === c).map((t) => [x(t.x), y(t.y)] as [number, number])
      if (pts.length < 3) return
      const hull = d3.polygonHull(pts)
      if (!hull) return
      svg
        .append('path')
        .attr('d', `M${hull.map((p) => p.join(',')).join('L')}Z`)
        .attr('fill', clusterColour(c))
        .attr('fill-opacity', 0.07)
        .attr('stroke', clusterColour(c))
        .attr('stroke-opacity', 0.25)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3')
    })

    // The analogy: king → queen must be parallel to man → woman. That
    // parallelism *is* the claim, so both arrows are drawn together.
    if (showAnalogy) {
      const pairs: [string, string][] = [
        ['king', 'queen'],
        ['man', 'woman'],
        ['prince', 'princess'],
        ['boy', 'girl'],
      ]
      const defs = svg.append('defs')
      defs
        .append('marker')
        .attr('id', 'emb-arrow')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 9)
        .attr('refY', 5)
        .attr('markerWidth', 5)
        .attr('markerHeight', 5)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M0,0 L10,5 L0,10 z')
        .attr('fill', 'var(--text-faint)')

      pairs.forEach(([a, b]) => {
        const ta = tokens.find((t) => t.word === a)!
        const tb = tokens.find((t) => t.word === b)!
        svg
          .append('line')
          .attr('x1', x(ta.x))
          .attr('y1', y(ta.y))
          .attr('x2', x(tb.x))
          .attr('y2', y(tb.y))
          .attr('stroke', 'var(--text-faint)')
          .attr('stroke-width', 1.2)
          .attr('stroke-dasharray', '4 3')
          .attr('marker-end', 'url(#emb-arrow)')
          .attr('opacity', 0)
          .transition()
          .duration(motionMs)
          .attr('opacity', 0.75)
      })
    }

    const g = svg
      .selectAll('g.emb-pt')
      .data(tokens)
      .enter()
      .append('g')
      .attr('class', 'emb-pt')
      .attr('transform', (d) => `translate(${x(d.x)},${y(d.y)})`)
      .style('cursor', 'pointer')
      .on('click', (_e, d) => setSel(d.word))

    g.append('circle')
      .attr('r', (d) => (d.word === sel ? 7 : 4.5))
      .attr('fill', (d) => clusterColour(d.cluster))
      .attr('stroke', 'var(--surface)')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', 0)
      .attr('y', -11)
      .attr('text-anchor', 'middle')
      .attr('class', 'emb-label')
      .attr('font-weight', (d) => (d.word === sel ? 700 : 500))
      .text((d) => d.word)
  }, [tokens, sel, showAnalogy])

  /* ── Panel 3: cosine similarity ────────────────────────────────────── */
  useEffect(() => {
    const svg = d3.select(barsRef.current)
    const motionMs = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 180
    const rowH = 18
    const others = tokens
      .filter((t) => t.word !== current.word)
      .map((t) => ({ word: t.word, cluster: t.cluster, s: cosine(current.vec, t.vec) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, 8)

    const W = 420
    const H = others.length * rowH + 8
    const LABEL = 62
    svg.attr('viewBox', `0 0 ${W} ${H}`)

    const x = d3
      .scaleLinear()
      .domain([Math.min(0, d3.min(others, (d) => d.s) ?? 0), 1])
      .range([LABEL, W - 38])

    const clusterColour = d3
      .scaleOrdinal<string, string>()
      .domain(CLUSTERS as unknown as string[])
      .range(['#4a3aa7', '#2a78d6', '#1baf7a', '#eb6834'])

    const rows = svg.selectAll<SVGGElement, (typeof others)[number]>('g.emb-bar').data(others, (d) => d.word)
    rows.exit().remove()

    const enter = rows.enter().append('g').attr('class', 'emb-bar')
    enter.append('text').attr('class', 'emb-bar-label').attr('x', LABEL - 8).attr('text-anchor', 'end')
    // 4px rounded data-end anchored at the baseline, per the mark spec.
    enter.append('rect').attr('class', 'emb-bar-fill').attr('height', 9).attr('rx', 3)
    enter.append('text').attr('class', 'emb-bar-val').attr('x', W - 34)

    const all = enter.merge(rows)
    all.transition().duration(motionMs).attr('transform', (_d, i) => `translate(0,${i * rowH + 12})`)
    all.select('text.emb-bar-label').text((d) => d.word)
    all
      .select('rect.emb-bar-fill')
      .attr('x', x(0))
      .attr('y', -7)
      .attr('fill', (d) => clusterColour(d.cluster))
      .transition()
      .duration(motionMs)
      .attr('width', (d) => Math.max(1, x(d.s) - x(0)))
    all
      .select<SVGTextElement>('text.emb-bar-val')
      .transition()
      .duration(motionMs)
      .textTween(function (d) {
        // Count up from whatever is on screen, so switching tokens reads as the
        // same bars moving rather than a new chart appearing.
        const prev = parseFloat(this.textContent ?? '0') || 0
        const i = d3.interpolateNumber(prev, d.s)
        return (t) => i(t).toFixed(2)
      })
  }, [current, tokens])

  return (
    <div className="emb">
      <div className="emb-tokens">
        {tokens.map((t) => (
          <button
            key={t.word}
            className={`emb-chip${t.word === sel ? ' on' : ''}`}
            onClick={() => setSel(t.word)}
          >
            {t.word}
          </button>
        ))}
      </div>

      <div className="emb-panel">
        <div className="emb-panel-head">
          <span className="emb-panel-title">1 · The row</span>
          <span className="emb-panel-note">
            <code>W_emb[{tokens.findIndex((t) => t.word === sel)}]</code> — {DIMS} of d_model dims
          </span>
        </div>
        <svg ref={stripRef} className="emb-svg" />
        <div className="emb-scale">
          <span>−1</span>
          <i className="emb-ramp" />
          <span>+1</span>
        </div>
      </div>

      <div className="emb-panel">
        <div className="emb-panel-head">
          <span className="emb-panel-title">2 · The space</span>
          <button className="emb-toggle" onClick={() => setShowAnalogy((v) => !v)}>
            {showAnalogy ? 'hide' : 'show'} the gender axis
          </button>
        </div>
        <svg ref={spaceRef} className="emb-svg" />
        <p className="emb-caption">
          Projected to 2D. Same-cluster tokens land together, and the four dashed arrows stay
          parallel — that shared offset is the direction you add to <b>king</b> to get{' '}
          <b>queen</b>. Click any point.
        </p>
      </div>

      <div className="emb-panel">
        <div className="emb-panel-head">
          <span className="emb-panel-title">3 · Nearest by cosine</span>
          <span className="emb-panel-note">to {sel}</span>
        </div>
        <svg ref={barsRef} className="emb-svg" />
      </div>
    </div>
  )
}

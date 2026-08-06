import { useLayoutEffect, useMemo, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import type { MathBlock } from '../data/types'
import Prose from './Prose'

/** Below this the equation is unreadable, so let it scroll instead. */
const MIN_SCALE = 0.62

/** A single KaTeX expression rendered in display mode. */
export function Tex({ tex, inline = false }: { tex: string; inline?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null)

  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { displayMode: !inline, throwOnError: false, output: 'html' })
    } catch {
      // Never let a malformed expression take the panel down — show the source.
      return `<code>${tex.replace(/</g, '&lt;')}</code>`
    }
  }, [tex, inline])

  /*
   * KaTeX cannot wrap, so a wide equation simply overflows the panel. Scale the
   * whole expression down until it fits. Font-size is the right lever because
   * KaTeX is em-based throughout, so the height reflows too — a transform would
   * leave the original height behind and gap the layout.
   */
  useLayoutEffect(() => {
    if (inline) return
    const el = ref.current
    if (!el) return

    const fit = () => {
      el.style.fontSize = ''
      const avail = el.clientWidth
      const needed = el.scrollWidth
      if (avail > 0 && needed > avail + 1) {
        el.style.fontSize = `${Math.max(MIN_SCALE, avail / needed).toFixed(3)}em`
      }
    }

    fit()
    // The panel width is viewport-driven, so a resize is the only thing that
    // changes the available space.
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [html, inline])

  return (
    <span
      ref={ref}
      className={inline ? 'tex-inline' : 'tex-display'}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

export default function MathBlockView({ block }: { block: MathBlock }) {
  return (
    <div className="mathblock">
      {block.title && <div className="mathblock-title">{block.title}</div>}

      <div className="mathblock-signpost">In notation</div>
      <Tex tex={block.tex} />

      {block.where && (
        <>
          <div className="mathblock-signpost where-signpost">What the symbols mean</div>
          <dl className="where">
            {block.where.map((w) => (
              <div className="where-row" key={w.sym}>
                <dt>
                  <Tex tex={w.sym} inline />
                </dt>
                <dd><Prose className="where-text">{w.means}</Prose></dd>
              </div>
            ))}
          </dl>
        </>
      )}

      {block.worked && (
        <div className="worked">
          <div className="mathblock-signpost">With real numbers</div>
          {block.worked.map((w, i) => (
            <div className="worked-row" key={i}>
              <Tex tex={w.tex} />
              {w.caption && <div className="worked-caption">{w.caption}</div>}
            </div>
          ))}
        </div>
      )}

      {block.note && <Prose className="mathblock-note">{block.note}</Prose>}
    </div>
  )
}

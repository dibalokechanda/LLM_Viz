import type { Block, Variant } from '../data/types'
import { useModel } from '../ModelContext'
import { evalTemplate } from '../content/expr'
import Prose from './Prose'
import Section from './Section'
import MathBlockView from './Math'
import FigureView from './Figure'
import CodeBlockView from './Code'
import { rawImplementationSnippet, transformersSnippet } from '../content/transformersSnippets'

/**
 * The instantiated block. Everything here is the *variant's* content, rendered
 * against the loaded model's dimensions — which is what makes selecting a node
 * in the lineage map "render the block" rather than merely describe it.
 */
export default function VariantDetail({ block, variant }: { block: Block; variant: Variant }) {
  const { dims, model } = useModel()

  // Cost rows are templates over the loaded model's dims — see content/expr.ts.
  // Evaluated here rather than at load so the same content re-reads against
  // whichever model is current.
  const costs = variant.cost?.map((row) => ({
    ...row,
    value: evalTemplate(row.value, dims),
    note: row.note ? evalTemplate(row.note, dims) : undefined,
  }))

  let n = 0
  const next = () => ++n

  return (
    <div className="vd">
      <div className="vd-head">
        <h3>{variant.full}</h3>
        <div className="vd-meta">
          <span className={`ln-role role-${variant.role}`}>{variant.role}</span>
          <span className="vd-year">{variant.year}</span>
          {variant.paper && (
            <a className="vd-paper" href={variant.paper.url} target="_blank" rel="noopener noreferrer">
              {variant.paper.authors ?? 'paper'} ↗
            </a>
          )}
        </div>
        {variant.fixes && (
          <div className="vd-fixes">
            <span className="vd-fixes-label">What it fixes</span>
            <p>{variant.fixes}</p>
          </div>
        )}
      </div>

      <div className="vd-body">
        {/* The live numbers come first. They are the reason a config was
            loaded at all, and they change what everything below means. */}
        {costs && costs.length > 0 && (
          <Section index={next()} title={`At ${dims.name}`} count={costs.length} defaultOpen>
            <div className="cost-table">
              {costs.map((c) => (
                <div className={`cost-row${c.key ? ' is-key' : ''}`} key={c.label}>
                  <div className="cost-label">{c.label}</div>
                  <div className="cost-value">{c.value}</div>
                  {c.note && <div className="cost-note">{c.note}</div>}
                </div>
              ))}
            </div>
            {!model && (
              <p className="cost-hint">
                No model loaded — these use a reference 8B shape. Load a config to see real numbers.
              </p>
            )}
          </Section>
        )}

        <Section index={next()} title="How it works" defaultOpen>
          <Prose>{variant.detail.join('\n\n')}</Prose>
        </Section>

        {variant.figures && variant.figures.length > 0 && (
          <Section index={next()} title="See it" count={variant.figures.length} defaultOpen>
            {variant.figures.map((f, i) => <FigureView figure={f} key={i} />)}
          </Section>
        )}

        {variant.math && variant.math.length > 0 && (
          <Section index={next()} title="Formula walkthrough" count={variant.math.length}>
            {variant.math.map((m, i) => <MathBlockView block={m} key={i} />)}
          </Section>
        )}

        <Section index={next()} title="Raw implementation · Raschka reference" defaultOpen>
          <CodeBlockView block={rawImplementationSnippet(block, variant)} />
        </Section>

        <Section index={next()} title="Using it with Transformers · framework contrast">
          <CodeBlockView block={transformersSnippet(block, variant)} />
        </Section>

        {variant.code && variant.code.length > 0 && (
          <Section index={next()} title="Reference code" count={variant.code.length}>
            {variant.code.map((c, i) => <CodeBlockView block={c} key={i} />)}
          </Section>
        )}

        {variant.example && (
          <Section index={next()} title="Example" defaultOpen>
            <div className="example">
              <div className="example-half">
                {variant.example.beforeLabel && <div className="example-label">{variant.example.beforeLabel}</div>}
                <div className={`example-text${variant.example.mono ? ' mono' : ''}`}>{variant.example.before}</div>
              </div>
              <div className="arrow-sep">↓</div>
              <div className="example-half">
                {variant.example.afterLabel && <div className="example-label">{variant.example.afterLabel}</div>}
                <div className={`example-text${variant.example.mono ? ' mono' : ''}`}>{variant.example.after}</div>
              </div>
            </div>
          </Section>
        )}

        {variant.distinctions && variant.distinctions.length > 0 && (
          <Section index={next()} title="Easy to confuse with" count={variant.distinctions.length} defaultOpen>
            {variant.distinctions.map((d) => (
              <div className="distinction" key={d.title}>
                <h4>{d.title}</h4>
                <p>{d.body}</p>
              </div>
            ))}
          </Section>
        )}

        {variant.usedBy && variant.usedBy.length > 0 && (
          <Section index={next()} title="Where it appears" count={variant.usedBy.length} defaultOpen>
            <div className="usedby-row">
              {variant.usedBy.map((m) => <span className="usedby-chip" key={m}>{m}</span>)}
            </div>
          </Section>
        )}

        <Section index={next()} title={[block.label, 'its job in the model'].join(': ')}>
          <Prose className="role-text">{block.role.join('\n\n')}</Prose>
          <div className="io-row">
            <div><span>in</span><code>{block.io.in}</code></div>
            <div><span>out</span><code>{block.io.out}</code></div>
          </div>
          {block.caveat && <div className="block-caveat">{block.caveat}</div>}
        </Section>
      </div>
    </div>
  )
}

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Block, BlockId, CodeBlock, Concept, CostRow, MathBlock, Variant } from '../data/types'
import { blocks } from '../content'
import { parseBlockFile, serializeBlockFile } from '../content/parse'
import { checkTemplate, evalTemplate } from '../content/expr'
import { REFERENCE_DIMS } from '../ModelContext'
import RichText from './RichText'
import { mathToText, textToMath } from './mathText'
import Icon from '../components/Icon'

/**
 * The content editor.
 *
 * Edits the same `content/<block>.md` files you would edit by hand and writes
 * them back through the dev server, so there is exactly one copy of the
 * content and it stays in the repository. Nothing lives in browser storage;
 * closing the tab loses nothing that was saved and saves nothing that wasn't.
 *
 * Fields are grouped by how they behave rather than by where they sit in the
 * file: prose gets a rich-text box, lists get a list editor, and the formula
 * fields get a plain input with live validation, because a cost template is
 * code and pretending otherwise would hide its errors.
 */

/**
 * Two tabs, not six.
 *
 * Everything a variant *says* — prose, trade-offs, maths, code, cost — is one
 * continuous surface, because that is one piece of writing and splitting it
 * across tabs meant hunting for the tab holding the next field. `details` is
 * kept separate because it is identity rather than content: the label, year,
 * role and paper are set once when a variant is created and then left alone.
 */
type Tab = 'content' | 'details'

const TABS: { id: Tab; label: string }[] = [
  { id: 'content', label: 'Content' },
  { id: 'details', label: 'Details' },
]

/** A titled band in the single content surface. */
function Sec({
  title,
  hint,
  count,
  children,
}: {
  title: string
  hint?: string
  count?: number
  children: React.ReactNode
}) {
  return (
    <section className="ce-sec">
      <div className="ce-sec-head">
        <h4>{title}</h4>
        {count !== undefined && count > 0 && <span className="ce-sec-count">{count}</span>}
        {hint && <span className="ce-sec-hint">{hint}</span>}
      </div>
      {children}
    </section>
  )
}

/**
 * A textarea that grows to fit its content.
 *
 * On one continuous surface an internally-scrolling box is a trap: the content
 * is hidden and the page scroll fights the field scroll. Counting rows does not
 * work either, since trade-off entries are sentences that wrap. So measure.
 */
function AutoTextarea({
  value,
  onChange,
  className = 'ce-textarea',
  minRows = 3,
  ...rest
}: {
  value: string
  onChange: (v: string) => void
  className?: string
  minRows?: number
} & Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, 'value' | 'onChange' | 'className'>) {
  const ref = useRef<HTMLTextAreaElement>(null)

  const fit = useCallback(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto' // collapse first, or it can only ever grow
    el.style.height = `${Math.max(el.scrollHeight, minRows * 22)}px`
  }, [minRows])

  useLayoutEffect(fit, [fit, value])

  return (
    <textarea
      {...rest}
      ref={ref}
      className={className}
      value={value}
      onChange={(e) => {
        onChange(e.target.value)
        fit()
      }}
    />
  )
}

/** A simple one-per-line list editor — trade-offs, used-by, symbol legends. */
function ListEditor({
  label,
  items,
  onChange,
  placeholder,
}: {
  label: string
  items: string[]
  onChange: (v: string[]) => void
  placeholder?: string
}) {
  return (
    <label className="ce-field">
      <span className="ce-label">{label}</span>
      <AutoTextarea
        placeholder={placeholder ?? 'One per line'}
        value={items.join('\n')}
        onChange={(v) => onChange(v.split('\n').map((s) => s.trim()).filter(Boolean))}
      />
      <span className="ce-hint">One per line</span>
    </label>
  )
}

function TextField({
  label,
  value,
  onChange,
  hint,
  mono,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  hint?: string
  mono?: boolean
}) {
  return (
    <label className="ce-field">
      <span className="ce-label">{label}</span>
      <input
        className={`ce-input${mono ? ' mono' : ''}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint && <span className="ce-hint">{hint}</span>}
    </label>
  )
}

export default function ContentEditor({
  initialBlock,
  onClose,
}: {
  initialBlock?: BlockId | null
  onClose: () => void
}) {
  const [blockId, setBlockId] = useState<BlockId>(initialBlock ?? blocks[0].id)
  const [variantId, setVariantId] = useState<string>('')
  const [tab, setTab] = useState<Tab>('content')
  const [draft, setDraft] = useState<Block | null>(null)
  const [dirty, setDirty] = useState(false)
  const [status, setStatus] = useState<{ kind: 'idle' | 'saving' | 'ok' | 'err'; msg?: string }>({
    kind: 'idle',
  })

  // Work on a copy of the parsed block. Loading from the server rather than the
  // bundled import means the editor always opens whatever is on disk now, even
  // if a hand edit landed since the page loaded.
  useEffect(() => {
    let cancelled = false
    setStatus({ kind: 'idle' })
    fetch(`/api/content/${blockId}.md`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((j: { text: string }) => {
        if (cancelled) return
        const { block } = parseBlockFile(j.text, `${blockId}.md`)
        setDraft(block)
        setVariantId(block.variants[0]?.id ?? '')
        setDirty(false)
      })
      .catch((e) => {
        if (cancelled) return
        // Fall back to the bundled copy so the editor still opens read-only
        // when the content server is not running.
        const fallback = blocks.find((b) => b.id === blockId) ?? null
        setDraft(fallback ? structuredClone(fallback) : null)
        setVariantId(fallback?.variants[0]?.id ?? '')
        setStatus({
          kind: 'err',
          msg: `Content server not reachable (${e.message}) — showing the bundled copy. Run \`npm run dev\` to enable saving.`,
        })
      })
    return () => {
      cancelled = true
    }
  }, [blockId])

  const variant = useMemo(
    () => draft?.variants.find((v) => v.id === variantId),
    [draft, variantId],
  )

  /** Apply a change to the currently selected variant. */
  const patchVariant = useCallback(
    (patch: Partial<Variant>) => {
      setDraft((d) =>
        d
          ? { ...d, variants: d.variants.map((v) => (v.id === variantId ? { ...v, ...patch } : v)) }
          : d,
      )
      setDirty(true)
    },
    [variantId],
  )

  const patchBlock = useCallback((patch: Partial<Block>) => {
    setDraft((d) => (d ? { ...d, ...patch } : d))
    setDirty(true)
  }, [])

  const save = useCallback(async () => {
    if (!draft) return
    setStatus({ kind: 'saving' })
    try {
      const text = serializeBlockFile(draft)
      // Round-trip before writing: if what we produced cannot be read back,
      // the bug is ours and the file should not be touched.
      parseBlockFile(text, `${draft.id}.md`)
      const res = await fetch(`/api/content/${draft.id}.md`, {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const j = await res.json()
      if (!res.ok) throw new Error(j.error ?? `HTTP ${res.status}`)
      setDirty(false)
      setStatus({ kind: 'ok', msg: `Saved content/${draft.id}.md` })
    } catch (e) {
      setStatus({ kind: 'err', msg: e instanceof Error ? e.message : String(e) })
    }
  }, [draft])

  // Cmd/Ctrl-S saves; Esc closes unless there is unsaved work.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault()
        void save()
      }
      if (e.key === 'Escape' && !dirty) onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [save, dirty, onClose])

  return (
    <div className="ce" role="dialog" aria-modal="true" aria-label="Content editor">
      <header className="ce-head">
        <div className="ce-head-text">
          <span className="ce-eyebrow">
            <Icon name="pencil" size={15} /> Editing content/{blockId}.md
          </span>
          <h2>{draft?.label ?? '…'}</h2>
        </div>

        <div className="ce-actions">
          {status.kind === 'err' && <span className="ce-status err">{status.msg}</span>}
          {status.kind === 'ok' && <span className="ce-status ok">{status.msg}</span>}
          {dirty && <span className="ce-status dirty">unsaved</span>}
          <button className="ce-save" onClick={() => void save()} disabled={!dirty || status.kind === 'saving'}>
            {status.kind === 'saving' ? 'Saving…' : 'Save  ⌘S'}
          </button>
          <button
            className="ce-close"
            onClick={() => {
              if (!dirty || window.confirm('Discard unsaved changes?')) onClose()
            }}
          >
            ✕
          </button>
        </div>
      </header>

      <div className="ce-body">
        <nav className="ce-nav">
          <div className="ce-nav-group">Blocks</div>
          {blocks.map((b) => (
            <button
              key={b.id}
              className={`ce-nav-item${b.id === blockId ? ' on' : ''}`}
              onClick={() => {
                if (dirty && !window.confirm('Discard unsaved changes?')) return
                setBlockId(b.id)
              }}
            >
              <span className="ce-nav-ord">§{b.ordinal}</span>
              {b.label}
            </button>
          ))}

          {draft && (
            <>
              <div className="ce-nav-group">Variants</div>
              <button
                className={`ce-nav-item${variantId === '' ? ' on' : ''}`}
                onClick={() => setVariantId('')}
              >
                <span className="ce-nav-ord">◆</span>
                Block itself
              </button>
              {draft.variants.map((v) => (
                <button
                  key={v.id}
                  className={`ce-nav-item${v.id === variantId ? ' on' : ''}`}
                  onClick={() => setVariantId(v.id)}
                >
                  <span className="ce-nav-ord">{v.year}</span>
                  {v.label}
                </button>
              ))}
            </>
          )}
        </nav>

        <main className="ce-main">
          {!draft ? (
            <p className="ce-empty">Loading…</p>
          ) : variantId === '' ? (
            /* ── The block itself ── */
            <div className="ce-pane">
              <h3 className="ce-pane-title">The block</h3>
              <TextField label="Label" value={draft.label} onChange={(label) => patchBlock({ label })} />
              <TextField
                label="Tagline"
                value={draft.tagline}
                onChange={(tagline) => patchBlock({ tagline })}
                hint="One line, shown on the card in the stack"
              />
              <div className="ce-field">
                <span className="ce-label">What this position is for</span>
                <RichText
                  value={draft.role.join('\n\n')}
                  onChange={(md) => patchBlock({ role: md.split(/\n{2,}/).filter(Boolean) })}
                  minHeight={200}
                />
              </div>
              <div className="ce-row">
                <TextField label="Input shape" mono value={draft.io.in} onChange={(v) => patchBlock({ io: { ...draft.io, in: v } })} />
                <TextField label="Output shape" mono value={draft.io.out} onChange={(v) => patchBlock({ io: { ...draft.io, out: v } })} />
              </div>
              <TextField
                label="Caveat"
                value={draft.caveat ?? ''}
                onChange={(caveat) => patchBlock({ caveat: caveat || undefined })}
                hint="Shown when the one-variant-per-block model genuinely misdescribes some models"
              />
            </div>
          ) : !variant ? (
            <p className="ce-empty">Pick a variant.</p>
          ) : (
            <div className="ce-pane">
              <div className="ce-tabs">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    className={`ce-tab${tab === t.id ? ' on' : ''}`}
                    onClick={() => setTab(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {/* One continuous surface. Everything a variant says sits in a
                  single scroll, in the order it appears in the panel, so
                  writing runs straight through instead of hunting for the tab
                  that holds the next field. Only the identity fields are held
                  back, because those are set once and then left alone. */}
              {tab === 'content' && (
                <>
                  <Sec title="What it is" hint="The main prose. Blank line between paragraphs.">
                    <RichText
                      value={variant.detail.join('\n\n')}
                      onChange={(md) => patchVariant({ detail: md.split(/\n{2,}/).filter(Boolean) })}
                      minHeight={260}
                    />
                  </Sec>

                  <Sec title="What it fixes" hint="One sentence on what was wrong with what came before.">
                    <RichText
                      value={variant.fixes ?? ''}
                      onChange={(md) => patchVariant({ fixes: md || undefined })}
                      placeholder="What was wrong with what came before…"
                      minHeight={90}
                    />
                  </Sec>

                  <Sec title="The maths" count={variant.math?.length} hint="One equation per block. Plain text — see the key under each.">
                    <MathEditor blocks={variant.math ?? []} onChange={(math) => patchVariant({ math })} />
                  </Sec>

                  <Sec title="Used by" count={variant.usedBy?.length} hint="Models that ship this. Rendered as chips.">
                    <ListEditor
                      label="Models"
                      items={variant.usedBy ?? []}
                      onChange={(usedBy) => patchVariant({ usedBy })}
                    />
                  </Sec>

                  <Sec title="Code" count={variant.code?.length}>
                    <CodeEditor blocks={variant.code ?? []} onChange={(code) => patchVariant({ code })} />
                  </Sec>

                  <Sec
                    title="Concept map"
                    count={variant.concepts?.length}
                    hint="Cards form a map for this variant. Add sub-cards to explain a card in more detail."
                  >
                    <ConceptEditor
                      concepts={variant.concepts ?? []}
                      onChange={(concepts) => patchVariant({ concepts: concepts.length > 0 ? concepts : undefined })}
                    />
                  </Sec>

                  <Sec title="Cost table" count={variant.cost?.length}>
                    <CostEditor rows={variant.cost ?? []} onChange={(cost) => patchVariant({ cost })} />
                  </Sec>
                </>
              )}

              {tab === 'details' && (
                <>
                  <div className="ce-row">
                    <TextField label="Short label" value={variant.label} onChange={(label) => patchVariant({ label })} />
                    <TextField label="Year" value={String(variant.year)} onChange={(y) => patchVariant({ year: Number(y) || variant.year })} />
                  </div>
                  <TextField label="Full name" value={variant.full} onChange={(full) => patchVariant({ full })} />
                  <TextField label="Tagline" value={variant.tagline} onChange={(tagline) => patchVariant({ tagline })} hint="One line, shown on the lineage node" />
                  <label className="ce-field">
                    <span className="ce-label">Role</span>
                    <select
                      className="ce-input"
                      value={variant.role}
                      onChange={(e) => patchVariant({ role: e.target.value as Variant['role'] })}
                    >
                      {['origin', 'refinement', 'branch', 'synthesis', 'legacy', 'frontier'].map((r) => (
                        <option key={r} value={r}>{r}</option>
                      ))}
                    </select>
                    <span className="ce-hint">Drives the node's colour and shape in the lineage map</span>
                  </label>
                  <div className="ce-row">
                    <TextField label="Paper title" value={variant.paper?.title ?? ''} onChange={(t) => patchVariant({ paper: { ...(variant.paper ?? { url: '' }), title: t } })} />
                    <TextField label="Authors" value={variant.paper?.authors ?? ''} onChange={(a) => patchVariant({ paper: { ...(variant.paper ?? { title: '', url: '' }), authors: a } })} />
                  </div>
                  <TextField label="Paper URL" mono value={variant.paper?.url ?? ''} onChange={(u) => patchVariant({ paper: { ...(variant.paper ?? { title: '' }), url: u } })} />
                </>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

/* ────────────────────────────── sub-editors ────────────────────────────── */

function MathEditor({ blocks: mb, onChange }: { blocks: MathBlock[]; onChange: (m: MathBlock[]) => void }) {
  return (
    <>
      {mb.map((m, i) => (
        <div className="ce-card" key={i}>
          <div className="ce-card-head">
            <span>Equation {i + 1}</span>
            <button className="ce-del" onClick={() => onChange(mb.filter((_, j) => j !== i))}>remove</button>
          </div>
          <AutoTextarea
            className="ce-textarea mono"
            minRows={6}
            spellCheck={false}
            value={mathToText(m)}
            onChange={(text) => onChange(mb.map((x, j) => (i === j ? textToMath(text) : x)))}
          />
          <span className="ce-hint">
            Caption on the first line · TeX between <code>$$</code> · then optional{' '}
            <code>where</code>, <code>worked</code> and <code>note</code> sections. Legend and
            worked lines are <code>- item — description</code>, separated by an em dash.
          </span>
        </div>
      ))}
      <button
        className="ce-add"
        onClick={() => onChange([...mb, { tex: '', title: '' }])}
      >
        + Add an equation
      </button>
    </>
  )
}

function CodeEditor({ blocks: cb, onChange }: { blocks: CodeBlock[]; onChange: (c: CodeBlock[]) => void }) {
  const patch = (i: number, p: Partial<CodeBlock>) =>
    onChange(cb.map((c, j) => (i === j ? { ...c, ...p } : c)))

  return (
    <>
      {cb.map((c, i) => (
        <div className="ce-card" key={i}>
          <div className="ce-card-head">
            <span>Snippet {i + 1}</span>
            <button className="ce-del" onClick={() => onChange(cb.filter((_, j) => j !== i))}>remove</button>
          </div>
          <div className="ce-row">
            <TextField label="Header" value={c.title ?? ''} onChange={(title) => patch(i, { title })} />
            <label className="ce-field">
              <span className="ce-label">Language</span>
              <select className="ce-input" value={c.language} onChange={(e) => patch(i, { language: e.target.value as CodeBlock['language'] })}>
                {['python', 'json', 'bash', 'text'].map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
            </label>
          </div>
          <label className="ce-field">
            <span className="ce-label">Source</span>
            <AutoTextarea
              className="ce-textarea mono"
              minRows={6}
              value={c.code}
              spellCheck={false}
              onChange={(code) => patch(i, { code })}
            />
          </label>
          <label className="ce-field">
            <span className="ce-label">Note</span>
            <AutoTextarea minRows={2} value={c.note ?? ''} onChange={(note) => patch(i, { note: note || undefined })} />
          </label>
        </div>
      ))}
      <button className="ce-add" onClick={() => onChange([...cb, { language: 'python', code: '' }])}>+ Add a snippet</button>
    </>
  )
}

const CONCEPT_KINDS: Concept['kind'][] = ['idea', 'formula', 'method', 'metric', 'pitfall', 'tradeoff']

function moveItem<T>(items: T[], from: number, to: number): T[] {
  if (to < 0 || to >= items.length) return items
  const next = [...items]
  const [item] = next.splice(from, 1)
  next.splice(to, 0, item)
  return next
}

function nextConceptId(concepts: Concept[], label = 'concept'): string {
  const used = new Set<string>()
  const collect = (cards: Concept[]) => cards.forEach((card) => {
    used.add(card.id)
    if (card.children) collect(card.children)
  })
  collect(concepts)

  const base = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'concept'
  let candidate = base
  let suffix = 2
  while (used.has(candidate)) candidate = `${base}-${suffix++}`
  return candidate
}

function newConcept(id: string): Concept {
  return {
    id,
    label: 'New concept',
    kind: 'idea',
    summary: 'A one-sentence summary shown on the card.',
    detail: [],
  }
}

function ConceptCardEditor({
  concept,
  title,
  depth,
  makeId,
  onChange,
  onRemove,
  onMove,
}: {
  concept: Concept
  title: string
  depth: number
  makeId: (label?: string) => string
  onChange: (concept: Concept) => void
  onRemove: () => void
  onMove: (direction: -1 | 1) => void
}) {
  const patch = (change: Partial<Concept>) => onChange({ ...concept, ...change })
  const children = concept.children ?? []
  const addChild = () => patch({ children: [...children, newConcept(makeId('subconcept'))] })
  const patchChild = (index: number, child: Concept) =>
    patch({ children: children.map((current, currentIndex) => (currentIndex === index ? child : current)) })
  const removeChild = (index: number) => {
    const next = children.filter((_, currentIndex) => currentIndex !== index)
    patch({ children: next.length > 0 ? next : undefined })
  }

  return (
    <div className={`ce-card ce-concept-card${depth > 0 ? ' is-child' : ''}`}>
      <div className="ce-card-head">
        <span>{title}</span>
        <span className="ce-card-actions">
          <button className="ce-move" onClick={() => onMove(-1)} aria-label={`Move ${title} up`} title="Move up">↑</button>
          <button className="ce-move" onClick={() => onMove(1)} aria-label={`Move ${title} down`} title="Move down">↓</button>
          <button className="ce-del" onClick={onRemove}>remove</button>
        </span>
      </div>
      <div className="ce-row">
        <TextField label="Card id" mono value={concept.id} onChange={(id) => patch({ id })} hint="Unique within this variant" />
        <label className="ce-field">
          <span className="ce-label">Kind</span>
          <select className="ce-input" value={concept.kind} onChange={(event) => patch({ kind: event.target.value as Concept['kind'] })}>
            {CONCEPT_KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
          </select>
        </label>
      </div>
      <TextField label="Title" value={concept.label} onChange={(label) => patch({ label })} />
      <TextField label="Card summary" value={concept.summary} onChange={(summary) => patch({ summary })} hint="The short text visible on the map card" />
      <div className="ce-field">
        <span className="ce-label">Card detail</span>
        <RichText
          value={(concept.detail ?? []).join('\n\n')}
          onChange={(markdown) => patch({ detail: markdown.split(/\n{2,}/).filter(Boolean) })}
          placeholder="What this concept means, why it matters, and how it connects."
          minHeight={130}
        />
      </div>

      {depth === 0 && (
        <div className="ce-concept-children">
          <div className="ce-concept-subhead">Sub-cards <span>Expand from this card in the map</span></div>
          {children.map((child, index) => (
            <ConceptCardEditor
              key={child.id}
              concept={child}
              title={`Sub-card ${index + 1}`}
              depth={1}
              makeId={makeId}
              onChange={(next) => patchChild(index, next)}
              onRemove={() => removeChild(index)}
              onMove={(direction) => patch({ children: moveItem(children, index, index + direction) })}
            />
          ))}
          <button className="ce-add" onClick={addChild}>+ Add a sub-card</button>
        </div>
      )}
    </div>
  )
}

function ConceptEditor({ concepts, onChange }: { concepts: Concept[]; onChange: (concepts: Concept[]) => void }) {
  const makeId = useCallback((label?: string) => nextConceptId(concepts, label), [concepts])

  return (
    <>
      <p className="ce-note">
        Each card is a concise idea in this variant. A card can have direct sub-cards; click a card in the map to read its detail.
      </p>
      {concepts.map((concept, index) => (
        <ConceptCardEditor
          key={concept.id}
          concept={concept}
          title={`Card ${index + 1}`}
          depth={0}
          makeId={makeId}
          onChange={(next) => onChange(concepts.map((current, currentIndex) => (currentIndex === index ? next : current)))}
          onRemove={() => onChange(concepts.filter((_, currentIndex) => currentIndex !== index))}
          onMove={(direction) => onChange(moveItem(concepts, index, index + direction))}
        />
      ))}
      <button className="ce-add" onClick={() => onChange([...concepts, newConcept(makeId())])}>+ Add a concept card</button>
    </>
  )
}

/**
 * Cost rows are formulas, so they get a plain input and live validation rather
 * than a rich editor. The preview column shows what each row renders to right
 * now, which is the only way to tell a working template from a plausible one.
 */
function CostEditor({ rows, onChange }: { rows: CostRow[]; onChange: (r: CostRow[]) => void }) {
  const patch = (i: number, p: Partial<CostRow>) =>
    onChange(rows.map((r, j) => (i === j ? { ...r, ...p } : r)))

  return (
    <>
      <p className="ce-note">
        Values are templates over the loaded model's dimensions. Text outside braces is literal;
        each <code>{'{…}'}</code> is arithmetic. Available names:{' '}
        <code>nLayer dModel nHead nKvHead dHead dFF vocab ctx ropeTheta nExperts nActive nShared window kvBytes</code>{' '}
        and functions <code>bytes si num fixed sci round min max</code>.
      </p>
      {rows.map((r, i) => {
        const errs = [...checkTemplate(r.value, REFERENCE_DIMS), ...checkTemplate(r.note ?? '', REFERENCE_DIMS)]
        return (
          <div className={`ce-card${errs.length ? ' has-err' : ''}`} key={i}>
            <div className="ce-card-head">
              <span>Row {i + 1}</span>
              <button className="ce-del" onClick={() => onChange(rows.filter((_, j) => j !== i))}>remove</button>
            </div>
            <TextField label="Label" value={r.label} onChange={(label) => patch(i, { label })} />
            <TextField label="Value template" mono value={r.value} onChange={(value) => patch(i, { value })} />
            <TextField label="Note template" mono value={r.note ?? ''} onChange={(note) => patch(i, { note: note || undefined })} />
            <label className="ce-check">
              <input type="checkbox" checked={Boolean(r.key)} onChange={(e) => patch(i, { key: e.target.checked || undefined })} />
              Emphasise this row — the number that matters most
            </label>
            {/* Live preview: the only way to tell a working template from a
                plausible-looking one is to see what it actually renders. */}
            <div className="ce-preview">
              <span className="ce-preview-tag">at {REFERENCE_DIMS.name}</span>
              <b>{evalTemplate(r.value, REFERENCE_DIMS)}</b>
              {r.note && <i>{evalTemplate(r.note, REFERENCE_DIMS)}</i>}
            </div>
            {errs.length > 0 && <div className="ce-err">{errs.join(' · ')}</div>}
          </div>
        )
      })}
      <button className="ce-add" onClick={() => onChange([...rows, { label: '', value: '' }])}>+ Add a row</button>
    </>
  )
}

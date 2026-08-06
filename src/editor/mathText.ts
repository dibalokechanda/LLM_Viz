import type { MathBlock } from '../data/types'

/**
 * One equation, as one text block.
 *
 * A MathBlock has five parts — caption, TeX, symbol legend, worked numbers and
 * a note — and five separate widgets made writing one feel like filling in a
 * form. Written out it is just a short document, so that is what this edits.
 * The shape is still parsed back into a MathBlock, so the rendered equation,
 * its legend and its worked substitutions are unchanged.
 */
export function mathToText(m: MathBlock): string {
  const out: string[] = []
  if (m.title) out.push(m.title)
  out.push('$$', m.tex.trim(), '$$')
  if (m.where?.length) {
    out.push('', 'where')
    for (const w of m.where) out.push(`- ${w.sym} — ${w.means}`)
  }
  if (m.worked?.length) {
    out.push('', 'worked')
    for (const w of m.worked) out.push(`- ${w.tex}${w.caption ? ` — ${w.caption}` : ''}`)
  }
  if (m.note) out.push('', 'note', m.note)
  return out.join('\n')
}

export function textToMath(text: string): MathBlock {
  const lines = text.split('\n')

  // Anything before the opening $$ is the caption.
  const open = lines.findIndex((l) => l.trim() === '$$')
  const close = open >= 0 ? lines.findIndex((l, i) => i > open && l.trim() === '$$') : -1
  const title = open > 0 ? lines.slice(0, open).join(' ').trim() || undefined : undefined
  const tex = open >= 0 && close > open ? lines.slice(open + 1, close).join('\n').trim() : text.trim()

  // Keyword sections after the equation, each running to the next keyword.
  const rest = close > 0 ? lines.slice(close + 1) : []
  let mode: 'where' | 'worked' | 'note' | null = null
  const where: NonNullable<MathBlock['where']> = []
  const worked: NonNullable<MathBlock['worked']> = []
  const note: string[] = []

  for (const raw of rest) {
    const line = raw.trim()
    if (line === 'where' || line === 'worked' || line === 'note') {
      mode = line
      continue
    }
    if (!line) continue
    const item = line.replace(/^[-*]\s*/, '')
    // "symbol — meaning". Em dash is the separator; a hyphen would collide with
    // minus signs, which appear in almost every one of these.
    const [head, ...tail] = item.split(' — ')
    if (mode === 'where') where.push({ sym: head.trim(), means: tail.join(' — ').trim() })
    else if (mode === 'worked') worked.push({ tex: head.trim(), caption: tail.join(' — ').trim() || undefined })
    else if (mode === 'note') note.push(line)
  }

  /*
   * Assembled in this exact order, not field-by-field onto a starter object.
   * The content file is YAML and js-yaml writes keys in insertion order, so
   * building `tex` first would silently reorder every equation in the file on
   * the first save — a diff touching all 72 of them and changing nothing.
   */
  const block: MathBlock = { tex }
  const ordered: MathBlock = {
    ...(title ? { title } : {}),
    tex: block.tex,
    ...(where.length ? { where } : {}),
    ...(worked.length ? { worked } : {}),
    ...(note.length ? { note: note.join(' ') } : {}),
  }
  return ordered
}

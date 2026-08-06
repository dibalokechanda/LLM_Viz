import { marked } from 'marked'
import TurndownService from 'turndown'
import { gfm } from 'turndown-plugin-gfm'

/**
 * HTML ↔ Markdown for the rich-text fields.
 *
 * TipTap edits HTML; the content files store Markdown and have to stay
 * readable and hand-editable, so the editor converts on the way in and back on
 * the way out. Built on `marked` and `turndown` rather than hand-rolled regex,
 * because the editor now offers the full formatting range — headings, tables,
 * images, code blocks, colour, alignment — and a partial converter would
 * silently drop whatever it did not recognise.
 *
 * Two things need care:
 *
 * **Heading levels are shifted.** The content file uses `##` to mark a variant
 * section and `###` for its `fixes`, so a heading typed in prose must never
 * land on those. The editor's H1/H2/H3 are stored as `####`/`#####`/`######`,
 * which the section splitter ignores by construction.
 *
 * **Marks Markdown cannot express survive as inline HTML.** Underline,
 * highlight, colour and alignment have no Markdown syntax, so they round-trip
 * as `<u>`, `<mark>` and `<span style>` — which the app renders through
 * `rehype-raw`. Turndown is told to keep those tags rather than strip them.
 */

/* ────────────────────────────── Markdown → HTML ────────────────────────────── */

marked.setOptions({ gfm: true, breaks: false })

export function mdToHtml(md: string): string {
  if (!md.trim()) return ''
  // Lift stored heading levels back into the editor's range before parsing:
  // `######` → h3, `#####` → h2, `####` → h1.
  const lifted = md
    .replace(/^######[ \t]+/gm, '### ')
    .replace(/^#####[ \t]+/gm, '## ')
    .replace(/^####[ \t]+/gm, '# ')
  return marked.parse(lifted, { async: false }) as string
}

/* ────────────────────────────── HTML → Markdown ────────────────────────────── */

const turndown = new TurndownService({
  headingStyle: 'atx',
  codeBlockStyle: 'fenced',
  bulletListMarker: '-',
  emDelimiter: '*',
  strongDelimiter: '**',
  // `---` would be ambiguous with the front-matter fence at the top of the
  // file, so use the other legal rule syntax.
  hr: '***',
})

turndown.use(gfm) // tables, strikethrough, task lists

/**
 * Turndown escapes every Markdown-significant character in text, which is right
 * for arbitrary web pages and wrong for this content. Left alone it rewrites
 * `d_model` as `d\_model` and `\sqrt{d_h}` as `\\sqrt{d\_h}` — so the first
 * save would mangle every equation in the prose and produce a diff touching
 * almost every line.
 *
 * Escape only what would change the *structure* of the document: a line that
 * would otherwise be read as a list item, heading or quote. Inline `_` and `*`
 * are left as typed, which is the right trade for technical prose written by
 * someone who knows Markdown.
 */
turndown.escape = (text: string) =>
  text
    /*
     * Anchored to the very start of the text node, with no /g and no /m.
     *
     * Turndown calls this per text node, not per line, so a multiline or
     * global anchor matches inside a paragraph: `\`un\` + \`believ\`` hands
     * this function the fragment " + " and a line-start rule escapes the plus.
     * A node that begins with the marker and no leading whitespace is the only
     * case that can actually be re-read as a block on the way back in.
     */
    .replace(/^([-*+>])(\s)/, '\\$1$2')
    .replace(/^(\d+)\.(\s)/, '$1\\.$2')
    .replace(/^(#{1,6})(\s)/, '\\$1$2')

/** GFM strikethrough is `~~text~~`; the plugin emits a single tilde. */
turndown.addRule('strikethrough', {
  filter: ['del', 's'],
  replacement: (content) => `~~${content}~~`,
})

/**
 * Keep the tags that carry marks Markdown has no syntax for. Without this
 * turndown unwraps them and the formatting is lost on the first save.
 */
turndown.keep(['u', 'mark', 'sub', 'sup'])

/** Push the editor's headings down into the range reserved for prose. */
turndown.addRule('shiftedHeadings', {
  filter: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
  replacement: (content, node) => {
    const level = Number((node as HTMLElement).tagName[1])
    // h1→####, h2→#####, h3→###### and anything deeper pins at ######.
    const hashes = '#'.repeat(Math.min(6, level + 3))
    return `\n\n${hashes} ${content.trim()}\n\n`
  },
})

/**
 * A `<span>` carrying inline style (colour, from the palette) is meaningful;
 * a bare one is TipTap scaffolding and should be unwrapped.
 */
turndown.addRule('styledSpan', {
  filter: (node) =>
    node.nodeName === 'SPAN' && Boolean((node as HTMLElement).getAttribute('style')),
  replacement: (content, node) =>
    `<span style="${(node as HTMLElement).getAttribute('style')}">${content}</span>`,
})

/** Paragraphs with text-align keep it as inline HTML; plain ones stay Markdown. */
turndown.addRule('alignedParagraph', {
  filter: (node) => {
    if (node.nodeName !== 'P') return false
    const style = (node as HTMLElement).getAttribute('style') ?? ''
    return /text-align:\s*(center|right|justify)/.test(style)
  },
  replacement: (content, node) =>
    `\n\n<p style="${(node as HTMLElement).getAttribute('style')}">${content}</p>\n\n`,
})

export function htmlToMd(html: string): string {
  if (!html || html === '<p></p>') return ''
  return (
    turndown
      .turndown(html)
      // Turndown pads list markers out to a fixed width (`-   item`). Harmless
      // to a renderer, but it means every save rewrites every list it touches,
      // burying the real edit in the diff. Normalise to one space.
      .replace(/^(\s*)([-*+])[ \t]{2,}/gm, '$1$2 ')
      .replace(/^(\s*)(\d+\.)[ \t]{2,}/gm, '$1$2 ')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  )
}

import { useState, type ReactNode } from 'react'
import type { CodeBlock } from '../data/types'

/*
 * Lightweight, dependency-free highlighting. A full highlighter is a large
 * dependency for a handful of short snippets, so this tokenises with one regex
 * and emits React spans, which keeps it XSS-safe (no dangerouslySetInnerHTML)
 * and in step with the theme. The snippets are authored to stay on single
 * lines per string, so no multi-line string state is needed.
 */
const PY_KEYWORDS =
  'from|import|as|def|class|return|if|elif|else|for|while|in|not|and|or|is|with|try|except|finally|raise|lambda|yield|await|async|pass|None|True|False|global|assert'

// Order matters: comments and strings win before keywords so a keyword inside
// a string is not recoloured.
const PY_RE = new RegExp(
  [
    '(#.*)', // 1 comment
    `("""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*')`, // 2 string
    `\\b(${PY_KEYWORDS})\\b`, // 3 keyword
    '(@[A-Za-z_]\\w*)', // 4 decorator
    '\\b(\\d+\\.?\\d*)\\b', // 5 number
    '\\b([A-Za-z_]\\w*)(?=\\()', // 6 call
  ].join('|'),
  'g',
)

const BASH_RE = /(#.*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g

export function highlight(code: string, language: CodeBlock['language']): ReactNode[] {
  const re = language === 'bash' ? BASH_RE : PY_RE
  const out: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  let key = 0
  re.lastIndex = 0
  while ((m = re.exec(code))) {
    if (m.index > last) out.push(code.slice(last, m.index))
    const cls =
      m[1] !== undefined
        ? 'tok-comment'
        : m[2] !== undefined
          ? 'tok-string'
          : m[3] !== undefined
            ? 'tok-keyword'
            : m[4] !== undefined
              ? 'tok-decorator'
              : m[5] !== undefined
                ? 'tok-number'
                : 'tok-call'
    out.push(
      <span className={cls} key={key++}>
        {m[0]}
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < code.length) out.push(code.slice(last))
  return out
}

export function CodeBlockView({ block }: { block: CodeBlock }) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(block.code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1400)
    } catch {
      /* clipboard blocked; nothing to do */
    }
  }

  return (
    <figure className="code-block">
      <div className="code-head">
        <span className="code-title">{block.title ?? 'Code'}</span>
        <span className="code-lang">{block.language}</span>
        {block.source && (
          <a className="code-source" href={block.source.url} target="_blank" rel="noopener noreferrer">
            {block.source.label} ↗
          </a>
        )}
        <button className={`code-copy${block.source ? '' : ' is-solo'}`} onClick={copy} title="Copy code">
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre className="code-body">
        <code>{highlight(block.code, block.language)}</code>
      </pre>
      {block.note && <figcaption className="code-note">{block.note}</figcaption>}
    </figure>
  )
}

export default CodeBlockView

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'

/**
 * Renders the prose written in the content files.
 *
 * The editor offers the full formatting range, so the reader has to understand
 * the same range or a save would silently change how a paragraph looks:
 *
 *   · `remark-gfm`    tables and strikethrough
 *   · `remark-math`   `$…$` and `$$…$$`, rendered by KaTeX
 *   · `rehype-raw`    the inline HTML that carries underline, highlight,
 *                     colour and alignment — marks Markdown cannot express
 *
 * `rehype-raw` means the Markdown may contain arbitrary HTML. That is safe
 * here and only here: this content ships with the app and is written by whoever
 * builds it. It is never user-submitted and never fetched at runtime.
 */
export default function Prose({ children, className }: { children: string; className?: string }) {
  return (
    <div className={`markdown-content${className ? ` ${className}` : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          // Images are stored relative to content/, which Vite serves from the
          // public root in dev and copies to the bundle in build.
          img: ({ src, alt }) => (
            <img className="prose-img" src={typeof src === 'string' ? resolveAsset(src) : undefined} alt={alt ?? ''} />
          ),
          // Wide tables must scroll inside their own box rather than pushing
          // the whole detail panel sideways.
          table: ({ children: kids }) => (
            <div className="prose-table-wrap">
              <table>{kids}</table>
            </div>
          ),
          a: ({ href, children: kids }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {kids}
            </a>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}

/** `assets/foo.png` in a content file → the URL the built site serves it from. */
function resolveAsset(src: string): string {
  if (/^(https?:|data:|\/)/.test(src)) return src
  return `${import.meta.env.BASE_URL}content-assets/${src.replace(/^assets\//, '')}`
}

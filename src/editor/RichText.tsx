import { useEffect, useRef, useState } from 'react'
import { useEditor, EditorContent, type Editor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Placeholder from '@tiptap/extension-placeholder'
import Link from '@tiptap/extension-link'
import Underline from '@tiptap/extension-underline'
import Highlight from '@tiptap/extension-highlight'
import TextAlign from '@tiptap/extension-text-align'
import Image from '@tiptap/extension-image'
import { TextStyle, Color } from '@tiptap/extension-text-style'
import { Table, TableRow, TableHeader, TableCell } from '@tiptap/extension-table'
import CodeBlockLowlight from '@tiptap/extension-code-block-lowlight'
import { common, createLowlight } from 'lowlight'
import {
  Bold, Italic, Underline as UnderlineIcon, Strikethrough, Code, Highlighter,
  Heading1, Heading2, Heading3, List, ListOrdered, Quote, Minus,
  AlignLeft, AlignCenter, AlignRight, Link as LinkIcon, Image as ImageIcon,
  Table as TableIcon, Terminal, Palette, Rows3, Columns3, Trash2, X,
} from 'lucide-react'
import type { EditorView } from '@tiptap/pm/view'
import { htmlToMd, mdToHtml } from './markdown'

const lowlight = createLowlight(common)

/**
 * The prose editor.
 *
 * Speaks Markdown at its edges and HTML in the middle: content arrives as the
 * Markdown stored in the content file, is edited as rich text, and converts
 * back on every change, so the file stays readable and hand-editable — the
 * editor is one way to write this content, not the only way.
 *
 * The full formatting range is available. Two of them come with a caveat worth
 * knowing, documented in `markdown.ts`: headings are stored two levels down so
 * they cannot collide with the `##` section markers that pair prose with a
 * variant, and marks Markdown cannot express (underline, highlight, colour,
 * alignment) round-trip as inline HTML.
 */

/** Colour palette. The app's own validated hues, so they hold in both themes. */
const COLORS = [
  { label: 'Default', value: '' },
  { label: 'Muted', value: '#7d7a73' },
  { label: 'Blue', value: '#2a78d6' },
  { label: 'Green', value: '#1baf7a' },
  { label: 'Orange', value: '#eb6834' },
  { label: 'Violet', value: '#4a3aa7' },
  { label: 'Amber', value: '#eda100' },
  { label: 'Red', value: '#d03b3b' },
]

/**
 * Store a pasted or dropped image in `content/assets/` and insert a relative
 * reference. Images belong in the repository next to the prose that uses them,
 * not as base64 inside the Markdown, which would make the file unreadable and
 * unreviewable.
 */
async function uploadImage(file: File): Promise<string | null> {
  const body = new FormData()
  body.append('file', file)
  try {
    const res = await fetch('/api/asset', { method: 'POST', body })
    if (!res.ok) return null
    const j = (await res.json()) as { path?: string }
    return j.path ?? null
  } catch {
    return null
  }
}

function insertImageFiles(view: EditorView, files: FileList | null | undefined): boolean {
  const imgs = files ? Array.from(files).filter((f) => f.type.startsWith('image/')) : []
  if (imgs.length === 0) return false
  for (const file of imgs) {
    void uploadImage(file).then((path) => {
      if (!path) return
      const node = view.state.schema.nodes.image?.create({ src: path, alt: file.name })
      if (node) view.dispatch(view.state.tr.replaceSelectionWith(node).scrollIntoView())
    })
  }
  return true
}

function Btn({
  onRun,
  active,
  title,
  children,
}: {
  onRun: () => void
  active?: boolean
  title: string
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      className={`rt-btn${active ? ' on' : ''}`}
      title={title}
      aria-label={title}
      onMouseDown={(e) => {
        e.preventDefault() // keep the selection while clicking the toolbar
        onRun()
      }}
    >
      {children}
    </button>
  )
}

function Toolbar({ editor }: { editor: Editor }) {
  const [showColors, setShowColors] = useState(false)
  const [showTable, setShowTable] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const popRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (popRef.current && !popRef.current.contains(e.target as Node)) {
        setShowColors(false)
        setShowTable(false)
      }
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  const c = () => editor.chain().focus()

  return (
    <div className="rt-bar" ref={popRef}>
      <Btn title="Heading 1" active={editor.isActive('heading', { level: 1 })} onRun={() => c().toggleHeading({ level: 1 }).run()}><Heading1 size={14} /></Btn>
      <Btn title="Heading 2" active={editor.isActive('heading', { level: 2 })} onRun={() => c().toggleHeading({ level: 2 }).run()}><Heading2 size={14} /></Btn>
      <Btn title="Heading 3" active={editor.isActive('heading', { level: 3 })} onRun={() => c().toggleHeading({ level: 3 }).run()}><Heading3 size={14} /></Btn>

      <span className="rt-sep" />

      <Btn title="Bold" active={editor.isActive('bold')} onRun={() => c().toggleBold().run()}><Bold size={14} /></Btn>
      <Btn title="Italic" active={editor.isActive('italic')} onRun={() => c().toggleItalic().run()}><Italic size={14} /></Btn>
      <Btn title="Underline" active={editor.isActive('underline')} onRun={() => c().toggleUnderline().run()}><UnderlineIcon size={14} /></Btn>
      <Btn title="Strikethrough" active={editor.isActive('strike')} onRun={() => c().toggleStrike().run()}><Strikethrough size={14} /></Btn>
      <Btn title="Inline code" active={editor.isActive('code')} onRun={() => c().toggleCode().run()}><Code size={14} /></Btn>
      <Btn title="Highlight" active={editor.isActive('highlight')} onRun={() => c().toggleHighlight().run()}><Highlighter size={14} /></Btn>

      <div className="rt-pop-host">
        <Btn title="Text colour" active={showColors} onRun={() => { setShowColors((v) => !v); setShowTable(false) }}><Palette size={14} /></Btn>
        {showColors && (
          <div className="rt-pop">
            {COLORS.map((col) => (
              <button
                key={col.label}
                type="button"
                className="rt-color"
                onMouseDown={(e) => {
                  e.preventDefault()
                  if (col.value) c().setColor(col.value).run()
                  else c().unsetColor().run()
                  setShowColors(false)
                }}
              >
                <i style={{ background: col.value || 'var(--text)' }} />
                {col.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <span className="rt-sep" />

      <Btn title="Bullet list" active={editor.isActive('bulletList')} onRun={() => c().toggleBulletList().run()}><List size={14} /></Btn>
      <Btn title="Numbered list" active={editor.isActive('orderedList')} onRun={() => c().toggleOrderedList().run()}><ListOrdered size={14} /></Btn>
      <Btn title="Quote" active={editor.isActive('blockquote')} onRun={() => c().toggleBlockquote().run()}><Quote size={14} /></Btn>
      <Btn title="Code block" active={editor.isActive('codeBlock')} onRun={() => c().toggleCodeBlock().run()}><Terminal size={14} /></Btn>
      <Btn title="Divider" onRun={() => c().setHorizontalRule().run()}><Minus size={14} /></Btn>

      <span className="rt-sep" />

      <Btn title="Align left" active={editor.isActive({ textAlign: 'left' })} onRun={() => c().setTextAlign('left').run()}><AlignLeft size={14} /></Btn>
      <Btn title="Align centre" active={editor.isActive({ textAlign: 'center' })} onRun={() => c().setTextAlign('center').run()}><AlignCenter size={14} /></Btn>
      <Btn title="Align right" active={editor.isActive({ textAlign: 'right' })} onRun={() => c().setTextAlign('right').run()}><AlignRight size={14} /></Btn>

      <span className="rt-sep" />

      <Btn
        title="Link"
        active={editor.isActive('link')}
        onRun={() => {
          const prev = editor.getAttributes('link').href as string | undefined
          const href = window.prompt('Link URL', prev ?? 'https://')
          if (href === null) return
          if (!href) c().unsetLink().run()
          else c().extendMarkRange('link').setLink({ href }).run()
        }}
      >
        <LinkIcon size={14} />
      </Btn>

      <Btn title="Insert image" onRun={() => fileRef.current?.click()}><ImageIcon size={14} /></Btn>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          insertImageFiles(editor.view, e.target.files)
          e.target.value = ''
        }}
      />

      <div className="rt-pop-host">
        <Btn title="Table" active={showTable} onRun={() => { setShowTable((v) => !v); setShowColors(false) }}><TableIcon size={14} /></Btn>
        {showTable && (
          <div className="rt-pop">
            <button type="button" className="rt-color" onMouseDown={(e) => { e.preventDefault(); c().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run(); setShowTable(false) }}>
              <TableIcon size={13} /> Insert 3 × 3
            </button>
            <button type="button" className="rt-color" onMouseDown={(e) => { e.preventDefault(); c().addRowAfter().run() }}><Rows3 size={13} /> Add row</button>
            <button type="button" className="rt-color" onMouseDown={(e) => { e.preventDefault(); c().addColumnAfter().run() }}><Columns3 size={13} /> Add column</button>
            <button type="button" className="rt-color" onMouseDown={(e) => { e.preventDefault(); c().deleteRow().run() }}><X size={13} /> Delete row</button>
            <button type="button" className="rt-color" onMouseDown={(e) => { e.preventDefault(); c().deleteColumn().run() }}><X size={13} /> Delete column</button>
            <button type="button" className="rt-color" onMouseDown={(e) => { e.preventDefault(); c().deleteTable().run(); setShowTable(false) }}><Trash2 size={13} /> Delete table</button>
          </div>
        )}
      </div>

      <span className="rt-hint">markdown · saved to the content file</span>
    </div>
  )
}

export default function RichText({
  value,
  onChange,
  placeholder = 'Write…',
  minHeight = 120,
}: {
  value: string
  onChange: (markdown: string) => void
  placeholder?: string
  minHeight?: number
}) {
  // Guard against the round trip fighting the caret: when our own onChange
  // bubbles back as a new `value`, the content is already correct.
  const lastEmitted = useRef(value)

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ heading: { levels: [1, 2, 3] }, codeBlock: false }),
      Placeholder.configure({ placeholder }),
      Underline,
      Highlight,
      TextStyle,
      Color,
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      Link.configure({ openOnClick: false }),
      Image.configure({ inline: false }),
      Table.configure({ resizable: true }),
      TableRow,
      TableHeader,
      TableCell,
      CodeBlockLowlight.configure({ lowlight }),
    ],
    content: mdToHtml(value),
    editorProps: {
      handlePaste: (view, event) => insertImageFiles(view, event.clipboardData?.files),
      handleDrop: (view, event) => insertImageFiles(view, (event as DragEvent).dataTransfer?.files),
    },
    onUpdate: ({ editor: ed }) => {
      const md = htmlToMd(ed.getHTML())
      lastEmitted.current = md
      onChange(md)
    },
  })

  // Re-seed only when the value changed elsewhere — switching variant, say.
  useEffect(() => {
    if (!editor) return
    if (value === lastEmitted.current) return
    lastEmitted.current = value
    editor.commands.setContent(mdToHtml(value), { emitUpdate: false })
  }, [value, editor])

  if (!editor) return null

  return (
    <div className="rt">
      <Toolbar editor={editor} />
      <EditorContent className="rt-body" style={{ minHeight }} editor={editor} />
    </div>
  )
}

import { useState, type ReactNode } from 'react'

/**
 * A numbered, collapsible section.
 *
 * The panel carries a lot of material, so most sections start closed. Their
 * headers then double as a table of contents — you can see what exists and how
 * much of it there is without scrolling through all of it.
 */
export default function Section({
  index,
  title,
  count,
  defaultOpen = false,
  children,
}: {
  index: number
  title: string
  /** Shown on the right of the header, e.g. how many variants or equations. */
  count?: number
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <section className={`sec ${open ? 'is-open' : ''}`}>
      <button className="sec-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="sec-num">{String(index).padStart(2, '0')}</span>
        <span className="sec-title">{title}</span>
        {count !== undefined && <span className="sec-count">{count}</span>}
        <svg className="sec-chev" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
          <path
            d="M9 6l6 6-6 6"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
      {open && <div className="sec-body">{children}</div>}
    </section>
  )
}

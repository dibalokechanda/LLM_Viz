/**
 * Stroke icon set. All drawn on a 24×24 grid with a 1.6 stroke and round caps,
 * inheriting `currentColor` so they pick up whatever the surrounding text does.
 */

export type IconName =
  // offline
  | 'documents'
  | 'load'
  | 'tag'
  | 'filter'
  | 'chunks'
  | 'vector'
  | 'graph'
  | 'compress'
  | 'database'
  // online
  | 'chat'
  | 'braces'
  | 'branch'
  | 'pencil'
  | 'expand'
  | 'fanout'
  | 'split'
  | 'ghostdoc'
  | 'search'
  | 'merge'
  | 'layers'
  | 'sort'
  | 'chart'
  | 'brackets'
  | 'sparkle'
  | 'checkdoc'
  | 'gauge'
  | 'check'
  // platform / control plane
  | 'sync'
  | 'shield'
  | 'box'
  | 'target'
  | 'rollback'
  | 'pulse'
  | 'crop'
  | 'bolt'
  // concept kinds
  | 'bulb'
  | 'fx'
  | 'steps'
  | 'warning'
  | 'scale'

const P: Record<IconName, React.ReactNode> = {
  documents: (
    <>
      <path d="M8 2h6l4 4v12a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z" />
      <path d="M14 2v4h4" />
      <path d="M4 6v14a2 2 0 0 0 2 2h9" />
    </>
  ),
  load: (
    <>
      <path d="M12 3v10" />
      <path d="M8.5 9.5 12 13l3.5-3.5" />
      <path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
    </>
  ),
  tag: (
    <>
      <path d="M20.5 13.5 13 21a2 2 0 0 1-2.8 0l-6.2-6.2a2 2 0 0 1 0-2.8L11.5 4.5a2 2 0 0 1 1.4-.6h5.6a2 2 0 0 1 2 2v5.6a2 2 0 0 1-.6 1.4z" />
      <circle cx="16.5" cy="7.5" r="1.3" />
    </>
  ),
  filter: <path d="M3 4h18l-7 8.5V20l-4 1.5v-9z" />,
  chunks: (
    <>
      <rect x="3" y="4" width="18" height="4.2" rx="1.2" />
      <rect x="3" y="10" width="18" height="4.2" rx="1.2" />
      <rect x="3" y="16" width="18" height="4.2" rx="1.2" />
    </>
  ),
  vector: (
    <>
      <path d="M4 20 20 4" />
      <path d="M20 10V4h-6" />
      <circle cx="6" cy="18" r="1.6" />
    </>
  ),
  graph: (
    <>
      <circle cx="6" cy="6" r="2.6" />
      <circle cx="18" cy="7" r="2.6" />
      <circle cx="12" cy="18" r="2.6" />
      <path d="M8.4 7.2 15.6 6.4M7.2 8.4l3.4 7.2M16.6 9.4 13.4 15.8" />
    </>
  ),
  compress: (
    <>
      <path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" />
      <path d="M9 9 4.5 4.5M15 9l4.5-4.5M9 15l-4.5 4.5M15 15l4.5 4.5" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5.5" rx="8" ry="3" />
      <path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13" />
      <path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
    </>
  ),
  chat: <path d="M21 14a2 2 0 0 1-2 2H8l-4 4V5a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z" />,
  braces: (
    <>
      <path d="M9 3H8a2 2 0 0 0-2 2v3.5a2 2 0 0 1-2 2 2 2 0 0 1 2 2V19a2 2 0 0 0 2 2h1" />
      <path d="M15 3h1a2 2 0 0 1 2 2v3.5a2 2 0 0 0 2 2 2 2 0 0 0-2 2V19a2 2 0 0 1-2 2h-1" />
    </>
  ),
  branch: (
    <>
      <circle cx="6" cy="5" r="2.4" />
      <circle cx="18" cy="5" r="2.4" />
      <circle cx="12" cy="19" r="2.4" />
      <path d="M6 7.4v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-3" />
      <path d="M12 13.4v3.2" />
    </>
  ),
  pencil: (
    <>
      <path d="M16.5 3.5a2.6 2.6 0 0 1 3.7 3.7L7.6 19.8 2.8 21.2l1.4-4.8z" />
      <path d="M15 5.4 18.6 9" />
    </>
  ),
  expand: (
    <>
      <path d="M12 5v14M5 12h14" />
      <circle cx="12" cy="12" r="9" />
    </>
  ),
  fanout: (
    <>
      <circle cx="4.5" cy="12" r="2" />
      <circle cx="19.5" cy="5" r="2" />
      <circle cx="19.5" cy="12" r="2" />
      <circle cx="19.5" cy="19" r="2" />
      <path d="M6.5 12h11M6.4 11.2 17.6 5.6M6.4 12.8l11.2 5.6" />
    </>
  ),
  split: (
    <>
      <rect x="3" y="9" width="6" height="6" rx="1.4" />
      <rect x="15" y="3" width="6" height="6" rx="1.4" />
      <rect x="15" y="15" width="6" height="6" rx="1.4" />
      <path d="M9 12h3a1 1 0 0 0 1-1V7a1 1 0 0 1 1-1h1M9 12h3a1 1 0 0 1 1 1v4a1 1 0 0 0 1 1h1" />
    </>
  ),
  ghostdoc: (
    <>
      <path
        d="M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"
        strokeDasharray="3 2.5"
      />
      <path d="M12 10.5l.9 2.6 2.6.9-2.6.9-.9 2.6-.9-2.6-2.6-.9 2.6-.9z" />
    </>
  ),
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.8" />
      <path d="M15.4 15.4 21 21" />
    </>
  ),
  merge: (
    <>
      <path d="M3 5h4l5 7M3 19h4l5-7" />
      <path d="M12 12h8" />
      <path d="M17.5 8.5 21 12l-3.5 3.5" />
    </>
  ),
  layers: (
    <>
      <rect x="3" y="3" width="12" height="12" rx="2" />
      <path d="M8 19a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2v-9a2 2 0 0 0-2-2" />
    </>
  ),
  sort: (
    <>
      <path d="M3 6h13M3 12h9M3 18h5" />
      <path d="M19 5v14M16.5 16.5 19 19l2.5-2.5" />
    </>
  ),
  chart: (
    <>
      <path d="M3 21h18" />
      <rect x="4" y="11" width="4" height="8" rx="1" />
      <rect x="10" y="5" width="4" height="14" rx="1" />
      <rect x="16" y="14" width="4" height="5" rx="1" />
    </>
  ),
  brackets: (
    <>
      <path d="M9 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3" />
      <path d="M15 3h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3" />
      <path d="M9.5 12h5" />
    </>
  ),
  sparkle: (
    <>
      <path d="M12 3l1.9 5.6 5.6 1.9-5.6 1.9L12 18l-1.9-5.6L4.5 10.5l5.6-1.9z" />
      <path d="M18.5 16.5l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" />
    </>
  ),
  checkdoc: (
    <>
      <path d="M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" />
      <path d="M14 3v4h4" />
      <path d="M9.5 14l2 2 4-4.5" />
    </>
  ),
  gauge: (
    <>
      <path d="M4 18a9 9 0 1 1 16 0" />
      <path d="M12 18l4.5-5" />
      <circle cx="12" cy="18" r="1.4" />
    </>
  ),
  check: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12.2l2.8 2.8L16 9" />
    </>
  ),
  sync: (
    <>
      <path d="M20.5 11.5A8.5 8.5 0 0 0 5.6 6.4M3.5 12.5a8.5 8.5 0 0 0 14.9 5.1" />
      <path d="M20.5 5.5v6h-6M3.5 18.5v-6h6" />
    </>
  ),
  shield: (
    <>
      <path d="M12 2.5 20 6v6.2c0 4.7-3.3 8.3-8 9.3-4.7-1-8-4.6-8-9.3V6z" />
      <path d="M8.8 12.2 11 14.4l4.3-4.5" />
    </>
  ),
  box: (
    <>
      <path d="M12 2.5 20.5 7v10L12 21.5 3.5 17V7z" />
      <path d="M3.5 7 12 11.6 20.5 7M12 11.6v9.9" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="12" cy="12" r="4.4" />
      <circle cx="12" cy="12" r="1.1" fill="currentColor" stroke="none" />
    </>
  ),
  rollback: (
    <>
      <path d="M3.5 10h10.5a5.5 5.5 0 1 1 0 11H9" />
      <path d="M7 6 3.2 10 7 14" />
    </>
  ),
  crop: (
    <>
      <path d="M6 2v14a2 2 0 0 0 2 2h14" />
      <path d="M2 6h14a2 2 0 0 1 2 2v14" />
    </>
  ),
  bolt: <path d="M13.5 2 5 13.5h5.5l-1 8.5L19 10.5h-5.5z" />,
  pulse: (
    <>
      <path d="M2.5 12h4l2.5-6.5L14 18.5l2.5-6.5h5" />
      <circle cx="21.5" cy="12" r="0.9" fill="currentColor" stroke="none" />
    </>
  ),
  bulb: (
    <>
      <path d="M9 18h6M10 21h4" />
      <path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2h5c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z" />
    </>
  ),
  fx: (
    <>
      <path d="M4 20c2 0 3-1 3-4V8c0-3 1-4 3-4" />
      <path d="M5 12h5" />
      <path d="M13 10l7 8M20 10l-7 8" />
    </>
  ),
  steps: (
    <>
      <path d="M4 19h4v-4h4v-4h4V7h4" />
      <circle cx="4" cy="19" r="1.4" />
    </>
  ),
  warning: (
    <>
      <path d="M12 3.5 22 20H2z" />
      <path d="M12 10v4.5" />
      <circle cx="12" cy="17.4" r="0.9" />
    </>
  ),
  scale: (
    <>
      <path d="M3 8h18M8.5 4.5 4 8l4.5 3.5" />
      <path d="M21 16H3M15.5 12.5 20 16l-4.5 3.5" />
    </>
  ),
}

export default function Icon({
  name,
  size = 20,
  className,
}: {
  name: IconName
  size?: number
  className?: string
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      {P[name]}
    </svg>
  )
}

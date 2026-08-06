import { useEffect } from 'react'
import { useNodesInitialized, useUpdateNodeInternals } from '@xyflow/react'

/**
 * Force React Flow to capture handle bounds for every node after mount.
 *
 * React Flow computes an edge's endpoints from its source and target nodes'
 * `internals.handleBounds`, which are normally captured by a ResizeObserver
 * registered on each node. When that observer does not deliver — and some
 * embedded/preview browsers never fire it at all — the nodes still paint
 * perfectly while `handleBounds` stays undefined and **every edge is silently
 * dropped**. The symptom is a diagram of disconnected cards with nothing in the
 * console, because React Flow treats "cannot place this edge yet" as a
 * wait-and-retry rather than an error. That failure mode cost a long debugging
 * session; this component exists so it cannot happen again.
 *
 * `useUpdateNodeInternals` measures straight from the DOM and does not depend
 * on the observer, so one call after paint makes measurement deterministic.
 * Combined with the explicit `width`/`height` declared on every node in
 * StackView and LineageMap, nothing about the layout relies on measurement
 * succeeding on its own.
 *
 * Must be rendered as a child of `<ReactFlow>`, which is what puts it inside
 * the flow's store context.
 */
export default function EnsureMeasured({ ids }: { ids: string[] }) {
  const updateNodeInternals = useUpdateNodeInternals()
  const initialized = useNodesInitialized()

  useEffect(() => {
    if (initialized || ids.length === 0) return

    /*
     * Retry rather than fire once. A single post-paint call is enough for the
     * flow that mounts with the page, but not for one that mounts later — the
     * lineage map opens while the previous overlay is still tearing down, and a
     * lone call can land before its nodes are in the document. Retrying until
     * React Flow reports the nodes initialised makes it timing-independent.
     */
    const timers = [0, 120, 400, 900].map((delay) =>
      window.setTimeout(() => updateNodeInternals(ids), delay),
    )
    return () => timers.forEach(window.clearTimeout)
  }, [ids, initialized, updateNodeInternals])

  return null
}

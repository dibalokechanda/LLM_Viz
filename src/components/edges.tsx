import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from '@xyflow/react'

interface EdgeData extends Record<string, unknown> {
  travelled?: boolean
  parallel?: boolean
}

/** Standard vertical hop between consecutive stages. */
export function PipelineEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  label,
  markerEnd,
}: EdgeProps) {
  const d = (data ?? {}) as EdgeData
  const [path, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 10,
  })

  const cls = [d.travelled && 'edge-travelled', d.parallel && 'edge-parallel'].filter(Boolean).join(' ')

  return (
    <>
      <BaseEdge id={id} path={path} className={cls} markerEnd={markerEnd} />
      {label && (
        <EdgeLabelRenderer>
          {/* Sized for the canvas, which sits at ~0.75 zoom — so these run
              larger than equivalent panel text to land at the same optical size. */}
          <div className="edge-chip" style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}>
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

/**
 * The one edge that joins the offline and online halves: the index is written
 * by ingestion and read by retrieval.
 *
 * Routed by hand up and over the top of the canvas. A direct path would cut
 * straight through the query-processing column that sits between the two.
 */
export function JoinEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, label }: EdgeProps) {
  const r = 12
  const out = sourceX + 44
  const inn = targetX - 44
  // Clear the lane headers, which sit above the first row of nodes.
  const topY = Math.min(sourceY, targetY) - 190

  const path = [
    `M ${sourceX} ${sourceY}`,
    `L ${out - r} ${sourceY}`,
    `Q ${out} ${sourceY} ${out} ${sourceY - r}`,
    `L ${out} ${topY + r}`,
    `Q ${out} ${topY} ${out + r} ${topY}`,
    `L ${inn - r} ${topY}`,
    `Q ${inn} ${topY} ${inn} ${topY + r}`,
    `L ${inn} ${targetY - r}`,
    `Q ${inn} ${targetY} ${inn + r} ${targetY}`,
    `L ${targetX} ${targetY}`,
  ].join(' ')

  const labelX = (out + inn) / 2
  const labelY = topY

  return (
    <>
      <BaseEdge id={id} path={path} className="edge-join" markerEnd={markerEnd} />
      <EdgeLabelRenderer>
        <div
          className="edge-chip edge-chip-join"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
        >
          {label ?? 'Reads the index'}
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

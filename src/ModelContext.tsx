import { createContext, useContext } from 'react'
import type { BlockId, Dims, ModelSpec } from './data/types'

export interface ModelContextValue {
  /** The loaded model, or null when exploring the design space unanchored. */
  model: ModelSpec | null
  /**
   * The currently selected variant per block. Starts as the model's inferred
   * path, but the user can move off it — that is the point of the app, and the
   * UI marks any block where the two disagree.
   */
  path: Partial<Record<BlockId, string>>
  /** Dimensions used for every live cost readout. Falls back to a reference 8B. */
  dims: Dims
  /** Move a block to a different variant. */
  setVariant: (block: BlockId, variant: string) => void
  /** Snap a block back to whatever the loaded model actually uses. */
  resetBlock: (block: BlockId) => void
  /** Blocks whose selection differs from the loaded model's. */
  diverged: Set<BlockId>
  /** Blocks the config could not resolve. */
  unresolved: Set<BlockId>
}

/** Used when nothing is loaded, so cost readouts always have real numbers. */
export const REFERENCE_DIMS: Dims = {
  name: 'reference 8B',
  nLayer: 32,
  dModel: 4096,
  nHead: 32,
  nKvHead: 8,
  dHead: 128,
  dFF: 14336,
  vocab: 128256,
  ctx: 131072,
  ropeTheta: 500000,
  kvBytes: 2,
}

export const ModelContext = createContext<ModelContextValue | null>(null)

export function useModel() {
  const ctx = useContext(ModelContext)
  if (!ctx) throw new Error('useModel must be used inside <ModelContext.Provider>')
  return ctx
}

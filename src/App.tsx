import { useCallback, useMemo, useState } from 'react'
import '@xyflow/react/dist/style.css'

import type { Block, BlockId, ModelSpec, Variant } from './data/types'
import { blockById, blocks, defaultPath, variantOf } from './content'
import { ModelContext, REFERENCE_DIMS, type ModelContextValue } from './ModelContext'
import StackView from './components/StackView'
import LineageMap from './components/LineageMap'
import ConceptMap from './components/ConceptMap'
import VariantDetail from './components/VariantDetail'
import ModelBar from './components/ModelBar'
import ContentEditor from './editor/ContentEditor'
import Icon from './components/Icon'

/*
 * No ReactFlowProvider here on purpose. The stack and the lineage map are two
 * independent React Flow instances that are on screen at the same time, and a
 * shared provider means a shared store — the second instance to mount wipes the
 * first one's node measurements, and both then render their nodes with no edges
 * at all. Each <ReactFlow> creates its own store when no provider is above it,
 * and each carries a distinct `id` so their marker definitions cannot collide.
 * Nothing here calls useReactFlow, so there is nothing a provider would buy.
 */
export default function App() {
  const [model, setModel] = useState<ModelSpec | null>(null)
  const [overrides, setOverrides] = useState<Partial<Record<BlockId, string>>>({})
  const [selectedId, setSelectedId] = useState<BlockId | null>(null)
  const [lineageFor, setLineageFor] = useState<BlockId | null>(null)
  const [conceptMapFor, setConceptMapFor] = useState<{ block: Block; variant: Variant } | null>(null)
  // Authoring is a dev affordance: it writes to content/*.md through the local
  // content server, which the built site has no counterpart for.
  const [editing, setEditing] = useState<BlockId | null | false>(false)

  /**
   * The path is three layers deep, most specific first: what the user picked,
   * then what the model's config says, then the block's default. Keeping the
   * user's overrides separate from the model's path is what lets the UI mark a
   * block as "edited" and offer to snap it back.
   */
  const path = useMemo(
    () => ({ ...defaultPath, ...(model?.path ?? {}), ...overrides }),
    [model, overrides],
  )

  const diverged = useMemo(() => {
    const s = new Set<BlockId>()
    if (!model) return s
    for (const b of blocks) {
      const mine = path[b.id]
      const theirs = model.path[b.id]
      if (theirs && mine !== theirs) s.add(b.id)
    }
    return s
  }, [model, path])

  const unresolved = useMemo(() => new Set(model?.unresolved ?? []), [model])

  const setVariant = useCallback((block: BlockId, variant: string) => {
    setOverrides((o) => ({ ...o, [block]: variant }))
    setSelectedId(block)
  }, [])

  const resetBlock = useCallback((block: BlockId) => {
    setOverrides(({ [block]: _drop, ...rest }) => rest)
  }, [])

  const loadModel = useCallback((m: ModelSpec) => {
    setModel(m)
    setOverrides({}) // a new model means a new path; keeping edits would misattribute them
  }, [])

  const ctx: ModelContextValue = useMemo(
    () => ({
      model,
      path,
      dims: model?.dims ?? REFERENCE_DIMS,
      setVariant,
      resetBlock,
      diverged,
      unresolved,
    }),
    [model, path, setVariant, resetBlock, diverged, unresolved],
  )

  const selectedBlock = selectedId ? blockById.get(selectedId) : undefined
  const selectedVariant = selectedId ? variantOf(selectedId, path[selectedId]) : undefined
  const lineageBlock = lineageFor ? blockById.get(lineageFor) : undefined

  return (
    <ModelContext.Provider value={ctx}>
      <div className="app">
        <header className="topbar">
          <h1>LLM Internals</h1>
          <p className="topbar-sub">
            Thirteen fixed positions in the forward pass. Every model is a path through them.
          </p>
          <div className="topbar-spacer" />
          {import.meta.env.DEV && (
            <button
              className="edit-toggle"
              onClick={() => setEditing(selectedId)}
              title="Edit the Markdown behind this app"
            >
              <Icon name="pencil" size={14} /> Edit content
            </button>
          )}
          <ModelBar model={model} onLoad={loadModel} onClear={() => { setModel(null); setOverrides({}) }} />
        </header>

        {model && (
          <div className="modelstrip">
            <span className="ms-name">
              <Icon name="box" size={15} /> {model.label}
            </span>
            <span className={`ms-source ${model.source}`}>
              {model.source === 'hf' ? 'live from config.json' : model.source === 'preset' ? 'bundled config' : 'manual'}
            </span>
            <span className="ms-dims">
              {model.dims.nLayer}L · d{model.dims.dModel} · {model.dims.nHead}h/{model.dims.nKvHead}kv ·
              {' '}{(model.dims.ctx / 1024).toFixed(0)}k ctx
              {model.dims.nExperts ? ` · ${model.dims.nActive}/${model.dims.nExperts} experts` : ''}
            </span>
            {diverged.size > 0 && (
              <span className="ms-diverged">
                {diverged.size} block{diverged.size > 1 ? 's' : ''} moved off this model
              </span>
            )}
            {unresolved.size > 0 && (
              <span className="ms-unresolved">
                {unresolved.size} not determinable from config.json
              </span>
            )}
          </div>
        )}

        <div className={`workspace${selectedBlock ? '' : ' is-detail-empty'}`}>
          <StackView selectedId={selectedId} onSelect={setSelectedId} onExplore={setLineageFor} />

          <aside className={`detail${selectedBlock ? '' : ' is-empty'}`}>
            {selectedBlock && selectedVariant ? (
              <>
                <div className="detail-head llm-detail-head">
                  <div className="detail-eyebrow">
                    <span className="detail-icon">
                      <Icon name={selectedBlock.icon} size={20} />
                    </span>
                    <span className={`llm-detail-slot ${selectedBlock.slot}`}>
                      {selectedBlock.slot === 'layer' ? 'repeating layer' : selectedBlock.slot}
                    </span>
                    <span className="detail-ord">§{selectedBlock.ordinal}</span>
                    <button className="close-btn" onClick={() => setSelectedId(null)} aria-label="Close detail panel">
                      ✕
                    </button>
                  </div>
                  <h2>{selectedBlock.label}</h2>
                  <p className="detail-tagline">{selectedBlock.tagline}</p>
                  <button className="map-btn llm-lineage-btn" onClick={() => setLineageFor(selectedBlock.id)}>
                    <span className="map-btn-glyph">
                      <Icon name="graph" size={17} />
                    </span>
                    Explore the lineage map
                    <span className="map-btn-count">{selectedBlock.variants.length}</span>
                  </button>
                  {selectedVariant.concepts && selectedVariant.concepts.length > 0 && (
                    <button
                      className="map-btn llm-concept-btn"
                      onClick={() => setConceptMapFor({ block: selectedBlock, variant: selectedVariant })}
                    >
                      <span className="map-btn-glyph">
                        <Icon name="bulb" size={17} />
                      </span>
                      Explore the concept map
                      <span className="map-btn-count">{selectedVariant.concepts.length}</span>
                    </button>
                  )}
                  {diverged.has(selectedBlock.id) && (
                    <button className="llm-reset-btn" onClick={() => resetBlock(selectedBlock.id)}>
                      Reset to {model?.label}
                    </button>
                  )}
                </div>
                <VariantDetail block={selectedBlock} variant={selectedVariant} />
              </>
            ) : (
              <div className="empty-detail">
                <div className="big"><Icon name="layers" size={30} /></div>
                <p>
                  Pick a block to read the variant it is currently rendered as. Open its lineage to
                  see every other option, when each arrived, and what problem it was solving.
                </p>
                <p className="empty-hint">
                  Load a model above and the thirteen blocks fill in with the variants its config
                  implies — plus the parameter counts and cache sizes those choices cost.
                </p>
              </div>
            )}
          </aside>
        </div>

        {lineageBlock && <LineageMap block={lineageBlock} onClose={() => setLineageFor(null)} />}

        {conceptMapFor && (
          <ConceptMap
            block={conceptMapFor.block}
            variant={conceptMapFor.variant}
            onClose={() => setConceptMapFor(null)}
          />
        )}

        {editing !== false && (
          <ContentEditor initialBlock={editing} onClose={() => setEditing(false)} />
        )}
      </div>
    </ModelContext.Provider>
  )
}

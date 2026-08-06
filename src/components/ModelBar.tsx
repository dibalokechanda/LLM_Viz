import { useState } from 'react'
import type { ModelSpec } from '../data/types'
import { presets } from '../data/models'
import { HFError, loadModel } from '../lib/hf'

/**
 * The model loader. Two ways in: a Hugging Face id, fetched live, or a preset.
 *
 * Presets are not a fallback for a broken fetch — they are the answer for the
 * gated repos (Llama, Gemma) that an anonymous browser request can never read,
 * and for working offline. They carry the same shape as a fetched model, so
 * nothing downstream can tell the difference.
 */
export default function ModelBar({
  model,
  onLoad,
  onClear,
}: {
  model: ModelSpec | null
  onLoad: (m: ModelSpec) => void
  onClear: () => void
}) {
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      onLoad(await loadModel(input))
      setInput('')
    } catch (err) {
      if (err instanceof HFError) setError({ message: err.message, hint: err.hint })
      else setError({ message: err instanceof Error ? err.message : 'Something went wrong' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mb">
      <div className="mb-credit" aria-label="Author and affiliation">
        <div className="mb-credit-person">
          <span className="mb-credit-name">Dibaloke Chanda</span>
          <span className="mb-credit-affiliation">Marquette University · PhD Student · Department of Computer Science</span>
        </div>
        <a
          className="mb-credit-link"
          href="https://www.linkedin.com/in/dibaloke-chanda/"
          target="_blank"
          rel="noopener noreferrer"
        >
          LinkedIn ↗
        </a>
      </div>
      <form className="mb-form" onSubmit={submit}>
        <input
          className="mb-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Qwen/Qwen3-8B  ·  or paste a Hugging Face URL"
          spellCheck={false}
          aria-label="Hugging Face model id"
        />
        <button className="mb-go" disabled={busy || !input.trim()}>
          {busy ? 'Reading config…' : 'Trace it'}
        </button>
      </form>

      <div className="mb-presets">
        <span className="mb-presets-label">or</span>
        {presets.map((p) => (
          <button
            key={p.id}
            className={`mb-preset${model?.id === p.id ? ' on' : ''}`}
            onClick={() => {
              setError(null)
              onLoad(p)
            }}
          >
            {p.label}
          </button>
        ))}
        {model && (
          <button className="mb-clear" onClick={onClear}>
            clear
          </button>
        )}
      </div>

      {error && (
        <div className="mb-error" role="alert">
          <b>{error.message}</b>
          {error.hint && <span>{error.hint}</span>}
        </div>
      )}
    </div>
  )
}

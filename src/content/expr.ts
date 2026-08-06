/**
 * A tiny expression language for the live cost tables.
 *
 * Everything else in a block is prose or data and moves to Markdown cleanly.
 * Cost rows are the exception: they are *formulas* evaluated against whatever
 * model is loaded, so "KV cache at full context" has to stay arithmetic over
 * that model's dimensions rather than a fixed string.
 *
 * Rather than leave those 23 entries behind in TypeScript — the one part of the
 * content you could not edit — they become templates:
 *
 *     value: "{bytes(2 * nLayer * nKvHead * dHead * ctx * kvBytes)}"
 *     value: "{nKvHead} KV heads for {nHead} query heads"
 *     value: "{fixed(dFF / dModel, 2)}×"
 *
 * Text outside braces is literal; each `{...}` is an expression over the loaded
 * model's dims. Deliberately not `eval` and deliberately not Turing-complete:
 * arithmetic, parentheses, and a fixed set of formatting helpers. A typo yields
 * a visible `?` in the table rather than a crashed panel or an executed string.
 */
import type { Dims } from '../data/types'

/* ────────────────────────────── formatting ────────────────────────────── */

const fmtBytes = (n: number) =>
  n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(0)} MB` : `${(n / 1024).toFixed(1)} KB`

const fmtSi = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(0)} M` : Math.round(n).toLocaleString()

/**
 * Functions available inside an expression. Each returns a number except the
 * formatters, which return a string — the evaluator allows that only at the
 * top level of a `{...}`, which is where formatting belongs.
 */
const FUNCS: Record<string, (...a: number[]) => number | string> = {
  /** Human bytes: 2.1 GB / 67 MB / 512.0 KB. */
  bytes: (n) => fmtBytes(n),
  /** Human count: 5.64 B / 176 M / 1,024. */
  si: (n) => fmtSi(n),
  /** Thousands-separated integer. */
  num: (n) => Math.round(n).toLocaleString(),
  /** Scientific notation, for counts too large to read as digits. */
  sci: (n, d = 2) => n.toExponential(d),
  /** Fixed decimal places. */
  fixed: (n, d = 0) => n.toFixed(d),
  round: Math.round,
  floor: Math.floor,
  ceil: Math.ceil,
  min: Math.min,
  max: Math.max,
  abs: Math.abs,
  sqrt: Math.sqrt,
  log2: Math.log2,
  pow: (a, b) => a ** b,
}

/** Dims fields an expression may name, plus a couple of derived conveniences. */
function scopeOf(d: Dims): Record<string, number> {
  return {
    nLayer: d.nLayer,
    dModel: d.dModel,
    nHead: d.nHead,
    nKvHead: d.nKvHead,
    dHead: d.dHead,
    dFF: d.dFF,
    vocab: d.vocab,
    ctx: d.ctx,
    ropeTheta: d.ropeTheta ?? 0,
    nExperts: d.nExperts ?? 0,
    nActive: d.nActive ?? 0,
    nShared: d.nShared ?? 0,
    window: d.window ?? 0,
    kvBytes: d.kvBytes ?? 2,
    // Derived, because they appear in nearly every cost table and spelling
    // them out each time invites arithmetic slips.
    kvPerToken: 2 * d.nLayer * d.nKvHead * d.dHead * (d.kvBytes ?? 2),
    pi: Math.PI,
  }
}

/* ────────────────────────────── parser ────────────────────────────── */

type Tok = { t: 'num' | 'id' | 'op' | 'str'; v: string }

function lex(src: string): Tok[] {
  const out: Tok[] = []
  let i = 0
  while (i < src.length) {
    const c = src[i]
    if (/\s/.test(c)) { i++; continue }
    if (/[0-9.]/.test(c)) {
      let j = i
      while (j < src.length && /[0-9._eE]/.test(src[j])) j++
      // Allow 1e6 and 10_000 style literals.
      out.push({ t: 'num', v: src.slice(i, j).replace(/_/g, '') })
      i = j
      continue
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++
      out.push({ t: 'id', v: src.slice(i, j) })
      i = j
      continue
    }
    // String literals, for the fallback arm of a conditional: "not set".
    if (c === '"' || c === "'") {
      const j = src.indexOf(c, i + 1)
      if (j < 0) throw new Error('unterminated string')
      out.push({ t: 'str', v: src.slice(i + 1, j) })
      i = j + 1
      continue
    }
    // Two-character comparisons first, so ">=" is not read as ">" then "=".
    const two = src.slice(i, i + 2)
    if (['>=', '<=', '==', '!='].includes(two)) { out.push({ t: 'op', v: two }); i += 2; continue }
    if ('+-*/%(),?:<>'.includes(c)) { out.push({ t: 'op', v: c }); i++; continue }
    throw new Error(`unexpected character "${c}"`)
  }
  return out
}

/** Values flowing through the evaluator. Strings only come from literals and formatters. */
type Value = number | string

const asNum = (v: Value, what: string): number => {
  if (typeof v === 'number') return v
  throw new Error(`${what} needs a number, got text "${v}"`)
}

/**
 * Recursive descent, loosest binding first:
 *   ternary → or ('?' ternary ':' ternary)?
 *   or      → cmp (('>'|'<'|'>='|'<='|'=='|'!=') cmp)?
 *   cmp     → term (('+'|'-') term)*
 *   term    → unary (('*'|'/'|'%') unary)*
 *   unary   → '-'? atom
 *   atom    → number | string | ident | ident '(' args ')' | '(' ternary ')'
 *
 * The ternary is what lets content say `{ropeTheta ? num(ropeTheta) : "not set"}`
 * — several cost rows need a fallback when a model simply has no such dimension.
 */
function parse(toks: Tok[], scope: Record<string, number>): Value {
  let p = 0
  const peek = () => toks[p]
  const eat = (v: string) => {
    if (toks[p]?.v !== v) throw new Error(`expected "${v}"`)
    p++
  }

  const atom = (): Value => {
    const tk = toks[p]
    if (!tk) throw new Error('unexpected end of expression')
    if (tk.t === 'num') { p++; return Number(tk.v) }
    if (tk.t === 'str') { p++; return tk.v }
    if (tk.t === 'id') {
      p++
      if (peek()?.v === '(') {
        eat('(')
        const args: number[] = []
        if (peek()?.v !== ')') {
          args.push(asNum(ternary(), `argument to ${tk.v}()`))
          while (peek()?.v === ',') { eat(','); args.push(asNum(ternary(), `argument to ${tk.v}()`)) }
        }
        eat(')')
        const fn = FUNCS[tk.v]
        if (!fn) throw new Error(`unknown function "${tk.v}"`)
        return fn(...args)
      }
      if (!(tk.v in scope)) throw new Error(`unknown name "${tk.v}"`)
      return scope[tk.v]
    }
    if (tk.v === '(') { eat('('); const v = ternary(); eat(')'); return v }
    throw new Error(`unexpected "${tk.v}"`)
  }

  const unary = (): Value => {
    if (peek()?.v === '-') { p++; return -asNum(unary(), 'negation') }
    if (peek()?.v === '+') { p++; return unary() }
    return atom()
  }

  const term = (): Value => {
    let v = unary()
    while (peek()?.t === 'op' && '*/%'.includes(peek().v)) {
      const op = toks[p++].v
      const r = asNum(unary(), op)
      const l = asNum(v, op)
      v = op === '*' ? l * r : op === '/' ? l / r : l % r
    }
    return v
  }

  const sum = (): Value => {
    let v = term()
    while (peek()?.t === 'op' && '+-'.includes(peek().v)) {
      const op = toks[p++].v
      const r = term()
      // `+` doubles as string concatenation, so a conditional can build a whole
      // phrase: `window == 0 ? "not set" : num(window) + " tokens"`.
      if (op === '+' && (typeof v === 'string' || typeof r === 'string')) {
        v = `${v}${r}`
      } else {
        const l = asNum(v, op)
        v = op === '+' ? l + asNum(r, op) : l - asNum(r, op)
      }
    }
    return v
  }

  const cmp = (): Value => {
    const v = sum()
    const op = peek()
    if (op?.t === 'op' && ['>', '<', '>=', '<=', '==', '!='].includes(op.v)) {
      p++
      const r = sum()
      const l = typeof v === 'number' && typeof r === 'number' ? [v, r] : [String(v), String(r)]
      switch (op.v) {
        case '>': return l[0] > l[1] ? 1 : 0
        case '<': return l[0] < l[1] ? 1 : 0
        case '>=': return l[0] >= l[1] ? 1 : 0
        case '<=': return l[0] <= l[1] ? 1 : 0
        case '==': return l[0] === l[1] ? 1 : 0
        default: return l[0] !== l[1] ? 1 : 0
      }
    }
    return v
  }

  const ternary = (): Value => {
    const cond = cmp()
    if (peek()?.v !== '?') return cond
    eat('?')
    const yes = ternary()
    eat(':')
    const no = ternary()
    // Truthiness matches JS for the cases that matter: 0 and "" are false.
    return cond === 0 || cond === '' ? no : yes
  }

  const value = ternary()
  if (p !== toks.length) throw new Error(`unexpected "${toks[p].v}"`)
  return value
}

/* ────────────────────────────── public API ────────────────────────────── */

/**
 * Fill `{...}` holes in a template against the model's dims.
 *
 * Never throws: a bad expression renders as `?` and reports itself through
 * `onError`, so one typo in a content file degrades a single table cell rather
 * than taking down the panel around it.
 */
export function evalTemplate(
  template: string,
  dims: Dims,
  onError?: (msg: string) => void,
): string {
  const scope = scopeOf(dims)
  return template.replace(/\{([^{}]+)\}/g, (_m, src: string) => {
    try {
      const v = parse(lex(src), scope)
      if (typeof v === 'string') return v
      // Bare numbers get thousands separators; anything fractional keeps 2dp.
      return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)
    } catch (e) {
      onError?.(`${src.trim()} — ${e instanceof Error ? e.message : 'bad expression'}`)
      return '?'
    }
  })
}

/** True when the template parses cleanly against a probe. Used by the validator. */
export function checkTemplate(template: string, dims: Dims): string[] {
  const errs: string[] = []
  evalTemplate(template, dims, (m) => errs.push(m))
  return errs
}

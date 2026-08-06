/**
 * `turndown-plugin-gfm` ships no types. It is a tiny plugin surface — the
 * whole API is a handful of functions taking a TurndownService — so a local
 * declaration is cheaper and more honest than pulling in a stub package.
 */
declare module 'turndown-plugin-gfm' {
  import type TurndownService from 'turndown'
  type Plugin = (service: TurndownService) => void
  /** Tables, strikethrough, task lists and fenced code, all at once. */
  export const gfm: Plugin
  export const tables: Plugin
  export const strikethrough: Plugin
  export const taskListItems: Plugin
  export const highlightedCodeBlock: Plugin
}

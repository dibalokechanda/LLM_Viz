# Editing the content

Everything this app says lives in the thirteen `.md` files next to this one — one
per block of the forward pass. There is no other copy. Edit them by hand, or
open the app with `npm run dev` and click **Edit content**; both write the same
files, so you can move between the two freely.

After any edit:

```bash
npm run check
```

That parses every file, resolves every lineage arrow, and evaluates every cost
formula against two probe models. It prints the exact file and field for
anything wrong, which is a much better experience than finding out in the
browser.

## Shape of a file

```markdown
---
id: qkv                      ← must match the filename
label: Query / Key / Value   ← shown on the card
ordinal: '5'
icon: fanout                 ← a name from src/components/Icon.tsx
slot: layer                  ← input | layer | output
tagline: How many distinct keys and values the heads share
io:
  in: x [B, T, d_model]
  out: q [B, n_head, T, d_head] · k, v [B, n_kv, T, d_head]
defaultVariant: gqa          ← which variant shows before a model is loaded

lineage:                     ← the arrows in the lineage map
  - from: mha
    to: mqa
    kind: fixes              ← derives | fixes | combines | replaces | inspires
    label: collapse all KV heads to one

variants:                    ← everything structured, per variant
  - id: mha
    label: MHA
    full: Multi-Head Attention
    year: 2017
    role: origin             ← origin | refinement | branch | synthesis | legacy | frontier
    tagline: Every query head gets its own key and value head
    paper: { title: …, url: …, authors: … }
    math: [ … ]
    code: [ … ]
    figures: [ … ]
    usedBy: [ … ]
    cost: [ … ]
---

## role

The prose about what this *position* does, independent of which variant fills
it. Blank line between paragraphs. Markdown works: **bold**, *italic*, `code`.

## mha

The prose for the `mha` variant — its "What it is" section. One `##` heading per
variant id.

### fixes

The one sentence on what was wrong with what came before. Optional.
```

### Formatting available in prose

Bold, italic, underline, strikethrough, inline code, highlight and colour;
headings, bullet and numbered lists, quotes, dividers, links, images, tables and
fenced code blocks; and `$…$` maths. All of it works whether you type Markdown by
hand or use the in-app editor — the two round-trip byte-for-byte.

Two things behave in a way worth knowing:

- **Headings in prose are stored two levels down.** What the editor calls H1/H2/H3
  is written as `####`/`#####`/`######`. That keeps `##` and `###` reserved for the
  section markers that pair prose with a variant, so a heading you write can never
  be mistaken for structure.
- **Underline, highlight, colour and text alignment round-trip as inline HTML**
  (`<u>`, `<mark>`, `<span style>`), because Markdown has no syntax for them. They
  render correctly; they just look like HTML in the file.

Images pasted or dropped into the editor are written to `public/content-assets/`
and referenced as `assets/<name>` — they live in the repository next to the prose
rather than as base64 inside it.

**Front matter is data, the body is prose.** That split is the whole format:
the things you rewrite most (paragraphs) are plain Markdown, and the things
with structure (equations, code, figures) are YAML, where a `|` block holds TeX
and source code verbatim with no escaping to get wrong.

## Two rules worth knowing

**Every variant needs a `## <id>` section.** The id in `variants:` and the
heading in the body are what pair structure with prose. `npm run check` tells
you when one is missing or orphaned.

**Lineage arrows must name real variants.** An arrow pointing at an id that
does not exist would silently vanish from the map; the checker catches it.

## Maths

Each equation is one text block, in the editor and in the file alike:

```
Cache reduction              ← caption (optional first line)
$$
rac{	ext{bytes}_{MQA}}{	ext{bytes}_{MHA}} = rac{1}{n_{head}}
$$

where                        ← optional symbol legend
- n_{head} — number of query heads

worked                       ← optional worked substitutions
- rac{1}{32} = 3.1\% — 32-head model: 2.1 GB becomes 67 MB

note                         ← optional prose under the block
Arithmetic intensity rises by the same factor.
```

Legend and worked lines use an **em dash** (`—`) as the separator, not a hyphen —
minus signs appear in almost every equation.

## Cost formulas

Cost rows are the one field that is not plain text — they are computed against
whichever model is loaded, so the same row reads differently at 8B and 671B:

```yaml
cost:
  - label: KV cache at full context
    value: "{bytes(2 * nLayer * nKvHead * dHead * ctx * kvBytes)}"
    note: "{fixed(nHead / max(nKvHead, 1), 0)}× smaller than MHA"
    key: true                ← emphasise this row
```

Text outside braces is literal; each `{…}` is arithmetic.

**Names:** `nLayer` `dModel` `nHead` `nKvHead` `dHead` `dFF` `vocab` `ctx`
`ropeTheta` `nExperts` `nActive` `nShared` `window` `kvBytes` `kvPerToken` `pi`

**Functions:** `bytes` (→ `2.10 GB`) · `si` (→ `5.64 B`) · `num` (→ `1,024`) ·
`fixed(x, dp)` · `sci` · `round` `floor` `ceil` `min` `max` `abs` `sqrt` `pow`

Comparisons and a conditional are available, which matters because a dimension
may be absent — a dense model has no `nExperts`, and missing values read as `0`:

```yaml
value: '{window == 0 ? "not set" : num(window) + " tokens"}'
```

A broken formula renders as `?` in that one cell rather than breaking the panel,
and `npm run check` reports it with the expression that failed.

## Concept maps

A variant can also expose a card map for ideas that deserve more structure than
one long section of prose. Add cards under that variant's front matter, or use
the **Concept map** section of the in-app editor:

```yaml
concepts:
  - id: key-value-sharing
    label: Share key/value heads
    kind: method                 # idea | formula | method | metric | pitfall | tradeoff
    summary: Several query heads reuse each key/value head.
    detail:
      - This is the longer Markdown explanation shown after a reader selects the card.
    children:
      - id: cache-effect
        label: Smaller cache
        kind: metric
        summary: Fewer key/value heads reduce memory per token.
        detail:
          - A direct sub-card of `key-value-sharing`.
```

The map lays out top-level cards around the selected variant. A card with
`children` gains an expand control and reveals its direct sub-cards. Card ids
must be unique within the variant; `npm run check` validates ids, kinds, titles
and summaries.

## What is not here

Two things stayed in TypeScript because they are programs, not prose:

- **Figure components** (`src/components/Figure*.tsx`) — the code that draws a
  heatmap or an embedding projection. The *data* for each figure is in the
  content file under `figures:`, so what a diagram shows is editable here; how
  it is drawn is not.
- **Model detection** (`src/lib/hf.ts`) — the rules that read a Hugging Face
  `config.json` and decide which variant each block sits on. Adding a variant
  means teaching the detector about it there.

## Safety

The in-app editor writes a copy of the previous revision to `.content-backups/`
before every save, and refuses to write anything that will not parse back. Even
so, these files are the only copy of a lot of writing — keep them in git.

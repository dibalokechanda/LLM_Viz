---
id: pattern
label: Attention Pattern
ordinal: '7'
icon: target
slot: layer
tagline: Which token pairs are allowed to interact
io:
  in: scores [B, n_head, T, T]
  out: scores with disallowed pairs set to −∞
defaultVariant: causal
lineage:
  - from: bidirectional
    to: causal
    kind: fixes
    label: a generator must not see the future
  - from: causal
    to: window
    kind: derives
    label: drop the long tail — most of it is unused
  - from: window
    to: sink
    kind: fixes
    label: evicting the first tokens destroys the model
  - from: window
    to: dilated
    kind: derives
    label: skip stride-wise to widen reach
  - from: window
    to: interleaved
    kind: fixes
    label: alternate windowed and global layers
  - from: causal
    to: interleaved
    kind: derives
    label: keep some layers fully global
  - from: dilated
    to: nsa
    kind: inspires
    label: learn the sparsity instead of fixing it
  - from: sink
    to: nsa
    kind: inspires
    label: compress far context rather than dropping it
variants:
  - id: bidirectional
    label: Bidirectional
    full: Full (unmasked) attention
    year: 2017
    role: legacy
    tagline: Every token sees every other token
    concepts:
      - id: bidirectional-complete-graph
        label: A complete token graph
        kind: method
        summary: Every query position can score every key position in the sequence.
        detail:
          - >-
            The mask contains no causal or locality restriction, so each row of the attention
            matrix has T legal entries. This lets an encoder use both left and right context.
      - id: bidirectional-generation-limit
        label: Not a decoder policy
        kind: tradeoff
        summary: Seeing future tokens prevents direct left-to-right autoregressive generation.
        detail:
          - >-
            Bidirectional attention is ideal for representation learning, classification, and
            masked-token objectives. A causal decoder must hide later positions at inference time.
    figures:
      - kind: heatmap
        title: Full attention, 10 tokens
        'n': 10
        mask: full
        caption: Every cell open. 100 of 100 pairs computed.
    usedBy:
      - BERT
      - RoBERTa
      - ViT
      - sentence-transformers
  - id: causal
    label: Causal
    full: Causal (lower-triangular) masking
    year: 2018
    role: origin
    tagline: Token i sees tokens 0…i and nothing after
    concepts:
      - id: future-mask
        label: Hide future tokens
        kind: method
        summary: A lower-triangular mask gives each position access only to the prefix available at generation time.
        detail:
          - >-
            Future scores receive negative infinity before softmax, so they receive exactly zero
            probability and cannot leak information into the prediction.
      - id: parallel-training
        label: Train all positions together
        kind: tradeoff
        summary: The mask preserves autoregressive causality while allowing the full training sequence to be processed in parallel.
        detail:
          - >-
            Training computes every allowed query-key pair in one matrix operation. Generation still
            proceeds one token at a time because the next token is not known yet.
      - id: prefix-cache
        label: Reuse the visible prefix
        kind: metric
        summary: Causal attention is why past keys and values can be cached during decoding.
        detail:
          - >-
            Each new token only needs fresh query, key, and value vectors for itself; the prior
            visible prefix has already been projected and can be read from the KV cache.
    math:
      - title: The mask
        tex: M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
        note: >-
          −∞ rather than deletion, because it must survive the softmax: exp(−∞) = 0, so the
          forbidden pairs get exactly zero probability and contribute no gradient.
      - title: Pairs actually computed
        tex: \frac{T(T+1)}{2} \approx \frac{T^2}{2}
        note: >-
          Half the full matrix — a constant factor, not a change in order. Everything below this
          variant is an attempt to change the order.
    figures:
      - kind: heatmap
        title: Causal mask, 10 tokens
        'n': 10
        mask: causal
        focusRow: 6
        caption: >-
          Shading is each cell's share of its row. Early rows attend to few tokens so each gets a
          large share; later rows spread thin. That entropy difference is the positional signal NoPE
          relies on.
    cost:
      - label: Pairs at full context
        value: '{sci(ctx * (ctx + 1) / 2, 2)}'
        note: per head, per layer
      - label: Score matrix memory
        value: '{fixed(ctx * ctx * nHead * 2 / 1073741824, 1)} GB'
        note: if materialised — which is exactly why FlashAttention does not
        key: true
    usedBy:
      - every decoder-only LLM
  - id: window
    label: Sliding window
    full: Sliding-window (local) attention
    year: 2020
    role: branch
    tagline: Attend only to the last w tokens
    paper:
      title: 'Longformer: The Long-Document Transformer'
      url: https://arxiv.org/abs/2004.05150
    concepts:
      - id: window-local-neighborhood
        label: Limit each row to a neighborhood
        kind: method
        summary: A query sees only a fixed number of earlier positions in its causal window.
        detail:
          - >-
            The legal attention band has width w rather than T, so per-layer memory and compute
            scale linearly with sequence length for a fixed window.
      - id: window-depth-reach
        label: Depth carries information outward
        kind: tradeoff
        summary: Distant tokens can interact only after their information crosses several local layers.
        detail:
          - >-
            A window is not the same as an absolute context limit: the receptive field grows with
            depth. Direct retrieval of one far-away token, however, remains unavailable in one layer.
    math:
      - title: Windowed mask
        tex: M_{ij} = \begin{cases} 0 & i - w < j \le i \\ -\infty & \text{otherwise} \end{cases}
      - title: Receptive field through depth
        tex: R = L \cdot (w - 1) + 1
        worked:
          - tex: 32 \cdot 4095 + 1 = 131{,}041
            caption: Mistral-7B — nominal reach, not reliable retrieval range
    figures:
      - kind: heatmap
        title: Sliding window, w = 4
        'n': 12
        mask: window
        w: 4
        focusRow: 9
        caption: >-
          The band is constant-width, so cost per token stops growing. Note that token 9 has
          completely lost tokens 0–5.
    cost:
      - label: Window
        value: '{window == 0 ? "not set" : num(window) + " tokens"}'
      - label: Nominal receptive field
        value: '{window == 0 ? "—" : num(nLayer * (window - 1) + 1) + " tokens"}'
        note: through layer stacking — reliable retrieval is far shorter
      - label: KV cache ceiling
        value: '{window == 0 ? "—" : bytes(2 * nLayer * nKvHead * dHead * min(window, ctx) * kvBytes)}'
        note: constant — the cache stops growing once the window fills
        key: true
    usedBy:
      - Longformer
      - Mistral-7B
      - Gemma-2 (alternating)
      - Phi-3-small
  - id: sink
    label: Attention sinks
    full: StreamingLLM — window plus retained initial tokens
    year: 2023
    role: refinement
    tagline: Keep the first few tokens forever, window the rest
    paper:
      title: Efficient Streaming Language Models with Attention Sinks
      url: https://arxiv.org/abs/2309.17453
      authors: Xiao et al.
    concepts:
      - id: sink-two-regions
        label: Preserve anchors and a recent window
        kind: method
        summary: The mask retains a few initial tokens alongside the newest local positions.
        detail:
          - >-
            The cache has two persistent regions: fixed sink slots at the beginning and a rolling
            suffix. Queries can always attend to both even after millions of streamed tokens.
      - id: sink-activation-anchor
        label: Sinks stabilise attention mass
        kind: idea
        summary: Initial positions become preferred destinations when local context is evicted.
        detail:
          - >-
            The retained tokens need not carry important semantic content. They provide stable
            locations for attention probability that would otherwise be forced onto poor local keys.
    figures:
      - kind: heatmap
        title: Window of 4 plus 2 retained sinks
        'n': 12
        mask: sink
        w: 4
        sinks: 2
        focusRow: 10
        caption: >-
          The two left columns stay open for every row. Cheap — two extra tokens of cache — and the
          difference between stable streaming and collapse.
    usedBy:
      - StreamingLLM
      - vLLM / llama.cpp streaming modes
      - GPT-OSS (learned sink logits)
  - id: dilated
    label: Dilated
    full: Dilated / strided attention
    year: 2019
    role: branch
    tagline: Attend to every r-th token to widen reach cheaply
    paper:
      title: Generating Long Sequences with Sparse Transformers
      url: https://arxiv.org/abs/1904.10509
    concepts:
      - id: dilated-strided-keys
        label: Sample distant keys at a stride
        kind: method
        summary: A query attends to selected positions separated by a fixed dilation interval.
        detail:
          - >-
            The pattern skips many intervening tokens while retaining long-range landmarks. It can
            be combined with a local band so near-token detail is not lost.
      - id: dilated-aliasing-risk
        label: Reach is not full coverage
        kind: tradeoff
        summary: Dilation widens a layer's span but leaves blind spots between sampled positions.
        detail:
          - >-
            Different strides or layers can cover those gaps, but a single fixed pattern assumes
            that relevant information lies on its chosen grid.
    figures:
      - kind: heatmap
        title: Dilated, stride 3
        'n': 12
        mask: dilated
        w: 3
        focusRow: 11
        caption: >-
          Token 11 reaches all the way to token 0 while attending to only 4 keys — but the tokens
          between are invisible to it at this layer.
    usedBy:
      - Sparse Transformer
      - LongNet
      - BigBird (as one component)
  - id: interleaved
    label: Interleaved local/global
    full: Alternating windowed and full-attention layers
    year: 2024
    role: synthesis
    tagline: Most layers windowed, every n-th layer global
    concepts:
      - id: interleaved-layer-schedule
        label: Alternate communication budgets
        kind: method
        summary: Most layers use a local mask while scheduled layers expose the full sequence.
        detail:
          - >-
            Global layers periodically refresh long-range communication; local layers perform the
            cheaper majority of token mixing. The ratio is an architectural schedule, not a runtime
            switch for one layer.
      - id: interleaved-global-dominance
        label: Global layers set the memory floor
        kind: metric
        summary: A small number of full-attention layers can dominate KV-cache cost at long context.
        detail:
          - >-
            Reducing local windows helps, but each global layer still stores and attends over T
            entries. The design balances quality against a nonzero global-memory baseline.
    math:
      - title: Cache with a local:global ratio
        tex: \text{cache} \propto n_{\text{global}} \cdot T + n_{\text{local}} \cdot w
        worked:
          - tex: \tfrac{1}{6}\cdot 48 \cdot 128\text{k} + \tfrac{5}{6}\cdot 48 \cdot 1024
            caption: Gemma-3 style 5:1 — the 8 global layers dominate entirely
        note: 'The saving is real but bounded: you can never get below what the global layers alone cost.'
    figures:
      - kind: blocks
        title: Gemma-2 layer schedule
        rows:
          - label: layers 0–3
            boxes:
              - text: local w=4096
              - text: global
                filled: true
              - text: local w=4096
              - text: global
                filled: true
          - label: cache cost
            boxes:
              - text: bounded
              - text: grows with T
                filled: true
              - text: bounded
              - text: grows with T
                filled: true
        caption: >-
          Filled layers are the ones whose cache grows without limit. Halving their number halves
          the long-context cache.
    usedBy:
      - Gemma-2 (1:1)
      - Gemma-3 (5:1)
      - Llama-4
      - Command R+
  - id: nsa
    label: Learned sparse
    full: Natively trainable sparse attention
    year: 2025
    role: frontier
    tagline: Learn which blocks to attend to, instead of fixing the pattern
    paper:
      title: 'Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention'
      url: https://arxiv.org/abs/2502.11089
      authors: DeepSeek
    concepts:
      - id: nsa-content-selector
        label: Select blocks from the input
        kind: method
        summary: A learned mechanism chooses which key blocks are visible for the current query.
        detail:
          - >-
            Sparsity can follow token content instead of a universal geometric mask. The selector
            is trained with the model so it can change its retrieval pattern across examples.
      - id: nsa-kernel-alignment
        label: Learned sparsity must still be regular
        kind: tradeoff
        summary: The selected pattern needs block structure that an accelerator can execute efficiently.
        detail:
          - >-
            Arbitrary token-level choices destroy the dense-kernel advantage. Native sparse designs
            constrain the selector so learning flexibility and hardware-friendly layout coexist.
    usedBy:
      - DeepSeek (research)
      - MoBA and related lines
---

## role

Attention is quadratic because by default every query looks at every key. This block decides which of those `T²` pairs actually exist — the mask.

Two different jobs get conflated here. **Causality** is a correctness constraint: a language model must not see the future, and this is non-negotiable for a decoder. **Sparsity** is an efficiency choice: even among legal pairs, most contribute nothing, so skipping them saves time and memory.

Everything below causal masking is a bet about which pairs matter. Get the bet right and you buy linear-ish scaling; get it wrong and the model cannot see the token it needed.

## bidirectional

No mask at all. Correct for encoders — BERT, embedding models, vision transformers — where the whole input is available at once and the task is to build a representation rather than continue a sequence.

Included here because it is the thing causal masking is defined against, and because a config with `is_decoder: false` is telling you the model belongs on this branch and will not generate text.

## causal

Set every score where `j > i` to −∞ before the softmax. This is the mask that makes next-token prediction a valid training objective: each position predicts its successor using only what precedes it, so all `T` predictions can be trained in parallel from a single forward pass.

It is also what makes the KV cache possible. Because position `i`'s keys and values never depend on anything after `i`, they are final the moment they are computed and can be reused for every subsequent step.

### fixes

A model that can see the next token during training learns to copy it and predicts nothing at inference.

## window

Restrict each query to the `w` most recent keys. Cost becomes `O(T·w)` — linear in sequence length — and the KV cache stops growing entirely once it reaches `w`, which is the bigger practical win.

The obvious objection is that the model can no longer see far back. The standard answer is stacking: a window of `w` at each of `L` layers gives an effective receptive field of `L·w`, because layer 2 attends to tokens that already aggregated their own window at layer 1. Mistral-7B's 4096 window over 32 layers nominally reaches 131k.

That argument is weaker than it sounds. Information does propagate, but it is repeatedly averaged on the way, so exact retrieval from far back degrades badly even when the receptive field nominally covers it. Sliding windows are good at fluency over long text and poor at needle-in-a-haystack.

### fixes

Causal attention keeps paying for distant tokens that measured attention weight says are almost never used.

## sink

The observed failure: run a windowed model past its window and quality does not degrade gracefully, it collapses. The cause is that softmax must sum to one. When a head has nothing it genuinely wants to attend to, it has to put its probability mass *somewhere*, and models learn to dump it on the first few tokens — which are visible to every position and therefore make a reliable default.

Those tokens are **attention sinks**. Their content is irrelevant; their availability is not. Evict them and every head's excess mass is forced onto real tokens, corrupting the representations.

The fix is four lines: always keep the first 4 tokens in the cache, plus the sliding window. That alone enables stable generation over millions of tokens with no fine-tuning.

### fixes

Evicting the very first tokens from a sliding-window cache causes perplexity to explode — far more than their content could possibly justify.

## dilated

Skip: attend to positions `i, i−r, i−2r, …`. The same number of keys now spans `r` times the distance. Usually paired with a local window and alternated across heads, so some heads see fine detail and others see coarse span.

Elegant on paper, awkward in practice — strided gathers are unfriendly to memory coalescing, so the theoretical saving often fails to materialise as wall-clock speed.

### fixes

A sliding window's reach grows only with depth. Striding buys distance within a single layer.

## interleaved

Make the choice per layer rather than per model. Gemma-2 alternates one local layer with one global; Gemma-3 uses a 5:1 ratio. The global layers provide genuine long-range retrieval, the local layers provide most of the compute saving, and the KV cache is dominated by the few global layers.

This is the variant that most clearly breaks this app's one-choice-per-block framing, which is why it is modelled explicitly rather than rounded to "window". A config with a `layer_types` array containing more than one distinct value is telling you the model is here.

### fixes

Pure sliding-window models cannot retrieve from far back; pure global attention costs too much. Neither answer is right for every layer.

## nsa

Combine three paths per query: a **compressed** path over coarse block summaries of the whole history, a **selected** path over the handful of blocks the compressed scores rank highest, and a **local** sliding window. The selection is learned and differentiable, so the sparsity is trained end-to-end rather than imposed.

The other half of the contribution is hardware alignment — selecting whole contiguous blocks rather than individual tokens, so the gather stays coalesced and the theoretical saving actually shows up as wall-clock speed. That is the step where most earlier sparse-attention work failed.

### fixes

Every fixed sparsity pattern is a guess about where information lives. A guess that is wrong for a given input cannot be corrected.

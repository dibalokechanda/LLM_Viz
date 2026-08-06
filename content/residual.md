---
id: residual
label: Residual & Placement
ordinal: '11'
icon: merge
slot: layer
tagline: How sub-layers attach to the stream running through the model
io:
  in: x [B, T, d_model]
  out: x + f(norm(x)), same shape
defaultVariant: pre-ln
lineage:
  - from: post-ln
    to: pre-ln
    kind: fixes
    label: keep the skip path clean of normalisation
  - from: post-ln
    to: deepnorm
    kind: fixes
    label: keep post-norm, scale the residual instead
  - from: pre-ln
    to: peri-ln
    kind: derives
    label: normalise the output too, not just the input
  - from: pre-ln
    to: parallel
    kind: derives
    label: run attention and FFN off the same input
  - from: pre-ln
    to: mhc
    kind: fixes
    label: replace one overloaded stream with stably mixed parallel streams
variants:
  - id: post-ln
    label: Post-LN
    full: Post-layer normalization
    year: 2017
    role: origin
    tagline: 'Add first, then normalise: LN(x + f(x))'
    concepts:
      - id: post-ln-normalised-output
        label: Normalise the combined signal
        kind: method
        summary: The residual branch and sub-layer output are added before LayerNorm is applied.
        detail:
          - >-
            Each layer presents a controlled-scale output to the next layer. This was the original
            transformer arrangement and remains common in encoder-style architectures.
      - id: post-ln-gradient-route
        label: The skip path crosses a norm
        kind: tradeoff
        summary: Backpropagation through the residual stream also passes through LayerNorm at every layer.
        detail:
          - >-
            That placement can make very deep stacks hard to optimise without careful schedules
            and warm-up. Pre-norm moves the normalisation inside the branch to create a clearer
            identity gradient path.
    math:
      - title: Post-LN
        tex: x_{l+1} = \text{LN}\big(x_l + f(x_l)\big)
        note: >-
          The identity path is inside the normalisation, so there is no clean gradient highway from
          output to input.
    usedBy:
      - Transformer (2017)
      - BERT
      - early NMT
  - id: pre-ln
    label: Pre-LN
    full: Pre-layer normalization
    year: 2019
    role: refinement
    tagline: 'Normalise first, then add: x + f(LN(x))'
    concepts:
      - id: normalize-branch
        label: Normalize before the transformation
        kind: method
        summary: Pre-LN feeds a normalized residual stream into attention or the FFN before adding the result back.
        detail:
          - >-
            The sub-layer sees controlled activations, while the residual branch itself keeps an
            unnormalized identity path from the input toward the output.
      - id: gradient-highway
        label: The skip path stays clean
        kind: formula
        summary: Repeated residual additions create a direct route for both activations and gradients across depth.
        detail:
          - >-
            This identity-like path makes deep stacks easier to optimize because information does not
            have to pass through a normalization operation at every residual hop.
      - id: final-normalization
        label: A final norm still matters
        kind: pitfall
        summary: Residual variance accumulates across layers, so Pre-LN architectures typically normalize before the output head.
        detail:
          - >-
            Moving the internal norms does not remove the need to control the final activation scale.
            The last normalization stabilizes the logits presented to the output head.
    paper:
      title: On Layer Normalization in the Transformer Architecture
      url: https://arxiv.org/abs/2002.04745
    math:
      - title: Pre-LN
        tex: x_{l+1} = x_l + f\big(\text{LN}(x_l)\big)
        note: >-
          Unrolled: x_L = x_0 + Σ f_l(LN(x_l)). The input reaches the output through a sum of clean
          identity terms, which is exactly the gradient highway post-norm lacks.
      - title: Why a final norm is required
        tex: \text{Var}(x_L) \approx \text{Var}(x_0) + \sum_{l=1}^{L}\text{Var}(f_l)
        note: >-
          Variance accumulates across depth. Without a final normalisation the logits would be
          scaled by roughly √L.
    figures:
      - kind: blocks
        title: Where the norm sits
        rows:
          - label: post-LN
            boxes:
              - text: x
              - text: + f(x)
              - text: LayerNorm
                filled: true
            arrow: gradient passes through the norm every layer
          - label: pre-LN
            boxes:
              - text: x
              - text: LN → f
                dashed: true
              - text: + x
                filled: true
            arrow: gradient reaches x directly — the skip is untouched
    cost:
      - label: Norms per layer
        value: '2'
        note: one before attention, one before the FFN
      - label: Total including the final norm
        value: '{nLayer * 2 + 1}'
        key: true
    usedBy:
      - GPT-2/3
      - Llama (all)
      - Mistral
      - Qwen
      - essentially every modern LLM
  - id: peri-ln
    label: Sandwich / Peri-LN
    full: Normalization on both sides of the sub-layer
    year: 2022
    role: refinement
    tagline: Normalise the branch input and its output
    concepts:
      - id: peri-ln-two-boundaries
        label: Bound both sides of a branch
        kind: method
        summary: Peri-LN normalises before a sub-layer and again before its output rejoins the stream.
        detail:
          - >-
            The incoming representation is well-conditioned for the branch, and the branch
            contribution is rescaled before accumulation. The residual stream itself remains a
            direct path across layers.
      - id: peri-ln-outlier-control
        label: Control accumulated outliers
        kind: pitfall
        summary: The second normalisation limits unusually large branch updates in deep networks.
        detail:
          - >-
            This addresses activation growth that can hurt high-precision training and later
            quantisation. It adds another normalisation operation compared with ordinary pre-LN.
    math:
      - title: Sandwich
        tex: x_{l+1} = x_l + \text{LN}_{\text{post}}\big(f(\text{LN}_{\text{pre}}(x_l))\big)
    usedBy:
      - Gemma-2
      - Grok-1
      - OLMo-2
  - id: parallel
    label: Parallel blocks
    full: Parallel attention and feed-forward
    year: 2021
    role: branch
    tagline: Run attention and FFN on the same input, add both
    concepts:
      - id: parallel-shared-input
        label: One normalised input, two branches
        kind: method
        summary: Attention and the FFN consume the same layer input and their updates are summed together.
        detail:
          - >-
            The FFN no longer waits for the attention output, reducing the serial dependency in a
            transformer layer. Both branches still contribute to a single residual update.
      - id: parallel-interaction-order
        label: Fewer sequential refinements
        kind: tradeoff
        summary: Parallelism improves scheduling but removes within-layer attention-to-FFN conditioning.
        detail:
          - >-
            In a sequential block, the FFN transforms attention's newly mixed representation. A
            parallel block delays that interaction until the following layer, trading ordering for
            hardware efficiency.
    math:
      - title: Parallel formulation
        tex: x_{l+1} = x_l + \text{attn}(\text{LN}(x_l)) + \text{ffn}(\text{LN}(x_l))
        note: >-
          Compare the sequential form, where ffn sees the post-attention stream. Here the FFN never
          observes what attention just wrote — that is the expressivity being traded away.
    usedBy:
      - GPT-J
      - GPT-NeoX
      - PaLM
      - Falcon
  - id: deepnorm
    label: DeepNorm
    full: DeepNorm — scaled post-normalization
    year: 2022
    role: branch
    tagline: Keep post-norm, scale the residual to survive depth
    paper:
      title: 'DeepNet: Scaling Transformers to 1,000 Layers'
      url: https://arxiv.org/abs/2203.00555
      authors: Microsoft
    concepts:
      - id: deepnorm-residual-scaling
        label: Scale the residual branch deliberately
        kind: formula
        summary: A depth-dependent coefficient controls how much old state and new update are combined.
        detail:
          - >-
            DeepNorm changes the magnitude of the residual path and paired parameter initialisation
            so that activations and gradients stay controlled across hundreds or thousands of layers.
      - id: deepnorm-coupled-recipe
        label: Scaling and initialisation are coupled
        kind: pitfall
        summary: The coefficient is not a drop-in multiplier independent of the training recipe.
        detail:
          - >-
            Its stability argument assumes matching initialisation and depth. Applying only the
            forward scaling to a pre-existing model changes the optimisation problem rather than
            recreating DeepNorm.
    math:
      - title: Scaled post-norm
        tex: x_{l+1} = \text{LN}(\alpha \cdot x_l + f(x_l)), \qquad \alpha = (2L)^{1/4}
    usedBy:
      - DeepNet
      - BEiT-3
  - id: mhc
    label: mHC
    full: Manifold-Constrained Hyper-Connections
    year: 2025
    role: frontier
    tagline: Carry several residual streams and constrain how each sub-layer mixes them
    fixes: A single residual stream must simultaneously preserve, route, and rewrite every feature; unconstrained multi-stream mixing becomes unstable at depth.
    paper:
      title: mHC — Manifold-Constrained Hyper-Connections
      url: https://arxiv.org/abs/2512.24880
    math:
      - title: Multi-stream residual update (schematic)
        tex: X_l\in\mathbb{R}^{K\times B\times T\times d},\quad u_l=P_lX_l,\quad X_{l+1}=M_lX_l+Q_l f(u_l)
        where:
          - sym: K
            means: number of parallel residual streams
          - sym: P_l, Q_l
            means: learned pre- and post-mappings around an ordinary-width sub-layer
          - sym: M_l
            means: constrained residual-stream mixing matrix
        note: >-
          This describes the data flow, not a single implementation signature. mHC constrains the
          residual mixing to a doubly stochastic manifold and bounds the pre/post mappings for depth stability.
    concepts:
      - id: parallel-residual-streams
        label: Parallel residual streams
        kind: idea
        summary: Instead of one d_model-wide communication channel, the layer carries K same-width streams in parallel.
        detail:
          - >-
            The attention or MoE sub-layer itself can remain ordinary width. The extra capacity lives
            in the residual pathway, which reads a mixture of streams and writes its result back across them.
      - id: pre-post-mappings
        label: Read and write mappings
        kind: method
        summary: A pre-mapping produces the sub-layer input from the streams; a post-mapping distributes the sub-layer output back into them.
      - id: doubly-stochastic-constraint
        label: Doubly stochastic residual mixing
        kind: formula
        summary: Constraining the stream-mixing matrix to nonnegative rows and columns that each sum to one limits amplification and collapse.
        detail:
          - >-
            The constraint is not cosmetic regularization. It is what makes repeated mixing well-behaved
            across many layers, where a freely learned dense residual matrix could explode or erase a stream.
    usedBy:
      - mHC research models
---

## role

Every sub-layer writes back into a residual stream rather than replacing it: `x + f(x)`. That skip connection is what makes depth trainable — gradients reach layer 1 through the identity path without passing through 80 transformations.

It also gives the modern reading of a transformer: the residual stream is a shared communication channel of width `d_model`, and each attention or FFN block reads from it, computes something, and adds its result back. Nothing overwrites; everything accumulates.

The design space is almost entirely about **where the normalisation goes relative to the skip**. That sounds like a detail. It decides whether a 100-layer model trains at all.

## post-ln

The original arrangement. It has better final quality when it converges — the normalisation after each addition keeps the stream tightly controlled — but it is notoriously hard to train. Gradients must pass through a LayerNorm at every layer on their way back, and past roughly 12 layers this makes training diverge without a learning-rate warmup carefully tuned for the depth.

The warmup requirement is the tell. If a paper spends a paragraph on its warmup schedule, it is probably post-norm.

## pre-ln

Move the norm inside the branch. The residual path becomes a pure identity from input to output, so gradients flow to the earliest layers unimpeded. Models train stably at 100+ layers, often with no warmup at all.

The trade is that the residual stream now grows in magnitude with depth — nothing normalises the accumulation — so a final norm before the output head is mandatory. Slightly worse final quality than post-norm at equal depth, comprehensively better in practice because you can actually build the deep model.

Universal in modern LLMs. Assume pre-norm unless a config says otherwise.

### fixes

Post-norm puts a normalisation on the gradient path at every layer, so deep models cannot be trained without delicate warmup — and often not even then.

## peri-ln

Add a second norm to the branch *output* before the addition: `x + LN(f(LN(x)))`. Keeps the pure-identity skip path that makes pre-norm trainable while bounding what each branch contributes.

Gemma-2 uses this, and it is increasingly common in very large models where activation outliers are a practical problem rather than a theoretical one.

### fixes

Pre-norm lets the residual stream grow without bound across depth, and at very large scale that growth produces activation outliers that destabilise training and wreck quantisation.

## parallel

Compute `x + attn(LN(x)) + ffn(LN(x))` instead of chaining them. Both branches read the same normalised input, so they run concurrently and share one normalisation. GPT-J and PaLM report roughly 15% higher throughput at large scale.

Quality is very slightly worse at small scale and reportedly indistinguishable at large. It has not become standard, mostly because the gain is a systems gain that depends on your parallelism strategy.

### fixes

Sequential sub-layers create a dependency chain: the FFN cannot start until attention finishes, which is a synchronisation point on every layer in every device.

## deepnorm

Keep post-normalisation but multiply the residual by a depth-derived constant `α` and scale the initialisation by `β`, both chosen so the expected update magnitude stays bounded no matter how deep the stack. The paper trains a 1000-layer transformer, which was the point.

A striking result that did not become standard — the constants depend on the architecture, and pre-norm was already good enough for the depths anyone actually wanted.

### fixes

Pre-norm gave up post-norm's quality advantage to gain trainability. DeepNorm asks whether both were available.

## mhc

mHC replaces the single residual stream with several parallel streams. Before an attention or MoE sub-layer, a learned **pre-mapping** mixes those streams into an ordinary-width input. After the sub-layer, a learned **post-mapping** writes its result back across the parallel streams. The expensive attention or MoE computation does not need to be K times wider.

The point is not merely more channels. The residual mixing is constrained to a doubly stochastic manifold, while the read/write mappings are bounded. Those restrictions make repeated cross-stream mixing stable at depth: no stream can become an unbounded amplifier or silently disappear just because an unconstrained residual matrix drifted.

This is a residual-path mechanism, not a new attention rule. The attention, SSM, or MoE selected elsewhere in the layer still supplies `f`; mHC changes the multi-stream state that `f` reads from and writes to.

### fixes

A single residual stream has to carry every preserved feature and every new write. Naively adding multiple streams creates another problem: unconstrained mixing can amplify, collapse, or entangle them as layers accumulate. mHC adds the multi-stream pathway with constraints designed for stable composition.

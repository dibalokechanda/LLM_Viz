---
id: scores
label: Score Normalization
ordinal: '8'
icon: chart
slot: layer
tagline: Turning raw logits into the weights that mix values
io:
  in: masked scores [B, n_head, T, T]
  out: probabilities, rows summing to 1
defaultVariant: softmax
lineage:
  - from: softmax
    to: softcap
    kind: fixes
    label: bound the logits before they saturate
  - from: softmax
    to: differential
    kind: fixes
    label: cancel the noise floor with a second map
  - from: softmax
    to: sigmoid-attn
    kind: replaces
    label: drop the sum-to-one constraint
variants:
  - id: softmax
    label: Softmax
    full: Scaled dot-product softmax
    year: 2017
    role: origin
    tagline: Exponentiate and normalise across the row
    concepts:
      - id: relative-logits
        label: Scores become relative preferences
        kind: formula
        summary: Softmax turns unbounded attention logits into nonnegative weights that can be compared within one row.
        detail:
          - >-
            Increasing one logit raises its share of the attention distribution while decreasing the
            shares of competing keys. Only score differences matter.
      - id: weighted-read
        label: The weights read values
        kind: method
        summary: Each probability weight controls how much of one value vector contributes to the output.
        detail:
          - >-
            The attention result is a weighted average of values. The sum-to-one rule makes the scale
            of that read stable but also forces every head to allocate all of its mass somewhere.
      - id: stable-softmax
        label: Subtract the row maximum
        kind: pitfall
        summary: Implementations shift logits before exponentiating to avoid numerical overflow.
        detail:
          - >-
            Subtracting the same maximum from every score leaves softmax unchanged mathematically, but
            prevents exponentials from overflowing in lower-precision arithmetic.
    math:
      - title: Row softmax
        tex: p_{ij} = \frac{\exp(e_{ij})}{\sum_{k} \exp(e_{ik})}
        note: >-
          Implementations subtract the row max before exponentiating. exp(e) overflows fp16 above
          about 11, and attention logits routinely exceed that — the subtraction is a numerical
          necessity, not an optimisation.
      - title: The sum-to-one constraint
        tex: \sum_j p_{ij} = 1 \quad \text{always}
        note: >-
          Every downstream variant exists because of this line. A head with nothing worth attending
          to must still distribute a full unit of mass, which is exactly what creates attention
          sinks.
    distinctions:
      - title: FlashAttention does not change this maths
        body: >-
          FlashAttention is an exact reordering of the same computation — tiling the matrix and
          combining partial softmaxes with a running max — so it never materialises the T×T matrix.
          The output is bit-comparable up to floating-point associativity. It belongs to the
          implementation, not to this design space, which is why it is not a variant here.
    usedBy:
      - essentially every transformer
  - id: softcap
    label: Logit softcapping
    full: Tanh soft-capping of attention logits
    year: 2024
    role: refinement
    tagline: Squash logits through tanh before the softmax
    concepts:
      - id: softcap-score-range
        label: Compress only the extremes
        kind: method
        summary: Tanh leaves ordinary attention scores nearly linear and bounds pathological ones.
        detail:
          - >-
            The cap is applied after masking and scaling but before softmax. It prevents a single
            dot product from making an attention row effectively one-hot too early in training.
      - id: softcap-score-selectivity
        label: Stability trades some sharpness
        kind: tradeoff
        summary: The strongest score gaps are deliberately flattened near the cap.
        detail:
          - >-
            This can protect gradients and numerical range, but it also changes how selectively a
            head can focus. The cap value is therefore part of the model architecture.
    math:
      - title: Soft cap
        tex: e' = c \cdot \tanh\!\left(\frac{e}{c}\right)
        worked:
          - tex: e = 10,\ c = 50 \;\Rightarrow\; e' = 9.93
            caption: ordinary logits pass through nearly unchanged
          - tex: e = 200,\ c = 50 \;\Rightarrow\; e' = 50.0
            caption: a runaway logit is pinned at the cap
    usedBy:
      - Gemma-2
      - Grok-1
  - id: differential
    label: Differential
    full: Differential Attention
    year: 2024
    role: frontier
    tagline: Two softmax maps, subtracted, to cancel common-mode noise
    paper:
      title: Differential Transformer
      url: https://arxiv.org/abs/2410.05258
      authors: Microsoft Research
    concepts:
      - id: differential-two-maps
        label: Pair attention maps
        kind: method
        summary: Two independently projected attention distributions are formed over the same values.
        detail:
          - >-
            Their weighted difference acts like a learned common-mode rejection mechanism. The
            second map is not merely another head; it is paired with the first in the output.
      - id: differential-signed-output
        label: Allow negative evidence
        kind: tradeoff
        summary: Subtraction can suppress shared noise but yields signed, non-probabilistic combined weights.
        detail:
          - >-
            Each constituent map is a normal softmax, yet their difference no longer sums to one
            or stays non-negative. Subsequent scaling and residual handling must accommodate that.
    math:
      - title: The difference
        tex: >-
          \text{DiffAttn}(X) = \left[\text{softmax}\!\left(\frac{Q_1K_1^\top}{\sqrt{d}}\right) -
          \lambda\,\text{softmax}\!\left(\frac{Q_2K_2^\top}{\sqrt{d}}\right)\right]V
        note: >-
          Rows no longer sum to one, and weights may be negative — attention can now actively
          subtract a token's contribution, which ordinary softmax attention cannot express at all.
    usedBy:
      - DIFF Transformer (research)
  - id: sigmoid-attn
    label: Sigmoid attention
    full: Element-wise sigmoid attention
    year: 2024
    role: frontier
    tagline: Score each pair independently — no competition between tokens
    concepts:
      - id: sigmoid-attn-independent-edges
        label: Gate each edge independently
        kind: method
        summary: A sigmoid decides whether each query-key pair contributes without row-wise competition.
        detail:
          - >-
            Adding a relevant key does not force probability mass away from another relevant key.
            A query can attend weakly to nothing or strongly to many positions at once.
      - id: sigmoid-attn-scale-control
        label: Control row magnitude separately
        kind: pitfall
        summary: Sigmoid weights do not sum to one, so output scale can grow with visible context.
        detail:
          - >-
            Bias initialisation and normalization are needed to keep early rows near a useful scale.
            Replacing softmax alone without that control changes activation magnitudes substantially.
    math:
      - title: Independent gating
        tex: p_{ij} = \sigma\!\left(\frac{e_{ij}}{\sqrt{d_h}} + b\right)
        note: >-
          b is initialised near −log(T) so early-training rows sum to roughly one, matching
          softmax's scale. Without it the output magnitude grows with sequence length and training
          destabilises.
    usedBy:
      - research (Apple, and related lines)
---

## role

The masked scores are unbounded reals. Something must convert them into mixing weights, and that conversion decides how sharply a head can focus, whether it can decline to attend at all, and how numerically stable training is.

Softmax has held this position essentially unchallenged since 2017, so the space is shallower than elsewhere. What variation exists is about two of its properties: it **cannot output zero mass** (every token gets some weight, however small) and it **cannot bound its input** (large logits saturate it and kill the gradient).

## softmax

Exponentiate each score and divide by the row sum. The result is a probability distribution over the visible tokens, so attention is a convex combination of values — the output can never leave the convex hull of what it is mixing, which is a useful stability property.

The `1/√d_h` scale inside is load-bearing. Dot products of `d_h`-dimensional vectors have variance proportional to `d_h`, so without it the logits grow with head dimension, the softmax saturates, and the gradient vanishes before training starts.

## softcap

Pass the logits through `c · tanh(e/c)`. Values well inside `±c` are almost untouched; values outside are smoothly compressed toward the cap. A differentiable ceiling.

Gemma-2 applies it in two places — attention logits at `c = 50`, final output logits at `c = 30`. It solves the same problem QK-norm does, from the other end, and notably Gemma-3 dropped it in favour of QK-norm alone.

A config with `attn_logit_softcapping` is on this branch. It is worth knowing because it is incompatible with stock FlashAttention: the cap has to be applied inside the kernel, and running an uncapped kernel on a capped model silently changes the outputs.

### fixes

Attention logits can grow large enough to saturate softmax and destabilise training, and clipping them hard is not differentiable.

## differential

Compute two independent attention maps and subtract one from the other, scaled by a learned `λ`. The reasoning is borrowed from differential amplifiers in electronics: noise that appears in both maps is common-mode and cancels, while genuine signal — which differs between them — survives.

Reported to substantially improve long-context retrieval and reduce hallucination, and to shrink the activation outliers that make quantisation difficult.

The cost is direct: two attention maps per head. The paper compensates by halving the head count, keeping FLOPs roughly constant.

### fixes

Softmax spreads a small amount of probability across every irrelevant token. Over thousands of tokens that noise floor sums to real mass, drowning the signal in long context.

## sigmoid-attn

Apply a sigmoid per element instead of a softmax per row. Each (query, key) pair is judged on its own merits, weights fall in `[0, 1]` independently, and a row may sum to zero — meaning "nothing here is relevant", which softmax cannot say.

Also removes the row-wise reduction, which is the sequential bottleneck in attention kernels. Apple's analysis shows it is a universal approximator like softmax attention and trains stably given an appropriate bias initialisation.

### fixes

Softmax makes tokens compete for a fixed unit of mass, so adding a relevant token necessarily reduces the weight on every other one, and a head with nothing to attend to still must attend to something.

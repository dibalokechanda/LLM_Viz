---
id: positional
label: Positional Encoding
ordinal: '3'
icon: sync
slot: layer
tagline: How the model learns that order matters
io:
  in: q, k [B, n_head, T, d_head]
  out: q, k with position applied
defaultVariant: rope
caveat: >-
  Some models use two schemes at once — Llama-4 interleaves NoPE and RoPE layers, and DeepSeek
  splits each head into a positional and a position-free half. A single selection here is a
  simplification those models genuinely break.
lineage:
  - from: sinusoidal
    to: learned
    kind: replaces
    label: just learn the table
  - from: sinusoidal
    to: relative
    kind: fixes
    label: absolute indices do not generalise
  - from: learned
    to: relative
    kind: fixes
    label: hard ceiling at max_position_embeddings
  - from: relative
    to: t5-bias
    kind: derives
    label: bucket the distances, share a scalar
  - from: relative
    to: rope
    kind: derives
    label: rotate instead of add
  - from: t5-bias
    to: alibi
    kind: fixes
    label: drop the learning — a fixed slope works
  - from: rope
    to: linear-interp
    kind: fixes
    label: squeeze positions into the trained range
  - from: linear-interp
    to: dynamic-ntk
    kind: fixes
    label: stop damaging short-context quality
  - from: dynamic-ntk
    to: yarn
    kind: fixes
    label: scale by frequency band, not uniformly
  - from: rope
    to: llama3-rope
    kind: derives
    label: piecewise scaling on low frequencies
  - from: yarn
    to: llama3-rope
    kind: inspires
    label: same insight, simpler schedule
  - from: rope
    to: nope
    kind: replaces
    label: causal masking already leaks position
variants:
  - id: sinusoidal
    label: Sinusoidal
    full: Fixed sinusoidal position embeddings
    year: 2017
    role: origin
    tagline: Add a fixed sine/cosine pattern to the embedding
    paper:
      title: Attention Is All You Need
      url: https://arxiv.org/abs/1706.03762
    concepts:
      - id: sinusoidal-frequency-bank
        label: A bank of fixed frequencies
        kind: formula
        summary: Sine and cosine channels assign a deterministic phase pattern to every index.
        detail:
          - >-
            Slow channels vary over long distances and fast channels distinguish nearby positions.
            The table can be computed for lengths not encountered during training.
      - id: sinusoidal-additive-position
        label: Position enters before attention
        kind: tradeoff
        summary: The encoding is added to token embeddings, so content and absolute position share one stream.
        detail:
          - >-
            No parameters are learned for positions, but the model must disentangle location from
            lexical features in subsequent projections.
    math:
      - title: The encoding
        tex: >-
          PE_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad PE_{(p, 2i+1)} =
          \cos\!\left(\frac{p}{10000^{2i/d}}\right)
        note: >-
          The geometric frequency spread means low dimensions cycle every few tokens while high ones
          barely move across the whole sequence — a positional code at many scales at once. RoPE
          keeps exactly this frequency structure and changes only how it is applied.
    usedBy:
      - Transformer (2017)
      - early NMT models
  - id: learned
    label: Learned absolute
    full: Learned absolute position embeddings
    year: 2018
    role: legacy
    tagline: A trainable lookup table indexed by position
    concepts:
      - id: learned-position-table
        label: A row per training position
        kind: method
        summary: Position i selects a learned vector from a finite embedding table.
        detail:
          - >-
            The model can fit arbitrary position-specific behavior within its trained range. Unlike
            a formula, there is no meaningful row for an unseen index without resizing or adaptation.
      - id: learned-position-ceiling
        label: Context length is a parameter boundary
        kind: pitfall
        summary: The table's highest index is a hard architectural limit for an unmodified checkpoint.
        detail:
          - >-
            Extending the context requires creating new position rows and teaching the model how to
            use them; simply accepting longer token arrays is not equivalent.
    usedBy:
      - BERT
      - GPT-2
      - GPT-3
      - RoBERTa
      - ViT
  - id: relative
    label: Relative
    full: Relative position representations
    year: 2018
    role: branch
    tagline: Encode the distance between tokens, not their indices
    paper:
      title: Self-Attention with Relative Position Representations
      url: https://arxiv.org/abs/1803.02155
      authors: Shaw et al.
    concepts:
      - id: relative-distance-label
        label: Label pairs by distance
        kind: method
        summary: Attention receives a learned representation of i minus j rather than two absolute indices.
        detail:
          - >-
            The same relation, such as previous token, can be recognised wherever it appears. The
            distance term is injected into the score or value computation for each query-key pair.
      - id: relative-distance-range
        label: Bucket or clip long distances
        kind: tradeoff
        summary: Practical relative schemes share representations beyond a maximum distance.
        detail:
          - >-
            This controls parameter count and lets the model treat very large separations coarsely,
            but it removes an exact distinction between all far-away positions.
    math:
      - title: Relative term in the score
        tex: >-
          e_{ij} = \frac{(x_i W^Q)(x_j W^K + a_{ij}^K)^\top}{\sqrt{d_h}}, \qquad a_{ij} =
          w_{\text{clip}(i-j,\,k)}
        note: >-
          The clip is what keeps the parameter count finite — every distance past k shares one
          vector.
    usedBy:
      - Transformer-XL
      - DeBERTa
      - Music Transformer
  - id: t5-bias
    label: T5 bias
    full: Learned relative attention bias with log-spaced buckets
    year: 2019
    role: refinement
    tagline: One learned scalar per distance bucket, added to the logits
    concepts:
      - id: t5-bias-logit-offset
        label: Add a distance preference to logits
        kind: method
        summary: Each relative-distance bucket contributes one learned scalar per attention head.
        detail:
          - >-
            The bias is added after the QK dot product, so it changes which keys a query prefers
            without rotating or altering the residual-stream token vectors.
      - id: t5-bias-log-buckets
        label: Spend resolution near the query
        kind: tradeoff
        summary: Small distances get exact buckets while large distances are grouped logarithmically.
        detail:
          - >-
            Nearby word order is resolved precisely; distant context receives a coarser directional
            signal. This keeps the bias table small for long sequences.
    math:
      - title: Scalar bias
        tex: e_{ij} = \frac{q_i k_j^\top}{\sqrt{d_h}} + b_{\text{bucket}(i-j)}
        note: >-
          Cheap enough to survive into modern kernels, since adding a bias to logits fuses trivially
          into FlashAttention.
    usedBy:
      - T5
      - Flan-T5
      - UL2
  - id: alibi
    label: ALiBi
    full: Attention with Linear Biases
    year: 2021
    role: branch
    tagline: Subtract a distance penalty from the logits. No learning at all.
    paper:
      title: 'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation'
      url: https://arxiv.org/abs/2108.12409
      authors: Press et al.
    concepts:
      - id: alibi-head-slopes
        label: Give each head a distance slope
        kind: formula
        summary: A head-specific linear penalty is subtracted from scores as keys become more distant.
        detail:
          - >-
            Some heads remain local and others decay slowly, creating a fixed multi-scale bias
            without a position embedding table or learned relative vectors.
      - id: alibi-extrapolation-prior
        label: Continue the same rule past training length
        kind: tradeoff
        summary: The bias extrapolates because its formula is defined at every distance.
        detail:
          - >-
            This is a strong prior rather than a learned long-range representation. It can generalise
            cleanly to longer sequences, but cannot adapt its distance function per dataset.
    math:
      - title: The bias
        tex: >-
          e_{ij} = \frac{q_i k_j^\top}{\sqrt{d_h}} - m_h \cdot |i - j|, \qquad m_h =
          2^{-8h/n_{\text{head}}}
        note: >-
          For 8 heads the slopes are 1/2, 1/4, 1/8 … 1/256. The steep head sees only its immediate
          neighbourhood; the shallow head sees almost everything.
    figures:
      - kind: curve
        title: Logit penalty against distance, by head
        xLabel: distance |i − j|
        yLabel: penalty subtracted
        lines:
          - label: head 1 (m = 1/2)
            points:
              - - 0
                - 0
              - - 4
                - -2
              - - 8
                - -4
              - - 12
                - -6
              - - 16
                - -8
          - label: head 4 (m = 1/16)
            points:
              - - 0
                - 0
              - - 4
                - -0.25
              - - 8
                - -0.5
              - - 12
                - -0.75
              - - 16
                - -1
          - label: head 8 (m = 1/256)
            points:
              - - 0
                - 0
              - - 4
                - -0.016
              - - 8
                - -0.03
              - - 12
                - -0.05
              - - 16
                - -0.06
            dashed: true
        caption: >-
          Each head gets a different effective receptive field from one fixed rule — the diversity
          comes free, without any parameters.
    usedBy:
      - BLOOM
      - MPT
      - Falcon (early)
      - Baichuan-13B
  - id: rope
    label: RoPE
    full: Rotary Position Embedding
    year: 2021
    role: synthesis
    tagline: Rotate q and k by an angle proportional to position
    concepts:
      - id: paired-rotation
        label: Rotate pairs of features
        kind: method
        summary: RoPE treats adjacent query and key dimensions as two-dimensional planes and rotates them by position.
        detail:
          - >-
            Each feature pair gets its own frequency. Position is injected by a rotation, so vector
            length is preserved while phase changes predictably along the sequence.
      - id: relative-phase
        label: Dot products reveal relative distance
        kind: formula
        summary: The query-key dot product depends on the difference between positions, not only their absolute values.
        detail:
          - >-
            This relative phase relationship lets an attention head compare how far apart two tokens
            are while still using ordinary dot-product attention.
      - id: context-extension
        label: Long context needs a frequency policy
        kind: tradeoff
        summary: Extending RoPE means changing how its frequencies behave beyond the range seen in training.
        detail:
          - >-
            Linear interpolation, NTK-aware scaling, YaRN, and Llama-style scaling all protect a
            different balance between short-context behavior and long-context extrapolation.
    paper:
      title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding'
      url: https://arxiv.org/abs/2104.09864
      authors: Su et al.
    math:
      - title: Rotation, per dimension pair
        tex: >-
          \begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos p\theta_i &
          -\sin p\theta_i \\ \sin p\theta_i & \cos p\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\
          q_{2i+1} \end{pmatrix}, \qquad \theta_i = \beta^{-2i/d_h}
        where:
          - sym: p
            means: absolute position of the token
          - sym: \beta
            means: '`rope_theta` — 10 000 originally, 500 000 in Llama-3, up to 10⁷ in long-context models'
      - title: Why the dot product becomes relative
        tex: \langle R_m q,\; R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k
        note: >-
          Rotation matrices are orthogonal, so R_mᵀR_n = R_{n−m}. The absolute positions cancel and
          only the offset survives. This identity is the entire method.
      - title: What rope_theta controls
        tex: \lambda_i = \frac{2\pi}{\theta_i} = 2\pi\,\beta^{2i/d_h}
        worked:
          - tex: \beta = 10^4,\; i = d_h/2 \;\Rightarrow\; \lambda \approx 62{,}832
            caption: slowest pair completes one cycle in ~63k tokens
          - tex: \beta = 5\times10^5 \;\Rightarrow\; \lambda \approx 3.1\times10^6
            caption: 'Llama-3: slow pairs stretched far past the context, so they never wrap'
        note: >-
          Raising β stretches every wavelength. That is why "just increase rope_theta" is the
          crudest long-context fix — and why it degrades short-range precision, since the fast pairs
          stretch too.
    figures:
      - kind: curve
        title: Rotation angle against position, by dimension pair
        xLabel: token position
        yLabel: angle (radians)
        lines:
          - label: pair 0 (fast)
            points:
              - - 0
                - 0
              - - 1
                - 1
              - - 2
                - 2
              - - 3
                - 3
              - - 4
                - 4
              - - 5
                - 5
              - - 6
                - 6
          - label: pair 16 (mid)
            points:
              - - 0
                - 0
              - - 1
                - 0.3
              - - 2
                - 0.6
              - - 3
                - 0.9
              - - 4
                - 1.2
              - - 5
                - 1.5
              - - 6
                - 1.8
          - label: pair 63 (slow)
            points:
              - - 0
                - 0
              - - 1
                - 0.02
              - - 2
                - 0.04
              - - 3
                - 0.06
              - - 4
                - 0.08
              - - 5
                - 0.1
              - - 6
                - 0.12
            dashed: true
        caption: >-
          Fast pairs wrap within a handful of tokens and encode fine local offset. Slow pairs barely
          turn across the whole context and encode where you are globally. Long-context methods
          differ only in which of these bands they stretch.
    code:
      - title: PyTorch
        language: python
        code: |-
          def rope(x, pos, theta=10000.0):
              # x: [B, n_head, T, d_head]
              d = x.shape[-1]
              freqs = theta ** (-torch.arange(0, d, 2, device=x.device).float() / d)
              ang = pos[:, None] * freqs[None, :]              # [T, d/2]
              cos, sin = ang.cos(), ang.sin()

              x1, x2 = x[..., 0::2], x[..., 1::2]              # the 2D pairs
              return torch.stack([x1 * cos - x2 * sin,
                                  x1 * sin + x2 * cos], dim=-1).flatten(-2)

          # Applied to q and k only — never to v. Values carry content, not position,
          # and rotating them would corrupt what attention copies forward.
        note: >-
          HF implementations use a "half rotation" layout — pairing dim i with dim i + d/2 rather
          than adjacent dims. Mathematically equivalent, but weights are not interchangeable between
          the two conventions, which is a common porting bug.
    cost:
      - label: rope_theta
        value: '{ropeTheta == 0 ? "not set" : num(ropeTheta)}'
        note: higher stretches all wavelengths
      - label: Slowest wavelength
        value: '{ropeTheta == 0 ? "—" : num(round(2 * pi * ropeTheta)) + " tokens"}'
        note: if this is below the context length, the slowest pair wraps and position becomes ambiguous
        key: true
      - label: Trained context
        value: '{num(ctx)}'
      - label: Added parameters
        value: '0'
    usedBy:
      - Llama (all)
      - Mistral
      - Qwen
      - Gemma
      - DeepSeek
      - GPT-NeoX
      - PaLM
  - id: linear-interp
    label: Position Interpolation
    full: Linear position interpolation
    year: 2023
    role: refinement
    tagline: Divide positions by a factor so they fit the trained range
    paper:
      title: Extending Context Window of Large Language Models via Position Interpolation
      url: https://arxiv.org/abs/2306.15595
    concepts:
      - id: interpolation-coordinate-compression
        label: Compress inference coordinates
        kind: method
        summary: Long-context indices are divided by a scale factor before their RoPE angles are computed.
        detail:
          - >-
            A position at 16k can be presented to a model trained to 4k as coordinate 4k. Fine-tuning
            teaches the model to tolerate the denser spacing of these interpolated phases.
      - id: interpolation-local-resolution
        label: Long range trades local phase resolution
        kind: tradeoff
        summary: All frequency bands are compressed, including bands that already behaved well at short range.
        detail:
          - >-
            The method avoids unseen rotation angles but makes nearby positions less separable in
            phase space. Later scaling methods target only the problematic bands.
    math:
      - title: Interpolation
        tex: p' = \frac{p}{s}, \qquad s = \frac{L_{\text{new}}}{L_{\text{train}}}
        note: >-
          Equivalent to multiplying every θ by 1/s — a uniform stretch across all frequency bands,
          including the fast ones that did not need it.
    usedBy:
      - Llama-2 long-context fine-tunes
      - Code Llama (base method)
  - id: dynamic-ntk
    label: Dynamic NTK
    full: NTK-aware / dynamic base scaling
    year: 2023
    role: refinement
    tagline: Scale rope_theta with the sequence, not the positions
    concepts:
      - id: dynamic-ntk-base-change
        label: Change the RoPE base, not the index
        kind: method
        summary: Dynamic NTK scaling increases rope_theta as the current sequence grows.
        detail:
          - >-
            Low-frequency rotation rates stretch for a long sequence while the positions themselves
            retain their integer coordinates. The adjustment can be chosen at runtime from length.
      - id: dynamic-ntk-short-context
        label: Preserve the short-context regime
        kind: tradeoff
        summary: The unscaled base remains available when context stays near the original training length.
        detail:
          - >-
            This avoids applying long-context compression universally, but makes positional behavior
            depend on the sequence length used for the current forward pass.
    math:
      - title: Base scaling
        tex: >-
          \beta' = \beta \cdot \left(\frac{s \cdot L_{\text{cur}}}{L_{\text{train}}} - (s -
          1)\right)^{\frac{d_h}{d_h - 2}}
        note: >-
          The exponent is chosen so the fastest pair is left essentially untouched while the slowest
          is stretched by the full factor.
    usedBy:
      - Qwen (early long-context)
      - many community fine-tunes
      - CodeLlama variants
  - id: yarn
    label: YaRN
    full: Yet another RoPE extensioN
    year: 2023
    role: synthesis
    tagline: Interpolate slow bands, leave fast bands alone, rescale attention
    paper:
      title: 'YaRN: Efficient Context Window Extension of Large Language Models'
      url: https://arxiv.org/abs/2309.00071
      authors: Peng et al.
    concepts:
      - id: yarn-frequency-regions
        label: Treat frequency bands differently
        kind: method
        summary: YaRN leaves short-wavelength channels intact, interpolates long-wavelength channels, and blends between them.
        detail:
          - >-
            The piecewise policy respects local phase resolution while extending the slow bands that
            fail first outside the training range.
      - id: yarn-attention-temperature
        label: Correct attention sharpness
        kind: tradeoff
        summary: Frequency scaling is paired with an attention-temperature adjustment.
        detail:
          - >-
            Altering RoPE phases changes QK score statistics. YaRN explicitly compensates for that
            distribution shift, which adds tuning choices beyond a single extension factor.
    math:
      - title: Per-band interpolation
        tex: >-
          \theta'_i = (1 - \gamma_i)\frac{\theta_i}{s} + \gamma_i \theta_i, \qquad \gamma_i =
          \text{ramp}\!\left(\frac{\lambda_i}{L_{\text{train}}}\right)
        where:
          - sym: \gamma_i = 1
            means: fast band — untouched
          - sym: \gamma_i = 0
            means: slow band — fully interpolated
      - title: Attention temperature correction
        tex: \sqrt{1/t} = 0.1 \ln(s) + 1
        note: >-
          Applied as a scale on the logits. An empirical fit, but consistently worth a point or two
          of perplexity at high extension factors.
    usedBy:
      - Qwen2/3 long-context
      - Nous-Hermes long variants
      - DeepSeek-V2 (128k)
  - id: llama3-rope
    label: Llama-3 RoPE
    full: Piecewise low-frequency RoPE scaling
    year: 2024
    role: refinement
    tagline: YaRN's insight, stripped to three constants
    concepts:
      - id: llama3-rope-three-regions
        label: Keep, scale, or blend
        kind: method
        summary: Wavelength thresholds divide RoPE dimensions into unchanged, scaled, and transition regions.
        detail:
          - >-
            The schedule captures the useful core of band-aware scaling with a small set of constants.
            Low-frequency dimensions are slowed while high-frequency local-detail dimensions remain intact.
      - id: llama3-rope-configuration-contract
        label: Extension lives in configuration
        kind: pitfall
        summary: The exact thresholds and factor are checkpoint-specific RoPE parameters.
        detail:
          - >-
            Loading a model with a generic RoPE implementation but the wrong scaling configuration
            silently changes all Q/K rotations. Correct long-context behavior requires those values.
    math:
      - title: The schedule
        tex: >-
          \theta'_i = \begin{cases} \theta_i & \lambda_i < \frac{L}{\alpha_{\text{high}}} \\[4pt]
          \theta_i / s & \lambda_i > \frac{L}{\alpha_{\text{low}}} \\[4pt] \text{linear blend} &
          \text{otherwise}\end{cases}
        worked:
          - tex: s = 8,\ \alpha_{\text{low}} = 1,\ \alpha_{\text{high}} = 4
            caption: 'Llama-3.1: 8k → 128k'
    usedBy:
      - Llama-3.1
      - Llama-3.2
      - Llama-3.3
  - id: nope
    label: NoPE
    full: No positional encoding
    year: 2023
    role: frontier
    tagline: Add nothing — the causal mask already encodes position
    paper:
      title: The Impact of Positional Encoding on Length Generalization in Transformers
      url: https://arxiv.org/abs/2305.19466
    concepts:
      - id: nope-order-from-mask
        label: Causality leaks an order signal
        kind: idea
        summary: Earlier tokens have a different visibility pattern from later tokens even without position vectors.
        detail:
          - >-
            A causal mask makes each token's accessible prefix length depend on its location. Deep
            layers can exploit that asymmetry, especially when other architectural cues are present.
      - id: nope-no-distance-coordinate
        label: No explicit distance representation
        kind: tradeoff
        summary: NoPE removes a positional prior but gives attention no direct coordinate or relative-distance feature.
        detail:
          - >-
            This can avoid a mismatched encoding assumption, yet tasks requiring precise order or
            long-range distance reasoning may need RoPE or another positional mechanism elsewhere.
    usedBy:
      - Llama-4 (interleaved with RoPE)
      - research decoders
---

## role

Attention is a set operation. Permute the input tokens and — with no positional information — the output permutes identically. "Dog bites man" and "man bites dog" produce the same representation. Something has to break that symmetry.

Two questions organise the whole space. **Absolute or relative?** Does a token know it is at index 7, or only that it is three places after another token? Relative won, because language is overwhelmingly about relative structure and because relative schemes extrapolate past their training length far better.

And **where is it injected?** Early schemes add a vector to the embedding once, at the bottom. Modern ones intervene inside attention on every layer — which is why this block sits inside the repeating trunk rather than in the input stage.

## sinusoidal

Add a fixed vector to each token embedding at the bottom of the stack. Dimension `2i` gets a sine at wavelength `10000^(2i/d)`, dimension `2i+1` the matching cosine, so each dimension pair oscillates at its own frequency and the combination is unique per position.

The stated hope was extrapolation: since the function is defined for any position, a model could in principle handle sequences longer than it saw. In practice it does not — the attention patterns learned over trained positions simply do not transfer.

## learned

A `max_position × d_model` embedding matrix, added to the token embedding. This is what BERT and GPT-2 use, and it is the reason those models have an absolute, unnegotiable context limit: position 1025 in GPT-2 has no row in the table.

A config with `max_position_embeddings` and no `rope_theta` is almost always this. It works well within the trained range and does not work at all outside it.

### fixes

Why hand-design a function when a table can be learned along with everything else?

## relative

Add a learned vector indexed by `i − j` — the offset between query and key — inside the attention computation rather than at the embedding. Distances beyond a clip range share a bucket.

This is the conceptual turn the rest of the field is built on. Position becomes a property of a *pair* of tokens, which is both what language actually cares about and what allows a model to handle a length it never trained on.

### fixes

Absolute schemes make the model relearn "the previous word" separately at every position, and give it nothing at positions it never saw.

## t5-bias

Reduce the relative term to a single scalar added to the attention logit, indexed by a bucketed distance. Buckets are log-spaced: adjacent tokens get their own bucket, distant ones are pooled coarsely, which mirrors how precisely distance actually matters.

The bias is shared across layers in T5 — computed once, added everywhere.

### fixes

Shaw's per-distance *vectors* are expensive and mostly redundant — a scalar bias per bucket captures nearly the same thing.

## alibi

No embeddings, no learned parameters. Subtract `m · |i − j|` from every attention logit, where `m` is a fixed per-head slope from a geometric sequence. Nearby tokens are penalised little, distant ones heavily, and each head gets a different slope so heads specialise at different ranges.

Because the rule is defined for every distance, ALiBi extrapolates well — a model trained at 1k handles 2k+ with a modest quality cost, which was the paper's headline claim.

Its weakness is the same as its strength: the linear penalty is a hard-coded prior that attention *should* decay with distance. When a task genuinely needs a token 5000 positions back, ALiBi has already suppressed it.

### fixes

Even bucketed relative biases have to be learned, and learned biases are only meaningful over distances that appeared in training.

## rope

Treat each pair of dimensions in `q` and `k` as a point in a 2D plane and rotate it by `p·θ`, where `p` is the absolute position and `θ` is that pair's frequency. Do this to queries and keys before the dot product.

The elegance is what happens next. Rotating both vectors and then taking their inner product yields something that depends only on `m − n`, the *relative* offset — because a rotation by `mθ` dotted against a rotation by `nθ` is a rotation by `(m−n)θ`. You apply position absolutely and the attention score reads it relatively, with no extra terms and no kernel changes.

This is why RoPE won. It is relative in effect, absolute in implementation, adds no parameters, and leaves `q·kᵀ` a plain matmul that FlashAttention can consume unmodified.

The frequency spread is inherited from the sinusoidal scheme: low dimension pairs rotate fast (fine local position), high pairs rotate slowly (coarse global position). Every long-context extension method works by manipulating that spectrum, which is why they all belong to this subtree.

### fixes

Additive relative schemes need a custom attention kernel and add terms that do not interact with the content signal. Rotation gets relative position *for free* out of the dot product itself.

## linear-interp

Rather than letting position 8000 produce an unseen angle, divide every position by the extension factor so 8000 is treated as 2000. All angles stay inside the trained range; positions are simply packed more densely.

A few hundred fine-tuning steps recover most of the quality. Crude, and it uniformly degrades the fine-grained local resolution the fast dimension pairs were providing — which is precisely what NTK and YaRN set out to repair.

### fixes

A RoPE model at 2× its trained context sees rotation angles it has never encountered, and quality collapses immediately.

## dynamic-ntk

Instead of dividing positions, raise the base `β`. Because wavelength depends on β raised to a power of the dimension index, this stretches slow pairs a great deal and fast pairs barely at all — exactly the distribution you want, since only the slow pairs were running out of range.

"Dynamic" adds the second half: recompute the scale from the *current* sequence length at each step, so short sequences use the original β and pay nothing. That removes interpolation's short-context penalty entirely and needs no fine-tuning at all.

### fixes

Linear interpolation squeezes high-frequency pairs that were perfectly fine, damaging local resolution even at short lengths.

## yarn

Split the dimension pairs into three groups by wavelength. Pairs whose wavelength is much shorter than the trained context are left completely alone — they already see every angle. Pairs whose wavelength exceeds the context are fully interpolated. Between them, a ramp.

YaRN adds a second correction that is easy to miss: interpolation changes the *entropy* of the attention distribution, so it also rescales the attention temperature by a factor tied to the extension. That correction is a meaningful part of why YaRN beats plain NTK.

The result is the strongest RoPE extension in wide use — 32× extension with modest fine-tuning — and it appears in configs as `rope_scaling: {"rope_type": "yarn", ...}`.

### fixes

NTK scaling stretches frequencies by a smooth formula, but the right treatment is not smooth — some bands should be untouched and some fully interpolated.

## llama3-rope

Keep only the band-selective part. Wavelengths below a low-frequency threshold pass through unchanged, wavelengths above a high threshold are divided by the scale factor, and a linear blend covers the middle. No temperature correction, three constants.

This is what `"rope_type": "llama3"` means in a config, and it is how Llama-3.1 reaches 128k from an 8k-trained base.

### fixes

YaRN works but carries a ramp function, a temperature correction, and four hyperparameters. Most of the benefit comes from one idea.

## nope

A decoder-only transformer with a causal mask is *not* permutation-invariant, even with no positional encoding. Token 1 attends to one position, token 5 attends to five. That difference in attention-distribution entropy is itself a positional signal, and the model learns to read it.

NoPE decoder models match or beat explicit schemes at length generalisation on several benchmarks. The practical catch is that the signal is weak and indirect, so pure NoPE trains slower and is fragile at scale.

Its real home is hybrids. Llama-4 interleaves NoPE layers with RoPE layers — the RoPE layers handle local precision while the NoPE layers, unconstrained by any wavelength, carry very long-range structure. This is one of the cases where "one variant per block" genuinely misdescribes the model.

### fixes

Every positional scheme is a prior imposed on the model. If the architecture already leaks position, the prior is unnecessary constraint.

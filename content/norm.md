---
id: norm
label: Normalization
ordinal: '4'
icon: scale
slot: layer
tagline: Keeping activations in a range gradients can survive
io:
  in: x [B, T, d_model]
  out: x normalised, same shape
defaultVariant: rmsnorm
lineage:
  - from: layernorm
    to: rmsnorm
    kind: fixes
    label: the mean subtraction does nothing
  - from: layernorm
    to: qk-norm
    kind: derives
    label: normalise q and k, not just the stream
  - from: rmsnorm
    to: qk-norm
    kind: combines
    label: RMSNorm applied inside attention
  - from: rmsnorm
    to: dyt
    kind: replaces
    label: no statistics at all — just a squashing function
variants:
  - id: layernorm
    label: LayerNorm
    full: Layer Normalization
    year: 2016
    role: origin
    tagline: Centre and rescale each token by its own statistics
    paper:
      title: Layer Normalization
      url: https://arxiv.org/abs/1607.06450
      authors: Ba, Kiros, Hinton
    concepts:
      - id: layernorm-per-token-stats
        label: Statistics stay within a token
        kind: method
        summary: Mean and variance are reduced over channels, independently for every token position.
        detail:
          - >-
            LayerNorm does not mix examples or sequence positions. It recentres a token vector
            and divides by its own standard deviation, which makes the operation stable for small
            batches and variable sequence lengths.
      - id: layernorm-affine-recovery
        label: Learnable recovery
        kind: tradeoff
        summary: Per-channel gain and bias let the model restore useful scales and offsets.
        detail:
          - >-
            Normalisation removes raw magnitude and mean, then affine parameters give later
            layers a learned way to reintroduce them. The two reductions cost more than the
            simpler RMSNorm alternative.
    math:
      - title: LayerNorm
        tex: >-
          \text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu
          = \frac{1}{d}\sum_i x_i, \quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
        note: >-
          Two passes over the feature vector — one for μ, one for σ² — plus two learned vectors.
          Every part of this was assumed necessary for six years.
    usedBy:
      - Transformer (2017)
      - BERT
      - GPT-2
      - GPT-3
      - OPT
  - id: rmsnorm
    label: RMSNorm
    full: Root Mean Square Layer Normalization
    year: 2019
    role: refinement
    tagline: Drop the mean subtraction — only the scale mattered
    concepts:
      - id: rms-measure
        label: Normalize vector scale
        kind: formula
        summary: RMSNorm divides by root mean square magnitude rather than centering and scaling by variance.
        detail:
          - >-
            The operation keeps the representation at a controlled scale while avoiding the mean
            reduction used by LayerNorm.
      - id: learned-gain
        label: Keep a learned per-feature gain
        kind: method
        summary: A learned scale vector lets the model restore useful feature-wise amplitude after normalization.
        detail:
          - >-
            RMSNorm removes only global magnitude information. The learned gain gives the network a
            direct way to set the scale of individual features again.
      - id: numerical-floor
        label: Epsilon protects small activations
        kind: pitfall
        summary: A small epsilon keeps the denominator well behaved when the activation magnitude is near zero.
        detail:
          - >-
            The epsilon value is a configuration detail with real low-precision consequences. A
            faithful implementation must use the checkpoint's normalization epsilon.
    paper:
      title: Root Mean Square Layer Normalization
      url: https://arxiv.org/abs/1910.07467
      authors: Zhang & Sennrich
    math:
      - title: RMSNorm
        tex: \text{RMS}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
        note: No μ, no β. Compare against LayerNorm above — the deletions are the whole contribution.
    code:
      - title: PyTorch
        language: python
        code: |-
          class RMSNorm(nn.Module):
              def __init__(self, d, eps=1e-6):
                  super().__init__()
                  self.weight = nn.Parameter(torch.ones(d))
                  self.eps = eps

              def forward(self, x):
                  # float32 for the reduction even under bf16 training: the sum of
                  # squares over d_model terms overflows bf16's range at scale.
                  dtype = x.dtype
                  x = x.float()
                  x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                  return self.weight * x.to(dtype)
        note: >-
          The upcast is not optional. Skipping it is a classic source of NaNs partway through a long
          bf16 run.
    cost:
      - label: Params per norm
        value: '{num(dModel)}'
        note: one gain vector; LayerNorm would need twice this
      - label: Norms in the model
        value: '{num(nLayer * 2 + 1)}'
        note: two per layer under pre-norm, plus a final one
      - label: Total norm params
        value: '{num(dModel * (nLayer * 2 + 1))}'
        key: true
    usedBy:
      - Llama (all)
      - Mistral
      - Qwen
      - Gemma
      - DeepSeek
      - T5 (variant)
  - id: qk-norm
    label: QK-Norm
    full: Query–key normalization
    year: 2023
    role: branch
    tagline: Normalise q and k before the dot product
    concepts:
      - id: qk-norm-unit-directions
        label: Compare directions, bound magnitude
        kind: formula
        summary: Queries and keys are normalised before their dot product forms an attention logit.
        detail:
          - >-
            Unit-scale Q and K make the score primarily a directional similarity. The usual
            1/√d_head factor then operates in a controlled range instead of fighting unbounded
            projection norms.
      - id: qk-norm-softmax-margin
        label: Protect the softmax margin
        kind: pitfall
        summary: Bounding Q/K norms prevents a few logits from saturating softmax during training.
        detail:
          - >-
            Residual-stream normalisation cannot directly constrain the independently learned Q
            and K projections. Applying the guard at the score source preserves gradient signal
            in large, high-learning-rate runs.
    math:
      - title: Normalised scores
        tex: e_{ij} = \frac{\text{Norm}(q_i) \cdot \text{Norm}(k_j)^\top}{\sqrt{d_h}}
        note: >-
          With full L2 normalisation this is cosine similarity, bounded in [−1, 1] before scaling —
          a hard ceiling on the logit, not a soft discouragement.
    usedBy:
      - Gemma-2/3
      - Chameleon
      - ViT-22B
      - OLMo-2
      - Qwen3
  - id: dyt
    label: DyT
    full: Dynamic Tanh
    year: 2025
    role: frontier
    tagline: Replace normalisation with a learned tanh squash
    paper:
      title: Transformers without Normalization
      url: https://arxiv.org/abs/2503.10622
    concepts:
      - id: dyt-pointwise-transform
        label: A pointwise learned squash
        kind: method
        summary: DyT applies learned scale, slope, and offset around tanh without feature reductions.
        detail:
          - >-
            Each channel is transformed independently, so the operation can be implemented as
            elementwise kernels. Unlike a norm, it never measures the current token's aggregate
            statistics.
      - id: dyt-stability-assumption
        label: Replace a guarantee with a learned constraint
        kind: tradeoff
        summary: DyT is cheaper to parallelise but does not explicitly set a token's variance.
        detail:
          - >-
            The bounded nonlinearity can control outliers when trained well, but the architecture
            relies on learned parameters rather than the deterministic scale correction supplied
            by LayerNorm or RMSNorm.
    math:
      - title: DyT
        tex: \text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta
    usedBy:
      - research only
---

## role

Stacking 80 residual blocks means the activation scale compounds. Without intervention it drifts by orders of magnitude across depth, and a single badly-scaled layer produces gradients that either vanish or blow the run up.

Normalisation pins the scale at every layer. The design space is narrow — there is really only one question, *how much of the statistics do you actually need?* — but the answer turned out to be "less than everyone assumed", and removing the unnecessary part bought a measurable speedup at zero quality cost.

## layernorm

For each token independently, subtract the mean across the `d_model` features and divide by the standard deviation, then apply a learned gain and bias. Unlike BatchNorm the statistics come from one token, so nothing depends on batch size or on other sequences — which is what makes it usable for variable-length autoregressive text.

A config carrying `layer_norm_eps` or `layer_norm_epsilon` is using this.

## rmsnorm

Divide by the root mean square and skip the centring entirely. One reduction instead of two, one learned vector instead of two, and measured quality that matches LayerNorm across the board.

The claim is that the benefit of normalisation is **re-scaling invariance**, not re-centring invariance. Once the network is scale-stable, whatever the mean happens to be gets absorbed by the following linear layer.

Roughly a 7–10% end-to-end training speedup for free, which is why every model since Llama-1 uses it. `rms_norm_eps` in a config means this.

### fixes

LayerNorm computes and subtracts a mean that ablations show contributes essentially nothing to quality, at the cost of an extra reduction pass on every token in every layer.

## qk-norm

Apply a norm to the query and key vectors themselves, immediately before computing scores. This bounds the logit magnitude structurally rather than hoping it stays small.

It has become standard in very large or very long training runs precisely because loss spikes at scale are expensive: a diverged run at 10²⁵ FLOPs costs real money, and QK-norm removes one of the main causes.

### fixes

Attention logits can grow without bound during training. Once they do, softmax saturates, gradients vanish and the run diverges — a common large-scale failure that normalising the residual stream does not prevent.

## dyt

The observation is that a trained LayerNorm's input–output mapping looks remarkably like a scaled `tanh`. So use one: `DyT(x) = γ ⊙ tanh(αx) + β`, with `α` learned. Entirely pointwise — no statistics, no reduction, no cross-feature dependency.

Reported to match normalised transformers on language and vision at equal budget. Genuinely new, and included here because it is the first credible proposal to remove normalisation rather than refine it — but adoption in shipped models is still negligible.

### fixes

Every norm needs a reduction across the feature dimension, which is a synchronisation point that does not parallelise well and cannot be fused as freely as a pointwise op.

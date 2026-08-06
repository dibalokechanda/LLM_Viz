---
id: lmhead
label: Output Head
ordinal: '12'
icon: brackets
slot: output
tagline: The hidden state becomes a score for every token in the vocabulary
io:
  in: h [B, T, d_model]
  out: logits [B, T, vocab]
defaultVariant: untied-head
lineage:
  - from: untied-head
    to: tied-head
    kind: derives
    label: reuse the embedding matrix
  - from: untied-head
    to: softcap-head
    kind: fixes
    label: bound runaway output logits
  - from: untied-head
    to: mtp
    kind: derives
    label: predict several tokens, not one
variants:
  - id: untied-head
    label: Untied head
    full: Independent output projection
    year: 2017
    role: origin
    concepts:
      - id: vocabulary-projection
        label: Score every vocabulary item
        kind: formula
        summary: The output head projects the final hidden state into one logit for each vocabulary token.
        detail:
          - >-
            At every decoding step, the head multiplies a d_model-wide hidden state by a
            vocabulary-by-model-width matrix. The resulting logits are the raw choices for the next token.
      - id: independent-geometry
        label: Keep output weights independent
        kind: method
        summary: An untied head learns a separate output matrix instead of reusing the input embedding table.
        detail:
          - >-
            Input lookup and output prediction can therefore organize token space differently. That
            freedom costs one additional vocabulary-sized matrix.
      - id: vocabulary-matmul
        label: Vocabulary size sets generation cost
        kind: metric
        summary: The final matrix multiplication scales with vocabulary size and occurs for every generated token.
        detail:
          - >-
            On a large vocabulary, the output projection can be a meaningful part of decode-time
            FLOPs even when its parameter count is modest relative to the transformer trunk.
    tagline: Its own vocab × d_model matrix
    cost:
      - label: Head parameters
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
      - label: FLOPs per generated token
        value: '{fixed(2 * vocab * dModel / 1e9, 2)} GFLOP'
        note: for one token — compare against the whole trunk below
        key: true
      - label: Trunk FLOPs per token
        value: '{fixed(2 * 12 * nLayer * dModel * dModel / 1e9, 2)} GFLOP'
        note: approximate — 12·L·d² is the usual rule of thumb
    usedBy:
      - GPT-3
      - Llama-3-8B/70B
      - Mistral
  - id: tied-head
    label: Tied head
    full: Output head tied to the input embedding
    year: 2016
    role: refinement
    tagline: Transpose of the embedding matrix
    concepts:
      - id: tied-head-dual-use
        label: Read and score with one table
        kind: method
        summary: The output projection reuses the input embedding weights transposed.
        detail:
          - >-
            A hidden state scores each vocabulary row in the same geometry used to embed that
            token at the input. There is no independent output-head parameter matrix.
      - id: tied-head-vocab-budget
        label: Save where vocabulary is expensive
        kind: tradeoff
        summary: Tying removes V×d_model parameters but couples input and output representation learning.
        detail:
          - >-
            The saving is particularly important in compact models or large vocabularies. Larger
            models may keep an untied head when independent output features justify its cost.
    cost:
      - label: Added parameters
        value: '0'
        note: shares the embedding matrix
      - label: Saved
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
        key: true
    usedBy:
      - GPT-2
      - Gemma
      - Llama-3.2-1B/3B
      - Qwen3-0.6B
      - T5
  - id: softcap-head
    label: Capped head
    full: Output logit soft-capping
    year: 2024
    role: branch
    tagline: tanh-bound the final logits
    concepts:
      - id: softcap-head-smooth-bound
        label: Bound logits smoothly
        kind: method
        summary: A tanh maps extreme vocabulary logits into a finite range before softmax.
        detail:
          - >-
            Ordinary-sized logits are almost unchanged while very large positive or negative
            values flatten toward the cap. Unlike clipping, the transform remains differentiable.
      - id: softcap-head-confidence
        label: Limit numerical overconfidence
        kind: tradeoff
        summary: The cap stabilises softmax but deliberately changes the sharpest output distributions.
        detail:
          - >-
            It is an architectural training-time choice, not a harmless decoding temperature. The
            cap must be applied exactly where the model was trained to expect it.
    math:
      - title: Output cap
        tex: \text{logits}' = c \cdot \tanh(\text{logits}/c), \qquad c = 30
    usedBy:
      - Gemma-2
  - id: mtp
    label: Multi-token prediction
    full: Multi-Token Prediction heads
    year: 2024
    role: frontier
    tagline: Predict the next n tokens, not just the next one
    paper:
      title: Better & Faster Large Language Models via Multi-token Prediction
      url: https://arxiv.org/abs/2404.19737
      authors: Meta AI
    concepts:
      - id: mtp-future-supervision
        label: Supervise several future offsets
        kind: method
        summary: Auxiliary heads predict token t+1, t+2, and later positions from the same prefix state.
        detail:
          - >-
            The backbone receives a denser training signal than next-token loss alone provides.
            Each offset has an aligned label sequence and contributes to the joint objective.
      - id: mtp-training-serving-gap
        label: Training signal versus decode path
        kind: tradeoff
        summary: Extra heads improve representation learning but standard decoding still emits one verified token at a time.
        detail:
          - >-
            Multi-token prediction can support speculative methods, but the target model must
            verify proposals to retain its intended probability distribution.
    math:
      - title: Multi-head objective
        tex: \mathcal{L} = -\sum_{t}\sum_{i=1}^{n} \log P_{\theta_i}(x_{t+i} \mid x_{\le t})
        note: >-
          n = 1 recovers ordinary next-token prediction. Each head has its own parameters but shares
          the entire trunk.
    usedBy:
      - DeepSeek-V3
      - Meta research
      - Medusa (related)
---

## role

One matrix multiply from `d_model` to `vocab_size`, producing a logit per token. Preceded by the final normalisation that pre-norm architectures require.

Cheap during training, where it runs once per position in parallel. Expensive at decode, where it is a `d_model × 128000` matmul for a *single* token — often 10–15% of per-token latency for a small model, and the reason large vocabularies are not free.

## untied-head

A dedicated projection learned separately from the input embedding. Standard for models large enough that the extra parameters are a small fraction of the total.

Worth noticing how much work this one layer does at inference: a 128k-row matmul per generated token, with no reuse across steps and no cache to help.

## tied-head

Use `W_embᵀ` as the output projection. The logit for token `t` becomes the dot product between the final hidden state and `t`'s embedding — which is a clean statement of what the model is doing: *how close is my prediction to this token's vector?*

See the Token Embedding block for the parameter arithmetic; this is the same decision viewed from the other end of the network, and a config's `tie_word_embeddings` sets both at once.

### fixes

A separate output matrix doubles the vocabulary cost, which for a small model is a large fraction of the whole budget.

## softcap-head

Apply `c · tanh(logits/c)` before the softmax — Gemma-2 uses `c = 30`. Improves calibration and numerical behaviour in fp16.

It changes the sampling distribution's sharpness, so temperature settings tuned on an uncapped model do not transfer directly, which surprises people porting prompts between models.

### fixes

Output logits can grow large enough to make the final softmax numerically brittle and the model overconfident.

## mtp

Attach several heads, each predicting a token at a different future offset. Two independent benefits: a **denser training signal**, since each position now supervises `n` predictions and the model must plan slightly ahead; and **speculative decoding for free**, because the extra heads propose the next few tokens and the main model verifies them in one pass.

DeepSeek-V3 uses a single extra head this way and reports both a quality gain and a 1.8× decode speedup from self-speculation. The auxiliary heads can be dropped after training if only the quality benefit is wanted.

### fixes

Single-token prediction gives a training signal about only the immediate next step, and forces decoding to be strictly one token at a time.

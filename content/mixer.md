---
id: mixer
label: Sequence mixer
ordinal: '5'
icon: pulse
slot: layer
tagline: The operation that lets one token use information from other positions
io:
  in: x [B, T, d_model]
  out: mixed x [B, T, d_model]
defaultVariant: attention
caveat: >-
  The attention-specific cards below this one apply only to layers that use attention. A pure
  Mamba trunk has no Q/K/V projections, attention mask, score normalization, or KV cache; a
  hybrid interleaves both kinds of layer.
lineage:
  - from: attention
    to: hybrid-ssm
    kind: combines
    label: retain explicit retrieval where it pays off
  - from: s4
    to: mamba-s6
    kind: fixes
    label: make state updates input-selective
  - from: mamba-s6
    to: mamba2-ssd
    kind: derives
    label: expose a state-space dual form for hardware-efficient chunks
  - from: mamba-s6
    to: hybrid-ssm
    kind: combines
    label: alternate recurrent state with attention layers
variants:
  - id: attention
    label: Self-attention
    full: Explicit self-attention sequence mixing
    year: 2017
    role: origin
    tagline: Each token reads a weighted set of token values
    math:
      - title: Attention as a token-to-token mixer
        tex: \operatorname{Attn}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_h})V
        note: Each query constructs a content-dependent weighted read over the entire visible prefix.
    concepts:
      - id: explicit-pairs
        label: Explicit token pairs
        kind: method
        summary: Attention materializes interactions between query and key positions, so it can retrieve a distant token in one layer.
      - id: quadratic-prefill
        label: Quadratic prompt work
        kind: tradeoff
        summary: Dense attention forms a T by T score matrix during prompt processing, which makes very long contexts expensive.
    usedBy:
      - GPT
      - Llama
      - Qwen
      - Mistral
  - id: s4
    label: Structured SSM
    full: Structured state-space sequence model
    year: 2021
    role: branch
    tagline: Compress prefix history into a recurrent state with fixed dynamics
    paper:
      title: Efficiently Modeling Long Sequences with Structured State Spaces
      url: https://arxiv.org/abs/2111.00396
      authors: Gu et al.
    math:
      - title: Discrete state update
        tex: h_t=\bar{A}h_{t-1}+\bar{B}x_t,\qquad y_t=Ch_t+Dx_t
        note: A fixed transition matrix carries a bounded-size summary forward instead of storing every prior key and value.
    concepts:
      - id: recurrent-state
        label: Fixed-size recurrent state
        kind: idea
        summary: Decoding carries h_t forward; memory does not grow with the number of generated tokens.
      - id: fixed-dynamics
        label: Input-independent dynamics
        kind: pitfall
        summary: The same transition is applied at every position, making it difficult to selectively retain one token and discard another.
    usedBy:
      - S4
      - S4D
  - id: mamba-s6
    label: Mamba (S6)
    full: Mamba selective state-space model
    year: 2023
    role: refinement
    tagline: Let the current token choose what the recurrent state keeps or forgets
    fixes: Fixed SSM dynamics treat every input alike; language needs content-dependent retention, reset, and readout.
    paper:
      title: Mamba — Linear-Time Sequence Modeling with Selective State Spaces
      url: https://arxiv.org/abs/2312.00752
      authors: Gu & Dao
    math:
      - title: Selective recurrence
        tex: h_t=\bar{A}(\Delta_t)h_{t-1}+\bar{B}(\Delta_t,B_t)x_t,\qquad y_t=C_t h_t+D x_t
        where:
          - sym: \Delta_t, B_t, C_t
            means: input-dependent step size, write, and read parameters
          - sym: h_t
            means: recurrent state at position t
        note: S6 projects the current token to the parameters of its own state update, then evaluates the recurrence with a parallel scan during training.
    concepts:
      - id: input-selectivity
        label: Input-dependent selection
        kind: method
        summary: Delta, B, and C are functions of the token representation, allowing a relevant token to persist and irrelevant tokens to be forgotten.
        detail:
          - >-
            This is the central change from S4. The transition is no longer a fixed linear filter;
            each input controls how far the continuous dynamics advance and how the state is written and read.
      - id: scan-not-cache
        label: Scan during training; state during decoding
        kind: tradeoff
        summary: The recurrence is evaluated with a hardware-aware parallel scan over a prompt, then generation carries only the latest state forward.
        detail:
          - >-
            There is no attention matrix and no growing KV cache. The implementation still needs a
            model-specific convolution/state cache, but its size is independent of context length.
      - id: local-conv-gate
        label: Local convolution and gate
        kind: method
        summary: Mamba surrounds the selective SSM with a short causal convolution and a multiplicative gate, so the layer is more than the recurrence alone.
    usedBy:
      - Mamba
      - Falcon Mamba
      - MambaByte
  - id: mamba2-ssd
    label: Mamba-2 / SSD
    full: Mamba-2 through structured state-space duality
    year: 2024
    role: refinement
    tagline: Recast the selective recurrence so chunks map efficiently to matrix multiplication
    fixes: Mamba's selective scan is linear in length, but its original core is not the most efficient form on modern accelerators.
    paper:
      title: Transformers are SSMs — Generalized Models and Efficient Algorithms Through Structured State Space Duality
      url: https://arxiv.org/abs/2405.21060
      authors: Dao & Gu
    math:
      - title: Scalar-transition selective state space
        tex: h_t=a_t h_{t-1}+B_t x_t,\qquad y_t=C_t h_t
        note: The scalar or low-rank transition exposes a dual matrix-multiplication form within chunks while the chunk boundary still carries state.
    concepts:
      - id: state-space-duality
        label: State-space duality
        kind: formula
        summary: A recurrence and a structured causal matrix multiplication are two views of the same operator, enabling a fused chunked implementation.
      - id: chunk-boundaries
        label: Chunkwise execution
        kind: method
        summary: Compute within-chunk interactions as matrix products, then pass a compact state across chunks instead of a full KV cache.
    usedBy:
      - Mamba-2
      - state-spaces/mamba2
  - id: hybrid-ssm
    label: Attention + SSM hybrid
    full: Interleaved attention and selective state-space layers
    year: 2024
    role: synthesis
    tagline: Spend attention layers on explicit retrieval and SSM layers on linear-time mixing
    fixes: Pure SSMs have constant decoding state but can lose the direct associative lookup that attention provides; pure attention pays to store and score every visible token.
    paper:
      title: Jamba — A Hybrid Transformer-Mamba Language Model
      url: https://arxiv.org/abs/2403.19887
      authors: AI21 Labs
    math:
      - title: Interleaved trunk
        tex: x_{l+1}=\begin{cases}\operatorname{Mamba}(x_l),&l\in\mathcal{S}\\\operatorname{Attn}(x_l),&l\in\mathcal{A}\end{cases}
        note: >-
          The schedule is architectural: only layers in the attention set own Q/K/V and a KV cache.
    concepts:
      - id: layer-schedule
        label: Layer schedule
        kind: method
        summary: The model chooses which depths run an SSM mixer and which run attention; one block-wide label cannot describe every layer by itself.
      - id: selective-cache
        label: Cache only attention layers
        kind: tradeoff
        summary: The serving cache grows only for the attention layers, while SSM layers carry fixed-size recurrent state.
    usedBy:
      - Jamba
      - Zamba
---

## role

The sequence mixer is the part of a repeated layer that moves information across token positions. In the usual Transformer it is self-attention, and the next four cards unpack its projections, mask, score rule, and decode-time cache.

That assumption is no longer universal. State-space models carry a compact recurrent state instead of explicitly comparing the current token with every earlier key. Hybrids deliberately use both. This card makes that architectural fork visible before attention-specific details appear.

## attention

Self-attention computes an explicit content-addressed read. Every token emits a query; its dot products against visible keys decide how to blend value vectors. That is why an attention layer can make a one-hop reference from a pronoun to a distant name.

It also explains the downstream cards: Q/K/V, visibility patterns, score normalization, and KV caching are not generic language-model components. They are the implementation anatomy of this particular mixer.

## s4

A structured state-space model reads the stream in order. It maintains a state vector whose update is linear and whose size does not increase with the prefix. During training, the same operation can be expressed as a long convolution or parallel scan; during decoding, only the current state needs to remain.

S4 made that style of long-sequence modeling competitive, but its transition and input/output maps are fixed across tokens. For language, that means it has no direct way to decide that one word should be copied into memory while the next should be ignored.

## mamba-s6

Mamba makes the SSM selective. From the current token it predicts the step size and the write/read parameters of the recurrence. A token can therefore cause a fast decay, a long retention, or a strong readout rather than passing through one fixed filter.

The crucial distinction from attention is operational: Mamba has neither Q/K/V projections nor an attention-score matrix. At generation time it carries a fixed-size SSM state (and small convolution state) rather than an ever-growing KV cache. The training implementation uses a fused parallel scan so the sequential recurrence is not interpreted one token at a time in Python.

### fixes

Conventional SSMs apply the same dynamics at every position. That prevents content-dependent copy, reset, and selective forgetting, which are central to language.

## mamba2-ssd

Mamba-2 changes the parameterization so the selective SSM has a structured state-space-dual form. Within chunks, the recurrence can be executed like a matrix multiplication; across chunks, it still passes a small state. This maps much more effectively to the compute units that already make attention fast.

It is not attention in disguise. The dual form is a structured causal operator, not arbitrary token-to-token weights, and decode state stays bounded rather than becoming a KV cache.

### fixes

The original Mamba selective scan is linear-time, but its core operation does not use accelerator matrix-multiply hardware as efficiently as the SSD formulation can.

## hybrid-ssm

A hybrid alternates SSM layers with attention layers rather than insisting on one mechanism everywhere. Jamba is the clear example: Mamba handles much of the linear-time sequence mixing, while selected attention layers retain exact content-addressed retrieval.

Read the attention cards conditionally for such a model. They describe the attention layers only; the SSM layers bypass them and use recurrent state instead. The schedule is a first-class architecture choice, not a serving-time toggle.

### fixes

Attention offers direct associative lookup but pays quadratic prompt work and a growing KV cache. A selective SSM avoids those costs but may not match attention's retrieval behavior at every depth.

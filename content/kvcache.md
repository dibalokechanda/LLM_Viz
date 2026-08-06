---
id: kvcache
label: KV Cache
ordinal: '9'
icon: database
slot: layer
tagline: The memory that makes decoding fast and expensive
io:
  in: k, v for the current token
  out: k, v for all tokens so far
defaultVariant: grouped
lineage:
  - from: none
    to: full
    kind: replaces
    label: stop recomputing the prefix
  - from: full
    to: grouped
    kind: derives
    label: fewer KV heads to store
  - from: full
    to: sliding
    kind: fixes
    label: bound it — evict past the window
  - from: full
    to: paged
    kind: fixes
    label: stop pre-allocating for the worst case
  - from: grouped
    to: mla-latent
    kind: derives
    label: store a latent, not the heads
  - from: grouped
    to: quantized
    kind: combines
    label: fewer bits per stored element
  - from: paged
    to: quantized
    kind: combines
    label: both, independently
  - from: sliding
    to: cross-layer
    kind: inspires
    label: share across layers as well
variants:
  - id: none
    label: No cache
    full: Recompute the full prefix each step
    year: 2017
    role: legacy
    tagline: What training does, and what naive generation did
    concepts:
      - id: no-cache-prefix-replay
        label: Replay the prefix
        kind: method
        summary: Each decode step projects and attends over every earlier token again.
        detail:
          - >-
            The model performs the same key/value work repeatedly as the sequence grows. Training
            and prefill can use this full-sequence computation because they process many positions
            together; autoregressive decoding cannot.
      - id: no-cache-cubic-generation
        label: Repetition adds an order of cost
        kind: metric
        summary: Summing quadratic attention over growing prefixes makes naive generation cubic in length.
        detail:
          - >-
            A cache changes the decode problem by retaining the old projections once. It does not
            remove attention over prior tokens, but avoids recomputing how those tokens are stored.
    math:
      - title: Cost to generate n tokens
        tex: \sum_{i=1}^{n} O(i^2) = O(n^3) \quad\text{versus}\quad O(n^2) \text{ with a cache}
        note: >-
          Per-step attention is quadratic in prefix length; summing over steps adds another power.
          The cache removes an entire order.
    usedBy:
      - training
      - prefill (where it is the right thing to do)
  - id: full
    label: Full cache
    full: Store every key and value for every head
    year: 2019
    role: origin
    tagline: Keep it all, contiguously, per sequence
    concepts:
      - id: full-cache-append-only
        label: Append-only decode state
        kind: method
        summary: Every layer retains the keys and values produced for each earlier token.
        detail:
          - >-
            A new query projects only the newest token, then reads the accumulated K/V tensors.
            Contiguous storage makes this simple and fast when sequence lengths are known.
      - id: full-cache-reservation
        label: Capacity follows the longest request
        kind: tradeoff
        summary: Cache memory grows with layer count, heads, length, precision, and concurrent sequences.
        detail:
          - >-
            The representation is lossless and requires no indirection, but serving systems must
            reserve enough contiguous memory for requests that may reach their maximum length.
    cost:
      - label: Per token per layer
        value: '{fixed(2 * nHead * dHead * kvBytes / 1024, 1)} KB'
      - label: One sequence at full context
        value: '{bytes(2 * nLayer * nHead * dHead * ctx * kvBytes)}'
        key: true
      - label: Batch of 32
        value: '{bytes((2 * nLayer * nHead * dHead * ctx * kvBytes) * 32)}'
        note: this is what decides how many users fit on a GPU
    usedBy:
      - reference implementations
      - HF transformers default
  - id: grouped
    label: Grouped
    full: Cache one K/V per group (GQA/MQA)
    year: 2023
    role: refinement
    tagline: Store n_kv heads instead of n_head
    concepts:
      - id: persistent-state
        label: The cache is persistent attention state
        kind: idea
        summary: Decoding stores past keys and values so the full prefix does not need to be projected again.
        detail:
          - >-
            The cache grows token by token and layer by layer. Its shape is determined by sequence
            length, KV head count, head dimension, precision, and batch size.
      - id: grouped-storage
        label: Store groups, not every query head
        kind: method
        summary: GQA and MQA reduce cache size by storing fewer key/value heads than query heads.
        detail:
          - >-
            Query heads remain numerous, but multiple queries read the same cached K/V group. The
            memory term therefore follows n_kv rather than n_head.
      - id: serving-capacity
        label: Memory becomes serving capacity
        kind: metric
        summary: A smaller cache lets the same accelerator hold more concurrent requests or longer contexts.
        detail:
          - >-
            During autoregressive decoding, cache bandwidth and capacity often limit throughput before
            arithmetic throughput does, so this reduction has a direct operational effect.
    cost:
      - label: KV heads stored
        value: '{nKvHead} of {nHead}'
      - label: One sequence at full context
        value: '{bytes(2 * nLayer * nKvHead * dHead * ctx * kvBytes)}'
        key: true
      - label: Saved against full
        value: '{bytes(2 * nLayer * (nHead - nKvHead) * dHead * ctx * kvBytes)}'
    usedBy:
      - Llama-3
      - Mistral
      - Qwen
      - Gemma-2
  - id: sliding
    label: Sliding / evicting
    full: Bounded cache with eviction
    year: 2023
    role: branch
    tagline: Discard entries past the window
    concepts:
      - id: sliding-cache-eviction
        label: Keep a moving suffix
        kind: method
        summary: New K/V entries overwrite or evict entries older than a fixed attention window.
        detail:
          - >-
            Cache length is capped even as the conversation continues. The attention mask must
            match the retained suffix so the model never reads entries that have been discarded.
      - id: sliding-cache-context-loss
        label: Bounded memory loses direct history
        kind: tradeoff
        summary: A hard ceiling turns an unbounded cache into a local-memory model.
        detail:
          - >-
            The method is appropriate when distant context is rarely needed or is carried through
            layers indirectly. Sinks and periodic global layers are common repairs for information
            that would otherwise be evicted.
    cost:
      - label: Window
        value: '{num((window == 0 ? 4096 : window))} tokens'
      - label: Cache ceiling
        value: '{bytes(2 * nLayer * nKvHead * dHead * (window == 0 ? 4096 : window) * kvBytes)}'
        note: constant, whatever the conversation length
        key: true
    usedBy:
      - Mistral-7B
      - StreamingLLM
      - H2O
      - SnapKV
  - id: paged
    label: PagedAttention
    full: Paged KV cache (vLLM)
    year: 2023
    role: refinement
    tagline: Virtual memory for the cache — allocate in blocks
    paper:
      title: Efficient Memory Management for Large Language Model Serving with PagedAttention
      url: https://arxiv.org/abs/2309.06180
      authors: Kwon et al.
    concepts:
      - id: paged-cache-block-table
        label: Logical cache, physical blocks
        kind: method
        summary: Each sequence uses a block table that maps logical token positions to physical KV pages.
        detail:
          - >-
            A request can grow by acquiring fixed-size blocks instead of moving one large contiguous
            allocation. The attention kernel follows the table while reading the cache.
      - id: paged-cache-fragmentation
        label: Reclaim memory at request granularity
        kind: tradeoff
        summary: Paging reduces reservation waste and enables sharing, at the cost of indirection.
        detail:
          - >-
            Short or finished requests release blocks immediately for other sequences. Kernels and
            schedulers must handle non-contiguous addresses efficiently for the win to materialise.
    math:
      - title: Waste
        tex: >-
          \text{waste}_{\text{contiguous}} = \frac{L_{\max} - L_{\text{actual}}}{L_{\max}}, \qquad
          \text{waste}_{\text{paged}} \le \frac{B - 1}{L_{\text{actual}}}
        worked:
          - tex: \frac{128000 - 500}{128000} = 99.6\%
            caption: a 500-token reply in a 128k reservation
          - tex: \frac{15}{500} = 3\%
            caption: the same reply, 16-token blocks
    usedBy:
      - vLLM
      - TGI
      - TensorRT-LLM
      - SGLang
  - id: quantized
    label: Quantized
    full: Low-precision KV cache
    year: 2024
    role: branch
    tagline: Store K and V in 8 or 4 bits
    concepts:
      - id: quantized-cache-scale
        label: Store codes with scale metadata
        kind: method
        summary: Keys and values are represented in fewer bits and dequantized when attention reads them.
        detail:
          - >-
            A quantizer records a scale, often per token or group, alongside int8 or int4 codes.
            The precision change acts on each stored element and composes with head or length
            reductions.
      - id: quantized-cache-error
        label: Memory buys approximation error
        kind: tradeoff
        summary: Fewer cache bits increase capacity but perturb the attention keys and values.
        detail:
          - >-
            Keys tend to be more sensitive than values, and the safe granularity depends on the
            model and context length. Calibration and mixed-precision policies are part of the
            implementation, not optional details.
    cost:
      - label: Current precision
        value: '{kvBytes * 8}-bit'
      - label: At int8
        value: '{bytes(2 * nLayer * nKvHead * dHead * ctx * 1)}'
        note: usually under 1% quality cost
        key: true
      - label: At int4
        value: '{bytes(2 * nLayer * nKvHead * dHead * ctx * 0.5)}'
        note: measurable degradation on long-context retrieval
    usedBy:
      - llama.cpp
      - vLLM (fp8)
      - TensorRT-LLM
      - KIVI
  - id: mla-latent
    label: Latent cache
    full: MLA compressed latent cache
    year: 2024
    role: synthesis
    tagline: Cache a low-rank latent, not keys and values
    concepts:
      - id: latent-cache-compression
        label: Retain the compressed token state
        kind: method
        summary: A low-rank latent replaces the full per-head key and value cache.
        detail:
          - >-
            K and V are reconstructed through learned projections only when the current layer
            needs them. This changes the storage basis rather than merely reducing head count.
      - id: latent-cache-model-coupling
        label: Compression is architectural
        kind: tradeoff
        summary: The cache reduction depends on projections trained specifically for the latent format.
        detail:
          - >-
            MLA cannot be retrofitted by quantising or reshaping an ordinary checkpoint. Its
            decomposition, positional treatment, and serving kernels are part of one design.
    cost:
      - label: Cached per token
        value: 512 latent + 64 RoPE
      - label: At full context
        value: '{bytes((512 + 64) * nLayer * kvBytes * ctx)}'
        key: true
      - label: vs. this model's grouped cache
        value: >-
          {fixed((2 * nLayer * nKvHead * dHead * ctx * kvBytes) / ((512 + 64) * nLayer * kvBytes *
          ctx), 1)}× smaller
    usedBy:
      - DeepSeek-V2/V3/R1
      - Kimi K2
  - id: cross-layer
    label: Cross-layer sharing
    full: Cross-layer attention / YOCO
    year: 2024
    role: frontier
    tagline: Share one cache across several layers
    concepts:
      - id: cross-layer-cache-producers
        label: Designate cache-producing layers
        kind: method
        summary: A subset of layers writes K/V state that later layers reuse instead of duplicating it.
        detail:
          - >-
            Cross-layer attention separates layers that create reusable context from layers that
            consume it. The cache's depth multiplier falls with the sharing schedule.
      - id: cross-layer-cache-coupling
        label: Fewer states, less layer autonomy
        kind: tradeoff
        summary: Sharing removes redundant storage but constrains each layer's independent attention basis.
        detail:
          - >-
            The method changes information flow across depth, not just memory layout. It needs
            training and architecture support to preserve quality under the shared representation.
    cost:
      - label: Layers
        value: '{nLayer}'
      - label: Sharing every 4 layers
        value: '{bytes((2 * nLayer * nKvHead * dHead * ctx * kvBytes) / 4)}'
        note: the depth multiplier drops from n_layer to n_layer/4
        key: true
    usedBy:
      - YOCO
      - CLA (research)
      - Character.AI (reported)
---

## role

Generating token `n+1` requires attention over all `n` previous tokens. Recomputing their keys and values every step would make generation quadratic in output length; caching them makes it linear. Every deployed LLM caches.

The consequence is that inference becomes a memory-management problem rather than a compute problem. Weights are fixed; the cache grows with every token, every concurrent request, and every layer, and it is the thing that actually determines how many users a GPU can serve.

The variants split into two independent axes that this block deliberately keeps together, because in practice you choose both: **what** is stored (set by the attention scheme upstream) and **how** it is laid out and quantised (set by the serving engine).

## none

During training there is no cache and none is needed — the whole sequence is processed in one parallel forward pass, and the K/V tensors are transient activations discarded after the backward.

Doing the same at generation means re-encoding the entire prefix for every new token: `O(n²)` work to produce `n` tokens. Included as the baseline that makes the cache's value legible.

## full

The straightforward implementation: a pre-allocated `[batch, n_head, max_len, d_head]` tensor per layer for K and another for V, appended to each step.

Two problems, and the rest of this block is the response to them. It is **large** — often larger than the weights past 32k context. And pre-allocating to `max_len` means a request that generates 50 tokens reserves memory for 128 000, so most of the allocation is never touched.

## grouped

Not a caching technique so much as the caching *consequence* of choosing GQA or MQA upstream. It appears as its own variant here because from the serving engine's point of view the attention scheme is invisible — all it sees is a smaller tensor.

This is the single largest cache reduction most models get, and it is essentially free at `n_kv = 8`.

### fixes

A full cache stores one K and one V per attention head, and head count is exactly the multiplier you can least afford.

## sliding

Keep at most `w` entries per layer as a ring buffer. Memory becomes constant regardless of how long the conversation runs, which is what makes indefinite streaming possible at all.

Naively evicting the oldest entries collapses quality, for the attention-sink reason — so real implementations retain the first few tokens permanently. More selective policies (H2O, SnapKV) score entries by accumulated attention and evict the least-used instead of the oldest.

### fixes

Even a grouped cache grows without bound. Streaming applications need a ceiling, not a smaller slope.

## paged

Borrow the operating system's answer. Store the cache in fixed-size blocks of, say, 16 tokens, and keep a per-sequence block table mapping logical positions to physical blocks. Blocks are allocated on demand and need not be contiguous.

Waste drops to at most one partly-filled block per sequence — under 4% instead of 60–80%. The reported effect was a 2–4× increase in serving throughput with no model change at all.

**Copy-on-write** falls out of the design and is arguably the bigger win: several sequences sharing a prompt share its physical blocks, so parallel sampling and beam search stop duplicating the prefix. For agent workloads with a large shared system prompt this is enormous.

### fixes

Contiguous per-request allocation must reserve for the maximum possible length, so real serving systems were wasting 60–80% of their KV memory on reservations never used.

## quantized

Store the cache in int8 or int4 with per-channel or per-token scales. Halving or quartering the bytes is close to a free win at int8; int4 needs care.

Keys and values behave differently, which is the practical finding worth knowing: keys have strong per-channel outliers and want per-channel quantisation, while values are better behaved and tolerate per-token. Systems that treat them identically lose accuracy for no reason.

### fixes

Every structural trick reduces how many elements are cached. This reduces how many bits each one takes — an orthogonal axis that composes with all of them.

## mla-latent

The serving-side face of Multi-head Latent Attention: what is stored is a single `d_c`-dimensional latent per token plus a small positional path, independent of head count entirely.

Smaller than MQA's single-head cache while behaving like full MHA, which is why DeepSeek can serve 128k context economically. See the Query/Key/Value block for the absorption trick that makes it work.

### fixes

Grouped caching stores fewer heads but still stores real K and V vectors. The floor is set by head count, and MLA removes the floor.

## cross-layer

GQA shares KV across heads; cross-layer attention shares it across *layers*. Compute K and V once every few layers and let the intervening layers reuse them. YOCO pushes this to the limit — one global cache produced by the first half of the network and read by the second.

Another 2–8× on top of GQA, and it attacks the one multiplier nothing else touches. Still early, and it constrains the architecture in ways that are not yet well characterised.

### fixes

Every scheme so far reduces the cache within a layer. Nothing addresses the fact that the cache is also multiplied by depth.

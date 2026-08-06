---
id: qkv
label: Query / Key / Value
ordinal: '6'
icon: fanout
slot: layer
tagline: How many distinct keys and values the heads share
io:
  in: x [B, T, d_model]
  out: q [B, n_head, T, d_head] · k, v [B, n_kv, T, d_head]
defaultVariant: gqa
lineage:
  - from: mha
    to: mqa
    kind: fixes
    label: collapse all KV heads to one
  - from: mha
    to: gqa
    kind: derives
    label: keep several, not all
  - from: mqa
    to: gqa
    kind: fixes
    label: one head was too few — interpolate
  - from: mqa
    to: mla
    kind: inspires
    label: keep the goal, drop the method
  - from: gqa
    to: mla
    kind: combines
    label: compress instead of share
variants:
  - id: mha
    label: MHA
    full: Multi-Head Attention
    year: 2017
    role: origin
    tagline: Every query head gets its own key and value head
    paper:
      title: Attention Is All You Need
      url: https://arxiv.org/abs/1706.03762
      authors: Vaswani et al.
    concepts:
      - id: mha-independent-heads
        label: Independent retrieval subspaces
        kind: method
        summary: Each attention head owns its Q, K, and V projection slices.
        detail:
          - >-
            Distinct key/value heads let different queries retrieve from separately learned
            representations. This is the most expressive and direct multi-head formulation.
      - id: mha-cache-multiplier
        label: Head count becomes memory
        kind: metric
        summary: Every generated token stores one key and one value for every attention head.
        detail:
          - >-
            During decoding, the cache grows with n_head rather than only model width. This is
            why MHA is straightforward for training but expensive for long-context serving.
    math:
      - title: Scaled dot-product attention, per head
        tex: \text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}} + M\right) V_i
        where:
          - sym: Q_i = xW_i^Q
            means: queries for head $i$, shape $T \times d_h$
          - sym: K_i, V_i
            means: keys and values for head $i$ — under MHA, unique to it
          - sym: d_h
            means: "head dimension, usually $d_{\text{model}} / n_{\text{head}}$"
          - sym: M
            means: the mask — $-infty$ wherever attention is forbidden
        note: >-
          The 1/√dₕ is not cosmetic. Dot products of two random dₕ-dimensional vectors have variance
          proportional to dₕ, so without the scale the softmax saturates at large dₕ and the
          gradient vanishes.
      - title: KV cache size — the number that ended MHA
        tex: \text{bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{head}} \cdot d_h \cdot T \cdot b
        where:
          - sym: '2'
            means: one tensor for K, one for V
          - sym: T
            means: tokens cached so far
          - sym: b
            means: bytes per element — 2 for bf16
        worked:
          - tex: 2 \cdot 32 \cdot 32 \cdot 128 \cdot 4096 \cdot 2 = 2.1\ \text{GB}
            caption: Llama-2-7B at 4k context — one sequence
          - tex: 2 \cdot 32 \cdot 32 \cdot 128 \cdot 128000 \cdot 2 = 67\ \text{GB}
            caption: the same model at 128k — larger than the weights, for one user
        note: >-
          Note what is missing from this formula: batch size is a multiplier on top. Serving 32
          concurrent users at 4k context on an MHA-7B needs 67 GB of cache alongside 14 GB of
          weights.
    figures:
      - kind: tensor
        title: Shapes through the projection
        chain:
          - label: residual
            shape:
              - B
              - T
              - d_model
          - label: queries
            shape:
              - B
              - n_head
              - T
              - d_head
            via: x @ W_Q  →  reshape
          - label: keys
            shape:
              - B
              - n_head
              - T
              - d_head
            via: x @ W_K
            focus: true
          - label: values
            shape:
              - B
              - n_head
              - T
              - d_head
            via: x @ W_V
            focus: true
          - label: scores
            shape:
              - B
              - n_head
              - T
              - T
            via: q @ kᵀ / √d_head
          - label: context
            shape:
              - B
              - T
              - d_model
            via: p @ v  →  merge  →  W_O
        caption: >-
          The two focused rows are the ones that get cached. Everything else is transient —
          computed, used, discarded. That asymmetry is the whole design pressure at this block.
    code:
      - title: PyTorch
        language: python
        code: |-
          class MHA(nn.Module):
              def __init__(self, d_model, n_head):
                  super().__init__()
                  self.n_head, self.d_head = n_head, d_model // n_head
                  # One projection per role, each full width. This symmetry is
                  # exactly what MQA and GQA break.
                  self.wq = nn.Linear(d_model, n_head * self.d_head, bias=False)
                  self.wk = nn.Linear(d_model, n_head * self.d_head, bias=False)
                  self.wv = nn.Linear(d_model, n_head * self.d_head, bias=False)
                  self.wo = nn.Linear(n_head * self.d_head, d_model, bias=False)

              def forward(self, x, cache=None):
                  B, T, _ = x.shape
                  split = lambda t: t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
                  q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))

                  if cache is not None:                      # decode: append and re-read
                      k = torch.cat([cache.k, k], dim=2)
                      v = torch.cat([cache.v, v], dim=2)
                      cache.k, cache.v = k, v                # grows by n_head * d_head per token

                  o = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
                  return self.wo(o.transpose(1, 2).reshape(B, T, -1))
        note: >-
          The cache append is the line that matters. Under MHA it grows by n_head × d_head elements
          per token per layer; MQA cuts that to d_head.
    cost:
      - label: KV heads
        value: '{nHead} of {nHead}'
        note: one per query head
      - label: KV cache at full context
        value: '{bytes(2 * nLayer * nHead * dHead * ctx * kvBytes)}'
        note: '{nLayer} layers × {nHead} heads × {dHead} dims × {num(ctx)} tokens × 2 (K,V)'
        key: true
      - label: Q/K/V projection params
        value: '{si(3 * dModel * nHead * dHead * nLayer)}'
    distinctions:
      - title: Head count vs. head dimension
        body: >-
          Almost every model keeps n_head × d_head = d_model, so adding heads shrinks each one. That
          is a convention, not a law — some models (Gemma) set d_head independently, which is why
          you should read head_dim from the config rather than dividing.
    usedBy:
      - GPT-2
      - GPT-3
      - BERT
      - Llama-1
      - Llama-2-7B/13B
      - Phi-2
  - id: mqa
    label: MQA
    full: Multi-Query Attention
    year: 2019
    role: branch
    tagline: One key/value head, shared by every query head
    paper:
      title: 'Fast Transformer Decoding: One Write-Head is All You Need'
      url: https://arxiv.org/abs/1911.02150
      authors: Noam Shazeer
    concepts:
      - id: mqa-shared-memory
        label: Many queries, one memory bank
        kind: method
        summary: Query heads stay separate while all of them read a single K/V head.
        detail:
          - >-
            The same cached keys and values are broadcast logically to every query head. Efficient
            kernels share the storage rather than materialising repeated tensors.
      - id: mqa-diversity-budget
        label: Spend diversity for bandwidth
        kind: tradeoff
        summary: MQA cuts KV memory sharply but gives all query heads identical stored content.
        detail:
          - >-
            The architecture keeps multiple ways to ask a question but only one representation to
            retrieve. GQA later restores part of that key/value diversity with a small number of
            groups.
    math:
      - title: The broadcast
        tex: >-
          \text{head}_i = \text{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d_h}} + M\right) V \qquad
          \forall i \in [1, n_{\text{head}}]
        note: K and V have lost their subscript. That single missing index is the entire method.
      - title: Cache reduction
        tex: \frac{\text{bytes}_{\text{MQA}}}{\text{bytes}_{\text{MHA}}} = \frac{1}{n_{\text{head}}}
        worked:
          - tex: \frac{1}{32} = 3.1\%
            caption: '32-head model: 2.1 GB of cache becomes 67 MB'
        note: >-
          Arithmetic intensity rises by the same factor — the same FLOPs now read 32× less memory,
          which is why the speedup at decode is close to the full 32× on bandwidth-bound hardware.
    figures:
      - kind: blocks
        title: Who shares what
        rows:
          - label: MHA
            boxes:
              - text: Q₁
              - text: Q₂
              - text: Q₃
              - text: Q₄
          - label: ''
            boxes:
              - text: K₁V₁
                filled: true
              - text: K₂V₂
                filled: true
              - text: K₃V₃
                filled: true
              - text: K₄V₄
                filled: true
          - label: MQA
            boxes:
              - text: Q₁
              - text: Q₂
              - text: Q₃
              - text: Q₄
          - label: ''
            boxes:
              - text: K V  — one pair, broadcast to all four
                span: 4
                filled: true
        caption: >-
          Filled boxes are cached. MQA keeps the query diversity and throws away the key/value
          diversity.
    cost:
      - label: KV heads
        value: 1 of {nHead}
        note: '{nHead}× fewer than MHA'
      - label: KV cache at full context
        value: '{bytes(2 * nLayer * 1 * dHead * ctx * kvBytes)}'
        note: down from {bytes(2 * nLayer * nHead * dHead * ctx * kvBytes)} under MHA
        key: true
      - label: Params saved vs MHA
        value: '{si(2 * dModel * (nHead - 1) * dHead * nLayer)}'
    usedBy:
      - PaLM
      - Falcon-7B
      - StarCoder
      - Gemini (reported)
  - id: gqa
    label: GQA
    full: Grouped-Query Attention
    year: 2023
    role: synthesis
    tagline: Several query heads share each key/value pair, balancing quality with cache memory
    paper:
      title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints'
      url: https://arxiv.org/abs/2305.13245
      authors: Ainslie et al., Google
    concepts:
      - id: grouping-rule
        label: The grouping rule
        kind: method
        summary: Each query head keeps its own question, but reads from the key/value pair assigned to its group.
        detail:
          - >-
            GQA preserves all $n_{head}$ query projections, so different heads can still ask different
            questions of the context. The change is in memory: only $n_{kv}$ key/value projections are
            stored, and a fixed head-to-group rule tells each query which pair to read.
        children:
          - id: head-to-group
            label: Head-to-group assignment
            kind: formula
            summary: The ratio of query heads to KV groups says exactly how much sharing happens.
            detail:
              - >-
                For 32 query heads and 8 KV groups, each stored K/V pair serves four query heads. Push
                that number to one group and you have MQA; give every query head its own group and you
                are back at MHA.
          - id: no-materialized-copy
            label: Logical, not physical, sharing
            kind: pitfall
            summary: Efficient kernels look up a shared group; they never make four physical copies of K and V.
            detail:
              - >-
                A teaching implementation may use repeat_interleave so the tensor shapes look familiar.
                Production attention kernels keep the mapping implicit, which preserves the bandwidth and
                cache-memory saving.
      - id: cache-payoff
        label: The cache payoff
        kind: metric
        summary: KV-cache memory follows the number of KV groups, so fewer groups directly increase serving capacity.
        detail:
          - >-
            The cache holds keys and values for every earlier token in every layer. Reducing 32 KV heads
            to 8 cuts that persistent memory by four, while all 32 query heads remain available to form
            distinct attention patterns.
      - id: quality-dial
        label: A tunable quality dial
        kind: tradeoff
        summary: More KV groups move toward MHA quality; fewer groups move toward MQA-sized memory.
        detail:
          - >-
            GQA makes the trade-off explicit instead of forcing a choice between two extremes. The
            model designer chooses the smallest number of KV groups that meets the quality target and
            the cache budget.
    math:
      - title: Which key/value pair each query head reads
        tex: >-
          \operatorname{Attn}_i =
          \operatorname{softmax}\!\left(\frac{Q_i K_{g(i)}^\top}{\sqrt{d_h}} + M\right)V_{g(i)},
          \qquad r = \frac{n_{\text{head}}}{n_{kv}}, \quad
          g(i) = \left\lfloor \frac{i}{r} \right\rfloor
        where:
          - sym: r
            means: query heads sharing each KV group
          - sym: n_{kv}
            means: number of KV groups — `num_key_value_heads` in the config
          - sym: g(i)
            means: the group that query head $i$ reads
        note: >-
          The small function $g(i)$ is the whole method. When $n_{kv}=n_{head}$, every head maps to
          itself and the formula becomes MHA. When $n_{kv}=1$, every query reads the same pair and it
          becomes MQA.
      - title: Turning an MHA checkpoint into GQA
        tex: >-
          K_g^{\text{new}} = \frac{1}{|g|}\sum_{i \in g} K_i, \qquad V_g^{\text{new}} =
          \frac{1}{|g|}\sum_{i \in g} V_i
        note: >-
          Mean-pooling preserves the average behaviour of the heads that will share a group. In the
          paper's ablation, it outperformed both choosing one existing head and starting from a random
          projection, then required only a short uptraining run.
      - title: The cache saving, as a ratio
        tex: \frac{\text{KV bytes}_{\text{GQA}}}{\text{KV bytes}_{\text{MHA}}}
          = \frac{n_{kv}}{n_{\text{head}}} = \frac{1}{r}
        worked:
          - tex: n_{kv} = 8,\ n_{\text{head}} = 32 \;\Rightarrow\; 4\times\ \text{reduction}
            caption: Llama-3-8B
          - tex: n_{kv} = 8,\ n_{\text{head}} = 64 \;\Rightarrow\; 8\times\ \text{reduction}
            caption: Llama-3-70B — the same 8 groups, a bigger win
        note: >-
          The cache still has one K and one V tensor for every layer and token. GQA changes only the
          head count in that storage term, so the saving is easy to predict from the model config.
    figures:
      - kind: blocks
        title: Eight query heads, two groups
        rows:
          - label: queries
            boxes:
              - text: Q₁
              - text: Q₂
              - text: Q₃
              - text: Q₄
              - text: Q₅
              - text: Q₆
              - text: Q₇
              - text: Q₈
          - label: cached
            boxes:
              - text: K₁ V₁
                span: 4
                filled: true
              - text: K₂ V₂
                span: 4
                filled: true
        caption: >-
          Four query heads consult each group. Only two K/V pairs are cached instead of eight, which is
          a fourfold reduction without collapsing the query heads themselves.
      - kind: curve
        title: Quality against cache size
        xLabel: KV groups (log)
        yLabel: quality (normalised)
        lines:
          - label: observed
            points:
              - - 0
                - 0.72
              - - 1
                - 0.89
              - - 2
                - 0.96
              - - 3
                - 0.985
              - - 4
                - 0.995
              - - 5
                - 1
        xTicks:
          - at: 0
            label: 1 (MQA)
          - at: 2
            label: '4'
          - at: 3
            label: '8'
          - at: 5
            label: 32 (MHA)
        marks:
          - x: 3
            'y': 0.985
            label: the knee — where everyone sits
        caption: >-
          This is a schematic, not a benchmark curve. It captures the pattern reported in GQA studies:
          quality stays close to MHA until the number of groups becomes very small, then the trade-off
          becomes visible. Eight groups is often near that practical knee.
    code:
      - title: PyTorch
        language: python
        code: |-
          class GQA(nn.Module):
              def __init__(self, d_model, n_head, n_kv):
                  super().__init__()
                  self.n_head, self.n_kv = n_head, n_kv
                  self.rep = n_head // n_kv          # query heads per KV group
                  self.d_head = d_model // n_head
                  self.wq = nn.Linear(d_model, n_head * self.d_head, bias=False)
                  # The asymmetry: K and V project to n_kv heads, not n_head.
                  self.wk = nn.Linear(d_model, n_kv * self.d_head, bias=False)
                  self.wv = nn.Linear(d_model, n_kv * self.d_head, bias=False)
                  self.wo = nn.Linear(n_head * self.d_head, d_model, bias=False)

              def forward(self, x, cache=None):
                  B, T, _ = x.shape
                  q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
                  k = self.wk(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
                  v = self.wv(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)

                  if cache is not None:
                      k, v = cache.append(k, v)      # only n_kv heads are stored

                  # Expand groups to match query heads. repeat_interleave materialises
                  # the copies; fused kernels skip this and index the group directly,
                  # which is where the real bandwidth saving comes from.
                  k = k.repeat_interleave(self.rep, dim=1)
                  v = v.repeat_interleave(self.rep, dim=1)

                  o = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
                  return self.wo(o.transpose(1, 2).reshape(B, T, -1))
        note: >-
          repeat_interleave is useful for explaining the mapping, not for serving a model. Production
          kernels such as FlashAttention and vLLM receive n_kv and the group size directly, then index
          the shared K/V pair without materialising copies.
      - title: Reading it off a config
        language: json
        code: |-
          {
            "num_attention_heads": 32,
            "num_key_value_heads": 8,     // ← 8 ≠ 32, and ≠ 1, so: GQA with 4 heads per group
            "head_dim": 128,
            "num_hidden_layers": 32
          }
        note: >-
          These fields are enough to identify the design: 32 query heads and 8 KV heads mean four
          queries per group. If num_key_value_heads is absent, the conventional default is MHA.
    cost:
      - label: Groups
        value: '{nKvHead} KV heads for {nHead} query heads'
        note: '{fixed(nHead / max(nKvHead, 1), 0)} query heads share each KV head'
      - label: KV cache at full context
        value: '{bytes(2 * nLayer * nKvHead * dHead * ctx * kvBytes)}'
        note: >-
          {fixed(nHead / max(nKvHead, 1), 0)}× smaller than MHA's {bytes(2 * nLayer * nHead * dHead
          * ctx * kvBytes)}
        key: true
      - label: Cache per token
        value: '{fixed(2 * nLayer * nKvHead * dHead * kvBytes / 1024, 1)} KB'
        note: multiply by context length and batch size
    usedBy:
      - Llama-2-70B
      - Llama-3 (all sizes)
      - Mistral-7B
      - Qwen2/3
      - Gemma-2
      - Mixtral
  - id: mla
    label: MLA
    full: Multi-head Latent Attention
    year: 2024
    role: frontier
    tagline: Cache one low-rank latent, reconstruct K and V from it
    paper:
      title: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model'
      url: https://arxiv.org/abs/2405.04434
      authors: DeepSeek-AI
    concepts:
      - id: mla-compressed-state
        label: Cache a latent state
        kind: method
        summary: A down-projection compresses the token before K and V are reconstructed for attention.
        detail:
          - >-
            The persistent decode state is a low-dimensional latent rather than a full key and
            value tensor per head. Up-projections recover the representations used by the current
            attention calculation.
      - id: mla-absorption-condition
        label: Move reconstruction into weights
        kind: tradeoff
        summary: Algebraic absorption avoids rebuilding full cached K/V tensors at every decode step.
        detail:
          - >-
            The serving benefit depends on folding compatible projections into the query and output
            paths. Positional components that cannot be absorbed require a separate treatment,
            which is why MLA has more architectural constraints than GQA.
    math:
      - title: Compress, then reconstruct
        tex: >-
          c_t = W^{DKV} x_t \in \mathbb{R}^{d_c}, \qquad k_t^{(i)} = W^{UK}_i c_t, \qquad v_t^{(i)}
          = W^{UV}_i c_t
        where:
          - sym: c_t
            means: the cached latent — the *only* thing stored per token
          - sym: d_c
            means: kv_lora_rank, e.g. 512 against d_model = 5120
          - sym: W^{DKV}
            means: down-projection, shared across all heads
          - sym: W^{UK}_i
            means: per-head up-projection, never materialised at inference
      - title: The absorption that makes it free
        tex: >-
          q_t^{(i)\top} k_s^{(i)} = q_t^{(i)\top} W^{UK}_i c_s = \underbrace{\left(W^{UK\top}_i
          q_t^{(i)}\right)}_{\text{fold into } W^Q \text{ once}}{}^{\top} c_s
        note: >-
          Because W_UK does not depend on s, it can move to the query side and be pre-multiplied
          into W_Q. The score is then a dot product against the cached latent directly, so K is
          never reconstructed at decode.
      - title: Cache per token, compared
        tex: >-
          \text{MHA}: 2 n_h d_h \quad\big|\quad \text{GQA}: 2 n_{kv} d_h \quad\big|\quad \text{MLA}:
          d_c + d_h^{R}
        worked:
          - tex: '\text{MHA}: 2 \cdot 128 \cdot 128 = 32{,}768'
            caption: DeepSeek-V2 shape, hypothetical MHA
          - tex: '\text{GQA-8}: 2 \cdot 8 \cdot 128 = 2{,}048'
            caption: 16× less
          - tex: '\text{MLA}: 512 + 64 = 576'
            caption: 57× less than MHA, and 3.5× less than GQA-8
        note: d_h^R is the decoupled RoPE dimension — the small uncompressed path that carries position.
    figures:
      - kind: tensor
        title: What actually gets stored
        chain:
          - label: token
            shape:
              - B
              - '1'
              - d_model
          - label: latent
            shape:
              - B
              - '1'
              - d_c
            via: W_DKV — down-project
            focus: true
          - label: rope path
            shape:
              - B
              - '1'
              - d_rope
            via: W_KR + RoPE
            focus: true
          - label: reconstructed K
            shape:
              - B
              - n_head
              - '1'
              - d_head
            via: W_UK  (absorbed — never runs)
          - label: reconstructed V
            shape:
              - B
              - n_head
              - '1'
              - d_head
            via: W_UV  (absorbed — never runs)
        caption: >-
          Only the two focused rows are cached. The bottom two are what MHA would have stored; under
          MLA they exist only as algebra folded into the query and output projections.
      - kind: bars
        title: KV cache per token, elements
        categories:
          - MHA
          - GQA-8
          - MQA
          - MLA
        series:
          - label: elements cached per token per layer
            values:
              - 32768
              - 2048
              - 256
              - 576
        highlight:
          - 3
        showValues: true
        caption: >-
          DeepSeek-V2 geometry: 128 heads at d_head 128. MLA sits between MQA and GQA on cost while
          reporting quality above MHA — which is the claim that made it interesting.
    code:
      - title: Detecting it from a config
        language: json
        code: |-
          {
            "model_type": "deepseek_v3",
            "kv_lora_rank": 512,          // ← the tell. No other scheme has this key.
            "q_lora_rank": 1536,          // queries are compressed too, but not cached
            "qk_rope_head_dim": 64,       // the decoupled positional path
            "qk_nope_head_dim": 128,      // the compressed, position-free path
            "v_head_dim": 128
          }
        note: >-
          kv_lora_rank is checked before num_key_value_heads in the detector, because MLA configs
          also carry a num_key_value_heads that would otherwise read as MHA.
    cost:
      - label: Cached per token
        value: 512 latent + 64 RoPE dims
        note: independent of head count
      - label: KV cache at full context
        value: '{bytes((512 + 64) * nLayer * kvBytes * ctx)}'
        note: against {bytes(2 * nLayer * nKvHead * dHead * ctx * kvBytes)} for this model's GQA shape
        key: true
      - label: vs. MHA
        value: >-
          {fixed((2 * nLayer * nHead * dHead * ctx * kvBytes) / ((512 + 64) * nLayer * kvBytes *
          ctx), 1)}× smaller
    distinctions:
      - title: MLA vs. LoRA
        body: >-
          Both are low-rank factorisations and the config key says "lora", but they solve unrelated
          problems. LoRA factorises a *weight update* to make fine-tuning cheap, and is merged away
          at inference. MLA factorises the *activation* to make the cache small, and is load-bearing
          at inference. The shared vocabulary is unfortunate.
    usedBy:
      - DeepSeek-V2
      - DeepSeek-V3
      - DeepSeek-R1
      - Kimi K2
---

## role

Attention starts by giving the residual stream three jobs. A **query** says what the current token is looking for; a **key** says what each earlier token can be matched on; and a **value** is the information copied forward once a match has been chosen.

Most of this design space comes down to **one number**: how many distinct key/value heads sit behind the query heads. Queries are recomputed for the current step and discarded. Keys and values are different: every layer must retain them for every token seen so far. That persistent state is the KV cache, and at long context it often dominates both memory use and decode latency.

Every variant here is answering the same practical question: how much KV memory can we remove before attention quality noticeably changes?

## mha

The original. `n_head` query heads, `n_head` key heads, `n_head` value heads — a one-to-one correspondence, so each head runs a completely independent attention operation in its own `d_head`-dimensional subspace.

The motivation was representational: one attention head can only average one way. Splitting into heads lets one head track syntactic agreement while another tracks coreference and a third tracks position, and the output projection recombines them. That argument was about **query** diversity, and it is still sound.

What nobody was optimising for in 2017 was **autoregressive decoding at 128k context**. Training is compute-bound and parallel over the sequence, so the KV tensor is transient. Decoding is memory-bound and serial: you generate one token, and to do it you must re-read the entire cached K and V for every layer and every head. MHA makes that tensor as large as it can possibly be.

## mqa

Keep all `n_head` query heads. Project **one** key head and **one** value head, and broadcast them across every query head. The cache shrinks by exactly `n_head`×.

The insight is that the heads are not equally load-bearing. Query heads are what give attention its ability to look for different things at once, and they cost nothing to keep because they are not kept. Key and value heads are what cost, and Shazeer's bet was that most of their diversity is redundant.

The bet was half right. MQA is dramatically faster at decode, and on many tasks indistinguishable. But it degrades measurably on tasks needing fine-grained retrieval from long context, and it is *unstable to train* — models trained from scratch with MQA are more prone to loss spikes than their MHA counterparts. That combination is what left room for GQA.

### fixes

MHA caches one K and one V per head, so the cache scales with n_head and decoding becomes memory-bandwidth bound long before it becomes compute bound.

## gqa

GQA keeps every query head, but lets small groups of them share a key/value pair. The group count, `n_kv`, is the dial: `n_kv = n_head` gives MHA, `n_kv = 1` gives MQA, and values in between trade a little key/value diversity for a much smaller cache.

The practical insight is that GQA does not require training a model from scratch. Start with an MHA checkpoint, **mean-pool** the key and value projections within each future group, then uptrain briefly. This gives the new shared pair a sensible starting point and turns GQA into a migration path rather than a separate model family.

Why do so many models use eight groups? Empirically, quality is often close to MHA down to that range and declines more noticeably below it. Operationally, eight groups also map neatly onto common 8-way tensor-parallel deployments: each device can own a group without moving KV cache state between GPUs.

### fixes

MQA shares too aggressively for some workloads, while MHA stores more key/value diversity than long-context serving can comfortably afford. GQA gives the model a measured middle ground.

## mla

Stop caching K and V. Instead, project the token down to a single small **latent** `c` of rank `kv_lora_rank` (512 in DeepSeek-V2, against a 5120-wide residual stream), cache only that, and reconstruct the full per-head K and V from it with up-projections when needed.

The reason this is not just "compress the cache" is an algebraic trick. The up-projection matrices are constant, so they can be **absorbed into the query and output projections at inference time**: instead of decompressing `c` into K and then computing `qᵀk`, you fold `W_UK` into `W_Q` once and dot the query against the latent directly. The decompression never runs. You get MHA-shaped attention while touching an MQA-sized cache.

MLA is a genuine synthesis rather than a point on the MHA↔MQA dial. It takes MQA's goal (one cached object per token, not one per head) and GQA's constraint (do not sacrifice head diversity), and satisfies both by changing *what* is cached rather than *how many*. DeepSeek report it beating MHA on quality while caching less than GQA-8.

The cost is that it does not compose cleanly with RoPE — rotation is position-dependent, so it cannot be absorbed into a constant matrix. DeepSeek's answer is **decoupled RoPE**: split each head into a compressed part carrying no position and a small separate part carrying RoPE, and concatenate. That extra path is the untidy corner of an otherwise elegant method.

### fixes

GQA still caches real key and value vectors — it just caches fewer of them. The cache stays proportional to n_kv × d_head, and quality is bounded by how much head diversity you gave up.

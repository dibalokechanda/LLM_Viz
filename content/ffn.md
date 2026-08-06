---
id: ffn
label: Feed-Forward / MoE
ordinal: '10'
icon: branch
slot: layer
tagline: Where the parameters — and most of the knowledge — live
io:
  in: x [B, T, d_model]
  out: y [B, T, d_model]
defaultVariant: swiglu
lineage:
  - from: relu-mlp
    to: gelu-mlp
    kind: fixes
    label: smooth the kink
  - from: gelu-mlp
    to: geglu
    kind: derives
    label: add a multiplicative gate
  - from: geglu
    to: swiglu
    kind: derives
    label: swap GELU for SiLU
  - from: relu-mlp
    to: topk-moe
    kind: replaces
    label: many FFNs, run k of them
  - from: topk-moe
    to: switch
    kind: fixes
    label: k = 1 is enough, and half the cost
  - from: topk-moe
    to: expert-choice
    kind: fixes
    label: invert the choice to force balance
  - from: switch
    to: shared-expert
    kind: derives
    label: split finer, pin some always-on
  - from: swiglu
    to: shared-expert
    kind: combines
    label: each expert is a SwiGLU MLP
variants:
  - id: relu-mlp
    label: ReLU MLP
    full: Two-layer feed-forward with ReLU
    year: 2017
    role: origin
    tagline: Up-project 4×, ReLU, down-project
    paper:
      title: Attention Is All You Need
      url: https://arxiv.org/abs/1706.03762
    concepts:
      - id: relu-mlp-expand-contract
        label: Expand, select, contract
        kind: method
        summary: The FFN widens each token independently, applies ReLU, then returns to d_model.
        detail:
          - >-
            The expansion creates many feature detectors; ReLU selects the positive ones; the
            second projection recombines them into the residual stream. No tokens interact here.
      - id: relu-mlp-dead-units
        label: Hard gating
        kind: tradeoff
        summary: ReLU is cheap and sparse, but negative activations have exactly zero local gradient.
        detail:
          - >-
            Units can become inactive for a region of input space. Later smooth activations keep a
            graded response near zero while preserving the same two-projection structure.
    math:
      - title: The layer
        tex: \text{FFN}(x) = \max(0,\, xW_1 + b_1)\,W_2 + b_2
        where:
          - sym: W_1
            means: $d_{\text{model}} \times d_{ff}$, usually $d_{ff} = 4d_{\text{model}}$
          - sym: W_2
            means: $d_{ff} \times d_{\text{model}}$
        worked:
          - tex: P = 2 \cdot d_{\text{model}} \cdot d_{ff} = 8d_{\text{model}}^2
            caption: against 4d² for attention — two thirds of the layer
    usedBy:
      - Transformer (2017)
      - BERT (partly)
      - T5-1.0
  - id: gelu-mlp
    label: GELU MLP
    full: Feed-forward with Gaussian Error Linear Unit
    year: 2018
    role: refinement
    tagline: Same shape, smooth activation
    paper:
      title: Gaussian Error Linear Units (GELUs)
      url: https://arxiv.org/abs/1606.08415
    concepts:
      - id: gelu-mlp-probabilistic-gate
        label: Smooth probabilistic gate
        kind: method
        summary: GELU scales an activation continuously instead of discarding every negative value.
        detail:
          - >-
            Small positive and negative values pass with different fractional weights, giving the
            MLP a smooth transition around zero rather than ReLU's hard threshold.
      - id: gelu-mlp-dense-cost
        label: Same dense compute envelope
        kind: tradeoff
        summary: GELU improves optimisation behavior without changing the dense FFN parameter shape.
        detail:
          - >-
            The activation is more expensive than a max operation but the dominant cost remains
            the two matrix multiplications. It does not change the fact that every token uses every
            FFN parameter.
    math:
      - title: GELU
        tex: >-
          \text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,(x +
          0.044715x^3)\right]\right)
        note: >-
          The tanh form is the "gelu_new" approximation — it exists because the exact erf was slow
          on 2018 hardware. Both survive in configs, and they differ enough to change logits, so
          ports must match the original.
    figures:
      - kind: curve
        title: Activation shapes
        xLabel: x
        yLabel: f(x)
        lines:
          - label: ReLU
            points:
              - - -3
                - 0
              - - -2
                - 0
              - - -1
                - 0
              - - 0
                - 0
              - - 1
                - 1
              - - 2
                - 2
              - - 3
                - 3
            dashed: true
          - label: GELU
            points:
              - - -3
                - -0.004
              - - -2
                - -0.045
              - - -1
                - -0.159
              - - 0
                - 0
              - - 1
                - 0.841
              - - 2
                - 1.955
              - - 3
                - 2.996
        caption: >-
          GELU dips slightly negative around x ≈ −1 before recovering. That non-monotonic pocket is
          where its gradient advantage over ReLU comes from.
    usedBy:
      - GPT-2
      - GPT-3
      - BERT
      - ViT
      - GPT-NeoX
  - id: geglu
    label: GEGLU
    full: Gated Linear Unit with GELU
    year: 2020
    role: branch
    tagline: 'Two up-projections: one gates the other'
    paper:
      title: GLU Variants Improve Transformer
      url: https://arxiv.org/abs/2002.05202
      authors: Noam Shazeer
    concepts:
      - id: geglu-input-gate
        label: Let the input choose features
        kind: method
        summary: One projection produces values while a second projection produces a GELU gate.
        detail:
          - >-
            Their elementwise product lets the current token decide which expanded features are
            useful. This is more expressive than applying one fixed nonlinearity to one projection.
      - id: geglu-width-accounting
        label: Two up-projections share the budget
        kind: tradeoff
        summary: Gated FFNs need two expanded tensors, so their inner width is usually reduced.
        detail:
          - >-
            Comparing a 4d ReLU MLP with a 4d gated MLP is not a fair parameter comparison. Modern
            configurations choose a narrower gated width to keep FLOPs and parameters comparable.
    math:
      - title: The gate
        tex: >-
          \text{GEGLU}(x) = \big(\text{GELU}(xW_{\text{gate}}) \odot
          xW_{\text{up}}\big)W_{\text{down}}
        note: >-
          Three matrices instead of two. To keep the parameter count fixed, d_ff shrinks from 4d to
          about 8/3·d — which is why you see intermediate sizes like 11008 (Llama-2-7B) rather than
          a clean 4× of 4096.
    usedBy:
      - T5-1.1
      - Flan-T5
      - Gemma
  - id: swiglu
    label: SwiGLU
    full: Gated Linear Unit with Swish/SiLU
    year: 2020
    role: refinement
    tagline: The gated FFN essentially every modern model uses
    concepts:
      - id: wider-workspace
        label: Expand into a wider workspace
        kind: method
        summary: The feed-forward block projects each token from model width into a larger intermediate representation.
        detail:
          - >-
            Attention mixes information across positions; the FFN then transforms each position
            independently with much more feature capacity.
      - id: content-gate
        label: A learned gate selects features
        kind: formula
        summary: SwiGLU multiplies an up projection by a SiLU-gated projection before projecting back down.
        detail:
          - >-
            The multiplicative gate lets one learned pathway control which features from the other
            pathway are passed onward, making the nonlinearity more expressive than a single activation.
      - id: matched-parameter-budget
        label: Intermediate width is chosen deliberately
        kind: tradeoff
        summary: SwiGLU uses three matrices, so its intermediate size is reduced to keep the parameter budget comparable.
        detail:
          - >-
            A common choice is roughly eight-thirds of model width instead of the four-times width
            used by a two-matrix ReLU or GELU MLP.
    paper:
      title: GLU Variants Improve Transformer
      url: https://arxiv.org/abs/2002.05202
    math:
      - title: SwiGLU
        tex: >-
          \text{SwiGLU}(x) = \big(\text{SiLU}(xW_{\text{gate}}) \odot
          xW_{\text{up}}\big)W_{\text{down}}, \qquad \text{SiLU}(z) = z\,\sigma(z)
        where:
          - sym: W_{\text{gate}}, W_{\text{up}}
            means: both $d_{\text{model}} \times d_{ff}$
          - sym: W_{\text{down}}
            means: $d_{ff} \times d_{\text{model}}$
      - title: Why intermediate_size looks arbitrary
        tex: >-
          3 \cdot d_{\text{model}} \cdot d_{ff} \;\approx\; 8 \cdot d_{\text{model}}^2
          \;\Rightarrow\; d_{ff} \approx \tfrac{8}{3} d_{\text{model}}
        worked:
          - tex: \tfrac{8}{3} \cdot 4096 = 10{,}923 \;\to\; 11{,}008
            caption: Llama-2-7B — rounded to a multiple of 256
          - tex: \tfrac{8}{3} \cdot 4096 = 10{,}923 \;\to\; 14{,}336
            caption: Llama-3-8B — deliberately over the ratio, spending more on the FFN
    figures:
      - kind: tensor
        title: Three projections, one output
        chain:
          - label: input
            shape:
              - B
              - T
              - d_model
          - label: gate
            shape:
              - B
              - T
              - d_ff
            via: W_gate  →  SiLU
          - label: up
            shape:
              - B
              - T
              - d_ff
            via: W_up  (linear)
          - label: gated
            shape:
              - B
              - T
              - d_ff
            via: elementwise ×
            focus: true
          - label: output
            shape:
              - B
              - T
              - d_model
            via: W_down
    code:
      - title: PyTorch
        language: python
        code: |-
          class SwiGLU(nn.Module):
              def __init__(self, d_model, d_ff):
                  super().__init__()
                  self.gate = nn.Linear(d_model, d_ff, bias=False)
                  self.up   = nn.Linear(d_model, d_ff, bias=False)
                  self.down = nn.Linear(d_ff, d_model, bias=False)

              def forward(self, x):
                  # The multiplication is the point: "up" is never passed through a
                  # non-linearity, so the gate can zero a feature outright.
                  return self.down(F.silu(self.gate(x)) * self.up(x))
        note: >-
          No biases. Every modern LLM drops them from the FFN — they cost parameters and buy nothing
          once the layer is preceded by a normalisation.
    cost:
      - label: Params per layer
        value: '{si(3 * dModel * dFF)}'
        note: three matrices, d_model × d_ff each
      - label: Params across all layers
        value: '{si(3 * dModel * dFF * nLayer)}'
        key: true
      - label: Expansion ratio
        value: '{fixed(dFF / dModel, 2)}×'
        note: the 8/3 ≈ 2.67 heuristic, or wider if the model spends extra here
    usedBy:
      - Llama-2/3/4
      - Mistral
      - Qwen2/3
      - Gemma-2/3
      - DeepSeek
      - Phi-3
  - id: topk-moe
    label: Top-k MoE
    full: Sparsely-Gated Mixture of Experts
    year: 2017
    role: branch
    tagline: Many FFNs; a router picks k per token
    paper:
      title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer'
      url: https://arxiv.org/abs/1701.06538
      authors: Shazeer et al.
    concepts:
      - id: topk-moe-conditional-path
        label: Conditional parameter path
        kind: method
        summary: A router scores experts and sends each token through only its top-k FFNs.
        detail:
          - >-
            The selected expert outputs are weighted and added. Total parameter count can grow with
            the number of experts while per-token compute grows only with k.
      - id: topk-moe-routing-balance
        label: Routing is a systems problem
        kind: pitfall
        summary: Popular experts create overflow, communication imbalance, and unused capacity.
        detail:
          - >-
            Training needs a balancing signal or routing constraint, while distributed execution
            needs all-to-all token exchange. Sparse arithmetic alone does not make an MoE efficient.
    math:
      - title: Routing and combination
        tex: >-
          y = \sum_{i \in \text{TopK}(g)} \frac{\exp(g_i)}{\sum_{j \in \text{TopK}(g)} \exp(g_j)} \,
          E_i(x), \qquad g = xW_r
        where:
          - sym: W_r
            means: the router — a single $d_{\text{model}} \times E$ matrix, tiny next to the experts
          - sym: E_i
            means: expert $i$, itself a full FFN (SwiGLU in modern models)
          - sym: k
            means: '`num_experts_per_tok` — 2 in Mixtral, 8 in DeepSeek-V3'
        note: >-
          The softmax is over the selected k only, not all E. Normalising over all experts first and
          then selecting gives weights that do not sum to 1, which changes the output scale.
      - title: Load-balancing auxiliary loss
        tex: \mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i
        where:
          - sym: f_i
            means: fraction of tokens in the batch routed to expert $i$
          - sym: P_i
            means: mean router probability assigned to expert $i$
          - sym: \alpha
            means: weight, typically 0.01 — large enough to balance, small enough not to distort the task
        note: >-
          f is a hard count and not differentiable; P is its smooth surrogate. Multiplying them
          gives a term whose gradient flows through P while its magnitude tracks the real imbalance.
          Minimised when both are uniform at 1/E.
      - title: Total against active
        tex: >-
          P_{\text{total}} \approx E \cdot P_{\text{expert}}, \qquad P_{\text{active}} \approx k
          \cdot P_{\text{expert}}
        worked:
          - tex: >-
              8 \times 7\text{B}: \; P_{\text{total}} = 47\text{B}, \; P_{\text{active}} =
              13\text{B}
            caption: Mixtral — attention is shared, so it is not simply 8 × 7
          - tex: '\text{DeepSeek-V3}: \; 671\text{B} \to 37\text{B}'
            caption: 18× more knowledge than compute
    figures:
      - kind: routing
        title: Top-2 routing over eight experts
        tokens:
          - The
          - protein
          - folds
          - into
          - a
          - helix
        experts:
          - E₁
          - E₂
          - E₃
          - E₄
          - E₅
          - E₆
          - E₇
          - E₈
        routes:
          - token: 0
            expert: 2
            weight: 0.7
          - token: 0
            expert: 5
            weight: 0.3
          - token: 1
            expert: 0
            weight: 0.8
          - token: 1
            expert: 2
            weight: 0.2
          - token: 2
            expert: 0
            weight: 0.6
          - token: 2
            expert: 3
            weight: 0.4
          - token: 3
            expert: 2
            weight: 0.55
          - token: 3
            expert: 5
            weight: 0.45
          - token: 4
            expert: 2
            weight: 0.9
          - token: 4
            expert: 0
            weight: 0.1
          - token: 5
            expert: 0
            weight: 0.65
          - token: 5
            expert: 3
            weight: 0.35
        showLoad: true
        steps:
          - The router scores all eight experts for each token — one small matmul.
          - >-
            The top 2 scores win. Every other expert is skipped entirely for that token, which is
            where the compute saving comes from.
          - >-
            Edge thickness is the gate weight; the two experts' outputs are summed in that
            proportion.
          - >-
            Read the histogram: E₁ and E₃ take most of the traffic while E₅, E₇ and E₈ take none.
            This is collapse beginning, and it is the default behaviour without an auxiliary loss.
        caption: >-
          Six tokens, twelve routing decisions, and already three experts are idle. At batch scale
          this compounds — which is why the load-balancing term is not optional.
    code:
      - title: PyTorch
        language: python
        code: |-
          class TopKMoE(nn.Module):
              def __init__(self, d_model, d_ff, n_expert, k):
                  super().__init__()
                  self.k = k
                  self.router = nn.Linear(d_model, n_expert, bias=False)
                  self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(n_expert))

              def forward(self, x):
                  B, T, D = x.shape
                  flat = x.view(-1, D)
                  logits = self.router(flat)                       # [BT, E]

                  weights, idx = logits.topk(self.k, dim=-1)       # select first...
                  weights = weights.softmax(dim=-1)                # ...then normalise over the k

                  out = torch.zeros_like(flat)
                  for e, expert in enumerate(self.experts):
                      # Gather only the tokens routed here. This scatter/gather is why
                      # MoE is awkward on GPUs: the work per expert is data-dependent.
                      tok, slot = (idx == e).nonzero(as_tuple=True)
                      if tok.numel():
                          out[tok] += weights[tok, slot, None] * expert(flat[tok])
                  return out.view(B, T, D)
        note: >-
          Correctness-first. Production kernels sort tokens by expert and run one grouped GEMM,
          because this loop launches E separate kernels on ragged batches.
    cost:
      - label: Experts
        value: '{(nActive == 0 ? 2 : nActive)} of {(nExperts == 0 ? 8 : nExperts)} active per token'
      - label: Total FFN params
        value: '{si(3 * dModel * dFF * (nExperts == 0 ? 8 : nExperts) * nLayer)}'
        note: what the checkpoint weighs
      - label: Active FFN params
        value: '{si(3 * dModel * dFF * (nActive == 0 ? 2 : nActive) * nLayer)}'
        note: what each token pays for
        key: true
      - label: Sparsity ratio
        value: >-
          {fixed((nExperts == 0 ? 8 : nExperts) / (nActive == 0 ? 2 : nActive), 1)}× more knowledge
          than compute
    distinctions:
      - title: Sparse compute is not sparse memory
        body: >-
          An MoE activates 13B of 47B parameters, so it computes like a 13B model. It does not *fit*
          like one — all 47B sit in VRAM because any token might need any expert. MoE trades memory
          capacity for compute, which is a good trade on a datacentre GPU and a bad one on a laptop.
    usedBy:
      - Mixtral-8×7B
      - Mixtral-8×22B
      - Qwen2-MoE
      - Grok-1
      - DBRX
  - id: switch
    label: Switch
    full: Switch Transformer (top-1 routing)
    year: 2021
    role: refinement
    tagline: k = 1 — one expert per token
    paper:
      title: 'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity'
      url: https://arxiv.org/abs/2101.03961
      authors: Fedus, Zoph, Shazeer
    concepts:
      - id: switch-single-expert
        label: Top-1 dispatch
        kind: method
        summary: Each token is assigned to one expert rather than combining several expert outputs.
        detail:
          - >-
            Selecting a single path halves expert-side compute and communication relative to a
            common top-2 configuration. The router's selected probability is still used to scale
            that expert's response.
      - id: switch-capacity-overflow
        label: Fixed capacity per expert
        kind: pitfall
        summary: Each expert accepts a bounded number of tokens in a batch.
        detail:
          - >-
            Capacity protects throughput but can drop or reroute tokens when the router concentrates
            demand. The capacity factor trades wasted expert slots against overflow risk.
    math:
      - title: Expert capacity
        tex: C = \left\lceil \frac{T_{\text{batch}}}{E} \cdot f \right\rceil
        where:
          - sym: f
            means: capacity factor, typically 1.0–1.25
          - sym: C
            means: slots per expert; tokens beyond this are dropped
        note: >-
          f = 1.0 means zero slack: perfectly uniform routing fills every slot exactly. Real routing
          is never uniform, so f = 1.0 drops tokens and f = 2.0 wastes half the compute on padding.
    figures:
      - kind: bars
        title: 'Capacity factor: padding against dropping'
        categories:
          - f = 1.0
          - f = 1.25
          - f = 1.5
          - f = 2.0
        series:
          - label: tokens dropped (%)
            values:
              - 11
              - 4
              - 1.5
              - 0.2
          - label: compute wasted on padding (%)
            values:
              - 0
              - 18
              - 33
              - 52
        showValues: true
        caption: >-
          Illustrative shape of the trade. There is no free setting — the only real escape is to
          change the routing rule, which is what expert-choice does.
    usedBy:
      - Switch Transformer
      - GLaM (variant)
      - ST-MoE
  - id: expert-choice
    label: Expert Choice
    full: Expert-Choice Routing
    year: 2022
    role: branch
    tagline: Experts pick tokens, not the other way round
    paper:
      title: Mixture-of-Experts with Expert Choice Routing
      url: https://arxiv.org/abs/2202.09368
      authors: Zhou et al., Google
    concepts:
      - id: expert-choice-transposed-router
        label: Experts select their workload
        kind: method
        summary: Each expert takes its highest-scoring tokens instead of each token choosing experts.
        detail:
          - >-
            The router score matrix is used in the transposed direction: rows for experts choose a
            fixed quota of tokens. This directly fixes expert load before dispatch.
      - id: expert-choice-variable-token-paths
        label: Uneven token coverage
        kind: tradeoff
        summary: Load balance is guaranteed per expert, but tokens can receive different numbers of experts.
        detail:
          - >-
            Some tokens may be selected by several experts while others are selected by none. That
            changes the computation semantics from the familiar fixed top-k-per-token formulation.
    math:
      - title: Transposed selection
        tex: >-
          \text{token-choice: } \; \text{TopK}_{\text{experts}}(g_{t,\cdot}) \qquad\text{vs.}\qquad
          \text{expert-choice: } \; \text{TopC}_{\text{tokens}}(g_{\cdot,e})
        note: >-
          The same score matrix g, reduced along the other axis. Balance stops being something you
          encourage with a loss and becomes something the operation cannot violate.
    usedBy:
      - Google research models
      - used in training-side ablations more than shipped decoders
  - id: shared-expert
    label: Shared + fine-grained
    full: DeepSeekMoE — fine-grained experts with shared isolation
    year: 2024
    role: synthesis
    tagline: Split experts finer, pin a few always-on
    paper:
      title: 'DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models'
      url: https://arxiv.org/abs/2401.06066
      authors: Dai et al., DeepSeek
    concepts:
      - id: shared-expert-common-path
        label: Preserve a common path
        kind: method
        summary: Always-on shared experts handle broadly useful features while routed experts specialise.
        detail:
          - >-
            Every token receives the shared contribution, then a small set of fine-grained routed
            experts. Shared capacity prevents common knowledge from being redundantly relearned in
            every specialist.
      - id: shared-expert-combinatorial-space
        label: More combinations, steady active cost
        kind: tradeoff
        summary: Finer experts increase the number of possible routed paths without activating more of them.
        detail:
          - >-
            The router has a larger combinatorial choice set while each token still evaluates a
            small active subset. This makes routing quality and load balancing even more central to
            usable capacity.
    math:
      - title: The layer
        tex: >-
          y = \underbrace{\sum_{s=1}^{N_s} E_s^{\text{shared}}(x)}_{\text{always}} \;+\;
          \underbrace{\sum_{i \in \text{TopK}(g)} g_i \, E_i^{\text{routed}}(x)}_{\text{selected}}
        where:
          - sym: N_s
            means: '`n_shared_experts` — 1 in DeepSeek-V3'
          - sym: E^{\text{routed}}
            means: '`n_routed_experts` = 256, each of width `moe_intermediate_size`'
      - title: Combinatorial capacity
        tex: >-
          \binom{E}{k}: \quad \binom{8}{2} = 28 \qquad\longrightarrow\qquad \binom{64}{8} = 4.4
          \times 10^{9}
        note: >-
          Identical active parameter count. What changes is how many distinct functions the layer
          can express by combination — which is the argument for why fine-graining is free capacity.
      - title: Auxiliary-loss-free balancing (V3)
        tex: g'_i = g_i + b_i, \qquad b_i \leftarrow b_i + \gamma \cdot \text{sign}(\bar{f} - f_i)
        note: >-
          DeepSeek-V3 drops the auxiliary loss entirely. Instead a per-expert bias is nudged up when
          an expert is underused and down when overused. The bias steers *selection* but is excluded
          from the gate weight, so balancing never distorts the output — which was the standing
          objection to auxiliary losses.
    figures:
      - kind: routing
        title: One shared expert plus top-2 of six routed
        tokens:
          - The
          - protein
          - folds
          - into
        experts:
          - Shared
          - E₁
          - E₂
          - E₃
          - E₄
          - E₅
          - E₆
        shared:
          - 0
        routes:
          - token: 0
            expert: 0
            weight: 1
          - token: 0
            expert: 2
            weight: 0.6
          - token: 0
            expert: 5
            weight: 0.4
          - token: 1
            expert: 0
            weight: 1
          - token: 1
            expert: 1
            weight: 0.7
          - token: 1
            expert: 4
            weight: 0.3
          - token: 2
            expert: 0
            weight: 1
          - token: 2
            expert: 1
            weight: 0.5
          - token: 2
            expert: 3
            weight: 0.5
          - token: 3
            expert: 0
            weight: 1
          - token: 3
            expert: 2
            weight: 0.8
          - token: 3
            expert: 6
            weight: 0.2
        showLoad: true
        steps:
          - 'The dashed expert is shared: every token goes through it, no routing decision involved.'
          - The router then picks 2 of the 6 routed experts, exactly as in top-k MoE.
          - >-
            Because general knowledge is handled by the shared expert, the routed experts see a
            narrower job and specialise harder.
          - >-
            Compare the load histogram against the plain top-k figure: the shared expert absorbs the
            common traffic that would otherwise have concentrated on one or two routed experts.
      - kind: bars
        title: Total against active parameters
        categories:
          - Mixtral 8×7B
          - DBRX
          - DeepSeek-V3
          - Llama-3-70B
        series:
          - label: total (B)
            values:
              - 47
              - 132
              - 671
              - 70
          - label: active (B)
            values:
              - 13
              - 36
              - 37
              - 70
        showValues: true
        caption: >-
          The rightmost pair is dense — total and active are the same number, which is exactly what
          MoE exists to separate. DeepSeek-V3 activates fewer parameters than Llama-3-70B while
          holding nearly ten times as many.
    code:
      - title: Detecting it from a config
        language: json
        code: |-
          {
            "model_type": "deepseek_v3",
            "n_routed_experts": 256,        // fine-grained: many, narrow
            "num_experts_per_tok": 8,       // k
            "n_shared_experts": 1,          // ← the tell for this variant
            "moe_intermediate_size": 2048,  // per-expert width, far below intermediate_size
            "intermediate_size": 18432,     // the dense FFN width, used by the first few layers
            "first_k_dense_replace": 3      // layers 0-2 stay dense — routing is unstable early
          }
        note: >-
          first_k_dense_replace is a detail worth noticing: the first few layers are ordinary dense
          FFNs, because routing on barely-formed early representations does not work well.
    cost:
      - label: Experts
        value: >-
          {(nActive == 0 ? 8 : nActive)} routed of {(nExperts == 0 ? 256 : nExperts)}, plus
          {(nShared == 0 ? 1 : nShared)} shared
      - label: Total FFN params
        value: >-
          {si(3 * dModel * dFF * ((nExperts == 0 ? 256 : nExperts) + (nShared == 0 ? 1 : nShared)) *
          nLayer)}
      - label: Active FFN params
        value: >-
          {si(3 * dModel * dFF * ((nActive == 0 ? 8 : nActive) + (nShared == 0 ? 1 : nShared)) *
          nLayer)}
        key: true
      - label: Routing combinations
        value: '{(nExperts == 0 ? 256 : nExperts)} choose {(nActive == 0 ? 8 : nActive)}'
        note: against 8-choose-2 = 28 for a Mixtral-style layer
    usedBy:
      - DeepSeek-V2
      - DeepSeek-V3
      - DeepSeek-R1
      - Qwen3-MoE
      - Kimi K2
---

## role

Attention moves information *between* positions. This block is the only place a transformer does per-position computation with any depth to it, and it is where roughly two thirds of the parameters sit.

The standard framing is that attention routes and the FFN recalls: probing work consistently finds factual associations stored in these matrices, which is why editing methods like ROME target them and not the attention heads.

The design space splits into two questions that developed independently and then merged. **What non-linearity, and how is it gated?** — the dense line, from ReLU through GLU to SwiGLU. **Does every token pay for every parameter?** — the sparse line, from Shazeer's mixture-of-experts through Switch to DeepSeek. Modern frontier models answer both at once: a mixture of SwiGLU experts.

## relu-mlp

Two matrices and a non-linearity. Widen the residual stream by 4×, clip the negatives, project back. Applied identically and independently at every position.

The 4× is pure convention — it appears in the original paper with no ablation and was inherited by nearly everything for the next five years. The width matters because it sets how many "features" the layer can hold: read as a key–value memory, the up-projection's rows are keys and the down-projection's columns are the values retrieved when a key fires.

## gelu-mlp

Replace the hard gate with a soft one: weight each input by the probability that a standard normal falls below it. Small negatives pass through attenuated rather than annihilated, so units that drift negative can recover.

This is the GPT-2 / BERT-era default, and it is still what you will find in most encoder checkpoints. The `hidden_act` field in a config saying `gelu` or `gelu_new` means the model is on this branch.

### fixes

ReLU's hard zero kills the gradient for any unit that drifts negative, and the kink at the origin is a discontinuity in the derivative.

## geglu

Split the up-projection in two. One branch passes through the activation and becomes a **gate**; the other stays linear and becomes the **value**. Multiply them elementwise.

The result is multiplicative rather than additive: the layer can now suppress a feature entirely based on a different feature, which a pointwise non-linearity cannot express at all. Shazeer's paper is famously terse about *why* it works — the conclusion offers "divine benevolence" — but the empirical result held up and every gated variant beat its ungated counterpart.

### fixes

A pointwise activation applies the same fixed non-linearity everywhere. It cannot make the transformation depend on the input.

## swiglu

GEGLU with `SiLU(x) = x·σ(x)` in place of GELU. The two activations are numerically close, but SiLU needs only a sigmoid where GELU needs an erf or a cubic approximation.

This is the dense FFN of the modern era. Llama, Mistral, Qwen, Gemma, Phi, DeepSeek — all SwiGLU. When a config says `"hidden_act": "silu"` alongside a `gate_proj` in the weight names, this is what it is.

The three-matrix structure also explains a detail that confuses people reading configs: `intermediate_size` is *not* 4×`hidden_size` in these models. It is chosen so that 3 × d_model × d_ff lands near the 8 × d_model² a two-matrix FFN would have used, then rounded to something friendly to the hardware.

### fixes

Nothing wrong with GEGLU — SiLU is simply cheaper than GELU's erf and scored marginally better in the same ablation.

## topk-moe

Replace the single FFN with `E` independent FFNs — the **experts** — plus a small linear **router**. For each token the router scores every expert, the top `k` win, and only those `k` run. Their outputs are combined weighted by the router's softmax.

This decouples the two things a dense layer conflates. **Total** parameters set how much the model can know; **active** parameters set what each token costs. Mixtral-8×7B holds 47B parameters but activates 13B per token, so it costs roughly a 13B model to run and behaves closer to something much larger.

The catch is that the router is trained by gradient descent and gradient descent likes winners. Left alone, a few experts get chosen early, improve fastest because they see the most tokens, and get chosen more — until most experts are dead weight. Every MoE system therefore ships a **load-balancing loss** that penalises uneven routing, and getting that auxiliary term right is most of the difficulty of training an MoE.

### fixes

A dense FFN forces every token to pay for every parameter. Capacity and compute are welded together, so scaling knowledge means scaling cost.

## switch

Route each token to exactly one expert. Halves expert FLOPs and communication against top-2, and simplifies the implementation enormously — no combination step, no second gather.

The paper also introduced **expert capacity**, which is where MoE stops being elegant. Because hardware needs fixed-size tensors, each expert gets a fixed number of slots per batch. Tokens arriving at a full expert are **dropped** — they skip the FFN and pass through on the residual alone. The capacity factor trades wasted padding against dropped tokens, and no setting avoids both.

### fixes

Shazeer argued k ≥ 2 was necessary for the router to get a useful gradient. That turned out to be false, and k = 2 costs twice the expert compute and twice the communication.

## expert-choice

Invert the argmax. Instead of each token choosing its top-k experts, each expert chooses its top-c tokens from the batch. Load is then **perfectly balanced by construction** — every expert processes exactly c tokens, always. No auxiliary loss, no capacity factor, no dropped tokens.

The cost is that tokens no longer get a guaranteed allocation. A token unpopular with every expert may be picked by none and skip the FFN entirely, while an "interesting" token may be picked by many. That variable per-token compute is arguably a feature during training and a problem at inference — and it breaks causality, because an expert choosing its top tokens from the batch has looked at the whole batch, including future positions. That makes it unusable as-is for autoregressive decoding, which is why it stayed a training-side idea.

### fixes

Token-choice routing cannot guarantee balance, so it needs both an auxiliary loss and a capacity limit, and still drops tokens.

## shared-expert

**Fine-grained segmentation.** Rather than 8 experts of width `d_ff`, use 64 experts of width `d_ff/8` and route to 8 of them. Active parameters are unchanged, but the number of possible expert *combinations* explodes — from 28 for choose-2-of-8 to over four billion for choose-8-of-64. Each expert can now specialise narrowly because the combination carries the generality.

**Shared expert isolation.** Designate one or two experts that every token passes through unconditionally. General knowledge — syntax, common words, the basics every token needs — lives there once, instead of being redundantly relearned inside all 64 routed experts. The routed experts are then free to be genuinely specialised.

These two ideas are complementary and both are needed: fine-graining alone makes redundancy worse, because now 64 experts each need the common knowledge. The shared expert is what absorbs it.

This is the architecture behind DeepSeek-V3 at 671B total and 37B active — an 18× ratio that neither Mixtral's 8-of-8 nor Switch's 1-of-N reached.

### fixes

With a handful of large experts, every expert must independently learn the same general-purpose knowledge, wasting capacity on redundancy — and coarse experts cannot specialise sharply.

---
id: embedding
label: Token Embedding
ordinal: '2'
icon: vector
slot: input
tagline: Integer ids become vectors in the residual stream
io:
  in: ids [B, T]
  out: x [B, T, d_model]
defaultVariant: untied
lineage:
  - from: untied
    to: tied
    kind: derives
    label: reuse the same matrix at the output
  - from: untied
    to: factorized
    kind: fixes
    label: decouple vocab width from model width
  - from: tied
    to: scaled
    kind: derives
    label: rescale so the shared matrix suits both ends
variants:
  - id: untied
    label: Untied
    full: Independent input and output embeddings
    year: 2017
    role: origin
    tagline: Separate matrices for reading in and writing out
    concepts:
      - id: lookup-row
        label: A token becomes one learned row
        kind: idea
        summary: The input embedding looks up a dense vector for each integer token id.
        detail:
          - >-
            This is the step where a discrete symbol enters the continuous residual stream. Every
            later transformer operation works on that vector representation.
      - id: separate-directions
        label: Reading and predicting are separate jobs
        kind: method
        summary: Untied embeddings use one matrix for input lookup and another for output logits.
        detail:
          - >-
            The model is free to organize input meaning and output prediction differently. That
            flexibility costs a second vocabulary-by-model-width matrix.
      - id: vocabulary-cost
        label: Vocabulary size drives parameter cost
        kind: metric
        summary: The embedding and output matrices each scale with vocabulary size times model width.
        detail:
          - >-
            For smaller models, these tables can dominate the parameter budget. That pressure is what
            motivates weight tying and factorized embeddings.
    figures:
      - kind: embedding
        title: What a row of the embedding table actually holds
        steps:
          - >-
            Pick a token. Panel 1 is literally its row — d_model signed numbers, blue negative, red
            positive.
          - >-
            Panel 2 projects every row to 2D. Tokens used in similar contexts end up near each
            other, which nothing in the architecture asks for; it falls out of the training
            objective.
          - >-
            The four dashed arrows are the same offset. Add it to king and you land on queen — the
            sense in which directions, not just positions, carry meaning.
          - >-
            Panel 3 is the quantity the other two are gesturing at: cosine similarity between the
            selected row and every other.
        caption: >-
          Vectors are synthetic — generated with per-cluster structure and one deliberate semantic
          axis, so the analogy resolves cleanly. Real GloVe rows would need a megabyte of data to
          make the same three points. The behaviour is faithful; the numbers are illustrative.
    cost:
      - label: Input embedding
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
      - label: Output head
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
      - label: Combined
        value: '{fixed(2 * vocab * dModel / 1e6, 0)} M'
        key: true
      - label: ''
        value: ''
    usedBy:
      - GPT-3
      - Llama-2/3 (larger sizes)
      - Mistral
  - id: tied
    label: Tied
    full: Weight tying between embedding and output head
    year: 2016
    role: refinement
    tagline: One matrix, used at both ends
    paper:
      title: Using the Output Embedding to Improve Language Models
      url: https://arxiv.org/abs/1608.05859
      authors: Press & Wolf
    concepts:
      - id: tied-shared-geometry
        label: One vocabulary geometry
        kind: idea
        summary: Input lookup and output scoring use the same learned token directions.
        detail:
          - >-
            A row is selected when a token enters the model, and that same row is compared with
            the final hidden state when the model predicts a token. Input meaning and output
            preference therefore live in one shared coordinate system.
      - id: tied-coupled-gradients
        label: Coupled learning signal
        kind: tradeoff
        summary: Gradients from reading and predicting tokens update one parameter matrix.
        detail:
          - >-
            Sharing saves a vocabulary-sized matrix and regularises rare-token representations.
            It also removes the freedom for the input and output spaces to specialise separately,
            which is less attractive once that matrix is a small part of a large model.
    math:
      - title: Both directions, one matrix
        tex: x = W_{\text{emb}}[\text{id}], \qquad \text{logits} = h\,W_{\text{emb}}^\top
        worked:
          - tex: 128256 \times 2048 = 263\text{M saved}
            caption: Llama-3.2-1B — about 20% of the model
          - tex: 128256 \times 8192 = 1.05\text{B saved}
            caption: a 70B model — under 2%, so much less compelling
        note: >-
          The saving is constant in absolute terms and shrinking in relative terms, which is exactly
          why small models tie and large ones do not.
    cost:
      - label: Shared matrix
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
      - label: Saved against untied
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
        key: true
      - label: Share of a model this size
        value: '{fixed(100 * vocab * dModel / (12 * nLayer * dModel * dModel + vocab * dModel), 1)}%'
        note: rough — trunk parameters estimated at 12·L·d²
    usedBy:
      - Gemma (all)
      - Llama-3.2-1B/3B
      - Qwen3-0.6B
      - GPT-2
      - T5
  - id: factorized
    label: Factorized
    full: Factorized embedding parameterization
    year: 2019
    role: branch
    tagline: Embed small, then project up to d_model
    paper:
      title: 'ALBERT: A Lite BERT'
      url: https://arxiv.org/abs/1909.11942
    concepts:
      - id: factorized-bottleneck
        label: Separate lexical and contextual width
        kind: method
        summary: A narrow vocabulary table is projected into the wider residual stream.
        detail:
          - >-
            The vocabulary matrix uses an embedding width e while transformer layers use
            d_model. The projection is shared across all tokens, so only the expensive
            vocabulary-by-width term is narrowed.
      - id: factorized-cost-shape
        label: Savings versus an extra projection
        kind: tradeoff
        summary: Factorisation replaces V×d parameters with V×e plus e×d work and parameters.
        detail:
          - >-
            The saving is largest when vocabulary size is high and e is much smaller than
            d_model. The additional projection runs on every token, so modern decoder designs
            often prefer capacity in the direct table when the budget allows it.
    math:
      - title: Factorisation saving
        tex: V \cdot d \;\longrightarrow\; V \cdot e + e \cdot d
        worked:
          - tex: 30000\cdot 768 = 23\text{M} \;\to\; 30000\cdot128 + 128\cdot768 = 3.9\text{M}
            caption: ALBERT — a 6× reduction
    usedBy:
      - ALBERT
      - some multilingual encoders
  - id: scaled
    label: Scaled
    full: Embedding multiplied by √d_model
    year: 2017
    role: refinement
    tagline: Multiply the looked-up vector by √d before use
    concepts:
      - id: scaled-variance-match
        label: Match residual scale
        kind: formula
        summary: The lookup vector is multiplied by √d_model before it enters the first layer.
        detail:
          - >-
            Embedding entries initialise at a small scale. The multiplier brings their aggregate
            magnitude closer to the residual activations expected by subsequent attention and
            feed-forward layers.
      - id: scaled-forward-contract
        label: A forward-pass convention
        kind: pitfall
        summary: The scale belongs in the model computation, not in a converted checkpoint file.
        detail:
          - >-
            Omitting it preserves tensor shapes and produces valid logits, which makes the bug
            easy to miss. A faithful implementation applies the same multiplication at inference
            and training time.
    math:
      - title: Scaling
        tex: x = W_{\text{emb}}[\text{id}] \cdot \sqrt{d_{\text{model}}}
        worked:
          - tex: \sqrt{2048} \approx 45.3
            caption: Gemma-2-2B — a substantial multiplier, not a rounding detail
    distinctions:
      - title: A common porting bug
        body: >-
          The scale is applied in the model's forward pass, not baked into the stored weights.
          Reimplementations that skip it produce a model that runs, emits plausible-looking logits,
          and is quietly much worse — with no error anywhere to point at it.
    usedBy:
      - Transformer (2017)
      - Gemma
      - T5 (variant)
---

## role

A lookup: row `i` of a `vocab × d_model` matrix is the initial representation of token `i`. This is the only place discrete input becomes continuous, and everything after it is differentiable.

The design space is small but the numbers are not. At a 128k vocabulary and `d_model` 4096, this table is 500M parameters — around 7% of an 8B model and well over 20% of a 1B one. For small models the embedding is a serious fraction of the budget, which is why weight tying and factorisation exist.

## untied

The input embedding and the output projection are different matrices learned independently. Costs `2 × vocab × d_model` parameters, and is what large models generally do — once the model is big enough, the embedding is a small fraction of the total and the extra freedom is worth having.

A config with `tie_word_embeddings: false`, or without the key at all in older models, is here.

## tied

Use `W_emb` for the lookup and `W_embᵀ` for the output logits. Halves the embedding cost, and acts as a regulariser: the output gradient now also trains the input representation, which particularly helps rare tokens that appear seldom as input but must still be predictable.

Near-universal in small models, where the embedding would otherwise dominate. Gemma-2-2B, Qwen3-0.6B and Llama-3.2-1B all tie; their larger siblings often do not.

### fixes

Two separate matrices spend twice the parameters on what is arguably the same object — a map between token identity and vector space.

## factorized

Factor the table: `vocab × e` followed by `e × d_model`, with `e ≪ d_model`. Cost falls from `V·d` to `V·e + e·d`, which is a large saving when the vocabulary is big.

Rarely used in modern decoders — the extra matmul sits on every token and the parameter saving is less pressing than it was — but it is the clearest statement of the underlying point, that vocabulary width and model width are independent quantities.

### fixes

Tying forces the embedding width to equal the model width. But a token id needs far less capacity to represent than a contextualised hidden state does — the two dimensions are conflated for no reason.

## scaled

Present in the original transformer and revived by Gemma. Embedding rows initialise with small variance; multiplying by `√d_model` puts them on the same scale as the activations flowing through the residual stream, so the first layer does not receive an input orders of magnitude smaller than what later layers see.

It matters more under weight tying, where one matrix must serve both a lookup (wants small values) and a logit projection (wants larger ones). The scale reconciles the two.

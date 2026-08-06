---
id: tokenizer
label: Tokenizer
ordinal: '1'
icon: chunks
slot: input
tagline: Cutting text into the units the model actually sees
io:
  in: text (string)
  out: ids [B, T]
defaultVariant: byte-bpe
caveat: >-
  config.json does not record the tokenizer algorithm — that lives in tokenizer.json alongside the
  merges. A model loaded from Hugging Face will show this block as unresolved rather than guessed.
lineage:
  - from: word
    to: char
    kind: fixes
    label: unbounded vocabulary, no unseen words
  - from: word
    to: bpe
    kind: fixes
    label: keep frequent words, split rare ones
  - from: char
    to: bpe
    kind: fixes
    label: sequences far too long
  - from: bpe
    to: wordpiece
    kind: derives
    label: merge by likelihood, not frequency
  - from: bpe
    to: byte-bpe
    kind: fixes
    label: start from bytes — never fail on any input
  - from: wordpiece
    to: unigram
    kind: replaces
    label: prune from a superset instead of building up
  - from: byte-bpe
    to: blt
    kind: replaces
    label: learn the segmentation end to end
variants:
  - id: word
    label: Word-level
    full: Whitespace/word tokenization
    year: 2013
    role: legacy
    tagline: One token per word
    concepts:
      - id: word-atomic-units
        label: Treat words as atomic symbols
        kind: method
        summary: Each observed word type receives one vocabulary id and one embedding row.
        detail:
          - >-
            Segmentation is simple and common words are short, but morphology and spelling inside a
            word are invisible to the model once the lookup is made.
      - id: word-oov-boundary
        label: Unknown words are a hard boundary
        kind: pitfall
        summary: A word not in the fixed vocabulary cannot be represented precisely.
        detail:
          - >-
            The vocabulary grows with corpus diversity, yet an open-ended language always produces
            names, typos, and new forms outside it. Subword tokenizers exist to soften this failure.
    usedBy:
      - word2vec
      - pre-2016 NLP
  - id: char
    label: Character
    full: Character-level tokenization
    year: 2015
    role: branch
    tagline: One token per character
    concepts:
      - id: char-universal-alphabet
        label: Use a small compositional alphabet
        kind: method
        summary: Text is represented as individual characters rather than learned multi-character units.
        detail:
          - >-
            New spellings can be composed from known characters, which makes the representation
            robust to rare words and orthographic variation.
      - id: char-sequence-inflation
        label: Coverage costs sequence length
        kind: tradeoff
        summary: Fine units make even ordinary words many model steps long.
        detail:
          - >-
            Longer sequences consume context and raise attention cost. Character models must learn
            word and phrase structure over far more positions than subword models.
    usedBy:
      - char-RNN
      - ByT5
      - CANINE
  - id: bpe
    label: BPE
    full: Byte-Pair Encoding
    year: 2016
    role: origin
    tagline: Repeatedly merge the most frequent adjacent pair
    paper:
      title: Neural Machine Translation of Rare Words with Subword Units
      url: https://arxiv.org/abs/1508.07909
      authors: Sennrich et al.
    concepts:
      - id: bpe-greedy-merges
        label: Build units by frequent merges
        kind: method
        summary: Training repeatedly merges the most frequent adjacent symbol pair in the corpus.
        detail:
          - >-
            The learned merge list defines a hierarchy from characters or bytes to common subwords.
            Encoding replays that ordered list to construct the same units.
      - id: bpe-irreversible-history
        label: Early merges become the vocabulary
        kind: tradeoff
        summary: Greedy frequency choices are never reconsidered once added to the merge table.
        detail:
          - >-
            BPE is simple and effective, but a merge useful in one corpus can consume vocabulary
            capacity without helping another language or domain. Unigram LM takes a pruning route instead.
    math:
      - title: The merge rule
        tex: (a, b)^* = \arg\max_{(a,b)} \; \text{count}(ab), \qquad V \leftarrow V \cup \{ab\}
        note: >-
          Greedy and purely frequency-driven — no likelihood model, no lookahead. Applied at encode
          time in the same order it was learned, which is why the merge list must ship with the
          model.
    example:
      beforeLabel: text
      before: tokenization is unbelievable
      afterLabel: tokens
      after: '["token", "ization", " is", " un", "bel", "ievable"]'
      mono: true
    usedBy:
      - GPT-2 (byte variant)
      - RoBERTa
      - most modern LLMs
  - id: wordpiece
    label: WordPiece
    full: WordPiece
    year: 2016
    role: branch
    tagline: Merge the pair that most improves corpus likelihood
    concepts:
      - id: wordpiece-association-score
        label: Reward distinctive pairs
        kind: formula
        summary: WordPiece favours pairs that co-occur more than their individual frequencies predict.
        detail:
          - >-
            A very common pair is not automatically useful when its parts are independently common.
            The selection criterion approximates an association or likelihood gain rather than raw count.
      - id: wordpiece-greedy-segmentation
        label: Segment by longest legal continuation
        kind: method
        summary: Encoding chooses the longest vocabulary piece beginning at the current character position.
        detail:
          - >-
            Continuation markers preserve word-internal boundaries. If no complete segmentation is
            available, the tokenizer emits an unknown token rather than falling back to bytes.
    math:
      - title: Selection criterion
        tex: \arg\max_{(a,b)} \; \frac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}
        note: >-
          A pointwise mutual information score — it favours pairs that co-occur more than chance,
          not merely often.
    usedBy:
      - BERT
      - DistilBERT
      - ELECTRA
  - id: unigram
    label: Unigram LM
    full: Unigram language model tokenization
    year: 2018
    role: branch
    tagline: Start from a large vocabulary and prune it down
    paper:
      title: Subword Regularization
      url: https://arxiv.org/abs/1804.10959
      authors: Kudo
    concepts:
      - id: unigram-oversupply-first
        label: Start with too many pieces
        kind: method
        summary: Unigram training begins from a large candidate vocabulary and removes pieces that help least.
        detail:
          - >-
            The model evaluates how removing a piece changes corpus likelihood, allowing earlier
            segmentation choices to be revised instead of committing to one merge history.
      - id: unigram-probabilistic-segmentation
        label: More than one segmentation can matter
        kind: tradeoff
        summary: A probabilistic piece model supports sampling alternative tokenizations during training.
        detail:
          - >-
            This can improve robustness to segmentation boundaries, but requires dynamic programming
            rather than the simple ordered merge replay used by BPE.
    usedBy:
      - T5
      - ALBERT
      - XLNet
      - mBART
  - id: byte-bpe
    label: Byte-level BPE
    full: Byte-level Byte-Pair Encoding
    year: 2019
    role: refinement
    tagline: Run BPE over raw bytes — nothing is ever unrepresentable
    concepts:
      - id: byte-alphabet
        label: A universal starting alphabet
        kind: idea
        summary: Any text can be represented because every input is first expressed as bytes.
        detail:
          - >-
            Byte-level BPE starts from 256 byte values rather than from a language-specific character
            inventory. It can always fall back to those bytes when it sees an unfamiliar symbol.
      - id: learned-merges
        label: Learned merges make common text short
        kind: method
        summary: Frequent byte sequences are merged into reusable subword units during tokenizer training.
        detail:
          - >-
            The merge table spends vocabulary capacity where the training corpus says it is useful.
            Common patterns receive short encodings while rare strings remain decomposable.
      - id: token-budget
        label: Token count is a budget
        kind: tradeoff
        summary: A fair byte-level fallback can still use far more tokens for some languages and scripts.
        detail:
          - >-
            More tokens consume context, attention compute, and money. The tokenizer is therefore part
            of a model's effective context length, not merely a preprocessing detail.
    figures:
      - kind: bars
        title: Tokens for the same sentence across languages
        categories:
          - English
          - Spanish
          - Chinese
          - Hindi
          - Burmese
        series:
          - label: tokens (GPT-4 tokenizer)
            values:
              - 10
              - 13
              - 17
              - 32
              - 72
        showValues: true
        caption: >-
          Roughly the pattern reported across tokenizer fairness studies. Identical meaning, 7× the
          cost, and 7× the context consumed — a direct consequence of byte-level merges being
          learned on English-heavy corpora.
    example:
      beforeLabel: input with an unseen glyph
      before: café ☕ 日本
      afterLabel: always representable — worst case, one token per byte
      after: '["caf", "é", " ☕", " 日", "本"]'
      mono: true
    cost:
      - label: Vocabulary
        value: '{num(vocab)}'
        note: 256 byte base plus learned merges
      - label: Embedding parameters
        value: '{fixed(vocab * dModel / 1e6, 0)} M'
        note: vocab × d_model — often 10%+ of a small model
        key: true
    usedBy:
      - GPT-2/3/4
      - Llama-3
      - Mistral
      - Qwen
      - most modern LLMs
  - id: blt
    label: Byte Latent
    full: Byte Latent Transformer — tokenizer-free
    year: 2024
    role: frontier
    tagline: No fixed vocabulary; group bytes dynamically by entropy
    paper:
      title: 'Byte Latent Transformer: Patches Scale Better Than Tokens'
      url: https://arxiv.org/abs/2412.09871
      authors: Meta AI
    concepts:
      - id: blt-adaptive-patches
        label: Learn where bytes should be grouped
        kind: method
        summary: A byte stream is dynamically partitioned into variable-size patches before latent processing.
        detail:
          - >-
            Entropy or model state can allocate short patches to unpredictable text and long patches
            to redundant spans. The segmentation is part of the learned computation, not a fixed file.
      - id: blt-end-to-end-cost
        label: Remove vocabulary bias, add patching work
        kind: tradeoff
        summary: Byte coverage is universal, but the model must learn efficient grouping and byte-level structure.
        detail:
          - >-
            There is no out-of-vocabulary failure or fixed tokenizer fairness bias. Performance now
            depends on the patcher and latent encoder using sequence budget as effectively as subwords.
    usedBy:
      - Meta research
---

## role

A transformer operates on a fixed vocabulary of integer ids. Something has to decide what the units are, and that decision is upstream of everything — the model literally cannot represent a distinction the tokenizer erased.

The design space is a single trade-off with two bad ends. **Large units** (words) give short sequences but an unbounded vocabulary and no way to handle anything unseen. **Small units** (bytes) give a tiny vocabulary and perfect coverage but sequences so long that attention becomes unaffordable. Subword methods sit between them, and the argument is only about how to choose the split.

Almost every "why can't the model count the r's in strawberry" question resolves here rather than in the model.

## word

Split on whitespace and punctuation, assign each distinct word an id. Intuitive, and unusable: the vocabulary is unbounded, morphology is invisible (`run`/`running` are unrelated ids), and any word absent from training becomes `<UNK>` — information destroyed before the model sees it.

## char

A vocabulary of a few hundred, no unknown tokens ever, and perfect access to spelling. The problem is length: sequences grow roughly 5×, and since attention is quadratic that is a 25× cost increase for the same text.

Character models are notably good at exactly the tasks subword models fail — counting letters, reversing strings, character-level manipulation.

## bpe

Start from characters. Count every adjacent pair in the corpus, merge the most frequent into a new symbol, repeat until the vocabulary reaches the target size. The learned merge list *is* the tokenizer.

The outcome is that common words end up as single tokens while rare words decompose into meaningful pieces — `unbelievable` becomes `un` + `believ` + `able`. Frequency buys shortness where it is worth having.

### fixes

Word-level cannot handle unseen words; character-level makes sequences unaffordable. Learn the units from the data instead of fixing them.

## wordpiece

Same loop, different criterion: pick the merge maximising the likelihood gain, `count(ab) / (count(a)·count(b))`, rather than raw count. Slightly better vocabularies at slightly higher training cost. BERT's tokenizer, recognisable by its `##` continuation prefix.

### fixes

BPE merges whatever is most frequent, which is not the same as whatever is most useful — a frequent pair whose parts are individually just as frequent gains nothing.

## unigram

Work backwards. Begin with a large candidate vocabulary, fit a unigram language model over segmentations with EM, and iteratively drop the tokens whose removal costs the least likelihood.

Because it is probabilistic, a string has many possible segmentations with different probabilities — which enables **subword regularisation**: sample a different segmentation each epoch as data augmentation. Usually paired with SentencePiece.

### fixes

Both BPE and WordPiece build up greedily and commit to every merge. A wrong early merge is permanent.

## byte-bpe

Use the 256 byte values as the base alphabet and merge upward from there. Any byte sequence — any language, emoji, corrupted encoding, raw binary — is representable, and the base vocabulary is exactly 256.

This is the modern default. GPT-2 introduced it, `tiktoken` and most Llama-family tokenizers implement it. Its known weakness is that non-Latin scripts pay several bytes per character before merging, so the same text costs materially more tokens in Hindi or Thai than in English — a real fairness and cost issue, not a curiosity.

### fixes

Character-level BPE still needs a base alphabet, so an unseen Unicode character has no representation and becomes UNK. There are 150 000 code points and no corpus contains them all.

## blt

Drop the vocabulary. Operate on raw bytes and group them into **patches** dynamically, using a small byte-level model's next-byte entropy to decide where to cut: predictable stretches become long patches, surprising ones become short patches.

Compute is then allocated where the data is actually hard, rather than uniformly by a merge table fixed before training. Meta report matching Llama-3 quality with up to 50% fewer inference FLOPs.

Included because it is the first credible attack on the tokenizer as a *concept* rather than an attempt to build a better one. Nothing has shipped with it yet.

### fixes

Every fixed tokenizer commits to one segmentation for all inputs, taxes some languages, hides spelling, and is chosen before training with no gradient signal to correct it.

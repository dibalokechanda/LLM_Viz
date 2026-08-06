---
id: sampling
label: Sampling
ordinal: '13'
icon: sparkle
slot: output
tagline: Choosing one token from a distribution over 128,000
io:
  in: logits [B, vocab]
  out: one token id
defaultVariant: top-p
caveat: >-
  Not recorded in config.json. A model loaded from Hugging Face leaves this block unresolved —
  generation_config.json carries a suggested default, but the caller overrides it freely.
lineage:
  - from: greedy
    to: beam
    kind: fixes
    label: locally best is not globally best
  - from: greedy
    to: temperature
    kind: fixes
    label: deterministic output is repetitive
  - from: temperature
    to: top-k
    kind: fixes
    label: the tail is still reachable
  - from: top-k
    to: top-p
    kind: fixes
    label: a fixed k ignores how sharp the distribution is
  - from: top-p
    to: min-p
    kind: fixes
    label: scale the threshold to the top token
  - from: greedy
    to: speculative
    kind: derives
    label: draft cheaply, verify exactly
variants:
  - id: greedy
    label: Greedy
    full: Greedy decoding (argmax)
    year: 2014
    role: origin
    tagline: Always take the highest-probability token
    concepts:
      - id: greedy-local-choice
        label: Commit to the current mode
        kind: method
        summary: Greedy decoding selects the largest next-token logit at every step.
        detail:
          - >-
            No candidate set or random draw is constructed. The chosen token is appended and the
            model repeats the same argmax decision on the longer prefix.
      - id: greedy-search-limit
        label: No recovery from an early choice
        kind: tradeoff
        summary: A locally best token can lead to a worse complete sequence than a lower-ranked alternative.
        detail:
          - >-
            Greedy is fast, reproducible, and often appropriate for constrained tasks, but it
            explores exactly one trajectory through the model distribution.
    math:
      - title: Greedy
        tex: x_t = \arg\max_i P(i \mid x_{<t})
        note: Locally optimal at each step; globally nothing in particular.
    usedBy:
      - code generation
      - extraction
      - anything evaluated against a reference
  - id: beam
    label: Beam search
    full: Beam search
    year: 2014
    role: branch
    tagline: Keep b partial sequences and extend them all
    concepts:
      - id: beam-live-hypotheses
        label: Keep several partial futures
        kind: method
        summary: Beam search expands and ranks b partial sequences at each decoding step.
        detail:
          - >-
            Candidates are compared by cumulative log probability rather than only the newest
            token. Surviving beams retain different earlier commitments for the next expansion.
      - id: beam-length-bias
        label: Sequence score needs calibration
        kind: pitfall
        summary: Multiplying token probabilities naturally favours shorter sequences.
        detail:
          - >-
            Length normalization, end-token handling, and repetition constraints change beam
            behavior materially. Beam width alone does not define a decoding policy.
    math:
      - title: Length-normalised score
        tex: \text{score}(y) = \frac{1}{|y|^\alpha}\sum_{t} \log P(y_t \mid y_{<t})
        note: >-
          Without the length penalty, beam search systematically prefers short sequences — every
          extra token multiplies in another probability below 1.
    usedBy:
      - NMT
      - summarisation
      - ASR
  - id: temperature
    label: Temperature
    full: Temperature-scaled sampling
    year: 2015
    role: refinement
    tagline: Divide logits by T, then sample
    concepts:
      - id: temperature-logit-rescale
        label: Rescale logit gaps
        kind: formula
        summary: Dividing logits by T sharpens the distribution below one and flattens it above one.
        detail:
          - >-
            The rank order of tokens is unchanged, but probability mass is redistributed before
            sampling. At the zero-temperature limit, the policy approaches greedy argmax.
      - id: temperature-tail-risk
        label: Temperature alone leaves the tail open
        kind: tradeoff
        summary: A hot distribution can still assign nonzero mass to a very large vocabulary tail.
        detail:
          - >-
            Temperature controls diversity, not candidate quality. It is usually paired with top-k,
            top-p, or min-p to remove implausible tokens before the draw.
    math:
      - title: Temperature scaling
        tex: P_T(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
        worked:
          - tex: T \to 0 \;\Rightarrow\; \text{argmax}
            caption: recovers greedy
          - tex: T \to \infty \;\Rightarrow\; \text{uniform}
            caption: the model is ignored entirely
    figures:
      - kind: bars
        title: The same logits at three temperatures
        categories:
          - cat
          - dog
          - bird
          - rock
          - the
        series:
          - label: T = 0.5
            values:
              - 0.79
              - 0.17
              - 0.03
              - 0.01
              - 0
          - label: T = 1.0
            values:
              - 0.5
              - 0.25
              - 0.13
              - 0.08
              - 0.04
          - label: T = 1.5
            values:
              - 0.37
              - 0.24
              - 0.17
              - 0.13
              - 0.09
        showValues: true
        caption: >-
          Note the rightmost bars. Going from T = 0.5 to T = 1.5 raises "the" from effectively
          impossible to a 1-in-11 shot. Multiplied across a 128k vocabulary, that tail is where
          incoherence comes from.
    usedBy:
      - universal — always combined with a truncation method
  - id: top-k
    label: Top-k
    full: Top-k truncation
    year: 2018
    role: refinement
    tagline: Keep the k most likely tokens, renormalise, sample
    concepts:
      - id: topk-fixed-candidate-set
        label: Keep a fixed number of candidates
        kind: method
        summary: The k highest-probability tokens survive and are renormalized before sampling.
        detail:
          - >-
            A logit threshold derived from the kth token masks the rest of the vocabulary. The
            size of the sampling set is constant even when confidence changes.
      - id: topk-confidence-mismatch
        label: Fixed k ignores distribution shape
        kind: tradeoff
        summary: The same k can be too broad for a confident step and too narrow for an ambiguous one.
        detail:
          - >-
            Top-p and min-p adapt their retained set to probability mass or relative confidence,
            while top-k offers simple predictable bounds.
    math:
      - title: Top-k
        tex: >-
          V_k = \text{TopK}(P, k), \qquad P'(i) = \frac{P(i) \cdot \mathbb{1}[i \in V_k]}{\sum_{j
          \in V_k} P(j)}
    figures:
      - kind: ranked
        title: k = 5 over a flat distribution
        grades:
          - 3
          - 3
          - 2
          - 2
          - 2
          - 2
          - 2
          - 1
          - 1
          - 1
          - 1
          - 1
        maxGrade: 3
        caption: >-
          Darker is more probable. With many similarly plausible options, a fixed cut at 5 discards
          several tokens just as good as the ones it kept.
    usedBy:
      - GPT-2 era defaults
      - still common as a secondary filter
  - id: top-p
    label: Top-p (nucleus)
    full: Nucleus sampling
    year: 2019
    role: refinement
    concepts:
      - id: ranked-probabilities
        label: Start with ranked next-token probabilities
        kind: method
        summary: Top-p sorts candidates from most to least likely after the model produces its probability distribution.
        detail:
          - >-
            The candidate set is determined by the shape of the current distribution, not by a fixed
            number of tokens. A confident prediction and an uncertain prediction therefore behave differently.
      - id: nucleus-cutoff
        label: Keep the smallest sufficient nucleus
        kind: formula
        summary: Candidates are kept until their cumulative probability reaches the chosen threshold p.
        detail:
          - >-
            The retained tokens are renormalized before sampling. With p close to one, the method
            stays diverse while removing the very low-probability tail.
      - id: temperature-interaction
        label: Temperature changes the nucleus
        kind: tradeoff
        summary: Temperature reshapes probabilities before top-p chooses its variable-size candidate set.
        detail:
          - >-
            Lower temperature sharpens the distribution and often shrinks the nucleus; higher
            temperature flattens it and tends to admit more candidates. Tune the two together.
    tagline: Keep the smallest set whose probability sums to p
    paper:
      title: The Curious Case of Neural Text Degeneration
      url: https://arxiv.org/abs/1904.09751
      authors: Holtzman et al.
    math:
      - title: The nucleus
        tex: 'V_p = \min\left\{ V'' \subseteq V : \sum_{i \in V''} P(i) \ge p \right\}'
        worked:
          - tex: P = [0.9, 0.05, ...],\ p = 0.9 \;\Rightarrow\; |V_p| = 1
            caption: 'confident: one candidate'
          - tex: P = [0.02, 0.02, ...],\ p = 0.9 \;\Rightarrow\; |V_p| \approx 45
            caption: 'uncertain: the nucleus opens up'
    usedBy:
      - OpenAI API default
      - Anthropic API
      - nearly every chat interface
  - id: min-p
    label: Min-p
    full: Minimum-probability sampling
    year: 2024
    role: refinement
    tagline: Keep tokens at least p× as likely as the best one
    paper:
      title: 'Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs'
      url: https://arxiv.org/abs/2407.01082
    concepts:
      - id: minp-relative-floor
        label: Threshold relative to the winner
        kind: method
        summary: A candidate survives only when its probability is at least p times the best token's probability.
        detail:
          - >-
            The threshold follows the current confidence level. A decisive distribution keeps a
            small set; a flatter one can retain more alternatives without chasing a fixed mass.
      - id: minp-temperature-order
        label: Apply after reshaping probability
        kind: pitfall
        summary: Temperature changes both the leading probability and the relative floor.
        detail:
          - >-
            Min-p is evaluated on the distribution presented to the sampler. Changing temperature
            without reconsidering p can make the retained candidate set unexpectedly strict or loose.
    math:
      - title: Relative threshold
        tex: 'V_{\min p} = \{ i : P(i) \ge p \cdot \max_j P(j) \}'
        note: >-
          Scale-relative rather than mass-absolute. This is why it survives temperature changes that
          break top-p.
    usedBy:
      - llama.cpp
      - text-generation-webui
      - vLLM
      - creative-writing communities
  - id: speculative
    label: Speculative
    full: Speculative decoding
    year: 2023
    role: frontier
    tagline: Draft several tokens cheaply, verify them in one pass
    paper:
      title: Fast Inference from Transformers via Speculative Decoding
      url: https://arxiv.org/abs/2211.17192
      authors: Leviathan et al.
    concepts:
      - id: speculative-draft-verify
        label: Propose a block, verify in parallel
        kind: method
        summary: A cheap draft model emits several candidates that the target evaluates in one forward pass.
        detail:
          - >-
            Accepted draft tokens are appended together; the first rejected position is replaced
            with a target-model sample. The target remains the distributional authority.
      - id: speculative-acceptance-economics
        label: Speed depends on acceptance
        kind: metric
        summary: Useful speedup requires a draft that is cheap and agrees often with the target.
        detail:
          - >-
            A poor draft creates little accepted work and adds verification overhead. Draft size,
            target batch behavior, and cache implementation determine real serving gains.
    math:
      - title: Acceptance rule
        tex: >-
          \text{accept } x \sim q \text{ with prob. } \min\left(1, \frac{p(x)}{q(x)}\right); \text{
          else resample from } \max(0, p - q)
        where:
          - sym: p
            means: the large (target) model's distribution
          - sym: q
            means: the small (draft) model's distribution
        note: >-
          The residual resampling term is what makes the guarantee exact. Implementations that just
          accept-or-reject without it do change the output distribution.
      - title: Expected speedup
        tex: E[\text{tokens per pass}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
        where:
          - sym: \alpha
            means: per-token acceptance rate — typically 0.7–0.9
        worked:
          - tex: \alpha = 0.8,\ \gamma = 4 \;\Rightarrow\; 3.4\ \text{tokens per pass}
            caption: against 1 without speculation
    usedBy:
      - vLLM
      - TensorRT-LLM
      - llama.cpp
      - DeepSeek-V3 (self-speculative)
      - Medusa
      - EAGLE
---

## role

The model outputs a probability for every token. Exactly one must be emitted. This block is the rule that picks it, and it runs once per generated token, forever.

It is the only block here that is **not** part of the model. Nothing about it is trained, none of it is in `config.json`, and the same checkpoint behaves completely differently under different settings. That makes it the cheapest thing to change and the most commonly misattributed — a great deal of what people describe as model behaviour is really a decoding parameter.

The space is organised by one tension: **truncate too little** and you sample from the noisy tail, producing incoherence; **truncate too much** and you collapse into repetition. Every variant is a different theory of where the boundary between signal and noise lies.

## greedy

Deterministic and reproducible, which is why it is right for anything with a correct answer — code, extraction, classification, structured output.

It is wrong for open-ended text, and interestingly so: the most probable *token* at each step does not produce the most probable *sequence*, and greedy text degenerates into loops far more readily than sampling does. Human text is not the maximum-likelihood path through a language model, which is the observation that motivated everything below.

## beam

Maintain `b` candidate sequences, extend each by every possible token, keep the `b` best by cumulative log-probability. Finds higher-likelihood sequences than greedy, and dominated translation and summarisation for years.

It has largely disappeared from open-ended LLM generation, and the reason is instructive: it works *too* well. Maximising sequence likelihood produces bland, generic, repetitive text, because the highest-likelihood continuation of almost anything is a cliché. Beam search is right when there is a correct answer to find and wrong when there is a distribution to sample from.

### fixes

Greedy commits to a token before knowing what follows, and a locally suboptimal token often leads to a better sequence.

## temperature

Divide the logits by `T` before the softmax. `T < 1` sharpens toward greedy, `T > 1` flattens toward uniform, `T = 1` samples from the model's actual distribution.

On its own, temperature is a blunt instrument. Raising it to get variety also raises the probability of the long tail — and with a 128k vocabulary, the tail holds an enormous amount of total mass made of individually terrible options. This is precisely the problem truncation methods exist to solve, and why temperature is almost always paired with one.

### fixes

Greedy and beam are deterministic. Open-ended generation needs variety, and the model already has a distribution — use it.

## top-k

Sort, keep the top `k` (typically 40–50), zero the rest, renormalise. The tail becomes strictly unreachable rather than merely unlikely.

Its flaw is that `k` is fixed while the distribution's sharpness is not. After "The capital of France is" the model is nearly certain and `k = 50` admits 49 wrong answers. After "He opened the door and saw a" hundreds of continuations are reasonable and `k = 50` cuts off good ones. A constant count cannot track a varying entropy.

### fixes

Temperature alone leaves every one of 128,000 tokens reachable, including tens of thousands that are clearly wrong.

## top-p

Sort descending, accumulate probability until the running total reaches `p` (typically 0.9–0.95), keep exactly that set. The candidate count is now **dynamic**: a confident distribution yields a nucleus of one or two tokens, a flat one yields hundreds.

This is the default nearly everywhere, and the paper that introduced it is also the clearest statement of *why* maximum-likelihood decoding produces degenerate text — worth reading for that argument alone.

### fixes

Top-k fixes the number of candidates when what should be fixed is how much probability mass they cover.

## min-p

Set the threshold **relative to the top token**: keep every token with probability at least `p × P_max`. If the best token has 0.9 and `p = 0.1`, the floor is 0.09 and almost nothing survives. If the best has 0.05, the floor is 0.005 and many do.

The practical consequence is that min-p stays coherent at much higher temperatures than top-p, because raising temperature lifts the floor along with everything else. That makes it popular for creative writing, where people want the variety of `T = 2` without the incoherence.

### fixes

Top-p keeps accumulating until it reaches its mass target, so on a genuinely flat distribution it admits hundreds of poor tokens just to reach 0.95.

## speculative

Have a small draft model propose `γ` tokens. Run the large model **once** over all of them — the same weight read that would have produced one token now verifies several, because the bottleneck was memory, not compute. Accept the longest prefix consistent with the large model, reject the rest, repeat.

The crucial property is that a correctly implemented rejection-sampling step makes the output distribution **exactly** that of the large model. This is not an approximation and not a quality trade — it is the same distribution, produced faster. That is what separates it from every other speedup on this list.

Typical speedups are 2–3×, set by how often the draft model agrees. Self-speculation via multi-token-prediction heads removes the need for a separate draft model entirely.

### fixes

Decoding is memory-bandwidth bound: generating one token reads every weight in the model. The GPU is almost entirely idle, and batching one sequence cannot fix it.

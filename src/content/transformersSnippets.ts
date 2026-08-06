import type { Block, CodeBlock, Variant } from '../data/types'

/*
 * These are deliberately usage-first examples. Some choices are inference
 * controls (sampling and cache strategy), while others are baked into a
 * checkpoint or are architectural code you would use while training. The
 * snippets say which is which instead of treating config inspection as use.
 */
const CAUSAL_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

/*
 * The snippets below are intentionally compact, original teaching examples.
 * This map points readers to the corresponding full, runnable from-scratch
 * implementation in Sebastian Raschka's repository. Newer mechanisms such as
 * Mamba, mHC, and MoE use that GPT path as a comparison baseline rather than
 * implying that they are implemented upstream.
 */
const RASCHKA_REPO = 'https://github.com/rasbt/LLMs-from-scratch'
const RASCHKA_CH2 = `${RASCHKA_REPO}/tree/main/ch02/01_main-chapter-code`
const RASCHKA_CH3 = `${RASCHKA_REPO}/tree/main/ch03/01_main-chapter-code`
const RASCHKA_GPT = `${RASCHKA_REPO}/blob/main/ch04/01_main-chapter-code/gpt.py`
const RASCHKA_KV_CACHE = `${RASCHKA_REPO}/tree/main/ch04/03_kv-cache`

function raschkaReference(block: Block, variant: Variant, mode: 'implementation' | 'framework'): NonNullable<CodeBlock['source']> {
  const isExtension =
    (block.id === 'mixer' && variant.id !== 'attention') ||
    (block.id === 'residual' && variant.id === 'mhc') ||
    (block.id === 'ffn' && !['ffn', 'gelu'].includes(variant.id)) ||
    (block.id === 'scores' && variant.id !== 'scaled-dot') ||
    (block.id === 'sampling' && variant.id !== 'greedy')

  const url =
    block.id === 'tokenizer'
      ? RASCHKA_CH2
      : ['qkv', 'pattern', 'scores'].includes(block.id)
        ? RASCHKA_CH3
        : block.id === 'kvcache'
          ? RASCHKA_KV_CACHE
          : RASCHKA_GPT

  return {
    label: isExtension || mode === 'framework' ? 'Raschka GPT baseline' : 'Raschka reference',
    url,
  }
}

function withRaschkaReference(block: Block, variant: Variant, snippet: CodeBlock, mode: 'implementation' | 'framework'): CodeBlock {
  return { ...snippet, source: snippet.source ?? raschkaReference(block, variant, mode) }
}

const py = (...lines: string[]) => lines.join('\n')

function example(title: string, code: string, note: string): CodeBlock {
  return { title, language: 'python', code, note }
}

function hubLoad(modelId = CAUSAL_MODEL): string[] {
  return [
    'import torch',
    'from transformers import AutoModelForCausalLM, AutoTokenizer',
    '',
    `model_id = ${JSON.stringify(modelId)}`,
    'device = "cuda" if torch.cuda.is_available() else "cpu"',
    'tokenizer = AutoTokenizer.from_pretrained(model_id)',
    'model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()',
  ]
}

function generationCall(settings: string[], prompt = 'Give one practical explanation of this transformer feature.'): string[] {
  return [
    '',
    `inputs = tokenizer(${JSON.stringify(prompt)}, return_tensors="pt").to(device)`,
    'with torch.inference_mode():',
    '    output_ids = model.generate(',
    '        **inputs,',
    '        max_new_tokens=80,',
    ...settings.map((line) => `        ${line}`),
    '    )',
    'print(tokenizer.decode(output_ids[0], skip_special_tokens=True))',
  ]
}

function tokenizerSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'word') {
    return example(
      'Practical · train a word tokenizer',
      py(
        'from tokenizers import Tokenizer, models, pre_tokenizers, trainers',
        'from transformers import PreTrainedTokenizerFast',
        '',
        'tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))',
        'tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()',
        'trainer = trainers.WordLevelTrainer(',
        '    vocab_size=16_000, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]',
        ')',
        'tokenizer.train(["corpus.txt"], trainer)',
        'hf_tokenizer = PreTrainedTokenizerFast(',
        '    tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",',
        '    bos_token="[BOS]", eos_token="[EOS]",',
        ')',
        'hf_tokenizer.save_pretrained("./my-word-tokenizer")',
      ),
      'Use this when training a model from scratch. A word tokenizer is not interchangeable with the tokenizer that a pretrained checkpoint expects.',
    )
  }

  if (variant.id === 'char' || variant.id === 'blt') {
    return example(
      `Practical · ${variant.label} input path`,
      py(
        'import torch',
        '',
        'text = "naïve café"',
        '# UTF-8 bytes are an always-defined, tokenizer-free alphabet.',
        'byte_ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)',
        'print(byte_ids.tolist())',
        '',
        '# A byte/character model owns the embedding and sequence model itself.',
        'embedding = torch.nn.Embedding(256, 512)',
        'x = embedding(byte_ids.unsqueeze(0))  # [batch, bytes, d_model]',
        'print(x.shape)',
      ),
      variant.id === 'blt'
        ? 'Byte-latent models bypass AutoTokenizer. Feed UTF-8 bytes into the model-specific byte encoder supplied by that architecture.'
        : 'Character-style input is usually a custom PyTorch data path. Pretrained Transformers checkpoints should still receive their own AutoTokenizer output.',
    )
  }

  const checkpointByVariant: Record<string, string> = {
    bpe: 'openai-community/gpt2',
    'byte-bpe': 'openai-community/gpt2',
    wordpiece: 'google-bert/bert-base-uncased',
    unigram: 'google/mt5-small',
  }
  const modelId = checkpointByVariant[variant.id] ?? CAUSAL_MODEL
  return example(
    `Practical · use a ${variant.label} checkpoint tokenizer`,
    py(
      'from transformers import AutoTokenizer',
      '',
      `model_id = ${JSON.stringify(modelId)}`,
      'tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)',
      'batch = tokenizer(',
      '    ["Tokenization is part of the checkpoint contract."],',
      '    return_tensors="pt", padding=True, truncation=True, max_length=64,',
      '    return_offsets_mapping=True,',
      ')',
      'print(tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))',
      'print(batch["offset_mapping"][0].tolist())',
    ),
    'Choose a checkpoint that was trained with this tokenizer family. The offsets show exactly which source characters each model token covers.',
  )
}

function embeddingSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'factorized') {
    return example(
      'Practical · factorized embeddings with ALBERT',
      py(
        'import torch',
        'from transformers import AutoModel, AutoTokenizer',
        '',
        'model_id = "albert/albert-base-v2"',
        'tokenizer = AutoTokenizer.from_pretrained(model_id)',
        'model = AutoModel.from_pretrained(model_id).eval()',
        'inputs = tokenizer("Factorized embeddings save parameters.", return_tensors="pt")',
        '',
        'word_vectors = model.embeddings.word_embeddings(inputs.input_ids)',
        'hidden_vectors = model.encoder.embedding_hidden_mapping_in(word_vectors)',
        'print(word_vectors.shape, "->", hidden_vectors.shape)',
      ),
      'ALBERT is a concrete checkpoint family with a small vocabulary embedding followed by a projection into hidden size.',
    )
  }

  if (variant.id === 'scaled') {
    return example(
      'Practical · scale embeddings in a custom PyTorch model',
      py(
        'import math',
        'import torch.nn as nn',
        '',
        'class ScaledEmbedding(nn.Module):',
        '    def __init__(self, vocab_size: int, d_model: int):',
        '        super().__init__()',
        '        self.table = nn.Embedding(vocab_size, d_model)',
        '        self.scale = math.sqrt(d_model)',
        '',
        '    def forward(self, input_ids):',
        '        return self.table(input_ids) * self.scale',
      ),
      'Put this module in a model you train yourself. Do not apply the scale around a pretrained checkpoint unless its architecture was trained that way.',
    )
  }

  const tied = variant.id === 'tied'
  return example(
    `Practical · ${tied ? 'verify tied input/output weights' : 'use independent input and output weights'}`,
    py(
      ...hubLoad(),
      '',
      'input_table = model.get_input_embeddings()',
      'output_head = model.get_output_embeddings()',
      'same_storage = input_table.weight.data_ptr() == output_head.weight.data_ptr()',
      'print("tied in this checkpoint:", same_storage)',
      '',
      'input_ids = tokenizer("A useful embedding", return_tensors="pt").input_ids.to(device)',
      'token_vectors = input_table(input_ids)',
      'logits = output_head(token_vectors)',
      'print(token_vectors.shape, "->", logits.shape)',
    ),
    tied
      ? 'Weight tying is a checkpoint architecture choice. `same_storage=True` confirms that lookup and prediction share one parameter matrix.'
      : 'An untied head uses separate parameters for lookup and prediction. Select a checkpoint whose config has `tie_word_embeddings: false` when that is required.',
  )
}

function positionalSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'sinusoidal' || variant.id === 'learned') {
    const learned = variant.id === 'learned'
    return example(
      `Practical · ${variant.label} position module`,
      py(
        'import math',
        'import torch',
        'import torch.nn as nn',
        '',
        'batch, seq_len, d_model = 2, 128, 512',
        'x = torch.randn(batch, seq_len, d_model)',
        'positions = torch.arange(seq_len)',
        ...(learned
          ? [
              'position_table = nn.Embedding(4096, d_model)',
              'x = x + position_table(positions)[None, :, :]',
            ]
          : [
              'freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))',
              'pe = torch.zeros(seq_len, d_model)',
              'pe[:, 0::2] = torch.sin(positions[:, None] * freq)',
              'pe[:, 1::2] = torch.cos(positions[:, None] * freq)',
              'x = x + pe[None, :, :]',
            ]),
        'print(x.shape)',
      ),
      learned
        ? 'A learned table has a fixed training-time length. Resize or retrain it deliberately; loading a longer prompt does not create new learned positions.'
        : 'Sinusoids are a deterministic PyTorch tensor, so they can be generated for any requested sequence length.',
    )
  }

  if (variant.id === 't5-bias' || variant.id === 'relative') {
    return example(
      'Practical · use a relative-position checkpoint',
      py(
        'from transformers import AutoModelForSeq2SeqLM, AutoTokenizer',
        '',
        'model_id = "google-t5/t5-small"',
        'tokenizer = AutoTokenizer.from_pretrained(model_id)',
        'model = AutoModelForSeq2SeqLM.from_pretrained(model_id)',
        'print("relative buckets:", model.config.relative_attention_num_buckets)',
        'inputs = tokenizer("summarize: Relative position is a distance, not an index.", return_tensors="pt")',
        'print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True))',
      ),
      'T5 is a concrete Transformers family where relative position bias is part of the loaded architecture.',
    )
  }

  if (variant.id === 'alibi') {
    return example(
      'Practical · use an ALiBi checkpoint',
      py(
        'from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer',
        '',
        'model_id = "bigscience/bloom-560m"',
        'config = AutoConfig.from_pretrained(model_id)',
        'print("ALiBi enabled:", getattr(config, "alibi", False))',
        'tokenizer = AutoTokenizer.from_pretrained(model_id)',
        'model = AutoModelForCausalLM.from_pretrained(model_id).eval()',
        'inputs = tokenizer("ALiBi adds a distance slope to attention scores.", return_tensors="pt")',
        'print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True))',
      ),
      'ALiBi is not a `generate()` switch. Pick an ALiBi architecture such as BLOOM when it is a model requirement.',
    )
  }

  if (variant.id === 'nope') {
    return example(
      'Practical · no positional signal in a custom block',
      py(
        'import torch.nn as nn',
        '',
        'class NoPositionInput(nn.Module):',
        '    def __init__(self, vocab_size: int, d_model: int):',
        '        super().__init__()',
        '        self.embedding = nn.Embedding(vocab_size, d_model)',
        '',
        '    def forward(self, input_ids):',
        '        return self.embedding(input_ids)  # deliberately no position addition/rotation',
      ),
      'NoPE is an architectural training decision. A checkpoint trained with RoPE cannot safely be converted by simply removing its position module.',
    )
  }

  const ropeType: Record<string, string> = {
    rope: 'default',
    'linear-interp': 'linear',
    'dynamic-ntk': 'dynamic',
    yarn: 'yarn',
    'llama3-rope': 'llama3',
  }
  const rope = ropeType[variant.id] ?? 'default'
  const factor = rope === 'default' ? 1 : rope === 'llama3' ? 8 : rope === 'yarn' ? 4 : 2
  const parameters = rope === 'default'
    ? '{ "rope_type": "default" }'
    : rope === 'llama3'
      ? `{ "rope_type": "llama3", "factor": ${factor}, "original_max_position_embeddings": trained_length, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "rope_theta": config.rope_theta }`
      : rope === 'yarn'
        ? `{ "rope_type": "yarn", "factor": ${factor}, "original_max_position_embeddings": trained_length }`
        : `{ "rope_type": ${JSON.stringify(rope)}, "factor": ${factor} }`
  return example(
    `Practical · configure ${variant.label} before loading`,
    py(
      'from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer',
      '',
      `model_id = ${JSON.stringify(CAUSAL_MODEL)}`,
      'config = AutoConfig.from_pretrained(model_id)',
      'trained_length = config.max_position_embeddings',
      `config.max_position_embeddings = trained_length * ${factor}`,
      `config.rope_parameters = ${parameters}`,
      '# Pass the modified config while loading; validate quality at the new length.',
      'model = AutoModelForCausalLM.from_pretrained(model_id, config=config).eval()',
      'tokenizer = AutoTokenizer.from_pretrained(model_id)',
      'print("rope parameters:", model.config.rope_parameters)',
    ),
    rope === 'default'
      ? 'RoPE is supplied by the checkpoint architecture; this loads it normally.'
      : 'RoPE scaling is model-family and version dependent. Configure it before loading, then test long-context quality—an override does not replace long-context training.',
  )
}

function normSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'layernorm' || variant.id === 'rmsnorm') {
    const constructor = variant.id === 'layernorm' ? 'nn.LayerNorm(d_model, eps=1e-5)' : 'nn.RMSNorm(d_model, eps=1e-6)'
    return example(
      `Practical · ${variant.label} in PyTorch`,
      py(
        'import torch',
        'import torch.nn as nn',
        '',
        'd_model = 1024',
        `norm = ${constructor}`,
        'x = torch.randn(2, 128, d_model)',
        'y = norm(x)',
        'print(y.shape)',
      ),
      'This is the actual PyTorch normalization module you would place inside a custom Transformer block.',
    )
  }

  if (variant.id === 'qk-norm') {
    return example(
      'Practical · normalize Q and K before attention',
      py(
        'import torch',
        'import torch.nn.functional as F',
        '',
        'q = torch.randn(2, 8, 128, 64, device="cuda" if torch.cuda.is_available() else "cpu")',
        'k = torch.randn_like(q)',
        'v = torch.randn_like(q)',
        'q = F.rms_norm(q, (q.shape[-1],))',
        'k = F.rms_norm(k, (k.shape[-1],))',
        'context = F.scaled_dot_product_attention(q, k, v, is_causal=True)',
        'print(context.shape)',
      ),
      'QK normalization is a block implementation choice. Use it when training or modifying the attention module, not as a post-hoc operation on model outputs.',
    )
  }

  return example(
    'Practical · Dynamic Tanh normalization',
    py(
      'import torch',
      'import torch.nn as nn',
      '',
      'class DynamicTanh(nn.Module):',
      '    def __init__(self, d_model: int):',
      '        super().__init__()',
      '        self.alpha = nn.Parameter(torch.ones(d_model))',
      '        self.gamma = nn.Parameter(torch.ones(d_model))',
      '        self.beta = nn.Parameter(torch.zeros(d_model))',
      '',
      '    def forward(self, x):',
      '        return self.alpha * torch.tanh(self.gamma * x) + self.beta',
    ),
    'DyT needs to be part of the model that is trained. It is not a generic replacement for the normalization layers inside an existing checkpoint.',
  )
}

function qkvSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'mha') {
    return example(
      'Practical · multi-head attention with PyTorch',
      py(
        'import torch',
        'from torch import nn',
        '',
        'attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)',
        'x = torch.randn(2, 128, 512)',
        'causal_mask = torch.ones(128, 128, dtype=torch.bool).triu(1)',
        'y, _ = attention(x, x, x, attn_mask=causal_mask, need_weights=False)',
        'print(y.shape)',
      ),
      'Use `nn.MultiheadAttention` for a standard trainable MHA block. Its mask uses `True` for disallowed positions.',
    )
  }

  if (variant.id === 'mqa' || variant.id === 'gqa') {
    const kvHeads = variant.id === 'mqa' ? 1 : 2
    return example(
      `Practical · ${variant.label} with scaled dot-product attention`,
      py(
        'import torch',
        'import torch.nn.functional as F',
        '',
        'if not torch.cuda.is_available():',
        '    raise RuntimeError("This GQA example requires a CUDA PyTorch build.")',
        'device = "cuda"',
        'batch, q_heads, seq, head_dim = 2, 8, 128, 64',
        `kv_heads = ${kvHeads}  # ${variant.id === 'mqa' ? 'one shared K/V head' : 'one K/V head per query group'}`,
        'q = torch.randn(batch, q_heads, seq, head_dim, device=device)',
        'k = torch.randn(batch, kv_heads, seq, head_dim, device=device)',
        'v = torch.randn_like(k)',
        'y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)',
        'print(y.shape)',
      ),
      'PyTorch’s current GQA path is CUDA-only and expands grouped K/V heads for the attention computation. A serving checkpoint must still have been trained with matching Q and KV projections.',
    )
  }

  return example(
    'Practical · latent K/V cache projection',
    py(
      'import torch',
      'import torch.nn.functional as F',
      '',
      'batch, heads, seq, d_model, d_latent, head_dim = 2, 8, 128, 512, 64, 64',
      'x = torch.randn(batch, seq, d_model)',
      'Wq = torch.randn(d_model, heads * head_dim)',
      'Wdown = torch.randn(d_model, d_latent)',
      'Wk = torch.randn(d_latent, heads * head_dim)',
      'Wv = torch.randn(d_latent, heads * head_dim)',
      'q = (x @ Wq).view(batch, seq, heads, head_dim).transpose(1, 2)',
      'latent_cache = x @ Wdown  # store this smaller state',
      'k = (latent_cache @ Wk).view(batch, seq, heads, head_dim).transpose(1, 2)',
      'v = (latent_cache @ Wv).view(batch, seq, heads, head_dim).transpose(1, 2)',
      'y = F.scaled_dot_product_attention(q, k, v, is_causal=True)',
    ),
    'MLA changes the attention and cache projection code. Use an MLA checkpoint or train this kind of module; Transformers does not convert an ordinary MHA checkpoint into MLA.',
  )
}

function patternSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'bidirectional' || variant.id === 'causal') {
    const causal = variant.id === 'causal'
    return example(
      `Practical · ${variant.label} scaled dot-product attention`,
      py(
        'import torch',
        'import torch.nn.functional as F',
        '',
        'q = torch.randn(2, 8, 128, 64)',
        'k = torch.randn_like(q)',
        'v = torch.randn_like(q)',
        `y = F.scaled_dot_product_attention(q, k, v, is_causal=${causal ? 'True' : 'False'})`,
        'print(y.shape)',
      ),
      causal ? 'Set `is_causal=True` for an autoregressive decoder. It creates the lower-triangular visibility rule.' : 'Encoder-style full attention allows every token to read every other token.',
    )
  }

  if (variant.id === 'nsa') {
    return example(
      'Practical · learned sparse pattern with FlexAttention',
      py(
        'import torch',
        'from torch.nn.attention.flex_attention import create_block_mask, flex_attention',
        '',
        'def sparse_causal(b, h, q_idx, kv_idx):',
        '    local = (q_idx - kv_idx).abs() < 256',
        '    landmarks = (kv_idx % 128) == 0',
        '    return (q_idx >= kv_idx) & (local | landmarks)',
        '',
        'mask = create_block_mask(sparse_causal, B=1, H=8, Q_LEN=2048, KV_LEN=2048, device="cuda")',
        'q = k = v = torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.bfloat16)',
        'y = flex_attention(q, k, v, block_mask=mask)',
      ),
      'FlexAttention is a PyTorch route for experimenting with sparse masks. A learned selector must be trained and benchmarked as part of the architecture.',
    )
  }

  const maskByVariant: Record<string, string[]> = {
    window: ['allowed = (q_pos >= kv_pos) & ((q_pos - kv_pos) < 256)'],
    sink: ['allowed = (q_pos >= kv_pos) & (((q_pos - kv_pos) < 256) | (kv_pos < 4))'],
    dilated: ['allowed = (q_pos >= kv_pos) & (((q_pos - kv_pos) < 256) | ((q_pos - kv_pos) % 8 == 0))'],
    interleaved: ['allowed = q_pos >= kv_pos  # use this full mask on global layers; swap in a window on local layers'],
  }
  return example(
    `Practical · ${variant.label} mask`,
    py(
      'import torch',
      'import torch.nn.functional as F',
      '',
      'seq = 1024',
      'q_pos = torch.arange(seq)[:, None]',
      'kv_pos = torch.arange(seq)[None, :]',
      ...(maskByVariant[variant.id] ?? ['allowed = q_pos >= kv_pos']),
      'q = k = v = torch.randn(1, 8, seq, 64)',
      'y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)',
      'print(y.shape)',
    ),
    'The boolean mask is explicit and easy to test. For high-throughput long contexts, benchmark an optimized sparse kernel rather than assuming a dense mask saves compute.',
  )
}

function scoresSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'softmax') {
    return example(
      'Practical · fused scaled dot-product softmax attention',
      py(
        'import torch',
        'import torch.nn.functional as F',
        '',
        'q = torch.randn(2, 8, 128, 64, device="cuda" if torch.cuda.is_available() else "cpu")',
        'k = torch.randn_like(q)',
        'v = torch.randn_like(q)',
        'context = F.scaled_dot_product_attention(q, k, v, is_causal=True)',
        'print(context.shape)',
      ),
      'Prefer PyTorch scaled dot-product attention for standard softmax attention: it can select an optimized CUDA kernel automatically.',
    )
  }

  if (variant.id === 'softcap') {
    return example(
      'Practical · soft-cap attention logits',
      py(
        'import math',
        'import torch',
        '',
        'q = torch.randn(2, 8, 128, 64)',
        'k = torch.randn_like(q)',
        'v = torch.randn_like(q)',
        'cap = 30.0',
        'scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])',
        'scores = cap * torch.tanh(scores / cap)',
        'weights = scores.softmax(dim=-1)',
        'context = weights @ v',
      ),
      'Soft-capping changes the model math, so use it in a trainable/custom attention module or load a checkpoint trained with it.',
    )
  }

  if (variant.id === 'differential') {
    return example(
      'Practical · differential attention',
      py(
        'import torch',
        'import torch.nn.functional as F',
        '',
        'q1 = k1 = v = torch.randn(2, 8, 128, 64)',
        'q2 = torch.randn_like(q1)',
        'k2 = torch.randn_like(k1)',
        'noise_cancel = 0.5',
        'a1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)',
        'a2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)',
        'context = a1 - noise_cancel * a2',
      ),
      'Differential attention needs two learned attention maps. Train the projections and cancellation factor together; it is not exposed as a `generate()` option.',
    )
  }

  return example(
    'Practical · sigmoid attention weights',
    py(
      'import math',
      'import torch',
      '',
      'q = torch.randn(2, 8, 128, 64)',
      'k = torch.randn_like(q)',
      'v = torch.randn_like(q)',
      'scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])',
      'weights = torch.sigmoid(scores)  # independent gates, not a probability simplex',
      'context = weights @ v',
    ),
    'Sigmoid attention deliberately drops the softmax sum-to-one constraint. Treat the scale and stability behavior as part of training, not an inference toggle.',
  )
}

function cacheSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'paged') {
    return example(
      'Practical · paged serving with vLLM',
      py(
        '# pip install vllm',
        'from vllm import LLM, SamplingParams',
        '',
        'llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")',
        'params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=80)',
        'for output in llm.generate(["Explain paged KV cache."], params):',
        '    print(output.outputs[0].text)',
      ),
      'PagedAttention is a vLLM serving feature, not a cache implementation selected by `transformers.generate()`. vLLM still loads Hugging Face-format checkpoints.',
    )
  }

  if (variant.id === 'cross-layer') {
    return example(
      'Practical · share a cache object across layers',
      py(
        'import torch',
        '',
        'shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}',
        '',
        'def attention_layer(x, layer_name, make_kv):',
        '    if "source" not in shared_kv:',
        '        shared_kv["source"] = make_kv(x)  # first layer writes the state',
        '    k, v = shared_kv["source"]           # later layers reuse it',
        '    return k, v',
      ),
      'Cross-layer sharing changes the model architecture and training objective. This sketch shows the ownership pattern; use a YOCO-style checkpoint or train the modified model.',
    )
  }

  if (variant.id === 'mla-latent') {
    return example(
      'Practical · inspect an MLA-capable checkpoint',
      py(
        'from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer',
        '',
        'model_id = "deepseek-ai/DeepSeek-V2-Lite"  # choose a checkpoint your Transformers version supports',
        'config = AutoConfig.from_pretrained(model_id)',
        'print("latent rank:", getattr(config, "kv_lora_rank", None))',
        'tokenizer = AutoTokenizer.from_pretrained(model_id)',
        'model = AutoModelForCausalLM.from_pretrained(model_id).eval()',
        'inputs = tokenizer("What does latent KV caching store?", return_tensors="pt")',
        'print(tokenizer.decode(model.generate(**inputs, max_new_tokens=48)[0], skip_special_tokens=True))',
      ),
      'MLA is encoded by the checkpoint’s modeling class and config. Install a Transformers release that supports the chosen DeepSeek-style architecture; it cannot be enabled on a non-MLA model.',
    )
  }

  const settingsByVariant: Record<string, string[]> = {
    none: ['use_cache=False,'],
    full: ['use_cache=True,', 'cache_implementation="dynamic",'],
    grouped: ['use_cache=True,', 'cache_implementation="dynamic",'],
    sliding: ['use_cache=True,', 'cache_implementation="sliding_window",'],
    quantized: ['use_cache=True,', 'cache_implementation="quantized",', 'cache_config={"backend": "quanto", "nbits": 4},'],
  }
  const settings = settingsByVariant[variant.id] ?? ['use_cache=True,']
  return example(
    `Practical · generate with ${variant.label} caching`,
    py(
      ...hubLoad(),
      ...(variant.id === 'grouped'
        ? ['', 'print("Q heads:", model.config.num_attention_heads)', 'print("KV heads:", model.config.num_key_value_heads)']
        : []),
      ...generationCall(settings, 'Explain why a KV cache speeds up decoding.'),
    ),
    variant.id === 'grouped'
      ? 'GQA/MQA cache size comes from the checkpoint’s `num_key_value_heads`. The dynamic cache stores those heads; it does not convert MHA into GQA.'
      : 'This is a real Transformers generation setting. Cache support depends on the selected model and installed Transformers version.',
  )
}

function ffnSnippet(variant: Variant): CodeBlock {
  if (['relu-mlp', 'gelu-mlp', 'geglu', 'swiglu'].includes(variant.id)) {
    const activation = variant.id === 'relu-mlp' ? 'F.relu' : variant.id === 'gelu-mlp' ? 'F.gelu' : variant.id === 'geglu' ? 'F.gelu' : 'F.silu'
    const gated = variant.id === 'geglu' || variant.id === 'swiglu'
    return example(
      `Practical · ${variant.label} feed-forward module`,
      py(
        'import torch',
        'import torch.nn as nn',
        'import torch.nn.functional as F',
        '',
        'class FeedForward(nn.Module):',
        '    def __init__(self, d_model=512, d_ff=1536):',
        '        super().__init__()',
        `        self.up = nn.Linear(d_model, d_ff${gated ? ' * 2' : ''})`,
        '        self.down = nn.Linear(d_ff, d_model)',
        '',
        '    def forward(self, x):',
        `        ${gated ? 'gate, value = self.up(x).chunk(2, dim=-1)\n        return self.down(' + activation + '(gate) * value)' : 'return self.down(' + activation + '(self.up(x)))'}`,
        '',
        'print(FeedForward()(torch.randn(2, 128, 512)).shape)',
      ),
      'This is a directly usable PyTorch replacement for an FFN sublayer. Match its width and activation to the architecture before loading weights.',
    )
  }

  const topK = variant.id === 'switch' ? 1 : 2
  return example(
    `Practical · ${variant.label} routing`,
    py(
      'import torch',
      'import torch.nn as nn',
      '',
      'd_model, n_experts = 512, 8',
      'router = nn.Linear(d_model, n_experts, bias=False)',
      'experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 1536), nn.GELU(), nn.Linear(1536, d_model)) for _ in range(n_experts)])',
      'x = torch.randn(2, 128, d_model)',
      `weights, indices = router(x).softmax(-1).topk(${topK}, dim=-1)`,
      'output = torch.zeros_like(x)',
      'for expert_id, expert in enumerate(experts):',
      '    selected = (indices == expert_id).any(dim=-1)',
      '    output[selected] += expert(x[selected])',
      'print(output.shape)',
    ),
    variant.id === 'expert-choice'
      ? 'This is the common token-choice baseline. Expert Choice inverts the selection so each expert chooses its capacity; it requires a different dispatch policy.'
      : 'This compact implementation shows routing semantics, not a production all-to-all dispatcher. Use a distributed MoE runtime for large expert counts.',
  )
}

function mixerSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'mamba-s6') {
    return example(
      'Practical · load and run a Mamba checkpoint',
      py(
        'import torch',
        'from transformers import AutoModelForCausalLM, AutoTokenizer',
        '',
        'model_id = "state-spaces/mamba-130m-hf"',
        'device = "cuda" if torch.cuda.is_available() else "cpu"',
        'tokenizer = AutoTokenizer.from_pretrained(model_id)',
        'model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()',
        'inputs = tokenizer("A selective SSM carries state, not a KV cache.", return_tensors="pt").to(device)',
        'with torch.inference_mode():',
        '    output_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)',
        'print(tokenizer.decode(output_ids[0], skip_special_tokens=True))',
        'print("model type:", model.config.model_type)',
      ),
      'Use the checkpoint through the normal CausalLM interface. Its Mamba mixer is fixed by the architecture; there is no `use_cache` setting that turns it into attention.',
    )
  }

  if (variant.id === 'mamba2-ssd') {
    return example(
      'Practical · construct a small Mamba-2 language model',
      py(
        'import torch',
        'from transformers import Mamba2Config, Mamba2ForCausalLM',
        '',
        'config = Mamba2Config(',
        '    vocab_size=32_000, hidden_size=512, state_size=64,',
        '    num_hidden_layers=8, expand=2, conv_kernel=4,',
        ')',
        'model = Mamba2ForCausalLM(config)',
        'input_ids = torch.randint(config.vocab_size, (2, 128))',
        'logits = model(input_ids).logits',
        'print(logits.shape)',
      ),
      'This creates randomly initialized architecture code for experiments. To generate useful text, load a compatible trained Mamba-2 checkpoint instead of transferring Transformer weights.',
    )
  }

  if (variant.id === 'hybrid-ssm') {
    return example(
      'Practical · inspect an attention–Mamba hybrid',
      py(
        'from transformers import AutoConfig, AutoModelForCausalLM',
        '',
        'model_id = "ai21labs/Jamba-v0.1"',
        'config = AutoConfig.from_pretrained(model_id)',
        'print("model type:", config.model_type)',
        'print("layer schedule:", getattr(config, "layers_block_type", "model-specific"))',
        '# Loading the full model requires hardware sized for this checkpoint.',
        'model = AutoModelForCausalLM.from_config(config)',
      ),
      'Hybrid architectures choose a layer schedule at training time. Inspect the config to see which layers are Mamba and which are attention; it is not a per-request generation option.',
    )
  }

  if (variant.id === 's4') {
    return example(
      'Practical · a small recurrent SSM layer',
      py(
        'import torch',
        '',
        'batch, steps, d_model = 2, 128, 64',
        'x = torch.randn(batch, steps, d_model)',
        'a = torch.full((d_model,), 0.98)  # stable fixed transition',
        'state = torch.zeros(batch, d_model)',
        'outputs = []',
        'for x_t in x.unbind(dim=1):',
        '    state = a * state + x_t',
        '    outputs.append(state)',
        'y = torch.stack(outputs, dim=1)',
        'print(y.shape)',
      ),
      'This is the recurrent view of an SSM. Production S4 implementations use a structured multi-dimensional state and parallel kernels, but the state does not grow with sequence length.',
    )
  }

  return example(
    'Practical · self-attention mixer in PyTorch',
    py(
      'import torch',
      'import torch.nn.functional as F',
      '',
      'x = torch.randn(2, 128, 512)',
      'q = k = v = x.view(2, 128, 8, 64).transpose(1, 2)',
      'y = F.scaled_dot_product_attention(q, k, v, is_causal=True)',
      'y = y.transpose(1, 2).reshape_as(x)',
      'print(y.shape)',
    ),
    'This is the mixer whose internals the Q/K/V, pattern, score, and cache cards below unpack.',
  )
}

function residualSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'mhc') {
    return example(
      'Practical · constrained multi-stream residual wrapper',
      py(
        'import torch',
        'import torch.nn as nn',
        '',
        'def sinkhorn(logits, rounds=6):',
        '    mix = logits.exp()',
        '    for _ in range(rounds):',
        '        mix = mix / mix.sum(dim=-1, keepdim=True)',
        '        mix = mix / mix.sum(dim=-2, keepdim=True)',
        '    return mix  # approximately doubly stochastic',
        '',
        'batch, streams, steps, d_model = 2, 4, 128, 512',
        'state = torch.randn(batch, streams, steps, d_model)',
        'pre = nn.Parameter(torch.zeros(streams))',
        'post = nn.Parameter(torch.zeros(streams))',
        'mix_logits = nn.Parameter(torch.zeros(streams, streams))',
        'sublayer = nn.Linear(d_model, d_model, bias=False)',
        'branch_input = (torch.tanh(pre)[None, :, None, None] * state).sum(dim=1)',
        'branch_output = sublayer(branch_input)',
        'mixed_state = torch.einsum("ij,bjtd->bitd", sinkhorn(mix_logits), state)',
        'next_state = mixed_state + torch.tanh(post)[None, :, None, None] * branch_output[:, None]',
        'print(next_state.shape)',
      ),
      'This is a compact schematic of mHC’s pre-mapping, constrained residual mixing, and post-mapping. The paper’s exact parameterization and optimizer treatment should be retained for a faithful training run.',
    )
  }

  const bodyByVariant: Record<string, string[]> = {
    'post-ln': ['y = norm(x + attention(x))', 'y = norm(y + ffn(y))'],
    'pre-ln': ['y = x + attention(norm(x))', 'y = y + ffn(norm(y))'],
    'peri-ln': ['y = x + norm(attention(norm(x)))', 'y = y + norm(ffn(norm(y)))'],
    parallel: ['y = x + attention(norm(x)) + ffn(norm(x))'],
    deepnorm: ['alpha = (2 * 24) ** 0.25  # choose from your depth/training recipe', 'y = norm(alpha * x + attention(x))'],
  }
  return example(
    `Practical · ${variant.label} residual wrapper`,
    py(
      'import torch',
      'import torch.nn as nn',
      '',
      'd_model = 512',
      'norm = nn.RMSNorm(d_model)',
      'attention = nn.Linear(d_model, d_model, bias=False)  # stand-in for an attention sublayer',
      'ffn = nn.Sequential(nn.Linear(d_model, 1536), nn.SiLU(), nn.Linear(1536, d_model))',
      'x = torch.randn(2, 128, d_model)',
      ...(bodyByVariant[variant.id] ?? ['y = x + attention(norm(x))', 'y = y + ffn(norm(y))']),
      'print(y.shape)',
    ),
    'Residual placement is part of the block definition. It must agree with the weights and training recipe; do not rearrange it inside a pretrained checkpoint.',
  )
}

function lmheadSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'mtp') {
    return example(
      'Practical · multi-token prediction training heads',
      py(
        'import torch',
        'import torch.nn as nn',
        'import torch.nn.functional as F',
        '',
        'd_model, vocab, n_future = 512, 32_000, 3',
        'heads = nn.ModuleList([nn.Linear(d_model, vocab, bias=False) for _ in range(n_future)])',
        'hidden = torch.randn(2, 128, d_model)',
        'labels = torch.randint(vocab, (2, 128 + n_future))',
        'loss = sum(',
        '    F.cross_entropy(head(hidden).flatten(0, 1), labels[:, step:step + 128].flatten())',
        '    for step, head in enumerate(heads, start=1)',
        ') / n_future',
        'loss.backward()',
      ),
      'MTP adds training-time objectives and heads. A normal CausalLM checkpoint only predicts the next token unless it was trained with this architecture.',
    )
  }

  if (variant.id === 'softcap-head') {
    return example(
      'Practical · cap output logits before sampling',
      py(
        'import torch',
        '',
        'hidden = torch.randn(2, 512)',
        'lm_head = torch.nn.Linear(512, 32_000, bias=False)',
        'cap = 30.0',
        'logits = lm_head(hidden)',
        'capped_logits = cap * torch.tanh(logits / cap)',
        'next_token = capped_logits.argmax(dim=-1)',
      ),
      'Output soft-capping affects both training and decoding behavior. Use a checkpoint trained with it, or include it in your custom CausalLM head.',
    )
  }

  const tied = variant.id === 'tied-head'
  return example(
    `Practical · ${tied ? 'tie' : 'keep'} the language-model head`,
    py(
      'import torch',
      'from transformers import AutoModelForCausalLM, AutoTokenizer',
      '',
      `model_id = ${JSON.stringify(CAUSAL_MODEL)}`,
      'tokenizer = AutoTokenizer.from_pretrained(model_id)',
      'model = AutoModelForCausalLM.from_pretrained(model_id).eval()',
      ...(tied ? ['model.tie_weights()'] : []),
      'input_table = model.get_input_embeddings().weight',
      'output_table = model.get_output_embeddings().weight',
      'print("shared storage:", input_table.data_ptr() == output_table.data_ptr())',
      'logits = model(**tokenizer("Predict the next token", return_tensors="pt")).logits',
      'print(logits.shape)',
    ),
    tied
      ? 'Call `tie_weights()` only for a model architecture designed to share them. The pointer check verifies that the two modules use one parameter tensor.'
      : 'An untied output head is supplied by the loaded checkpoint. Its logits are the input to the sampler.',
  )
}

function samplingSnippet(variant: Variant): CodeBlock {
  if (variant.id === 'speculative') {
    return example(
      'Practical · assisted / speculative decoding',
      py(
        'import torch',
        'from transformers import AutoModelForCausalLM, AutoTokenizer',
        '',
        'target_id = "Qwen/Qwen2.5-1.5B-Instruct"',
        'draft_id = "Qwen/Qwen2.5-0.5B-Instruct"',
        'device = "cuda" if torch.cuda.is_available() else "cpu"',
        'tokenizer = AutoTokenizer.from_pretrained(target_id)',
        'target = AutoModelForCausalLM.from_pretrained(target_id).to(device).eval()',
        'draft = AutoModelForCausalLM.from_pretrained(draft_id).to(device).eval()',
        'inputs = tokenizer("Write a concise explanation of speculative decoding.", return_tensors="pt").to(device)',
        'out = target.generate(**inputs, assistant_model=draft, max_new_tokens=96, do_sample=False)',
        'print(tokenizer.decode(out[0], skip_special_tokens=True))',
      ),
      'Assisted decoding uses a small compatible draft model and preserves the target model’s greedy output when verification succeeds.',
    )
  }

  const settingsByVariant: Record<string, string[]> = {
    greedy: ['do_sample=False,', 'num_beams=1,'],
    beam: ['do_sample=False,', 'num_beams=4,', 'early_stopping=True,'],
    temperature: ['do_sample=True,', 'temperature=0.7,'],
    'top-k': ['do_sample=True,', 'top_k=50,', 'temperature=0.8,'],
    'top-p': ['do_sample=True,', 'top_p=0.9,', 'temperature=0.8,'],
    'min-p': ['do_sample=True,', 'min_p=0.08,', 'temperature=0.8,'],
  }
  return example(
    `Practical · ${variant.label} with generate()`,
    py(...hubLoad(), ...generationCall(settingsByVariant[variant.id] ?? ['do_sample=False,'], 'Write one sentence about practical text generation.')),
    'These are per-request Transformers controls. Sampling fields only take effect when `do_sample=True`; beam search instead uses `num_beams > 1` with sampling off.',
  )
}

export function transformersSnippet(block: Block, variant: Variant): CodeBlock {
  const snippet = (() => {
    switch (block.id) {
      case 'tokenizer': return tokenizerSnippet(variant)
      case 'embedding': return embeddingSnippet(variant)
      case 'positional': return positionalSnippet(variant)
      case 'norm': return normSnippet(variant)
      case 'mixer': return mixerSnippet(variant)
      case 'qkv': return qkvSnippet(variant)
      case 'pattern': return patternSnippet(variant)
      case 'scores': return scoresSnippet(variant)
      case 'kvcache': return cacheSnippet(variant)
      case 'ffn': return ffnSnippet(variant)
      case 'residual': return residualSnippet(variant)
      case 'lmhead': return lmheadSnippet(variant)
      case 'sampling': return samplingSnippet(variant)
    }
  })()
  return withRaschkaReference(block, variant, snippet, 'framework')
}

/*
 * Small from-first-principles equivalents of the operation selected above.
 * They are intentionally shorter than the production Transformers source:
 * no fused kernels, distributed dispatch, sharding, or compatibility layers.
 * The aim is to expose the data flow a reader would otherwise have to infer.
 */
function rawTokenizer(variant: Variant): CodeBlock {
  if (variant.id === 'word') {
    return example('Raw PyTorch · whitespace words', py(
      'import re',
      '',
      'vocab = {"[UNK]": 0, "attention": 1, "is": 2, "useful": 3}',
      'def encode(text: str) -> list[int]:',
      '    words = re.findall(r"\\w+|[^\\w\\s]", text.lower())',
      '    return [vocab.get(word, vocab["[UNK]"]) for word in words]',
      '',
      'print(encode("Attention is useful."))',
    ), 'Core idea only: a production word tokenizer also handles normalization, special tokens, truncation, and serialization.')
  }

  if (variant.id === 'char' || variant.id === 'blt') {
    return example(`Raw PyTorch · ${variant.label} bytes`, py(
      'import torch',
      '',
      'text = "naïve café"',
      'byte_ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)',
      'embedding = torch.nn.Embedding(256, 512)',
      'x = embedding(byte_ids)[None, :, :]  # [batch, byte positions, d_model]',
      'print(byte_ids.tolist(), x.shape)',
    ), 'UTF-8 gives a fixed 256-symbol alphabet. A byte-latent architecture adds its own learned patching or latent encoder after this step.')
  }

  if (variant.id === 'wordpiece') {
    return example('Raw PyTorch · greedy WordPiece segmentation', py(
      'def wordpiece(word: str, vocab: set[str]) -> list[str]:',
      '    pieces, start = [], 0',
      '    while start < len(word):',
      '        end, match = len(word), None',
      '        while start < end:',
      '            token = word[start:end] if start == 0 else "##" + word[start:end]',
      '            if token in vocab: match = token; break',
      '            end -= 1',
      '        if match is None: return ["[UNK]"]',
      '        pieces.append(match); start = end',
      '    return pieces',
      '',
      'print(wordpiece("playing", {"play", "##ing", "[UNK]"}))',
    ), 'WordPiece repeatedly takes the longest legal continuation, marking non-initial pieces with `##`.')
  }

  if (variant.id === 'unigram') {
    return example('Raw PyTorch · Unigram LM dynamic program', py(
      'import math',
      '',
      'scores = {"a": -1.0, "attention": -0.2, "att": -1.8, "ention": -1.0}',
      'def best_segment(text: str):',
      '    dp = [(0.0, [])] + [(math.inf, []) for _ in text]',
      '    for end in range(1, len(text) + 1):',
      '        for start in range(end):',
      '            piece = text[start:end]',
      '            if piece in scores and dp[start][0] - scores[piece] < dp[end][0]:',
      '                dp[end] = (dp[start][0] - scores[piece], dp[start][1] + [piece])',
      '    return dp[-1][1]',
      '',
      'print(best_segment("attention"))',
    ), 'A Unigram tokenizer chooses the lowest-cost segmentation over all pieces, rather than greedily applying merge ranks.')
  }

  return example(`Raw PyTorch · ${variant.label} merge loop`, py(
    'def apply_bpe(symbols: list[str], merge_rank: dict[tuple[str, str], int]):',
    '    while len(symbols) > 1:',
    '        pairs = list(zip(symbols, symbols[1:]))',
    '        best = min(pairs, key=lambda pair: merge_rank.get(pair, float("inf")))',
    '        if best not in merge_rank: break',
    '        merged, out, i = "".join(best), [], 0',
    '        while i < len(symbols):',
    '            if tuple(symbols[i:i + 2]) == best: out.append(merged); i += 2',
    '            else: out.append(symbols[i]); i += 1',
    '        symbols = out',
    '    return symbols',
    '',
    'print(apply_bpe(list("lowest"), {("l", "o"): 0, ("lo", "w"): 1}))',
  ), 'BPE repeatedly merges the lowest-ranked adjacent pair. Byte-level BPE starts from UTF-8 bytes, so every input remains representable.')
}

function rawEmbedding(variant: Variant): CodeBlock {
  if (variant.id === 'factorized') {
    return example('Raw PyTorch · factorized embedding', py(
      'import torch.nn as nn',
      '',
      'vocab, d_embed, d_model = 32_000, 128, 768',
      'lookup = nn.Embedding(vocab, d_embed)',
      'project = nn.Linear(d_embed, d_model, bias=False)',
      '',
      'def embed(input_ids):',
      '    return project(lookup(input_ids))  # [B, T] -> [B, T, d_model]',
    ), 'Factorization makes the vocabulary-sized matrix narrow, then expands token vectors once per token.')
  }

  if (variant.id === 'scaled') {
    return example('Raw PyTorch · scaled embedding', py(
      'import math',
      'import torch.nn as nn',
      '',
      'table = nn.Embedding(32_000, 768)',
      'def embed(input_ids):',
      '    return table(input_ids) * math.sqrt(table.embedding_dim)',
    ), 'The scale is applied immediately after lookup so embedding magnitude is comparable to the residual stream at initialization.')
  }

  const tied = variant.id === 'tied'
  return example(`Raw PyTorch · ${tied ? 'tied' : 'untied'} embedding and LM head`, py(
    'import torch.nn as nn',
    '',
    'vocab, d_model = 32_000, 768',
    'input_table = nn.Embedding(vocab, d_model)',
    'output_head = nn.Linear(d_model, vocab, bias=False)',
    ...(tied ? ['output_head.weight = input_table.weight  # one shared Parameter'] : ['# separate Parameter tensors: input and prediction can specialize independently']),
    '',
    'hidden = input_table(input_ids)',
    'next_token_logits = output_head(hidden)',
  ), tied ? 'The output projection reuses the lookup matrix transposed by `nn.Linear` convention.' : 'The two large vocabulary matrices are learned independently.')
}

function rawPositional(variant: Variant): CodeBlock {
  if (variant.id === 'sinusoidal') {
    return example('Raw PyTorch · sinusoidal positions', py(
      'import math',
      'import torch',
      '',
      'def sinusoidal_positions(length, d_model, device=None):',
      '    position = torch.arange(length, device=device)[:, None]',
      '    freq = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10_000.0) / d_model))',
      '    pe = torch.zeros(length, d_model, device=device)',
      '    pe[:, 0::2] = torch.sin(position * freq)',
      '    pe[:, 1::2] = torch.cos(position * freq)',
      '    return pe',
      'x = x + sinusoidal_positions(x.size(1), x.size(-1), x.device)',
    ), 'Every position has a deterministic vector; there is no learned table to look up.')
  }

  if (variant.id === 'learned') {
    return example('Raw PyTorch · learned absolute positions', py(
      'import torch',
      'import torch.nn as nn',
      '',
      'position_table = nn.Embedding(4096, 768)',
      'positions = torch.arange(x.size(1), device=x.device)',
      'x = x + position_table(positions)[None, :, :]',
    ), 'The maximum embedding index is a real train-time ceiling unless the position table is resized and adapted.')
  }

  if (variant.id === 'relative' || variant.id === 't5-bias') {
    return example('Raw PyTorch · relative distance bias', py(
      'import torch',
      'import torch.nn as nn',
      '',
      'n_heads, n_buckets, max_distance = 8, 32, 128',
      'bias_table = nn.Embedding(n_buckets, n_heads)',
      'q_pos = torch.arange(seq)[:, None]',
      'k_pos = torch.arange(seq)[None, :]',
      'distance = (k_pos - q_pos).clamp(-max_distance, max_distance)',
      'bucket = (distance + max_distance) * (n_buckets - 1) // (2 * max_distance)',
      'bias = bias_table(bucket).permute(2, 0, 1)  # [heads, query, key]',
      'scores = scores + bias[None, :, :, :]',
    ), 'T5 uses a more carefully bucketed distance function, but the key mechanism is a per-head lookup added to attention logits.')
  }

  if (variant.id === 'alibi') {
    return example('Raw PyTorch · ALiBi score slopes', py(
      'import torch',
      '',
      'slopes = torch.tensor([0.5 ** (i / 8) for i in range(8)])',
      'q_pos = torch.arange(seq)[:, None]',
      'k_pos = torch.arange(seq)[None, :]',
      'distance = (q_pos - k_pos).clamp_min(0)',
      'alibi = -slopes[:, None, None] * distance  # [heads, query, key]',
      'scores = scores + alibi[None, :, :, :]',
    ), 'Each head gets a fixed negative distance slope before softmax; no position vectors are added to the residual stream.')
  }

  if (variant.id === 'nope') {
    return example('Raw PyTorch · no positional encoding', py(
      '# x is just token embeddings. No position vector, rotation, or bias is added.',
      'x = token_embedding(input_ids)',
      'q, k, v = q_proj(x), k_proj(x), v_proj(x)',
    ), 'NoPE deliberately leaves position out of this stage. Any order signal must arise elsewhere in the architecture or data path.')
  }

  const setupByVariant: Record<string, { before: string[]; after: string[] }> = {
    'linear-interp': {
      before: ['positions = positions / 2.0  # compress a 2x longer context into trained coordinates'],
      after: [],
    },
    'dynamic-ntk': {
      before: ['factor, trained_length = 2.0, 4096', 'context_length = int(positions.max()) + 1', 'theta = base_theta * ((factor * context_length / trained_length) - (factor - 1)) ** (d_model / (d_model - 2))'],
      after: [],
    },
    yarn: {
      before: [],
      after: ['factor, trained_length = 4.0, 4096', 'wavelength = 2 * torch.pi / freq', 'ramp = ((trained_length / wavelength - 1.0) / (32.0 - 1.0)).clamp(0, 1)', 'freq = (1 - ramp) * (freq / factor) + ramp * freq  # blend interpolation and extrapolation'],
    },
    'llama3-rope': {
      before: [],
      after: ['factor, trained_length = 8.0, 8192', 'low, high = trained_length / 1.0, trained_length / 4.0', 'wavelength = 2 * torch.pi / freq', 'scaled = torch.where(wavelength > low, freq / factor, freq)', 'smooth = ((trained_length / wavelength - 1.0) / (4.0 - 1.0)).clamp(0, 1)', 'freq = torch.where((wavelength >= high) & (wavelength <= low), (1 - smooth) * (freq / factor) + smooth * freq, scaled)'],
    },
  }
  const ropeSetup = setupByVariant[variant.id] ?? { before: [], after: [] }
  return example(`Raw PyTorch · ${variant.label} rotation`, py(
    'import torch',
    '',
    'def rotate_half(x):',
    '    first, second = x.chunk(2, dim=-1)',
    '    return torch.cat((-second, first), dim=-1)',
    '',
    'positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)',
    'base_theta, d_model = 10_000.0, q.size(-1)',
    'theta = base_theta',
    ...ropeSetup.before,
    'freq = 1.0 / (theta ** (torch.arange(0, d_model, 2, device=q.device) / d_model))',
    ...ropeSetup.after,
    'angles = positions[:, None] * freq[None, :]',
    'cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None, :, :]',
    'sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None, :, :]',
    'q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin',
  ), 'RoPE rotates paired Q/K channels by a position-dependent phase. Long-context variants change the frequencies or coordinates before this rotation.')
}

function rawNorm(variant: Variant): CodeBlock {
  if (variant.id === 'layernorm') {
    return example('Raw PyTorch · LayerNorm', py(
      'mean = x.mean(dim=-1, keepdim=True)',
      'variance = (x - mean).square().mean(dim=-1, keepdim=True)',
      'y = (x - mean) * torch.rsqrt(variance + eps)',
      'y = y * weight + bias',
    ), 'LayerNorm removes both mean and scale per token.')
  }
  if (variant.id === 'rmsnorm') {
    return example('Raw PyTorch · RMSNorm', py(
      'rms = x.square().mean(dim=-1, keepdim=True)',
      'y = x * torch.rsqrt(rms + eps)',
      'y = y * weight',
    ), 'RMSNorm keeps the mean and only normalizes the root-mean-square magnitude.')
  }
  if (variant.id === 'qk-norm') {
    return example('Raw PyTorch · QK normalization', py(
      'def rms_norm(x, eps=1e-6):',
      '    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)',
      '',
      'q = rms_norm(q_proj(x))',
      'k = rms_norm(k_proj(x))',
      'scores = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)',
    ), 'Normalize only the query and key vectors, immediately before dot products.')
  }
  return example('Raw PyTorch · Dynamic Tanh', py(
    '# alpha, gamma, and beta are learned vectors with one value per channel.',
    'y = alpha * torch.tanh(gamma * x) + beta',
  ), 'DyT replaces statistic-based normalization with a learned bounded nonlinearity.')
}

function rawQkv(variant: Variant): CodeBlock {
  if (variant.id === 'mla') {
    return example('Raw PyTorch · multi-head latent attention', py(
      'latent = down_proj(x)                    # [B, T, d_latent] — cache this',
      'q = q_proj(x).view(B, T, H, Dh).transpose(1, 2)',
      'k = k_from_latent(latent).view(B, T, H, Dh).transpose(1, 2)',
      'v = v_from_latent(latent).view(B, T, H, Dh).transpose(1, 2)',
      'scores = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)',
      'context = scores.softmax(-1) @ v',
    ), 'MLA reduces what is stored between decode steps by reconstructing K/V from a smaller latent state.')
  }

  const kvHeads = variant.id === 'mha' ? 'Hq' : variant.id === 'mqa' ? '1' : 'Hq // group_size'
  return example(`Raw PyTorch · ${variant.label} projections`, py(
    'q = q_proj(x).view(B, T, Hq, Dh).transpose(1, 2)',
    `k = k_proj(x).view(B, T, ${kvHeads}, Dh).transpose(1, 2)`,
    `v = v_proj(x).view(B, T, ${kvHeads}, Dh).transpose(1, 2)`,
    ...(variant.id === 'mha' ? [] : ['k = k.repeat_interleave(Hq // k.size(1), dim=1)', 'v = v.repeat_interleave(Hq // v.size(1), dim=1)  # broadcast shared KV across query heads']),
    'scores = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)',
    'context = scores.softmax(-1) @ v',
  ), 'The number of projected K/V heads changes cache size. Repetition here expresses the logical grouping; optimized kernels avoid materializing a large copy.')
}

function rawPattern(variant: Variant): CodeBlock {
  const ruleByVariant: Record<string, string[]> = {
    bidirectional: ['allowed = torch.ones(seq, seq, dtype=torch.bool, device=q.device)'],
    causal: ['allowed = q_pos >= k_pos'],
    window: ['allowed = (q_pos >= k_pos) & ((q_pos - k_pos) < window)'],
    sink: ['allowed = (q_pos >= k_pos) & (((q_pos - k_pos) < window) | (k_pos < n_sinks))'],
    dilated: ['distance = q_pos - k_pos', 'allowed = (distance >= 0) & ((distance < window) | (distance % stride == 0))'],
    interleaved: ['local = (q_pos >= k_pos) & ((q_pos - k_pos) < window)', 'global_layer = layer_index % 4 == 0', 'allowed = (q_pos >= k_pos) if global_layer else local'],
    nsa: ['selected = learned_selector(q, k)  # learned boolean/top-k pattern', 'allowed = (q_pos >= k_pos) & selected'],
  }
  return example(`Raw PyTorch · ${variant.label} visibility`, py(
    'q_pos = torch.arange(seq, device=q.device)[:, None]',
    'k_pos = torch.arange(seq, device=q.device)[None, :]',
    ...(ruleByVariant[variant.id] ?? ['allowed = q_pos >= k_pos']),
    'scores = scores.masked_fill(~allowed[None, None], float("-inf"))',
    'weights = scores.softmax(dim=-1)',
  ), 'The mask decides which score entries are legal before softmax. Sparse production kernels avoid calculating the masked entries in the first place.')
}

function rawScores(variant: Variant): CodeBlock {
  if (variant.id === 'softcap') {
    return example('Raw PyTorch · logit soft-capping', py(
      'raw_scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)',
      'scores = softcap * torch.tanh(raw_scores / softcap)',
      'weights = scores.softmax(dim=-1)',
      'context = weights @ v',
    ), 'The tanh bounds extreme dot products while remaining smooth.')
  }
  if (variant.id === 'differential') {
    return example('Raw PyTorch · differential attention', py(
      'a1 = ((q1 @ k1.transpose(-2, -1)) * scale).softmax(dim=-1)',
      'a2 = ((q2 @ k2.transpose(-2, -1)) * scale).softmax(dim=-1)',
      'context = (a1 - lambda_value * a2) @ v',
    ), 'Two attention maps are learned, then one is subtracted to suppress shared noise.')
  }
  if (variant.id === 'sigmoid-attn') {
    return example('Raw PyTorch · sigmoid attention', py(
      'scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)',
      'weights = torch.sigmoid(scores)',
      'context = weights @ v',
    ), 'Each connection is independently gated; weights are not normalized to sum to one.')
  }
  return example('Raw PyTorch · scaled dot-product softmax', py(
    'scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)',
    'scores = scores.masked_fill(~causal_mask, float("-inf"))',
    'weights = torch.softmax(scores, dim=-1)',
    'context = weights @ v',
  ), 'This is the core attention calculation behind the fused PyTorch implementation.')
}

function rawCache(variant: Variant): CodeBlock {
  if (variant.id === 'none') {
    return example('Raw PyTorch · no KV cache', py(
      '# At decode step t, run attention over all t prefix tokens again.',
      'for token in generated_tokens:',
      '    logits = model(generated_tokens).logits[:, -1]',
      '    generated_tokens = torch.cat([generated_tokens, logits.argmax(-1, keepdim=True)], dim=-1)',
    ), 'No cache means K and V for the entire prefix are recomputed on every generated token.')
  }
  if (variant.id === 'sliding') {
    return example('Raw PyTorch · fixed sliding cache', py(
      'def update(cache, new_k, new_v, window=4096):',
      '    k = torch.cat([cache[0], new_k], dim=-2)[..., -window:, :]',
      '    v = torch.cat([cache[1], new_v], dim=-2)[..., -window:, :]',
      '    return k, v',
    ), 'Appending a new token evicts the oldest cached positions once the window is full.')
  }
  if (variant.id === 'paged') {
    return example('Raw PyTorch · paged cache layout', py(
      'page_size, pages = 16, []',
      'def append_token(k_t, v_t):',
      '    if not pages or len(pages[-1][0]) == page_size:',
      '        pages.append(([], []))',
      '    pages[-1][0].append(k_t); pages[-1][1].append(v_t)',
      '',
      '# A sequence stores page indices; pages can be shared/reclaimed independently.',
    ), 'Paged serving stores fixed-size KV blocks rather than one contiguous allocation per request.')
  }
  if (variant.id === 'quantized') {
    return example('Raw PyTorch · quantized KV values', py(
      'def quantize(x):',
      '    scale = x.abs().amax(dim=-1, keepdim=True) / 127',
      '    return (x / scale).round().clamp(-128, 127).to(torch.int8), scale',
      '',
      'k_int8, k_scale = quantize(k_t)',
      'k_for_attention = k_int8.float() * k_scale  # dequantize near the matmul',
    ), 'Real cache quantizers use block-wise metadata and optimized kernels, but the memory trade-off is this integer payload plus a scale.')
  }
  if (variant.id === 'mla-latent') {
    return example('Raw PyTorch · latent KV cache', py(
      'latent_t = down_proj(x_t)              # cache d_latent values, not H * Dh K/V values',
      'k_t = k_from_latent(latent_t)',
      'v_t = v_from_latent(latent_t)',
      'latent_cache = torch.cat([latent_cache, latent_t], dim=-2)',
    ), 'The cache owns a compressed latent; K and V are reconstructed for the attention calculation.')
  }
  if (variant.id === 'cross-layer') {
    return example('Raw PyTorch · cross-layer KV ownership', py(
      'shared_cache = {}',
      'def layer_forward(x, layer_id):',
      '    if layer_id == source_layer:',
      '        shared_cache["kv"] = make_kv(x)',
      '    k, v = shared_cache["kv"]',
      '    return attend(x, k, v)',
    ), 'Later layers read K/V produced by an earlier layer instead of owning independent cache tensors.')
  }
  const grouped = variant.id === 'grouped'
  return example(`Raw PyTorch · ${grouped ? 'grouped' : 'dynamic'} KV append`, py(
    'new_k = k_proj(x_t).view(B, 1, Hkv, Dh).transpose(1, 2)',
    'new_v = v_proj(x_t).view(B, 1, Hkv, Dh).transpose(1, 2)',
    'cache_k = torch.cat([cache_k, new_k], dim=-2)',
    'cache_v = torch.cat([cache_v, new_v], dim=-2)',
    ...(grouped ? ['# Hkv < Hq: queries are grouped onto these fewer cached K/V heads.'] : []),
  ), 'A conventional cache is two tensors per layer, shaped approximately [batch, KV heads, tokens, head dimension].')
}

function rawFfn(variant: Variant): CodeBlock {
  if (variant.id === 'relu-mlp' || variant.id === 'gelu-mlp') {
    const act = variant.id === 'relu-mlp' ? 'torch.relu' : 'torch.nn.functional.gelu'
    return example(`Raw PyTorch · ${variant.label}`, py(
      `hidden = ${act}(x @ W_up + b_up)`,
      'output = hidden @ W_down + b_down',
    ), 'Each token independently expands into a wider feature space, applies a nonlinearity, then projects back to model width.')
  }
  if (variant.id === 'geglu' || variant.id === 'swiglu') {
    const act = variant.id === 'geglu' ? 'torch.nn.functional.gelu' : 'torch.nn.functional.silu'
    return example(`Raw PyTorch · ${variant.label}`, py(
      'gate, value = (x @ W_up).chunk(2, dim=-1)',
      `hidden = ${act}(gate) * value`,
      'output = hidden @ W_down',
    ), 'A second projection creates a learned gate that controls which expanded features are passed through.')
  }
  if (variant.id === 'expert-choice') {
    return example('Raw PyTorch · Expert Choice dispatch', py(
      'scores = router(x).softmax(dim=-1)                  # [tokens, experts]',
      'for expert_id in range(n_experts):',
      '    token_ids = scores[:, expert_id].topk(capacity).indices',
      '    output[token_ids] += experts[expert_id](x[token_ids])',
    ), 'Experts select their highest-scoring tokens, so capacity is naturally balanced per expert.')
  }
  if (variant.id === 'shared-expert') {
    return example('Raw PyTorch · shared plus routed experts', py(
      'shared_output = shared_expert(x)',
      'router_scores = router(x).softmax(dim=-1)',
      'weights, expert_ids = router_scores.topk(k=2, dim=-1)',
      'routed_output = dispatch_to_selected_experts(x, expert_ids, weights)',
      'output = shared_output + routed_output',
    ), 'The shared expert handles always-useful features while fine-grained routed experts specialize.')
  }
  const k = variant.id === 'switch' ? 1 : 2
  return example(`Raw PyTorch · ${variant.label} routing`, py(
    'router_scores = router(x).softmax(dim=-1)',
    `weights, expert_ids = router_scores.topk(k=${k}, dim=-1)`,
    'output = torch.zeros_like(x)',
    'for route_slot in range(expert_ids.size(-1)):',
    '    for expert_id, expert in enumerate(experts):',
    '        selected = expert_ids[:, route_slot] == expert_id',
    '        output[selected] += expert(x[selected]) * weights[selected, route_slot, None]',
  ), 'Tokens take only their selected expert paths. Production MoE uses capacity limits and all-to-all communication around this routing step.')
}

function rawMixer(variant: Variant): CodeBlock {
  if (variant.id === 'mamba-s6') {
    return example('Raw PyTorch · selective S6 recurrence', py(
      'import torch',
      'import torch.nn.functional as F',
      '',
      '# x: [B, T, D]; A: [D, N]; B_t/C_t: [B, T, N]',
      'state = x.new_zeros(x.size(0), x.size(-1), A.size(-1))',
      'outputs = []',
      'for t in range(x.size(1)):',
      '    dt = F.softplus(delta[:, t]).unsqueeze(-1)           # token-selected step size',
      '    a_bar = torch.exp(dt * A.unsqueeze(0))',
      '    write = dt * B[:, t, None, :] * x[:, t, :, None]',
      '    state = a_bar * state + write',
      '    y_t = (state * C[:, t, None, :]).sum(dim=-1) + D * x[:, t]',
      '    outputs.append(y_t)',
      'y = torch.stack(outputs, dim=1)',
    ), 'The loop exposes the selective recurrence. Mamba executes its equivalent with a fused parallel scan rather than Python iteration.')
  }
  if (variant.id === 'mamba2-ssd') {
    return example('Raw PyTorch · SSD chunk recurrence', py(
      'def ssd_chunk(x, a, b, c, state):',
      '    # x: [B, L, D]; a/b/c: [B, L]; state: [B, D]',
      '    outputs = []',
      '    for t in range(x.size(1)):',
      '        state = a[:, t, None] * state + b[:, t, None] * x[:, t]',
      '        outputs.append(c[:, t, None] * state)',
      '    return torch.stack(outputs, dim=1), state',
      '',
      '# Production SSD evaluates the within-chunk causal operator with matmuls,',
      '# then carries `state` to the next chunk.',
    ), 'The scalar transition makes the causal operator dual to a structured matrix multiplication. This loop keeps the recurrence visible; the fast kernel replaces it within each chunk.')
  }
  if (variant.id === 'hybrid-ssm') {
    return example('Raw PyTorch · interleaved hybrid trunk', py(
      'for layer_index, layer in enumerate(layers):',
      '    x = norm(x)',
      '    if layer.kind == "mamba":',
      '        x = x + layer.selective_ssm(x, ssm_state[layer_index])',
      '    else:',
      '        x = x + layer.attention(x, kv_cache[layer_index])',
      '    x = x + layer.ffn(norm(x))',
    ), 'The layer schedule determines which state is owned: an SSM state for Mamba layers and a growing KV cache only for attention layers.')
  }
  if (variant.id === 's4') {
    return example('Raw PyTorch · fixed SSM recurrence', py(
      'state = x.new_zeros(batch, d_model)',
      'outputs = []',
      'for x_t in x.unbind(dim=1):',
      '    state = A * state + B * x_t',
      '    outputs.append(C * state + D * x_t)',
      'y = torch.stack(outputs, dim=1)',
    ), 'Every position uses the same A, B, C, and D. Selective SSMs replace those fixed terms with input-dependent values.')
  }
  return example('Raw PyTorch · causal self-attention mixer', py(
    'scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)',
    'scores = scores.masked_fill(~causal_mask, float("-inf"))',
    'y = scores.softmax(dim=-1) @ v',
  ), 'Self-attention explicitly mixes each query with visible value vectors. The next cards split this operation into its architectural choices.')
}

function rawResidual(variant: Variant): CodeBlock {
  if (variant.id === 'mhc') {
    return example('Raw PyTorch · mHC stream mixing', py(
      'def sinkhorn(logits, rounds=6):',
      '    matrix = logits.exp()',
      '    for _ in range(rounds):',
      '        matrix = matrix / matrix.sum(-1, keepdim=True)',
      '        matrix = matrix / matrix.sum(-2, keepdim=True)',
      '    return matrix',
      '',
      'branch_input = (pre_map[None, :, None, None] * streams).sum(dim=1)',
      'branch_output = sublayer(branch_input)',
      'streams = torch.einsum("ij,bjtd->bitd", sinkhorn(mix_logits), streams)',
      'streams = streams + post_map[None, :, None, None] * branch_output[:, None]',
    ), 'Streams carry the residual state. A constrained matrix mixes old streams; the sub-layer reads one mixture and writes its output back across them.')
  }
  const implementation: Record<string, string[]> = {
    'post-ln': ['x = norm(x + attention(x))', 'x = norm(x + ffn(x))'],
    'pre-ln': ['x = x + attention(norm(x))', 'x = x + ffn(norm(x))'],
    'peri-ln': ['x = x + norm(attention(norm(x)))', 'x = x + norm(ffn(norm(x)))'],
    parallel: ['h = norm(x)', 'x = x + attention(h) + ffn(h)'],
    deepnorm: ['x = norm(alpha * x + attention(x))', 'x = norm(alpha * x + ffn(x))'],
  }
  return example(`Raw PyTorch · ${variant.label} block`, py(...(implementation[variant.id] ?? implementation['pre-ln'])), 'The same attention and FFN functions sit in a different normalization/residual arrangement, which changes optimization behavior.')
}

function rawLmHead(variant: Variant): CodeBlock {
  if (variant.id === 'softcap-head') {
    return example('Raw PyTorch · capped LM logits', py(
      'logits = hidden @ output_weight.T',
      'logits = logit_cap * torch.tanh(logits / logit_cap)',
      'next_token = logits.argmax(dim=-1)',
    ), 'Cap only the pre-sampling logits; the sampler still consumes a normal vocabulary-sized logit vector.')
  }
  if (variant.id === 'mtp') {
    return example('Raw PyTorch · multi-token prediction losses', py(
      'loss = 0.0',
      'for offset, head in enumerate(prediction_heads, start=1):',
      '    logits = head(hidden[:, :-offset])',
      '    loss += cross_entropy(logits.flatten(0, 1), labels[:, offset:].flatten())',
      'loss = loss / len(prediction_heads)',
    ), 'Each auxiliary head predicts a different future offset, giving the backbone a denser training signal.')
  }
  const tied = variant.id === 'tied-head'
  return example(`Raw PyTorch · ${tied ? 'tied' : 'untied'} vocabulary projection`, py(
    'input_embedding = nn.Embedding(vocab_size, d_model)',
    'lm_head = nn.Linear(d_model, vocab_size, bias=False)',
    ...(tied ? ['lm_head.weight = input_embedding.weight'] : ['# lm_head.weight remains a separate learnable matrix']),
    'logits = lm_head(final_hidden_state)',
  ), 'The output head applies one vocabulary-wide dot product to each final hidden state.')
}

function rawSampling(variant: Variant): CodeBlock {
  if (variant.id === 'greedy') {
    return example('Raw PyTorch · greedy next token', py(
      'next_token = logits[:, -1].argmax(dim=-1, keepdim=True)',
    ), 'Choose the largest logit directly; no sampling distribution is constructed.')
  }
  if (variant.id === 'temperature') {
    return example('Raw PyTorch · temperature sampling', py(
      'probs = torch.softmax(logits[:, -1] / temperature, dim=-1)',
      'next_token = torch.multinomial(probs, num_samples=1)',
    ), 'Dividing logits by a lower temperature sharpens the distribution before multinomial sampling.')
  }
  if (variant.id === 'top-k') {
    return example('Raw PyTorch · top-k sampling', py(
      'last = logits[:, -1]',
      'values, _ = last.topk(k, dim=-1)',
      'cutoff = values[:, -1, None]',
      'filtered = last.masked_fill(last < cutoff, float("-inf"))',
      'next_token = torch.multinomial(filtered.softmax(-1), 1)',
    ), 'All but the k highest-scoring tokens receive negative infinity before softmax.')
  }
  if (variant.id === 'top-p') {
    return example('Raw PyTorch · nucleus (top-p) sampling', py(
      'last = logits[:, -1]',
      'sorted_logits, sorted_ids = last.sort(dim=-1, descending=True)',
      'cumulative = sorted_logits.softmax(-1).cumsum(dim=-1)',
      'remove = cumulative > top_p',
      'remove[:, 1:] = remove[:, :-1].clone(); remove[:, 0] = False',
      'sorted_logits[remove] = float("-inf")',
      'filtered = torch.full_like(last, float("-inf")).scatter(-1, sorted_ids, sorted_logits)',
      'next_token = torch.multinomial(filtered.softmax(-1), 1)',
    ), 'The cutoff is based on cumulative probability mass, so the number of retained tokens changes for each step.')
  }
  if (variant.id === 'min-p') {
    return example('Raw PyTorch · min-p sampling', py(
      'probs = logits[:, -1].softmax(dim=-1)',
      'threshold = min_p * probs.max(dim=-1, keepdim=True).values',
      'probs = probs.masked_fill(probs < threshold, 0.0)',
      'next_token = torch.multinomial(probs / probs.sum(dim=-1, keepdim=True), 1)',
    ), 'Min-p measures every candidate against the most likely token in the current distribution.')
  }
  if (variant.id === 'beam') {
    return example('Raw PyTorch · one beam-search expansion', py(
      'log_probs = logits[:, -1].log_softmax(dim=-1)',
      'candidate_scores = beam_scores[:, None] + log_probs',
      'best_scores, flat_ids = candidate_scores.flatten(1).topk(num_beams, dim=-1)',
      'parent_beam = flat_ids // vocab_size',
      'next_token = flat_ids % vocab_size',
    ), 'Beam search ranks cumulative log-probability across several partial sequences instead of choosing a single sampled token.')
  }
  return example('Raw PyTorch · speculative verification', py(
    'draft_tokens = draft_model.propose(prefix, n_tokens=4)',
    'target_logits = target_model(prefix + draft_tokens).logits',
    'accepted = first_position_where_target_rejects(target_logits, draft_tokens)',
    'prefix = torch.cat([prefix, draft_tokens[:accepted]], dim=-1)',
    'prefix = torch.cat([prefix, sample_target(target_logits[:, accepted])], dim=-1)',
  ), 'The fast draft proposes several tokens; the target verifies them in one forward pass and preserves the target distribution.')
}

export function rawImplementationSnippet(block: Block, variant: Variant): CodeBlock {
  const snippet = (() => {
    switch (block.id) {
      case 'tokenizer': return rawTokenizer(variant)
      case 'embedding': return rawEmbedding(variant)
      case 'positional': return rawPositional(variant)
      case 'norm': return rawNorm(variant)
      case 'mixer': return rawMixer(variant)
      case 'qkv': return rawQkv(variant)
      case 'pattern': return rawPattern(variant)
      case 'scores': return rawScores(variant)
      case 'kvcache': return rawCache(variant)
      case 'ffn': return rawFfn(variant)
      case 'residual': return rawResidual(variant)
      case 'lmhead': return rawLmHead(variant)
      case 'sampling': return rawSampling(variant)
    }
  })()
  return withRaschkaReference(block, variant, snippet, 'implementation')
}

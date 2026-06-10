# Pre-Training — AlmondT5

> *Before T5 can translate, summarize, or classify — it needs to learn language itself. Span corruption is how it does that.*

---

## Why T5 Pre-Training is Different

GPT pre-trains by predicting the next token — always one clear answer. BERT pre-trains by predicting masked tokens — ambiguous but token-level. T5 pre-trains by reconstructing corrupted spans — the hardest of the three.

```
GPT  objective : "The quick brown [?]"  →  "fox"          ← one answer
BERT objective : "The [MASK] brown fox" →  "quick"        ← one answer
T5   objective : "The <X> fox <Y> dog"  →  "<X> quick brown <Y> jumps over the lazy </s>"
                                                           ← reconstruct structure + content
```

T5's decoder must predict: the sentinel token in the right order, then the original tokens, then the next sentinel — while attending to a corrupted encoder input. It's not just filling blanks — it's reconstructing a structured sequence.

---

## Span Corruption — The Pre-Training Objective

**15% of tokens are selected for corruption. Selected tokens are grouped into contiguous spans.**

```
Original  : "The quick brown fox jumps over the lazy dog."
Corrupted : "The <extra_id_0> fox <extra_id_1> lazy dog."
Target    : "<extra_id_0> quick brown <extra_id_1> jumps over the </s>"
```

**Why spans instead of individual tokens (like BERT)?**

When multiple consecutive tokens form a meaningful unit — "New York City", "machine learning", "quick brown fox" — masking them individually gives partial context that makes prediction easier. Masking the entire span forces the model to reconstruct the full unit from surrounding context.

**Why sentinel tokens instead of [MASK]?**

BERT uses one [MASK] token regardless of span length. T5 uses unique sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...) for each corrupted span. This allows the decoder to output multiple spans in a single sequence, each clearly delimited by its sentinel. Without unique sentinels, the decoder would not know where one span ends and the next begins.

**Masking strategy:**

```
mask_prob             : 0.15  (15% of tokens selected)
mean_noise_span_length: 3.0   (average span = 3 tokens, geometric distribution)
num_extra_ids         : 100   (sufficient for ~6 spans per 128-token sequence)
```

With mean span length 3 and 15% mask ratio on a 128-token sequence:
```
Tokens to mask  : ~19 tokens
Expected spans  : ~6 spans
Sentinels needed: ~6 per sample
```

---

## Dataset

**Bilingual corpus: Wikipedia EN + Wikipedia ID**

```python
from datasets import load_dataset

en_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:25000]")
id_wiki = load_dataset("wikimedia/wikipedia", "20231101.id", split="train[:25000]")
```

**Why bilingual for pre-training?**

The downstream task is EN→ID translation. If the tokenizer and pre-trained representations only see English, the model will struggle to generate Indonesian during fine-tuning — it has never seen the language's vocabulary and structure during the representation-learning phase.

By pre-training on both languages, the model develops shared representations where semantically similar concepts in English and Indonesian are closer in embedding space. This is the same principle behind multilingual models like mBERT and mT5.

**Corpus preparation:**

```
Wikipedia raw text → filter short articles (< 20 chars, > 300 chars per chunk)
                  → split into sentence-level chunks
                  → save as single stream .txt
                  → train SentencePiece tokenizer from this corpus
                  → tokenize corpus → stream of token IDs
                  → apply span corruption at get_batch time (dynamic)
```

---

## Tokenizer — SentencePiece Unigram

AlmondT5 uses **SentencePiece Unigram**, not BPE. This is a deliberate departure from AlmondGPT and AlmondBERT.

**Why Unigram for T5?**

BPE merges by frequency — bottom-up, deterministic. One word always tokenizes the same way.

Unigram starts with a large seed vocabulary and prunes it by maximizing the likelihood of the training corpus under a unigram language model. The result: tokens that are linguistically more meaningful, not just statistically frequent pairs.

For translation specifically, this matters because:

```
BPE  : "translation" → ["trans", "##lation"]   ← byte-fragment split
Unigram: "translation" → ["▁translation"]       ← whole word if frequent enough
         "terjemahan"  → ["▁terjemahan"]        ← Indonesian preserved similarly
```

The `▁` marker explicitly marks word boundaries — the decoder knows exactly where one word ends and another begins when generating Indonesian output.

**Tokenizer config:**

```yaml
vocab_size       : 6000
seed_vocab_size  : 20000       # start large, prune to 6000
max_piece_length : 8
num_em_steps     : 2           # EM iterations for Unigram LM fitting
shrinking_factor : 0.75        # prune 25% each iteration
smoothing        : 1e-8
whitespace_marker: _           # ▁ marker for word boundaries
num_extra_ids    : 100         # <extra_id_0> ... <extra_id_99>
```

**Special token IDs:**

```
ID 0 : <pad>          ← padding + decoder start token
ID 1 : <unk>          ← unknown (rare at byte-level coverage)
ID 2 : </s>           ← end of sequence
ID 3 : <extra_id_0>   ← sentinel for span corruption
...
ID 102: <extra_id_99>
```

**Vocab size trade-off:**

6,000 tokens is small. Standard T5 uses 32,000. With 6,000 covering both English and Indonesian, many words fragment into 3-5 pieces. This is the primary quality bottleneck for this project — a deliberate constraint to keep compute feasible on a single T4 GPU.

---

## Architecture

### Encoder — Bidirectional + Modern

The encoder reads the corrupted input sequence with full bidirectional attention. Every token attends to every other token — identical to BERT, just without [MASK].

**Alternating Attention:**

Standard full attention is O(N²) — every token attends to every other. For longer sequences this becomes compute-heavy.

AlmondT5 encoder uses alternating attention:

```
Even blocks → Full global attention O(N²)  ← see all tokens
Odd blocks  → Local sliding window O(N×W)  ← see W nearest tokens only
```

With 6 encoder blocks:
```
Block 0 → Full attention
Block 1 → Local attention (window = 32)
Block 2 → Full attention
Block 3 → Local attention
Block 4 → Full attention
Block 5 → Local attention
```

Time complexity of alternating vs full:
```
Full attention (6 blocks)        : 6 × O(N²)
Alternating attention (3+3)      : 3 × O(N²) + 3 × O(N×W)
                                 = O(N²) dominated by global blocks
                                   but ~40-50% compute saved on local blocks
```

**Unpadding:**

In a standard padded batch, the model computes attention for PAD tokens — wasted compute. Unpadding removes all PAD tokens before entering attention, flattening `(B, T, C)` to `(TotalRealTokens, C)`.

The challenge: without batch structure, cross-sentence attention must be prevented. Each token carries a `batch_id`, and the attention mask is built from `batch_id` equality — tokens from different sentences get `-inf` attention score.

```python
same_batch = (valid_batch_ids.unsqueeze(0) == valid_batch_ids.unsqueeze(1))
attn_mask  = torch.zeros(...).masked_fill(~same_batch, float("-inf"))
```

After attention, tokens are re-padded back to `(B, T, C)` for the residual connection.

**GeGLU FFN (Encoder):**

```
gate   = GELU(Linear(x, 8d/3))
value  = Linear(x, 8d/3)
output = Linear(gate * value, d)
```

GELU gate provides smooth gradient through slightly negative values. The 8/3 expansion factor matches the parameter count of a standard 4× FFN with two matrices.

---

### Decoder — Autoregressive + KV Cache

The decoder generates output tokens one at a time. At each step, it needs:
1. **Self-attention (causal)** — attend to all previously generated tokens
2. **Cross-attention** — query the encoder output for relevant input context
3. **FFN** — transform the attended representation

**KV Cache:**

Without KV cache, every new token requires recomputing K and V for all previous tokens — O(N) K,V computations per step, O(N²) total across generation.

With KV cache, K and V from previous steps are stored and reused:

```python
# Step 1: generate token 1
K1, V1 = compute(token_1)
cache = {K: [K1], V: [V1]}

# Step 2: generate token 2
K2, V2 = compute(token_2)
keys   = [K1, K2]   # ← from cache + new
values = [V1, V2]
cache  = {K: [K1, K2], V: [V1, V2]}
```

Each step is now O(1) for K,V computation, O(n) for attention (attend to cached tokens). Total: O(N) instead of O(N²).

KV cache is applied to both decoder self-attention and cross-attention. For cross-attention specifically, the encoder output never changes — K and V from the encoder are computed once at the start and reused at every decoder step.

**Trade-off:** Memory grows linearly with sequence length. For max_decoder_len=128 and batch_size=32, this is manageable on a T4.

**SwiGLU FFN (Decoder):**

```
gate   = SiLU(Linear(x, 8d/3))    ← SiLU instead of GELU
value  = Linear(x, 8d/3)
output = Linear(gate * value, d)
```

SiLU (Sigmoid Linear Unit) = `x * sigmoid(x)`. Slightly smoother gradient than GELU for negative values. Both GeGLU and SwiGLU perform empirically better than ReLU — the choice between them at this scale is marginal.

---

### Tied Embeddings

AlmondT5 uses tied word embeddings:

```
encoder_embedding.weight == decoder_embedding.weight == lm_head.weight.T
```

Three separate matrices share the same underlying weights. This means:
- The encoder sees words the same way the decoder does
- The LM head projects back to the same space tokens came from
- Parameter count reduced by ~2 × vocab_size × embed_dim

At vocab_size=6000 and embed_dim=256, this saves ~3M parameters — significant at this scale.

---

## Positional Encoding — The Ablation

AlmondT5 was trained with two different positional encoding strategies to compare their effect on downstream translation quality.

### RoPE — Rotary Positional Encoding

Position is encoded as a rotation of the query and key vectors:

```
q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)
k_rotated = k * cos(θ) + rotate_half(k) * sin(θ)
```

The angle θ depends on position. What matters is the **relative angle between two positions** — not the absolute position itself. Two tokens that always appear close together will always have a small relative rotation, producing a consistently high dot product in attention.

RoPE is continuous — every position gets a unique, mathematically precise encoding. It generalizes well to lengths not seen during training because the rotation formula extends naturally.

### Relative Position Bias

Instead of rotating vectors, Relative Position Bias adds a learnable scalar bias to the attention score based on the distance between tokens:

```
Attention_score(i, j) = QᵢKⱼᵀ/√d + bias(i - j)
```

Distances are grouped into **buckets** with logarithmic spacing:

```
Buckets 0 to max_exact   : distance = bucket_id (exact, fine-grained for nearby tokens)
Buckets max_exact to max  : log-spaced (coarser for distant tokens)
Bucket max_buckets - 1    : clamped (all very distant tokens share the last bucket)
```

**Intuition:** nearby tokens get precise, individual distance encodings. Distant tokens share buckets — the model knows they're "far" but not exactly how far. This matches how language works: the exact distance between close words matters (subject-verb agreement), but whether a word is 50 or 60 positions away matters much less.

**Why Relative Bias won on BLEU:**

Pre-training loss was nearly identical (4.7165 vs 4.6998). But Relative Bias scored BLEU 4.71 vs RoPE 2.99 — a 57% difference.

Hypothesis: for seq2seq translation at small scale, the inductive bias of logarithmic bucket spacing better matches the translation signal. Translation often requires attending to nearby structural elements (subject, verb, object) with precise distance awareness, while RoPE's continuous encoding may not provide the same strong local proximity signal at 6,000 vocab and 256 embedding dimensions.

---

## Training Config

```yaml
embed_dim     : 256
n_heads       : 8
n_encoder_blocks: 6
n_decoder_blocks: 6
dropout       : 0.1
batch_size    : 32
learning_rate : 3e-4
weight_decay  : 0.01
epochs        : 15
eval_interval : 3
mask_prob     : 0.15
mean_span_len : 3.0
grad_clip     : 1.0
early_stopping_patience: 3
```

---

## Loss Curve

![Pre-Training Loss](assets/pretrain_loss.png)

Both models converge similarly. RoPE achieves slightly lower final eval loss (4.6998 vs 4.7165), but this difference does not translate to better downstream BLEU — see fine-tuning results.

---

## Span Corruption Output Sample

```
ENCODER INPUT:
"<extra_id_0> company <extra_id_1> early <extra_id_2> wentieth century
 and later became one of the <extra_id_3> companies in the country."

GENERATED (both models):
"<pad> <extra_id_0> <extra_id_1> of the <extra_id_2> district <extra_id_3> s </s>"

LABEL:
"The was founded in the t largest"
```

The model correctly predicts sentinel token order and generates plausible completions, but does not recover the exact original tokens. This is expected — with vocab_size=6,000 and a small model, span corruption is a hard task. The representations learned are still useful for downstream fine-tuning.

---

## What I Learned

**Span corruption forces structural understanding.**

Unlike BERT's token-level masking, span corruption requires the model to reason about where a span begins, what it contained, and where the next span starts — all while attending to a corrupted context. The decoder learns to output structured sentinel-delimited sequences, not just tokens. This structural learning is what makes T5 representations useful for generation tasks, not just classification.

**Pre-training loss and downstream quality can diverge.**

RoPE achieved lower pre-training loss but lower BLEU. The training objective (span corruption perplexity) is a proxy for representation quality, not a direct measure of translation ability. This is a fundamental insight in transfer learning: optimize the proxy, but evaluate on the target.

**Bilingual pre-training matters for translation.**

A monolingual pre-training corpus would have produced encoder representations with no exposure to Indonesian vocabulary. Fine-tuning cannot recover what pre-training never saw. Shared multilingual pre-training gives the model a foundation where English and Indonesian words already live in a partially shared space before translation training begins.

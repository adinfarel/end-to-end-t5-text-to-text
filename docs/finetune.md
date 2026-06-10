# Fine-Tuning — AlmondT5 EN→ID Translation

> *Pre-training gave the model language. Fine-tuning gives it a job.*

---

## From Span Corruption to Translation

After pre-training, AlmondT5 has learned:
- How to encode a corrupted sequence into contextual representations (encoder)
- How to decode structured sentinel-delimited output from those representations (decoder)
- Cross-attention: how to query encoder context at every decoder step

Fine-tuning for translation reuses all of this — the encoder now reads a full English sentence, and the decoder generates Indonesian. The mechanism is identical. Only the task framing changes.

```
Pre-training:
  Encoder input  : "The <extra_id_0> fox <extra_id_1> lazy dog"
  Decoder target : "<extra_id_0> quick brown <extra_id_1> jumps over the </s>"

Fine-tuning:
  Encoder input  : "translate English to Indonesian: I eat rice"
  Decoder target : "Aku makan nasi </s>"
```

The model already knows how to: read input → build representation → generate output token by token. Fine-tuning teaches it the mapping from English meaning to Indonesian words.

---

## Text-to-Text Format

Every task in T5 is framed as text-to-text. For translation, a prefix is prepended to the encoder input to signal which task to perform:

```
prefix + source sentence → target sentence

"translate English to Indonesian: Hello, how are you?"
→ "Halo, apa kabar?"
```

**Why a text prefix instead of a special token?**

T5's design philosophy is that all tasks should use the same format — natural language in, natural language out. A text prefix like `"translate English to Indonesian:"` is just more tokens that the encoder processes normally. No task-specific heads, no special tokens beyond what was learned during pre-training.

This means the same model and the same inference code handles all tasks — only the prefix changes.

---

## Teacher Forcing

During training, the decoder does not use its own generated output as input for the next step. Instead, it receives the ground truth target shifted one position to the right:

```
Target          : ["Aku", "makan", "nasi", "</s>"]
Decoder input   : [<pad>, "Aku", "makan", "nasi"]    ← ground truth, shifted right
Labels          : ["Aku", "makan", "nasi", "</s>"]    ← what the model must predict
```

The `<pad>` token (ID 0) serves as the decoder start token — signaling "begin generation."

**Why teacher forcing?**

Without it, the decoder receives its own (likely wrong) predictions as input during early training. Errors compound — a wrong token at step 1 corrupts step 2, step 2 corrupts step 3, and so on. The model never sees what the correct continuation looks like.

With teacher forcing, the decoder always receives the correct previous token regardless of what it predicted. Training is stable because the input distribution is consistent throughout.

**Trade-off:** teacher forcing creates an exposure bias — the model trains on ground truth inputs but must use its own predictions at inference time. For short sequences (max 128 tokens), this gap is manageable.

---

## Dataset

**OPUS Indonesian-English** — parallel translation corpus:

```python
from datasets import load_dataset
ds = load_dataset("kaitchup/opus-Indonesian-to-English", split="train[:25000]")
```

Format:
```jsonl
{"en_lang": "Okay ... it's ...", "id_lang": "Oke... itu... "}
{"en_lang": "Now you wanna drive.", "id_lang": "Kini kau mau mengemudi."}
{"en_lang": "You in pain?", "id_lang": "Kau kesakitan?"}
```

**Why OPUS and not a larger dataset?**

OPUS provides clean, parallel sentence pairs from subtitles and literary sources — good diversity of conversational and narrative language. At 25,000 pairs, the dataset is intentionally small to match the model's capacity. Training a 15M parameter model on 1M+ pairs would not improve quality proportionally but would dramatically increase compute time.

**Train/val split:** 80/20 (20,000 train, 5,000 val)

---

## Architecture Changes from Pre-Training

No architectural changes. The same encoder-decoder model is loaded from the pre-training checkpoint. Fine-tuning only:

1. Replaces the span corruption dataloader with a translation dataloader
2. Uses standard seq2seq cross-entropy loss (no sentinel structure)
3. Applies a smaller learning rate to avoid catastrophic forgetting

---

## Loss Function

Cross-entropy between decoder predictions and target tokens, with `ignore_index=-100` for padding:

```python
logits  = model(encoder_input, decoder_input)    # (B, tgt_len, vocab_size)
loss    = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100
)
```

Labels are constructed from the target token IDs with PAD positions set to -100. Only real target tokens contribute to the loss — padding is ignored.

---

## Training Config

```yaml
learning_rate           : 1e-4     (lower than pre-training 3e-4)
batch_size              : 32
epochs                  : 15
eval_interval           : 3
grad_clip               : 1.0
weight_decay            : 0.01
early_stopping_patience : 4
optimizer               : AdamW
ignore_index            : -100
prefix                  : "translate English to Indonesian: "
```

**Why lower learning rate than pre-training?**

The pre-trained weights encode general language representations built from 25,000 Wikipedia articles in two languages. A large learning rate would overwrite these representations — catastrophic forgetting. LR 1e-4 nudges the weights toward the translation task while preserving the structural knowledge from pre-training.

---

## Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)

BLEU measures n-gram overlap between the prediction and the reference translation:

```
BLEU = BP × exp(Σ wₙ × log pₙ)

pₙ = precision of n-grams (n=1,2,3,4)
BP = brevity penalty (penalize short predictions)
```

BLEU of 4.7 is low by academic standards (state-of-the-art EN-ID is 30+), but expected given:
- Vocab size 6,000 (many words fragment into subwords)
- Model size 15-20M (vs 60M+ for capable translation models)
- Training data 25,000 pairs (vs millions for production systems)

### chrF (Character F-score)

chrF measures character n-gram overlap — more forgiving than BLEU for morphologically rich languages like Indonesian. A prediction that gets close but not exact is rewarded by chrF but penalized by BLEU.

```
chrF = F-score of character n-gram precision and recall
```

chrF 21.3 for Relative Bias indicates the model is generating partially correct Indonesian morphology even when full word-level accuracy is low.

---

## Loss Curve

![Fine-Tuning Loss](assets/finetune_loss.png)

| Epoch | Rel.Bias Train | Rel.Bias Eval | RoPE Train | RoPE Eval |
|-------|---------------|--------------|-----------|----------|
| 3 | 3.6536 | 3.6486 | 3.6185 | 3.7097 |
| 6 | 3.4994 | 3.4581 | 3.2982 | 3.5070 |
| 9 | 3.0532 | **3.4338** | 2.6651 | **3.4480** |
| 12 | 2.9532 | 3.4978 | 2.5692 | **3.4430** |
| 15 | 2.3236 | 3.5100 | 2.9392 | 3.4557 |

Train loss continues to drop past epoch 9 while eval loss plateaus — classic sign of overfitting at this dataset size. Early stopping triggered correctly: best checkpoint is from epoch 9 for Relative Bias (3.4338) and epoch 12 for RoPE (3.4430).

---

## Evaluation Results

![Evaluation Scores](assets/eval_scores.png)

```
                  Relative Bias    RoPE
BLEU            :    4.7118        2.9945
chrF            :   21.2895       18.3580
```

**Relative Bias outperforms RoPE on both metrics despite similar loss.**

### Sample Predictions

```
[0] EN  : I said I wouldn't do it.
    REF : Aku bilang kalau aku tidak mau melakukannya.
    BIAS: Kurasa aku tidak bisa melakukan akukan aku.
    ROPE: Aku tidak akan membiarkannya.

[2] EN  : What a rush!
    REF : Cepatnya!
    BIAS: Apa itu!
    ROPE: Apa yang harus menemukannya!

[7] EN  : No, you go ahead.
    REF : Tidak, kamu duluan saja.
    BIAS: Tidak, jangan bukan apa.
    ROPE: Tidak, tidak.

[9] EN  : What?
    REF : Apa?
    BIAS: Apa?        ← correct ✓
    ROPE: Apa?        ← correct ✓
```

**Observations:**

- Both models handle very short, high-frequency phrases correctly ("What?" → "Apa?")
- Both struggle with longer, context-dependent sentences
- Relative Bias produces more grammatically Indonesian-sounding output even when wrong
- RoPE tends to produce more repetitive patterns ("aku akan menunjukkan aku akan menunjukkan")
- Neither model reliably handles proper nouns ("Wild Geese Gate", "Mercato") — expected with small vocab

---

## Why Relative Bias Outperforms RoPE Here

Pre-training loss was nearly identical: 4.7165 vs 4.6998. Fine-tuning loss was nearly identical: 3.4338 vs 3.4430. Yet BLEU differs by 57%.

This points to a **generation quality difference that cross-entropy loss cannot capture.**

**Hypothesis:**

Relative Bias encodes position as discrete, interpretable buckets. Nearby tokens (distance 1-4) get precise individual encodings. Distant tokens get logarithmically compressed buckets. This bucket structure creates a strong locality bias — the model is explicitly told "these two tokens are very close."

For translation, local structure matters most: subject-verb agreement, adjective-noun order, particle placement. Relative Bias's explicit local precision may provide a stronger signal for these short-range dependencies than RoPE's continuous rotation at 256 embedding dimensions.

RoPE is theoretically superior for long-context extrapolation. But for max_len=128 with a 6,000-token bilingual vocabulary and a 15M parameter model, the practical advantage may not emerge — and the inductive bias of explicit bucket distances may be more useful.

**Key takeaway:** positional encoding choice affects generation quality in ways that training loss cannot reveal. Ablation with BLEU/chrF is necessary to surface this difference.

---

## Limitations

**Vocabulary fragmentation:**
```
"terjemahan" (translation in Indonesian) may fragment into
["▁ter", "jemahan"] or ["▁ter", "je", "mahan"]
```
Each fragment must be generated in the right order. Errors at any step cascade. Larger vocab (16,000-32,000) would dramatically reduce this problem.

**Exposure bias:**
Teacher forcing during training vs autoregressive generation at inference creates a distribution mismatch. Scheduled sampling (gradually mixing ground truth and model predictions during training) would help but adds training complexity.

**Domain mismatch:**
Pre-training on Wikipedia (encyclopedic) and fine-tuning on OPUS subtitles (conversational/literary) creates a domain gap. The model's representations are calibrated for formal text but must generate casual conversational Indonesian.

**No beam search:**
AlmondT5 uses greedy decoding at inference — always picks the highest probability next token. Beam search (keep top-k hypotheses at each step) would improve BLEU by exploring more output candidates.

---

## What I Learned

**Fine-tuning a seq2seq model is not just "change the data."**

The shift from span corruption to translation requires the model to relearn what "correct output" means. In span corruption, the decoder generates sentinel tokens and original fragments. In translation, it generates fluent Indonesian. The first few fine-tuning epochs are spent unlearning the sentinel structure and learning the translation mapping — visible in the sharp loss drop from epoch 1 to epoch 3.

**BLEU and loss measure different things.**

Both models converged to similar loss values. BLEU differed by 57%. This is not a contradiction — cross-entropy loss measures how well the model assigns probability to the correct next token given the previous ground truth tokens (teacher forcing). BLEU measures how good the generated sequence looks when the model has to predict everything on its own. These are genuinely different tasks, and optimizing one does not guarantee the other.

**Translation is a harder version of span corruption.**

In span corruption, the model reconstructs specific tokens from a corrupted version of the same sentence — the answer is in the input. In translation, the model must map meaning across languages — the answer requires knowledge of a different language's grammar, vocabulary, and idioms. The pre-training objective prepares the model to use encoder representations, but the actual translation knowledge comes entirely from the 25,000 fine-tuning pairs. More data, larger vocab, and more parameters are the straightforward path to improvement.

**The pipeline is what matters.**

BLEU 4.71 is not impressive by any standard. But the pipeline — tokenizer training, bilingual pre-training, span corruption, teacher forcing fine-tuning, BLEU/chrF evaluation — works end-to-end and produces outputs that are recognizably attempting Indonesian translation. For a model built from scratch on a single T4 GPU, that is the real result.

# Tokenizer States

The place save Tokenizer states

```text
- filename
states.json
```

```python
# states
state = {
    "whitespace_marker": self.whitespace_marker,
    "num_extra_ids": self.num_extra_ids,
    "pad_token": self.pad_token,
    "unk_token": self.unk_token,
    "eos_token": self.eos_token,
    "special_tokens": self.special_tokens,
    "piece_to_id": self.piece_to_id,
    "piece_log_probs": self.piece_log_probs,
    "pad_token_id": self.pad_token_id,
    "unk_token_id": self.unk_token_id,
    "eos_token_id": self.eos_token_id,
    "decoder_start_token_id": self.decoder_start_token_id,
    "unk_log_prob": self.unk_log_prob,
}
```

> *Note: this file not tracked by Git because too large. Thank you >.<*
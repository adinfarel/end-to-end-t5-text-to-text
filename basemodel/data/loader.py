'''
basemodel/data/loader.py

Create Dataset, DataLoader and collate_fn for dataset training
'''

import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tokenizer.ulm import AlmondUnigramTokenizer

class WikipediaENIDDataset(Dataset):
    def __init__(self, data: list[str]) -> None:
        super().__init__()
        
        if not isinstance(data, list):
            raise TypeError(f"data must be list, got {type(data).__name__}")
        
        if not data:
            raise ValueError("data must be not empty.")
        
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> str:
        return self.data[index]

def load_corpus_lines(
    file_path: str | Path,
    min_chars: int = 20,
    max_chars: int | None = None,
) -> list[str]:
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {str(path)!r}")
    
    lines: list[str] = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if len(line) < min_chars:
                continue
            
            if max_chars is not None:
                line = line[:max_chars]
            
            lines.append(line)
    
    if not lines:
        raise ValueError(f"No valid lines found from corpus: {str(path)!r}")
    
    return lines

def train_val_split(
    data: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if not isinstance(train_ratio, float):
        raise TypeError(f"train_ratio must be float, got {type(train_ratio).__name__}")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    data = list(data)
     
    rng = random.Random(seed)
    rng.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    if not train_data:
        raise ValueError("train split is empty")

    if not val_data:
        raise ValueError("validation split is empty")
    
    return train_data, val_data

def pad_1d(
    ids: list[int],
    max_len: int,
    pad_id: int,
) -> tuple[list[int], list[int]]:
    ids = ids[:max_len]
    
    valid_len = len(ids)
    pad_len = max_len - valid_len
    
    padded = ids + pad_len * [pad_id]
    attn_mask = [1] * valid_len + [0] * pad_len
    
    return padded, attn_mask

def pad_labels(
    labels: list[int],
    max_len: int,
    ignore_index: int = -100,
) -> list[int]:
    labels = labels[:max_len]
    return labels + [ignore_index] * (max_len - len(labels))

def get_extra_id_token_id(
    tokenizer: AlmondUnigramTokenizer,
    extra_id: int,
) -> int:
    token = f"<extra_id_{extra_id}>"
    
    if token not in tokenizer.piece_to_id:
        raise ValueError(f"{token!r} not found in tokenizer vocab")
    
    return tokenizer.piece_to_id[token]

def random_segmentation(
    num_items: int,
    num_segments: int,
    rng: random.Random,
) -> list[int]:
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")

    if num_items < num_segments:
        raise ValueError(
            f"num_items must be >= num_segments, got {num_items=} {num_segments=}"
        )
        
    if num_segments == 1:
        return [num_items]
    
    cuts = sorted(rng.sample(range(1, num_items), num_segments - 1))
    
    segments: list[int] = []
    prev = 0
    
    for cut in cuts:
        segments.append(cut - prev)
        prev = cut
        
    segments.append(num_items - prev)
    
    return segments

def create_noise_mask(
    length: int,
    noise_density: float,
    mean_noise_span_length: float,
    rng: random.Random
) -> list[bool]:
    if length <= 0:
        return []

    if length <= 1:
        return [False] * length
    
    if not 0.0 < noise_density < 1.0:
        raise ValueError(f"noise_density must be between 0 and 1, got {noise_density}")

    if mean_noise_span_length <= 0:
        raise ValueError(
            f"mean_noise_span_length must be positive, got {mean_noise_span_length}"
        )
        
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = max(1, min(num_noise_tokens, length - 1))
    
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    num_noise_spans = max(1, num_noise_spans)
    num_noise_spans = min(num_noise_spans, num_noise_tokens)
    
    num_nonnoise_tokens = length - num_noise_tokens
    num_noise_spans = min(num_noise_spans, num_nonnoise_tokens)
    
    noise_span_lengths = random_segmentation(
        num_items=num_noise_tokens,
        num_segments=num_noise_spans,
        rng=rng
    )
    
    nonnoise_span_lengths = random_segmentation(
        num_items=num_nonnoise_tokens,
        num_segments=num_noise_spans,
        rng=rng
    )
    
    start_with_noise = rng.random() < noise_density
    
    mask: list[bool] = []
    
    for noise_len, nonnoise_len in zip(noise_span_lengths, nonnoise_span_lengths):
        if start_with_noise:
            mask.extend([True] * noise_len)
            mask.extend([False] * nonnoise_len)
        else:
            mask.extend([False] * nonnoise_len)
            mask.extend([True] * noise_len)
    
    mask = mask[:length]
    
    if len(mask) < length:
        mask.extend([False] * (length - len(mask)))
        
    return mask

def build_span_corruption_io(
    input_ids: list[int],
    tokenizer: AlmondUnigramTokenizer,
    noise_density: float,
    mean_noise_span_length: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Main span corupption function."""
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None")
    
    if not input_ids:
        return [tokenizer.eos_token_id], [tokenizer.eos_token_id]
    
    noise_mask = create_noise_mask(
        length=len(input_ids),
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        rng=rng
    )
    
    encoder_input_ids: list[int] = []
    labels: list[int] = []
    
    extra_id = 0
    i = 0
    
    while i < len(input_ids):
        is_noise = noise_mask[i]
        
        if not is_noise:
            encoder_input_ids.append(input_ids[i])
            i += 1
            continue
        
        sentinel_id = get_extra_id_token_id(tokenizer, extra_id)
        
        encoder_input_ids.append(sentinel_id)
        labels.append(sentinel_id)
        
        while i < len(input_ids) and noise_mask[i]:
            labels.append(input_ids[i])
            i += 1
        
        extra_id += 1
        
        if extra_id >= tokenizer.num_extra_ids:
            break
    
    while i < len(input_ids):
        if not noise_mask[i]:
            encoder_input_ids.append(input_ids[i])
        i += 1
    
    encoder_input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    
    return encoder_input_ids, labels

def make_t5_span_corruption_example(
    text: str,
    tokenizer: AlmondUnigramTokenizer,
    max_encoder_len: int,
    max_decoder_len: int,
    max_piece_length: int,
    noise_density: float,
    mean_noise_span_length: float,
    rng: random.Random
) -> tuple[list[int], list[int]]:
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None")
    
    input_ids = tokenizer.encode(
        text=text,
        max_piece_length=max_piece_length,
        add_eos=False,
    )
    
    # Leave space for EOS token and sentinel expansion.
    # Raw input can max_encoder_len-ish, but target may expand too.
    # For safety, keep raw ids bounded by encoder again.
    input_ids = input_ids[: max_encoder_len - 1]
    
    encoder_input_ids, labels = build_span_corruption_io(
        input_ids=input_ids,
        tokenizer=tokenizer,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        rng=rng,
    )
    
    encoder_input_ids = encoder_input_ids[:max_encoder_len] #type: ignore
    
    if encoder_input_ids[-1] != tokenizer.eos_token_id:
        encoder_input_ids[-1] = tokenizer.eos_token_id
    
    labels = labels[:max_decoder_len]
    
    if labels[-1] != tokenizer.eos_token_id:
        labels[-1] = tokenizer.eos_token_id

    return encoder_input_ids, labels

def shift_right_t5(
    labels: list[int],
    decoder_start_token_id: int,
) -> list[int]:
    """Decoder input ids
    
    Example:
    decoder_input_ids = [pad] + labels[:-1]
    """
    return [decoder_start_token_id] + labels[:-1]

class T5PretrainCollator:
    def __init__(
        self,
        tokenizer: AlmondUnigramTokenizer,
        max_encoder_len: int,
        max_decoder_len: int,
        max_piece_length: int,
        noise_density: float,
        mean_noise_span_length: float,
        label_ignore_index: int = -100,
        seed: int = 42
    ) -> None:
        self.tokenizer = tokenizer
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.max_piece_length = max_piece_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.label_ignore_index = label_ignore_index
        self.rng = random.Random(seed)

        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is None. Load/train tokenizer first.")

        if self.tokenizer.decoder_start_token_id is None:
            raise ValueError(
                "tokenizer.decoder_start_token_id is None. Load/train tokenizer first."
            )

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is None. Load/train tokenizer first.")
        
    def __call__(self, batch_texts: list[str]) -> dict[str, torch.Tensor]:
        encoder_batch: list[list[int]] = []
        encoder_mask_batch: list[list[int]] = []
        
        decoder_input_batch: list[list[int]] = []
        decoder_mask_batch: list[list[int]] = []
        
        labels_batch: list[list[int]] = []
        
        pad_id = self.tokenizer.pad_token_id
        decoder_start_id = self.tokenizer.decoder_start_token_id
        
        assert pad_id is not None
        assert decoder_start_id is not None
        
        for text in batch_texts:
            encoder_ids, labels = make_t5_span_corruption_example(
                text=text,
                tokenizer=self.tokenizer,
                max_decoder_len=self.max_decoder_len,
                max_encoder_len=self.max_encoder_len,
                max_piece_length=self.max_piece_length,
                noise_density=self.noise_density,
                mean_noise_span_length=self.mean_noise_span_length,
                rng=self.rng
            ) 
            
            decoder_input_ids = shift_right_t5(
                labels=labels,
                decoder_start_token_id=decoder_start_id
            )
            
            encoder_ids, encoder_attn_mask = pad_1d(
                ids=encoder_ids,
                max_len=self.max_encoder_len,
                pad_id=pad_id
            )
            
            decoder_input_ids, decoder_attn_mask = pad_1d(
                ids=decoder_input_ids,
                max_len=self.max_decoder_len,
                pad_id=pad_id,
            )
            
            labels = pad_labels(
                labels=labels,
                max_len=self.max_decoder_len,
                ignore_index=self.label_ignore_index
            )
            
            encoder_batch.append(encoder_ids)
            encoder_mask_batch.append(encoder_attn_mask)

            decoder_input_batch.append(decoder_input_ids)
            decoder_mask_batch.append(decoder_attn_mask)

            labels_batch.append(labels)

        return {
            "encoder_input_ids": torch.tensor(encoder_batch, dtype=torch.long),
            "encoder_attn_mask": torch.tensor(encoder_mask_batch, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_batch, dtype=torch.long),
            "decoder_attn_mask": torch.tensor(decoder_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
    
def create_pretrain_dataloaders(
    corpus_path: str | Path,
    tokenizer: AlmondUnigramTokenizer,
    batch_size: int,
    max_encoder_len: int,
    max_decoder_len: int,
    max_piece_length: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
    min_chars: int = 30,
    max_chars: int | None = 300,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
) -> tuple[DataLoader, DataLoader]:
    data = load_corpus_lines(
        file_path=corpus_path,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    
    train_data, val_data = train_val_split(
        data=data,
        train_ratio=train_ratio,
        seed=seed,
    )
    
    train_dataset = WikipediaENIDDataset(train_data)
    val_dataset = WikipediaENIDDataset(val_data)
    
    train_collator = T5PretrainCollator(
        tokenizer=tokenizer,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        max_piece_length=max_piece_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        seed=seed,
    )
    
    val_collator = T5PretrainCollator(
        tokenizer=tokenizer,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        max_piece_length=max_piece_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        seed=seed + 1,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader

def debug_one_span_corruption_example(
    text: str,
    tokenizer: AlmondUnigramTokenizer,
    max_encoder_len: int,
    max_decoder_len: int,
    max_piece_length: int,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    seed: int = 42,
) -> None:
    """
    Debug helper to inspect one text in the exact training format.
    """
    rng = random.Random(seed)

    encoder_ids, labels = make_t5_span_corruption_example(
        text=text,
        tokenizer=tokenizer,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        max_piece_length=max_piece_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        rng=rng,
    )

    assert tokenizer.decoder_start_token_id is not None

    decoder_input_ids = shift_right_t5(
        labels=labels,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
    )

    print("RAW TEXT:")
    print(text)
    print()

    print("ENCODER IDS:")
    print(encoder_ids)
    print("ENCODER TEXT:")
    print(tokenizer.decode(encoder_ids, skip_special_tokens=False))
    print()

    print("DECODER INPUT IDS:")
    print(decoder_input_ids)
    print("DECODER INPUT TEXT:")
    print(tokenizer.decode(decoder_input_ids, skip_special_tokens=False))
    print()

    print("LABEL IDS:")
    print(labels)
    print("LABEL TEXT:")
    print(tokenizer.decode(labels, skip_special_tokens=False))
    
if __name__ == "__main__":
    tokenizer = AlmondUnigramTokenizer("_", 100, -20.0)
    tokenizer.load('model/tokenizer/states.json')
    
    debug_one_span_corruption_example(
        text="Aku ingin bekerja di google tahun ini",
        tokenizer=tokenizer,
        max_decoder_len=64,
        max_encoder_len=64,
        max_piece_length=8,
        noise_density=0.8,
        mean_noise_span_length=2.0,
        seed=42
    )
'''
data/loader.py

Prepare dataset and build Dataset, DataLoader, and collate_fn
'''

import random
import jsonlines
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from functools import partial

from tokenizer.ulm import AlmondUnigramTokenizer


# HELPER
def load_jsonl(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"{str(file_path)!r} not found")
    
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected file .jsonl, got {str(file_path)!r}")
    
    with jsonlines.open(path, mode='r') as reader:
        content = list(reader)
    
    return content

def train_val_split(
    data: list[dict],
    train_ratio: float,
    seed: int
) -> tuple[list[dict], list[dict]]:
    if not isinstance(train_ratio, float):
        raise TypeError(f"train_ratio must be float, got {type(train_ratio).__name__}")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    rng = random.Random(seed)
    rng.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    if not train_data:
        raise ValueError("train split is empty")
    
    if not val_data:
        raise ValueError("val split is empty")
    
    return train_data, val_data

def truncate_with_eos(
    ids: list[int],
    max_len: int,
    eos_id: int,
) -> list[int]:
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    
    ids = ids[:max_len]
    
    if not ids:
        return [eos_id]
    
    if ids[-1] != eos_id:
        ids[-1] = eos_id
    
    return ids

# -----------------------------------------------------------------
def get_prefix(text: str | None, tokenizer: AlmondUnigramTokenizer, max_piece_length: int) -> list[int]:
    '''Translate once for prefix data'''
    if not text:
        text = "translate English to Indonesian: "
        
    prefix = tokenizer.encode(
        text=text,
        max_piece_length=max_piece_length,
        add_eos=False,
    )
    
    return prefix

class DatasetTranslate(Dataset):
    def __init__(self, data: list[dict]) -> None:
        if not isinstance(data, list):
            raise TypeError(f"data must be list of dict, got {type(data).__name__}")
        
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> dict:
        return self.data[index]

def pad_1d(
    ids: list[int],
    pad_id: int,
    max_len: int,
) -> tuple[list[int], list[int]]:
    ids = ids[:max_len]
    
    pad_len = max_len - len(ids)
    
    padded = ids + [pad_id] * pad_len
    attn_mask = [1] * len(ids) + [0] * pad_len
    
    return padded, attn_mask

def pad_labels(
    labels: list[int],
    max_len: int,
    ignore_index: int,
) -> list[int]:
    labels = labels[:max_len]
    labels = labels + [ignore_index] * (max_len - len(labels))
    return labels

def shift_labels(
    labels: list[int],
    decoder_start_id: int,
) -> list[int]:
    return [decoder_start_id] + labels[:-1]

def collate_fn(
    batch: list[dict],
    tokenizer: AlmondUnigramTokenizer,
    prefix: list[int],
    max_enc_len: int,
    max_dec_len: int,
    max_piece_len: int,
    ignore_index: int,
) -> dict:
    encoder_inputs_batch: list[list[int]] = []
    decoder_inputs_batch: list[list[int]] = []
    
    encoder_mask_batch: list[list[int]] = []
    decoder_mask_batch: list[list[int]] = []
    
    labels_batch: list[list[int]] = []
    
    decoder_start_id = tokenizer.decoder_start_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    
    assert decoder_start_id is not None
    assert pad_id is not None
    assert eos_id is not None
    
    for item in batch:
        en_text = item['en_lang']
        id_text = item['id_lang']
        
        if not isinstance(en_text, str) or not isinstance(id_text, str):
            raise TypeError("en_text or id_text must be strings.")
        
        en_text = en_text.strip()
        id_text = id_text.strip()
        
        if not en_text or not id_text:
            continue
        
        en_ids = tokenizer.encode(en_text, max_piece_length=max_piece_len)
        id_ids = tokenizer.encode(id_text, max_piece_length=max_piece_len)
                
        encoder_inputs = prefix + en_ids + [eos_id]
        labels = id_ids.copy() + [eos_id]
        
        encoder_inputs = truncate_with_eos(
            ids=encoder_inputs,
            max_len=max_enc_len,
            eos_id=eos_id
        )
        
        labels = truncate_with_eos(
            ids=labels,
            max_len=max_dec_len,
            eos_id=eos_id,
        )
        
        decoder_inputs = shift_labels(
            labels=labels,
            decoder_start_id=decoder_start_id
        )
        
        encoder_inputs, enc_attn_mask = pad_1d(
            encoder_inputs,
            pad_id=pad_id,
            max_len=max_enc_len
        )
        
        decoder_inputs, dec_attn_mask = pad_1d(
            decoder_inputs,
            pad_id=pad_id,
            max_len=max_dec_len
        )
        
        labels = pad_labels(
            labels=labels,
            max_len=max_dec_len,
            ignore_index=ignore_index,
        )

        encoder_inputs_batch.append(encoder_inputs)
        encoder_mask_batch.append(enc_attn_mask)
        
        decoder_inputs_batch.append(decoder_inputs)
        decoder_mask_batch.append(dec_attn_mask)
        
        labels_batch.append(labels)
    
    return {
        "encoder_input_ids": torch.tensor(encoder_inputs_batch, dtype=torch.long),
        "encoder_attn_mask": torch.tensor(encoder_mask_batch, dtype=torch.long),
        "decoder_input_ids": torch.tensor(decoder_inputs_batch, dtype=torch.long),
        "decoder_attn_mask": torch.tensor(decoder_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }

def create_dataloaders(
    data: list[dict],
    tokenizer: AlmondUnigramTokenizer,
    prefix_str: str,
    max_enc_len: int,
    max_dec_len: int,
    max_piece_length: int,
    ignore_index: int,
    shuffle: bool = True,
    batch_size: int = 32,
    num_workers: int =0
) -> DataLoader:
    prefix = get_prefix(text=prefix_str, tokenizer=tokenizer, max_piece_length=max_piece_length)
    
    custom_collate_fn = partial(
        collate_fn,
        tokenizer=tokenizer,
        prefix=prefix,
        max_enc_len=max_enc_len,
        max_dec_len=max_dec_len,
        max_piece_len=max_piece_length,
        ignore_index=ignore_index,
    )
    
    return DataLoader(
        dataset=DatasetTranslate(data),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        num_workers=num_workers
    )
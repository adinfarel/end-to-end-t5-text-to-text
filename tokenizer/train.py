"""
tokenizer/train.py

Training pipeline to get state tokenizer
"""

import random
from pathlib import Path
from collections.abc import Iterator

from tokenizer.ulm import AlmondUnigramTokenizer
from config.config import get_config


CONFIG = get_config()


def iter_corpus_lines(
    file_path: str | Path,
    min_chars: int = 20,
    max_chars: int | None = None,
) -> Iterator[str]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {str(path)!r}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if len(line) < min_chars:
                continue

            if max_chars is not None:
                line = line[:max_chars]

            yield line


def train_tokenizer() -> None:
    print(f"TRAINING PIPELINE {AlmondUnigramTokenizer.__name__!r}")

    print("INITIALIZE TOKENIZER")
    tokenizer = AlmondUnigramTokenizer(
        whitespace_marker=CONFIG.tokenizer.whitespace_marker,
        num_extra_ids=CONFIG.tokenizer.num_extra_ids,
    )

    print(f"LOAD DATASET {CONFIG.datasets.name}")
    texts = list(
        iter_corpus_lines(
            CONFIG.datasets.output_dataset_raw,
            min_chars=20,
            max_chars=300,
        )
    )

    random.seed(42)
    random.shuffle(texts)

    print(f"DATASET TOTAL: {len(texts)}")

    if not texts:
        raise ValueError("No valid corpus lines found for tokenizer training.")

    print("TRAINING TOKENIZER STARTING")
    tokenizer.train(
        texts=texts,
        vocab_size=CONFIG.tokenizer.vocab_size,
        seed_vocab_size=CONFIG.tokenizer.seed_vocab_size,
        max_piece_length=CONFIG.tokenizer.max_piece_length,
        num_em_steps=CONFIG.tokenizer.num_em_steps,
        shrinking_factor=CONFIG.tokenizer.shrinking_factor,
        smoothing=float(CONFIG.tokenizer.smoothing),
    )

    print("TRAINING FINISH")

    tokenizer.save(file_path=CONFIG.tokenizer.output_tokenizer_path)
    print(f"TOKENIZER STATE SAVE TO {CONFIG.tokenizer.output_tokenizer_path}")


if __name__ == "__main__":
    train_tokenizer()
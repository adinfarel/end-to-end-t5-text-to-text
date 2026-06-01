'''
tokenizer/ulm.py

SentencePiece-Unigram Language Model this tokenizer fit with T5 model instead of BPE so i prefer choose this
to prevent fragmentation token and corresponding paper t5
'''

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import islice
from collections.abc import Iterator, Iterable

class AlmondUnigramTokenizer:
    """SentencePiece-style Unigram tokenizer."""
    def __init__(
        self, 
        whitespace_marker: str = "_",
        num_extra_ids: int = 100
    ) -> None:
        if not isinstance(whitespace_marker, str):
            raise TypeError(
                f"whitespace_marker must be str, got {type(whitespace_marker).__name__}"
            )
        
        if len(whitespace_marker) != 1:
            raise ValueError(
                f"whitespace_marker must be exactly 1 character, got {whitespace_marker!r}"
            )
        
        if not isinstance(num_extra_ids, int):
            raise TypeError(
                f"num_extra_ids must be int, got {type(num_extra_ids).__name__}"
            )
        
        if num_extra_ids < 0:
            raise ValueError(
                f"num_extra_ids must be non-negative, got {num_extra_ids}"
            )
        
        self.whitespace_marker = whitespace_marker
        self.num_extra_ids = num_extra_ids
        
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "</s>"
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.eos_token,
            *[f"<extra_id_{i}>" for i in range(num_extra_ids)],
        ]
        
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}
        self.piece_log_probs: dict[str, float] = {}
        
        self.pad_token_id: int | None = None
        self.unk_token_id: int | None = None
        self.eos_token_id: int | None = None
        self.decoder_start_token_id: int | None = None
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize raw text into SentencePiece-style text.
        
        Example:
            "I love you" --> "_I_love_you"
        """
        
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
    
        text = text.strip()
        
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        
        normalized = self.whitespace_marker + text.replace(" ", self.whitespace_marker)
        
        return normalized
    
    def _denormalize_text(self, normalized_text: str) -> str:
        """
        Denormalize text into normal text.
        
        Example:
            "_I_love_you" --> "I love you"
        """
        
        if not isinstance(normalized_text, str):
            raise TypeError(
                f"normalized_text must be str, got {type(str).__name__}"
            )
        
        normalized_text = normalized_text.strip()
        
        if not normalized_text:
            return ""

        return normalized_text.replace(self.whitespace_marker, " ").strip()

    def _extract_candidate_pieces_from_text(
        self,
        text: str,
        max_piece_length: int,
    ) -> Counter[str]:
        """Extract substring candidate pieces from one raw text sample."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        if not isinstance(max_piece_length, int):
            raise TypeError(
                f"max_piece_length must be int, got {type(max_piece_length).__name__}"
            )
        
        if max_piece_length <= 0:
            raise ValueError(
                f"max_piece_length must be positive, got {max_piece_length}"
            )
        
        normalized = self._normalize_text(text)
        
        pieces: Counter[str] = Counter()
        
        if not normalized:
            return pieces
        
        for piece in self._iter_candidate_pieces(
            normalized_text=normalized,
            max_piece_length=max_piece_length
        ):
            pieces[piece] += 1
        
        return pieces
    
    def _iter_candidate_pieces(
        self,
        normalized_text: str,
        max_piece_length: int,
    ) -> Iterator[str]:
        """Yield candidate pieces from one normalized text."""
        if not isinstance(normalized_text, str):
            raise TypeError(
                f"normalized_text must be str, got {type(normalized_text).__name__}"
            )

        if not isinstance(max_piece_length, int):
            raise TypeError(
                f"max_piece_length must be int, got {type(max_piece_length).__name__}"
            )

        if max_piece_length <= 0:
            raise ValueError(
                f"max_piece_length must be positive, got {max_piece_length}"
            )
        
        n = len(normalized_text)
        
        for start in range(n):
            end_limit = min(n, start + max_piece_length)
            
            for end in range(start + 1, end_limit + 1):
                yield normalized_text[start:end]
    
    def _build_seed_vocab_from_texts(
        self,
        texts: Iterable[str],
        max_piece_length: int,
        seed_vocab_size: int,
    ):
        """Build initial seed vocabulary from raw text samples."""
        if not isinstance(max_piece_length, int):
            raise TypeError(
                f"max_piece_length must be int, got {type(max_piece_length).__name__}"
            )

        if max_piece_length <= 0:
            raise ValueError(
                f"max_piece_length must be positive, got {max_piece_length}"
            )

        if not isinstance(seed_vocab_size, int):
            raise TypeError(
                f"seed_vocab_size must be int, got {type(seed_vocab_size).__name__}"
            )

        if seed_vocab_size <= 0:
            raise ValueError(
                f"seed_vocab_size must be positive, got {seed_vocab_size}"
            )
        
        piece_counts: Counter[str] = Counter()
        required_chars: set[str] = set()
        
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    f"all texts must be str, got {type(text).__name__}"
                )

            normalized = self._normalize_text(text)

            if not normalized:
                continue
            
            required_chars.update(normalized)
            
            for piece in self._iter_candidate_pieces(
                normalized_text=normalized,
                max_piece_length=max_piece_length
            ):
                piece_counts[piece] += 1

        if not piece_counts:
            raise ValueError(
                f"cannot build seed vocab from empty texts; no candidate pieces found."
            )
        
        seed_vocab: Counter[str] = Counter()
        
        for char in sorted(required_chars):
            seed_vocab[char] = piece_counts.get(char, 1)
        
        remaining_budget = seed_vocab_size - len(seed_vocab)
        
        if remaining_budget <= 0:
            return seed_vocab
        
        for piece, count in piece_counts.most_common():
            if piece in seed_vocab:
                continue
            
            seed_vocab[piece] = count

            if len(seed_vocab) >= seed_vocab_size:
                break
        
        return seed_vocab

    def _initiliaze_vocabulary(
        self,
        seed_vocab: Counter[str],
    ) -> None:
        """Initialize vocabulary from seed vocab."""
        if not isinstance(seed_vocab, Counter):
            raise TypeError(
                f"seed_vocab must be collections.Counter, got {type(seed_vocab).__name__}"
            )
        
        if not seed_vocab:
            raise ValueError("seed_vocab cannot be empty")
        
        # Start with special tokens
        self.piece_to_id.clear()
        self.id_to_piece.clear()
        self.piece_log_probs.clear()
        
        next_id = 0
        
        for token in self.special_tokens:
            if token in self.piece_to_id:
                raise RuntimeError(f"duplicate special token detected: {token!r}")
            
            self.piece_to_id[token] = next_id
            self.id_to_piece[next_id] = token
            next_id += 1
        
        normal_pieces = [
            piece
            for piece, _ in seed_vocab.most_common()
            if piece not in self.piece_to_id
        ]
        
        if not normal_pieces:
            raise ValueError("seed_vocab contains no normal pieces after special tokens")
        
        total_count = sum(seed_vocab[piece] for piece in normal_pieces)
        
        if total_count <= 0:
            raise ValueError(f"total normal piece count must be positive, got {total_count}")
        
        for piece in normal_pieces:
            count = seed_vocab[piece]
            
            if count <= 0:
                continue
            
            self.piece_to_id[piece] = next_id
            self.id_to_piece[next_id] = piece
            
            probability = count / total_count
            self.piece_log_probs[piece] = math.log(probability)
            
            next_id += 1
        
        self.pad_token_id = self.piece_to_id.get(self.pad_token, None)
        self.unk_token_id = self.piece_to_id.get(self.unk_token, None)
        self.eos_token_id = self.piece_to_id.get(self.eos_token, None)
        self.decoder_start_token_id = self.eos_token_id
    
    def _is_initialized(self) -> bool:
        """Check if the vocabulary has been initialized."""
        return (
            bool(self.piece_to_id)
            and bool(self.id_to_piece)
            and bool(self.piece_log_probs)
            and self.pad_token_id is not None
            and self.unk_token_id is not None
            and self.eos_token_id is not None
            and self.decoder_start_token_id is not None
        )
    
    def _require_initialized(self) -> None:
        """Raise a clear error if tokenizer vocabulary is not initialized."""
        if not self._is_initialized():
            raise RuntimeError(
                "Tokenizer is not initialized. "
                "Call train(), load(), or _initialize_vocabulary() before using this method."
            )
    
    def _get_piece_log_prob(self, piece: str) -> float | None:
        """Get log probability of a piece or None if piece is not in vocab."""
        self._require_initialized()
        
        if not isinstance(piece, str):
            raise TypeError(f"piece must be str, got {type(piece).__name__}")
        
        if not piece:
            return None
        
        if piece in self.special_tokens:
            return None
        
        return self.piece_log_probs.get(piece, None)
    
    # MAIN IDEA
    def _viterbi_segment(self):
        pass
        
if __name__ == "__main__":
    tokenizer = AlmondUnigramTokenizer("_", num_extra_ids=10)
    # print(tokenizer._require_initialized()) 
    
    text = [
        "Aku bisa AI",
        "     Aku lolos \n\n big tech               "
    ]
    seed_vocab = tokenizer._build_seed_vocab_from_texts(
        text,
        max_piece_length=4,
        seed_vocab_size=30
    )
    
    print("SEED VOCAB SIZE:", len(seed_vocab))
    print("TOP 30 PIECES:")
    for piece, count in seed_vocab.most_common(30):
        print(f"{piece!r:<7} -> {count}")
    
    print("-"*40)

    for tx in text:
        NORM = tokenizer._normalize_text(tx)
        candidates = tokenizer._extract_candidate_pieces_from_text(
            tx,
            4
        )

        print(f"RAW     : {repr(tx)}")
        print(f"NORM    : {repr(NORM)}")
        print(f"DENORM  : {repr(tokenizer._denormalize_text(NORM))}")
        print(f"TOTAL CANDIDATES    : {len(candidates)}")
        print(f"\nTop 20:")
        for piece, count in candidates.most_common(20):
            print(f"{piece!r:<7} -> {count}")
        print("-"*40)
    
    print("\n\nINITIALIZING VOCABULARY...")
    tokenizer._initiliaze_vocabulary(seed_vocab)
    print(tokenizer._is_initialized())
    print(f"VOCAB SIZE: {len(tokenizer.piece_to_id)}")
    print("TOP 30 PIECES IN VOCAB:")
    for token in islice(tokenizer.piece_to_id, 30):
        print(f"{token!r:<7} -> {tokenizer.piece_to_id[token]} (log_prob={tokenizer.piece_log_probs.get(token, 'N/A')})")
        
    print(f"PAD TOKEN ID: {tokenizer.pad_token_id}")
    print(f"UNK TOKEN ID: {tokenizer.unk_token_id}")
    print(f"EOS TOKEN ID: {tokenizer.eos_token_id}")
    print(f"DECODER START TOKEN ID: {tokenizer.decoder_start_token_id}")
    
    test_pieces = [
        "_Aku",
        "_big",
        "_Google",
        "</s>",
        ""
    ]
    
    for piece in test_pieces:
        log_prob = tokenizer._get_piece_log_prob(piece)
        print(f"Piece: {piece!r:12s}, Log Prob: {log_prob}")
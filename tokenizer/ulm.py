'''
tokenizer/ulm.py

SentencePiece-Unigram Language Model this tokenizer fit with T5 model instead of BPE so i prefer choose this
to prevent fragmentation token and corresponding paper t5
'''

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from collections.abc import Iterator, Iterable

class AlmondUnigramTokenizer:
    """SentencePiece-style Unigram tokenizer."""
    def __init__(
        self, 
        whitespace_marker: str = "_",
        num_extra_ids: int = 100,
        unk_log_prob: float = -20.00,
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
        
        self.special_tokens: list[str] = [
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
        
        self.unk_log_prob = unk_log_prob
    
    def encode(
        self,
        text: str,
        max_piece_length: int,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode raw text into list of token IDs."""
        self._require_initialized()
        
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
        
        pieces = self._encode_pieces(
            text=text,
            max_piece_length=max_piece_length,
        )
        
        token_ids: list[int] = []; append_token_ids = token_ids.append
        
        assert self.unk_token_id is not None, "unk_token_id should not be None after initialization"
        assert self.eos_token_id is not None, "eos_token_id should not be None after initialization"
        
        for piece in pieces:
            token_id = self.piece_to_id.get(piece, self.unk_token_id)
            append_token_ids(token_id)
        
        if add_eos:
            append_token_ids(self.eos_token_id)
        
        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode list of token IDs back into raw text."""
        self._require_initialized()
        
        if not isinstance(token_ids, list):
            raise TypeError(
                f"token_ids must be list of int, got {type(token_ids).__name__}"
            )
        
        if not isinstance(skip_special_tokens, bool):
            raise TypeError(
                f"skip_special_tokens must be bool, got {type(skip_special_tokens).__name__}"
            )
        
        pieces: list[str] = []
        
        special_token_set = set(self.special_tokens)
        
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    f"all token_ids must be int, got {type(token_id).__name__}"
                )
            
            piece = self.id_to_piece.get(token_id, None)
            
            if piece is None:
                raise ValueError(f"token_id {token_id} is not in vocabulary")
            
            if skip_special_tokens and piece in special_token_set:
                continue
            
            pieces.append(piece)
        
        normalized_text = ''.join(pieces)
        
        return self._denormalize_text(normalized_text)
    
    def save(
        self,
        file_path: str | Path,
    ) -> None:
        """Save tokenizer state to JSON."""
        self._require_initialized()
        
        if not isinstance(file_path, (str, Path)):
            raise TypeError(
                f"file_path must be str or pathlib.Path, got {type(file_path).__name__}"
            )
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
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
        }
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer state saved to {path}")
    
    def load(
        self,
        file_path: str | Path,
    ) -> None:
        if not isinstance(file_path, (str, Path)):
            raise TypeError(
                f"file_path must be str or pathlib.Path, got {type(file_path).__name__}"
            )
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"tokenizer state file not found: {path}")
        
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        
        required_keys = {
            "whitespace_marker",
            "num_extra_ids",
            "pad_token",
            "unk_token",
            "eos_token",
            "special_tokens",
            "piece_to_id",
            "piece_log_probs",
            "pad_token_id",
            "unk_token_id",
            "eos_token_id",
            "decoder_start_token_id",
        }
        
        missing_keys = required_keys - set(state.keys())
        
        if missing_keys:
            raise ValueError(
                f"tokenizer state file is invalid; missing required keys: {missing_keys}"
            )
        
        self.whitespace_marker = state["whitespace_marker"]
        self.num_extra_ids = state["num_extra_ids"]
        
        self.pad_token = state["pad_token"]
        self.unk_token = state["unk_token"]
        self.eos_token = state["eos_token"]
        self.special_tokens = state["special_tokens"]
        
        self.piece_to_id = {
            str(piece): int(token_id)
            for piece, token_id in state["piece_to_id"].items()
        }
        self.id_to_piece = {
            int(token_id): str(piece)
            for piece, token_id in self.piece_to_id.items()
        }
        self.piece_log_probs = {
            str(piece): float(log_prob)
            for piece, log_prob in state["piece_log_probs"].items()
        }
        
        self.pad_token_id = int(state["pad_token_id"])
        self.unk_token_id = int(state["unk_token_id"])
        self.eos_token_id = int(state["eos_token_id"])
        self.decoder_start_token_id = int(state["decoder_start_token_id"])
        
        self._validate_loaded_state()
        
        print(f"Tokenizer state loaded from {path}")
    
    def _validate_loaded_state(self) -> None:
        """Validate tokenizer state after loading from file."""
        if not self.piece_to_id:
            raise ValueError("loaded tokenizer state has empty piece_to_id")
        
        if not self.id_to_piece:
            raise ValueError("loaded tokenizer state has empty id_to_piece")
        
        if not self.piece_log_probs:
            raise ValueError("loaded tokenizer state has empty piece_log_probs")
        
        if len(self.piece_to_id) != len(self.id_to_piece):
            raise ValueError(
                f"loaded tokenizer state has inconsistent piece_to_id and id_to_piece lengths: "
                f"{len(self.piece_to_id)} vs {len(self.id_to_piece)}"
            )
        
        for token in self.special_tokens:
            if token not in self.piece_to_id:
                raise ValueError(f"special token {token!r} is missing from loaded piece_to_id")
        
        expected_ids = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.eos_token: self.eos_token_id,
        }
        
        for token, expected_id in expected_ids.items():
            actual_id = self.piece_to_id.get(token, None)
            
            if actual_id != expected_id:
                raise ValueError(
                    f"special token {token!r} has inconsistent ID in loaded state: "
                    f"expected {expected_id}, got {actual_id}"
                )
        
        if self.decoder_start_token_id != self.eos_token_id:
            raise ValueError(
                f"decoder_start_token_id should be the same as eos_token_id in loaded state: "
                f"expected {self.eos_token_id}, got {self.decoder_start_token_id}"
            )
        
        if not isinstance(self.unk_log_prob, float):
            raise TypeError(
                f"unk_log_prob must be float, got {type(self.unk_log_prob).__name__}"
            )

        if self.unk_log_prob >= 0:
            raise ValueError(
                f"unk_log_prob should be negative log probability, got {self.unk_log_prob}"
            )
    
    @staticmethod
    def _logaddexp(a: float, b: float) -> float:
        """Stable log(exp(a) + exp(b)) computation."""
        
        if a == float("-inf"):
            return b
        
        if b == float("-inf"):
            return a
        
        maximum = max(a, b)
        
        return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))
    
    @staticmethod
    def _logsumexp(values: list[float]) -> float:
        """Stable log(sum(exp(values))) computation."""
        
        if not values:
            return float("-inf")
        
        maximum = max(values)
        
        if maximum == float("-inf"):
            return float("-inf")
        
        total = sum(math.exp(v - maximum) for v in values)
        
        return maximum + math.log(total)
            
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

    def _initialize_vocabulary(
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
    
    def _iter_valid_pieces_from_position(
        self,
        normalized_text: str,
        start: int,
        max_piece_length: int,
        allow_unk: bool = True,
    ) -> Iterator[tuple[int, str, float]]:
        """Yield valid segmentation edges from a start position."""
        self._require_initialized()

        if not isinstance(normalized_text, str):
            raise TypeError(
                f"normalized_text must be str, got {type(normalized_text).__name__}"
            )

        if not isinstance(start, int):
            raise TypeError(f"start must be int, got {type(start).__name__}")

        if not isinstance(max_piece_length, int):
            raise TypeError(
                f"max_piece_length must be int, got {type(max_piece_length).__name__}"
            )

        if not isinstance(allow_unk, bool):
            raise TypeError(f"allow_unk must be bool, got {type(allow_unk).__name__}")

        if start < 0 or start >= len(normalized_text):
            raise ValueError(
                f"start must be in range [0, {len(normalized_text) - 1}], got {start}"
            )
        
        n = len(normalized_text)
        end_limit = min(n, start + max_piece_length)
        found_valid_piece = False
        
        for end in range(start + 1, end_limit + 1):
            piece = normalized_text[start:end]
            log_prob = self._get_piece_log_prob(piece)
            
            if log_prob is None:
                continue
            
            found_valid_piece = True
            yield (end, piece, log_prob)
        
        if allow_unk and not found_valid_piece:
            yield (start + 1, self.unk_token, self.unk_log_prob)
            
    # MAIN IDEA
    def _viterbi_segment(
        self,
        normalized_text: str,
        max_piece_length: int,
    ) -> list[str]:
        """Find the best segmentation for normalized_text using Viberti DP."""
        self._require_initialized()
        
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
        
        if not normalized_text:
            return []
        
        n = len(normalized_text)
        # dp[i] = best log score for normalized_text[:i]
        dp: list[float] = [float("-inf")] * (n + 1)
        
        # bacpointer[i] = (previous_pos, chosen_piece)
        backpointer: list[tuple[int, str] | None] = [None] * (n + 1)
        
        dp[0] = 0.0
        
        for start in range(n):
            if dp[start] == float("-inf"):
                continue
            
            for end, piece, log_prob in self._iter_valid_pieces_from_position(
                normalized_text=normalized_text,
                start=start,
                max_piece_length=max_piece_length,
                allow_unk=True
            ):
                candidate_score = dp[start] + log_prob
                
                if candidate_score > dp[end]:
                    dp[end] = candidate_score
                    backpointer[end] = (start, piece)
        
        if dp[n] == float("-inf"):
            raise RuntimeError(
                f"failed to segment text with current vocabulary. "
                f"normalized_text={normalized_text!r}"
            )
        
        pieces: list[str] = []
        position: int = n
        
        while position > 0:
            prev = backpointer[position]
            
            if prev is None:
                raise RuntimeError(
                    f"broken Viterbi backpointer state. "
                    f"position={position}, normalized_text={normalized_text!r}"
                )
            
            prev_pos, piece = prev
            pieces.append(piece)
            position = prev_pos
        
        pieces.reverse()
        
        return pieces

    def _encode_pieces(
        self,
        text: str,
        max_piece_length: int,
    ) -> list[str]:
        """Encode raw text into list of pieces."""
        self._require_initialized()
        
        if not isinstance(text, str):
            raise TypeError(
                f"text must be str, got {type(text).__name__}"
            )
        
        if not isinstance(max_piece_length, int):
            raise TypeError(
                f"max_piece_length must be int, got {type(max_piece_length).__name__}"
            )
        
        if max_piece_length <= 0:
            raise ValueError(
                f"max_piece_length must be positive, got {max_piece_length}"
            )
            
        normalized = self._normalize_text(text)
        
        if not normalized:
            return []

        return self._viterbi_segment(
            normalized_text=normalized,
            max_piece_length=max_piece_length,
        )
        
    def _forward_backward_expected_counts(
        self,
        normalized_text: str,
        max_piece_length: int,
    ) -> tuple[dict[str, float], float]:
        """Run forward-backward over one normalized text."""
        self._require_initialized()
        
        if not isinstance(normalized_text, str):
            raise TypeError(f"normalized_text must be str, got {type(normalized_text).__name__}")

        if not isinstance(max_piece_length, int):
            raise TypeError(f"max_piece_length must be int, got {type(max_piece_length).__name__}")

        if max_piece_length <= 0:
            raise ValueError(f"max_piece_length must be positive, got {max_piece_length}")
        
        if not normalized_text:
            return {}, 0.0
        
        n = len(normalized_text)
        
        edges_by_start: list[list[tuple[int, str, float]]] = [[] for _ in range(n + 1)]
        
        for start in range(n):
            edges_by_start[start] = list(
                self._iter_valid_pieces_from_position(
                    normalized_text=normalized_text,
                    start=start,
                    max_piece_length=max_piece_length,
                    allow_unk=True
                )
            )
        
        alpha = [float("-inf")] * (n + 1)
        alpha[0] = 0.0
        
        for start in range(n):
            if alpha[start] == float("-inf"):
                continue
            
            for end, _piece, log_prob in edges_by_start[start]:
                alpha[end] = self._logaddexp(alpha[end], alpha[start] + log_prob)
        
        log_likelihood = alpha[n]
        
        if log_likelihood == float("-inf"):
            raise RuntimeError(
                f"failed to compute forward probabilities; no valid segmentations. "
                f"normalized_text={normalized_text!r}"
            )
        
        beta = [float("-inf")] * (n + 1)
        beta[n] = 0.0
        
        for start in range(n - 1, -1, -1):
            scores: list[float] = []
            
            for end, _piece, log_prob in edges_by_start[start]:
                if beta[end] == float("-inf"):
                    continue
                
                scores.append(log_prob + beta[end])
            
            beta[start] = self._logsumexp(scores)
        
        expected_counts: dict[str, float] = {}
        
        for start in range(n):
            if alpha[start] == float("-inf"):
                continue
            
            for end, piece, log_prob in edges_by_start[start]:
                if beta[end] == float("-inf"):
                    continue
                
                posterior_log_prob = alpha[start] + log_prob + beta[end] - log_likelihood
                posterior = math.exp(posterior_log_prob)
                
                if piece == self.unk_token:
                    continue
                
                expected_counts[piece] = expected_counts.get(piece, 0.0) + posterior
        
        return expected_counts, log_likelihood
    
    def _em_step(
        self,
        texts: list[str],
        max_piece_length: int,
        smoothing: float = 1e-8,
    ) -> float:
        """Run one EM update over raw text samples."""
        self._require_initialized()
        
        if not isinstance(texts, list):
            raise TypeError(
                f"texts must be list of str, got {type(texts).__name__}"
            )
        
        if not isinstance(max_piece_length, int):
            raise TypeError(
                f"max_piece_length must be int, got {type(max_piece_length).__name__}"
            )
        
        if max_piece_length <= 0:
            raise ValueError(
                f"max_piece_length must be positive, got {max_piece_length}"
            )
        
        if not isinstance(smoothing, float):
            raise TypeError(f"smoothing must be float, got {type(smoothing).__name__}")

        if smoothing < 0:
            raise ValueError(f"smoothing must be non-negative, got {smoothing}")
        
        total_expected_counts: dict[str, float] = {}
        total_log_likelihood = 0.0
        num_texts = 0
        
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"all texts must be str, got {type(text).__name__}")
            
            normalized = self._normalize_text(text)
            
            if not normalized:
                continue
            
            expected_counts, log_likelihood = self._forward_backward_expected_counts(
                normalized_text=normalized,
                max_piece_length=max_piece_length,
            )
            
            for piece, count in expected_counts.items():
                total_expected_counts[piece] = total_expected_counts.get(piece, 0.0) + count
                
            total_log_likelihood += log_likelihood
            num_texts += 1
        
        if num_texts == 0:
            raise ValueError("cannot run EM step on empty texts")
        
        if not total_expected_counts:
            raise RuntimeError("EM step produced empty expected counts")
        
        updated_counts: dict[str, float] = {}
        
        for piece in self.piece_log_probs:
            updated_counts[piece] = total_expected_counts.get(piece, 0.0) + smoothing
        
        normalizer = sum(expected_counts.values())
        
        if normalizer <= 0:
            raise RuntimeError(f"invalid EM normalizer: {normalizer}")
        
        for piece, count in updated_counts.items():
            probability = count / normalizer
            self.piece_log_probs[piece] = math.log(probability)
        
        average_negative_log_likelihood = -total_log_likelihood / num_texts
        
        return average_negative_log_likelihood

    def _prune_vocabulary(
        self,
        target_vocab_size: int,
    ) -> None:
        """Prune normal pieces to target vocab size while keeping:
        - all special tokens
        - all single-character pieces
        """
        self._require_initialized()
        
        if not isinstance(target_vocab_size, int):
            raise TypeError(
                f"target_vocab_size must be int, got {type(target_vocab_size).__name__}"
            )

        if target_vocab_size <= len(self.special_tokens):
            raise ValueError(
                "target_vocab_size must be larger than number of special tokens; "
                f"got target_vocab_size={target_vocab_size}, "
                f"num_special_tokens={len(self.special_tokens)}"
            )
    
        required_pieces = {
            piece
            for piece in self.piece_log_probs
            if len(piece) == 1
        }
        
        normal_budget = target_vocab_size - len(required_pieces) 
        
        if len(required_pieces) > normal_budget:
            normal_budget = len(required_pieces)
        
        sorted_pieces = sorted(
            self.piece_log_probs.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        
        kept_pieces: set[str] = set(required_pieces)
        
        for piece, log_prob in sorted_pieces:
            if len(kept_pieces) >= normal_budget:
                break
        
            kept_pieces.add(piece)
        
        old_log_probs = self.piece_log_probs
        
        self.piece_to_id.clear()
        self.id_to_piece.clear()
        self.piece_log_probs = {
            piece: old_log_probs[piece]
            for piece in kept_pieces
        }
        
        next_id = 0
        
        for token in self.special_tokens:
            self.piece_to_id[token] = next_id
            self.id_to_piece[next_id] = token
            next_id += 1
        
        for piece, _log_prob in sorted(
            self.piece_log_probs.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            self.piece_to_id[piece] = next_id
            self.id_to_piece[next_id] = piece
            next_id += 1
        
        self.pad_token_id = self.piece_to_id[self.pad_token]
        self.eos_token_id = self.piece_to_id[self.eos_token]
        self.unk_token_id = self.piece_to_id[self.unk_token]
        self.decoder_start_token_id = self.pad_token_id
    
    def train(
        self,
        texts: Iterable[str],
        vocab_size: int,
        seed_vocab_size: int,
        max_piece_length: int,
        num_em_steps: int = 5,
        shrinking_factor: float = 0.75,
        smoothing: float = 1e-8
    ) -> None:
        """Train simplified Unigram LM tokenizer."""
        if not isinstance(vocab_size, int):
            raise TypeError(f"vocab_size must be int, got {type(vocab_size).__name__}")

        if not isinstance(seed_vocab_size, int):
            raise TypeError(f"seed_vocab_size must be int, got {type(seed_vocab_size).__name__}")
        
        if not isinstance(num_em_steps, int):
            raise TypeError(f"num_em_steps must be int, got {type(num_em_steps).__name__}")
        
        if not isinstance(shrinking_factor, float):
            raise TypeError(f"shrinking_factor must be float, got {type(shrinking_factor).__name__}")
        
        if vocab_size <= len(self.special_tokens):
            raise ValueError(
                f"vocab_size must be greater than special token count "
                f"({len(self.special_tokens)}), got {vocab_size}"
            )
        
        if seed_vocab_size < vocab_size:
            raise ValueError(
                f"seed_vocab_size should be >= vocab_size, got "
                f"seed_vocab_size={seed_vocab_size}, vocab_size={vocab_size}"
            )
        
        if num_em_steps <= 0:
            raise ValueError(
                f"num_em_steps must be positive, got {num_em_steps}"
            )
        
        if not (0.0 < shrinking_factor < 1.0):
            raise ValueError(
                f"shrinking_factor must be between 0 and 1, got {shrinking_factor}"
            )
        
        texts = list(texts)
        
        if not texts:
            raise ValueError("texts must not be empty")
        
        seed_vocab = self._build_seed_vocab_from_texts(
            texts=texts,
            max_piece_length=max_piece_length,
            seed_vocab_size=seed_vocab_size,
        )
        
        self._initialize_vocabulary(seed_vocab=seed_vocab)
        
        print(f"Initial vocab size: {len(self.piece_to_id)}")
        
        for step in range(1, num_em_steps + 1):
            avg_null = self._em_step(
                texts=texts,
                max_piece_length=max_piece_length,
                smoothing=smoothing,
            )
        
            current_vocab_size = len(self.piece_to_id)
            
            if current_vocab_size > vocab_size:
                next_vocab_size = max(
                    vocab_size,
                    int(current_vocab_size * shrinking_factor)
                )

                self._prune_vocabulary(next_vocab_size)
            
            print(
                f"EM step {step}/{num_em_steps} | "
                f"avg_null={avg_null:.4f} | "
                f"vocab_size={len(self.piece_to_id)}"
            )
        
        final_nll = self._em_step(
            texts=texts,
            max_piece_length=max_piece_length,
            smoothing=smoothing,
        )
        
        if len(self.piece_to_id) > vocab_size:
            self._prune_vocabulary(vocab_size)
        
        print(
            f"Training complete | final_avg_nll={final_nll:.4f} | "
            f"final_vocab_size={len(self.piece_to_id)}"
        )
                
if __name__ == "__main__":
    texts = [
        "Aku lolos big tech",
        "Aku belajar tokenizer",
        "big tech membuka peluang",
        "Saya ingin bekerja di Google",
        "I want to work at Google",
        "Google is a big tech company",
    ]
    
    tokenizer = AlmondUnigramTokenizer(
        whitespace_marker="_",
        num_extra_ids=20,
        unk_log_prob=-20.00,
    )
    
    tokenizer.train(
        texts=texts,
        vocab_size=80,
        seed_vocab_size=200,
        max_piece_length=12,
        num_em_steps=5,
        shrinking_factor=0.75,
    )
    
    text = "Aku ingin bekerja di Google"
    
    ids = tokenizer.encode(
        text=text,
        max_piece_length=12,
        add_eos=True,
    )
    
    pieces = tokenizer._encode_pieces(
        text=text,
        max_piece_length=12
    )
    
    decoded = tokenizer.decode(ids)
    
    print(f"TEXT    : {text}")
    print(f"PIECES  : {pieces}")
    print(f"IDS     : {ids}")
    print(f"DECODED : {decoded}")
    print(f"VOCAB   :\n{tokenizer.piece_to_id}")
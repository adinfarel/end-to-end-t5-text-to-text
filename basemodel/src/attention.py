'''
basemodel/src/attention.py

Get score affinity each token, token query ask, token key answer, token value given
Query -> what information that i can looking for
Key -> what informatin that i can give
Value -> what real value token that u get
'''

from __future__ import annotations

from re import L
from turtle import forward
from typing import Literal

from sympy import per
import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel.src.pos_enc import RelativePositionBias, RotaryPositionalEncoding

PositionalEncoding = Literal["none", "relative_bias", "rope"]

def _build_key_padding_mask(
    attn_mask: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Convert attention mask from:
            attn_mask: (B, T), 1 valid, 0 pad
        
        into additive attention mask:
            (B, 1, 1, T), 0 valid, -inf pad
    """
    if attn_mask is None:
        return None
    
    if attn_mask.dim() != 2:
        raise ValueError(f"attn_mask must have shape (B, T), got {attn_mask.shape}")
    
    return torch.zeros(
        (*attn_mask.shape, ),
        device=attn_mask.device,
        dtype=dtype,
    )[:, None, None, :].masked_fill(
        attn_mask[:, None, None, :] == 0,
        float("-inf")
    )

def _build_causal_mask(
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
    query_position_offset: int = 0,
) -> torch.Tensor:
    """Build additive causal mask."""
    query_positions = (
        torch.arange(query_len, device=device) + query_position_offset
    )[:, None]
    
    key_positions = torch.arange(key_len, device=device)[None, :]
    
    disallow = key_positions > query_positions
    
    mask = torch.zeros(
        (1, 1, query_len, key_len),
        device=device,
        dtype=dtype,
    )
    
    return mask.masked_fill(disallow, float("-inf"))

def _build_sliding_window_mask(
    query_len: int,
    key_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
    query_position_offset: int = 0,
    causal: bool = False,
):
    """Build additive sliding-window mask.
    
    if causal=True
        query can only see previous/current keys inside window.
        
    if causal=False
        query can see left/right keys inside window.
    """
    if window_size <= 0:
        raise TypeError(f"window_size must be int, got {window_size}")
    
    query_positions = (
        torch.arange(query_len, device=device) + query_position_offset
    )[: None]
    
    key_positions = torch.arange(key_len, device=device)[None, :]
    
    distance = torch.abs(key_positions - query_positions)
    disallow = distance > window_size
    
    if causal:
        disallow = disallow | (key_positions > query_positions)
    
    mask = torch.zeros(
        (1, 1, query_len, key_len),
        device=device,
        dtype=dtype,
    )
    
    return mask.masked_fill(disallow, float("-inf"))

class EncoderSelfAttention(nn.Module):
    """Encoder self-attention."""
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        layer_idx: int,
        dropout: float = 0.1,
        position_encoding: PositionalEncoding = "relative_bias",
        use_unpadding: bool = False,
        use_alternating_attention: bool = False,
        window_size: int = 3,
        relative_num_buckets: int = 32,
        relative_max_distance: int = 128,
    ) -> None:
        super().__init__()
        
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by n_heads, got {embed_dim=} {n_heads=}"
            )

        if position_encoding not in {"none", "rope", "relative_bias"}:
            raise ValueError(
                "position_encoding must be one of {'none', 'rope', 'relative_bias'}, "
                f"got {position_encoding!r}"
            )
            
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_size = embed_dim // n_heads
        self.layer_idx = layer_idx
        self.dropout = dropout
        
        self.position_encoding = position_encoding
        self.use_unpadding = use_unpadding
        self.use_alternating_attention = use_alternating_attention
        self.window_size = window_size
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.rope = (
            RotaryPositionalEncoding(self.head_size)
            if self.position_encoding == "rope"
            else None
        )
        
        self.relative_bias = (
            RelativePositionBias(num_heads=self.n_heads, num_buckets=relative_num_buckets, max_distance=relative_max_distance, bidirectional=True)
            if self.position_encoding == "relative_bias"
            else None
        )
        
    def _shape_qkv(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_size
        ).transpose(1, 2)
    
    def _forward_padded(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        Q = self._shape_qkv(self.query(x), batch_size=B, seq_len=T)
        K = self._shape_qkv(self.key(x), batch_size=B, seq_len=T)
        V = self._shape_qkv(self.value(x), batch_size=B, seq_len=T)
        
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)
        
        final_mask = _build_key_padding_mask(attn_mask, dtype=Q.dtype)
        
        if self.relative_bias is not None:
            position_bias = self.relative_bias(
                query_len=T,
                key_len=T,
                device=x.device,
            )
            final_mask = position_bias if final_mask is None else final_mask + position_bias
        
        if self.use_alternating_attention and self.layer_idx % 2 != 0:
            sliding_mask = _build_sliding_window_mask(
                query_len=T,
                key_len=T,
                window_size=self.window_size,
                device=x.device,
                dtype=Q.dtype,
                causal=False
            )
            
            final_mask = sliding_mask if final_mask is None else final_mask + sliding_mask
        
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=final_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
    
    def _forward_unpadded(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unpadding path."""
        if attn_mask is None:
            raise ValueError(f"attn_mask is required when use_unpadding=True")
        
        B, T, C = x.shape
        
        non_pad_indices = torch.nonzero(
            attn_mask.flatten(),
            as_tuple=True
        )[0]
        
        x_flat = x.view(B * T, C)
        x_unpadded = x_flat[non_pad_indices]
        total_tokens = x_unpadded.shape[0]
        
        if total_tokens == 0:
            raise ValueError("all tokens are padding; cannot run unpadded attention.")
        
        Q = self.query(x_unpadded)
        K = self.query(x_unpadded)
        V = self.query(x_unpadded)
        
        Q = Q.view(1, total_tokens, self.n_heads, self.head_size).transpose(1, 2)
        K = K.view(1, total_tokens, self.n_heads, self.head_size).transpose(1, 2)
        V = V.view(1, total_tokens, self.n_heads, self.head_size).transpose(1, 2)
        
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)
        
        batch_ids = torch.arange(B, device=x.device).unsqueeze(1).expand(B, T).flatten()
        
        valid_batch_ids = batch_ids[non_pad_indices]
        
        same_batch = valid_batch_ids[:, None] == valid_batch_ids[None, :]
        
        final_mask = torch.zeros(
            (1, 1, total_tokens, total_tokens),
            device=x.device,
            dtype=Q.dtype
        ).masked_fill(~same_batch[None, None, ...], float("-inf"))
        
        if self.relative_bias is not None:
            position_bias = self.relative_bias(
                query_len=total_tokens,
                key_len=total_tokens,
                device=x.device,
            )
            final_mask = final_mask + position_bias
        
        if self.use_alternating_attention and self.layer_idx % 2 != 0:
            sliding_mask = _build_sliding_window_mask(
                query_len=total_tokens,
                key_len=total_tokens,
                window_size=self.window_size,
                device=x.device,
                dtype=Q.dtype,
                causal=False,
            )
            final_mask = final_mask + sliding_mask
        
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=final_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).contiguous().view(total_tokens, C)
        
        out_flat = torch.zeros(
            (B * T, C),
            device=x.device,
            dtype=x.dtype,
        )
        
        out_flat[non_pad_indices] = out
        
        out_repadded = out_flat.view(B, T, C)
        
        return self.proj(out_repadded)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, T, C). got {x.shape}")
    
        if self.use_unpadding:
            if attn_mask is None:
                raise ValueError("attn_mask is required when use_padding=True")
            return self._forward_padded(x, attn_mask)
        
        return self._forward_unpadded(x, attn_mask)

class DecoderSelfAttention(nn.Module):
    """Decoder causal self-attention."""
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: int,
        position_encoding: PositionalEncoding = "relative_bias",
        use_sliding_window: bool = False,
        window_size: int = 128,
        relative_num_buckets: int = 32,
        relative_max_distances: int = 128,
    ) -> None:
        super().__init__()
        
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by n_heads, got {embed_dim=} {n_heads=}"
            )
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_size = embed_dim // n_heads
        self.dropout = dropout
        
        self.position_encoding = position_encoding
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        
        self.relative_num_buckets = relative_num_buckets
        self.relative_max_distances = relative_max_distances
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.rope = (
            RotaryPositionalEncoding(self.head_size)
            if self.position_encoding == "rope"
            else None
        )
        
        self.relative_bias = (
            RelativePositionBias(num_heads=n_heads, num_buckets=relative_num_buckets, max_distance=relative_max_distances, bidirectional=False)
            if self.position_encoding == "relative_bias"
            else None
        )
        
        self.register_buffer("cache_key", None, persistent=False)
        self.register_buffer("cache_value", None, persistent=False)
    
    def clear_cache(self) -> None:
        self.cache_key = None
        self.cache_value = None
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, T, C), got {x.shape}")

        if use_cache and self.position_encoding == "relative_bias":
            raise NotImplementedError(
                "KV cache with relative_position_bias needs query position offset. "
                "Use no cache for relative-bias generation first, or use RoPE cache ablation."
            )
            
        B, T, C = x.shape
        
        past_len = (
            self.cache_key.shape[2]
            if use_cache and self.cache_key is not None
            else 0
        )
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        if self.rope is not None:
            Q = self.rope(Q, seq_offset=past_len)
            K = self.rope(K, seq_offset=past_len)
        
        if use_cache:
            if self.cache_key is None:
                self.cache_key = K
                self.cache_value = V
            else:
                self.cache_key = torch.cat([self.cache_key, K], dim=2)
                self.cache_value = torch.cat([self.cache_value, V], dim=2) #type: ignore
            
            K = self.cache_key
            V = self.cache_value
        
        key_len = K.shape[2]
        
        if self.use_sliding_window:
            final_mask = _build_sliding_window_mask(
                query_len=T,
                key_len=key_len,
                window_size=self.window_size,
                device=x.device,
                dtype=Q.dtype,
                query_position_offset=past_len,
                causal=True,
            )
        else:
            final_mask = _build_causal_mask(
                query_len=T,
                key_len=key_len,
                device=x.device,
                dtype=Q.dtype,
                query_position_offset=past_len
            )
        
        key_padding_mask = _build_key_padding_mask(attn_mask=attn_mask, dtype=Q.dtype)
        
        if key_padding_mask is not None and not use_cache:
            final_mask = final_mask + key_padding_mask
        
        if self.relative_bias is not None:
            position_bias = self.relative_bias(
                query_len=T,
                key_len=key_len,
                device=x.device,
            )
            final_mask = final_mask + position_bias
        
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=final_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)

class CrossAttention(nn.Module):
    """Decoder cross-attention."""
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim must be divisible by n_heads, got {embed_dim=} {n_heads=}")
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_size = embed_dim // n_heads
        self.dropout = dropout
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.register_buffer("cache_key", None, persistent=False)
        self.register_buffer("cache_value", None, persistent=False)
    
    def clear_cache(self) -> None:
        self.cache_key = None
        self.cache_value = None
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_attn_mask: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if decoder_hidden.dim() != 3:
            raise ValueError(
                f"decoder_hidden must have shape (B, T_dec, C), got {decoder_hidden.shape}"
            )

        if encoder_hidden.dim() != 3:
            raise ValueError(
                f"encoder_hidden must have shape (B, T_enc, C), got {encoder_hidden.shape}"
            )
        
        B, T_dec, C = encoder_hidden.shape
        B_enc, T_enc, C_enc = decoder_hidden.shape
        
        if B != B_enc:
            raise ValueError(f"batch mismatch: decoder B={B}, encoder B={B_enc}")

        if C != C_enc:
            raise ValueError(f"embed dim mismatch: decoder C={C}, encoder C={C_enc}")
        
        Q = self.query(decoder_hidden)
        Q = Q.view(B, T_dec, self.n_heads, self.head_size).transpose(1, 2)
        
        if use_cache and self.cache_key is not None:
            K = self.cache_key
            V = self.cache_value
        else:
            K = self.key(encoder_hidden)
            V = self.value(encoder_hidden)
            
            K = K.view(B_enc, T_enc, self.n_heads, self.head_size).transpose(1, 2)
            V = V.view(B_enc, T_enc, self.n_heads, self.head_size).transpose(1, 2)
            
            if use_cache:
                self.cache_key = K
                self.cache_value = V
        
        final_mask = _build_key_padding_mask(encoder_attn_mask, dtype=Q.dtype)
        
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V, #type: ignore
            attn_mask=final_mask,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T_dec, C)
        
        return self.proj(out)
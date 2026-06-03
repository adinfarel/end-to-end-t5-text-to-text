'''
basemodel/src/pos_enc.py

This is position encoding to get position of each token
'''

import math
import torch
import torch.nn as nn

class LearnedPositionEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        assert T <= self.max_len, "length of input ids large than max_len"
        arange = torch.arange(T, dtype=torch.long)
        return self.pos_emb(arange) + x

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, base: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.embed_dim, 2, dtype=torch.float32) / self.embed_dim))
        self.register_buffer(
            "inv_freq", inv_freq
        )
        
    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        T, C = x.shape[-2], x.shape[-1]
        assert C == self.embed_dim, "embedding dimension of input ids mismatch with embed_dim"
        
        l = torch.arange(T, dtype=self.inv_freq.dtype , device=x.device) + seq_offset #type: ignore
        theta = torch.einsum("i,j->ij", l, self.inv_freq)
        hat_theta = torch.cat([theta, theta], dim=-1)
        
        sin = torch.sin(hat_theta)
        cos = torch.cos(hat_theta)
        
        xu, xd = x[..., : C//2], x[..., C//2 :]
        hatx = torch.cat([-xd, xu], dim=-1)
        
        return x * cos + hatx * sin

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        
        if not isinstance(num_heads, int):
            raise TypeError(f"num_heads must be int, got {num_heads}")
        
        if not isinstance(num_buckets, int):
            raise TypeError(f"num_buckets must be int, got {num_buckets}")
        
        if not isinstance(max_distance, int):
            raise TypeError(f"max_distance must be int, got {max_distance}")
        
        if not isinstance(bidirectional, bool):
            raise TypeError(f"bidirectional must be bool, got {bidirectional}")
        
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if num_buckets <= 0:
            raise ValueError(f"num_buckets must be positive, got {num_buckets}")

        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")

        if bidirectional and num_buckets % 2 != 0:
            raise ValueError(
                "num_buckets must be even when bidirectional=True, "
                f"got {num_buckets}"
            )
        
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        """Map relative positions to bucket IDs."""
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
        
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            
            is_positive = relative_position > 0
            
            relative_buckets += is_positive.to(torch.long) * num_buckets
            
            relative_position = torch.abs(relative_position)
        else:
            num_buckets = self.num_buckets
            
            relative_position = -torch.minimum(
                relative_position,
                torch.zeros_like(relative_position)
            )
        
        max_exact = num_buckets // 2
        
        is_small = relative_position < max_exact
        
        relative_position_float = relative_position.to(torch.float32)
        
        large_position_bucket = max_exact + (
            torch.log(relative_position_float / max_exact + 1e-6)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        large_position_bucket = torch.minimum(
            large_position_bucket,
            torch.full_like(large_position_bucket, num_buckets - 1),
        )
        
        relative_buckets += torch.where(
            is_small,
            relative_position.to(torch.long),
            large_position_bucket,
        )
        
        return relative_buckets
    
    def forward(
        self,
        query_len: int,
        key_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build relative position bias."""
        if not isinstance(query_len, int):
            raise TypeError(f"query_len must be int, got {type(query_len).__name__}")

        if not isinstance(key_len, int):
            raise TypeError(f"key_len must be int, got {type(key_len).__name__}")

        if query_len <= 0:
            raise ValueError(f"query_len must be positive, got {query_len}")

        if key_len <= 0:
            raise ValueError(f"key_len must be positive, got {key_len}")
        
        context_position = torch.arange(
            query_len,
            dtype=torch.long,
            device=device,
        )[:, None] # (query_len, 1)
        
        memory_position = torch.arange(
            key_len,
            dtype=torch.long,
            device=device,
        )[None, :] # (1, key_len)
        
        relative_position = memory_position - context_position
        
        relative_buckets = self._relative_position_bucket(relative_position=relative_position)
        
        values = self.relative_attention_bias(relative_buckets)
        # (query_len, key_len, num_heads)
        
        values = values.permute(2, 0, 1).unsqueeze(0)
        # (1, num_heads, query_len, key_len)
        
        return values
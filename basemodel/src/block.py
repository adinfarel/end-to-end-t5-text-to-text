'''
basemodel/src/block.py

This is block to get wrap all piece of transformer in one block Encoder and Decoder 
'''

import torch
import torch.nn as nn
from basemodel.src.normalization import RMSNorm
from basemodel.src.attention import EncoderSelfAttention, DecoderSelfAttention, CrossAttention, PositionalEncoding
from basemodel.src.feedforward import GeGLU, SwiGLU

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float,
                 layer_idx: int, position_encoding: PositionalEncoding = "relative_bias", use_unpadding: bool = False,
                 use_alternating_attention: bool = False, window_size: int = 3, relative_num_buckets: int = 32,
                 relative_max_distances: int = 128) -> None:
        super().__init__()
        self.attn = EncoderSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            layer_idx=layer_idx,
            dropout=dropout,
            position_encoding=position_encoding,
            use_unpadding=use_unpadding,
            use_alternating_attention=use_alternating_attention,
            window_size=window_size,
            relative_num_buckets=relative_num_buckets,
            relative_max_distances=relative_max_distances,
        )
        
        self.ffwd = GeGLU(
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        self.ln1 = RMSNorm(embed_dim=embed_dim)
        self.ln2 = RMSNorm(embed_dim=embed_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float,
                 position_encoding: PositionalEncoding = "relative_bias", use_sliding_window: bool = False,
                 window_size: int = 3, relative_num_buckets: int = 32, relative_max_distances: int = 128) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.self_attn = DecoderSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            position_encoding=position_encoding,
            use_sliding_window=use_sliding_window,
            window_size=window_size,
            relative_num_buckets=relative_num_buckets,
            relative_max_distances=relative_max_distances
        )
        
        self.ffwd = SwiGLU(
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        self.ln1 = RMSNorm(embed_dim=embed_dim)
        self.ln2 = RMSNorm(embed_dim=embed_dim)
        self.ln3 = RMSNorm(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor, encoder_hidden: torch.Tensor, decoder_attn_mask: torch.Tensor | None = None, encoder_attn_mask: torch.Tensor | None = None, use_cache: bool = False) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), attn_mask=decoder_attn_mask, use_cache=use_cache)
        x = x + self.cross_attn(decoder_hidden=self.ln2(x), encoder_hidden=encoder_hidden, encoder_attn_mask=encoder_attn_mask, use_cache=use_cache)
        x = x + self.ffwd(self.ln3(x))
        return x

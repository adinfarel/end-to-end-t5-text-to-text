'''
basemodel/src/feedforward.py

Build feedforward to get capture non-linearity representation of model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_embed = int(embed_dim * (8/3))
        self.gate = nn.Linear(embed_dim, self.n_embed, bias=False)
        self.value = nn.Linear(embed_dim, self.n_embed, bias=False)
        self.down_proj = nn.Linear(self.n_embed, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        gate = F.silu(self.gate(x))
        hidden = value * gate
        out = self.down_proj(hidden)
        return self.dropout(out)

class GeGLU(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_embed = int(embed_dim * (8/3))
        self.gate = nn.Linear(embed_dim, self.n_embed, bias=False)
        self.value = nn.Linear(embed_dim, self.n_embed, bias=False)
        self.down_proj = nn.Linear(self.n_embed, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        gate = F.gelu(self.gate(x))
        hidden = value * gate
        out = self.down_proj(hidden)
        return self.dropout(out)
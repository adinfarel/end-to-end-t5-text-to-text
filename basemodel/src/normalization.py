'''
basemodel/src/normalization.py

Build normalization layer for normalize hidden states to prevent vanishing/exploding gradient, dying ReLU, saturated Tanh
'''

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps=1e-8):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return x_norm * self.weight

'''
basemodel/src/embedding.py

Give each tokens representation numeric in vector space
'''

import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        
        if isinstance(vocab_size, int):
            raise TypeError(f"vocab_size must be int, got{vocab_size}")
        
        if isinstance(embed_dim, int):
            raise TypeError(f"embed_dim must be int, got{embed_dim}")
        
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
    def forward(self, x: torch.Tensor | list) -> torch.Tensor:
        '''x: (B, T) --> x: (B, T, C)'''
        return self.embedding(x)
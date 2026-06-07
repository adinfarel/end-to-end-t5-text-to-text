'''
basemodel/src/t5.py

Create t5 model for this project
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box

from basemodel.src.embedding import Embedding
from basemodel.src.normalization import RMSNorm
from basemodel.src.block import EncoderBlock, DecoderBlock

class AlmondT5Model(nn.Module):
    def __init__(self, config: Box) -> None:
        super().__init__()
        self.embedding = Embedding(config.tokenizer.vocab_size, config.model.embed_dim)
        
        self.pad_token_id = config.tokenizer.pad_token_id
        self.eos_token_id = config.tokenizer.eos_token_id
        self.decoder_start_token_id = config.tokenizer.decoder_start_token_id
        
        self.encoder = nn.ModuleList(
            [EncoderBlock(
                embed_dim=config.model.embed_dim,
                n_heads=config.model.n_heads,
                dropout=config.model.dropout,
                layer_idx=i,
                position_encoding=config.model.position_encoding,
                use_unpadding=config.model.use_encoder_unpadding,
                use_alternating_attention=config.model.use_encoder_alternating_attention,
                window_size=config.model.window_size,
                relative_num_buckets=config.model.relative_num_buckets,
                relative_max_distances=config.model.relative_max_distances,
            )
             for i in range(config.model.n_encoder_blocks)]
        )
        
        self.decoder = nn.ModuleList(
            [DecoderBlock(
                embed_dim=config.model.embed_dim,
                n_heads=config.model.n_heads,
                dropout=config.model.dropout,
                position_encoding=config.model.position_encoding,
                use_sliding_window=config.model.use_decoder_sliding_window,
                window_size=config.model.window_size,
                relative_num_buckets=config.model.relative_num_buckets,
                relative_max_distances=config.model.relative_max_distances,
            ) for _ in range(config.model.n_decoder_blocks)]
        )
        
        self.encoder_ln_f = RMSNorm(embed_dim=config.model.embed_dim)
        self.decoder_ln_f = RMSNorm(embed_dim=config.model.embed_dim)
        self.lm_head = nn.Linear(config.model.embed_dim, config.tokenizer.vocab_size, bias=False)
        
        if getattr(config.model, "tie_word_embeddings", True):
            self.lm_head.weight = self.embedding.embedding.weight
    
    def clear_kv_cache(self) -> None:
        for block in self.decoder:
            if hasattr(block.self_attn, "clear_cache"):
                block.self_attn.clear_cache() #type: ignore
            
        for block in self.decoder:
            if hasattr(block.cross_attn, "clear_cache"):
                block.cross_attn.clear_cache() #type: ignore
    
    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embedding(encoder_input_ids)
        
        for block in self.encoder:
            x = block(
                x,
                attn_mask=encoder_attn_mask,
            )
        
        return self.encoder_ln_f(x)
    
    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        decoder_attn_mask: torch.Tensor | None = None,
        encoder_attn_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = self.embedding(decoder_input_ids)
        
        for block in self.decoder:
            x = block(
                x,
                encoder_hidden,
                decoder_attn_mask,
                encoder_attn_mask,
                use_cache
            )
        
        return self.decoder_ln_f(x)
        
    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        decoder_labels: torch.Tensor | None = None,
        enc_attn_mask: torch.Tensor | None = None,
        dec_attn_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoder_hidden = self.encode(
            encoder_inputs,
            enc_attn_mask,
        )
        
        decoder_hidden = self.decode(
            decoder_inputs,
            encoder_hidden,
            dec_attn_mask,
            enc_attn_mask,
            use_cache,
        )
        
        logits = self.lm_head(decoder_hidden)
        
        if decoder_labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                decoder_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attn_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        self.eval()
        self.clear_kv_cache()
            
        if encoder_input_ids.dim() != 2:
            raise ValueError(
                f"encoder_input_ids must have shape (B, T_enc), got {encoder_input_ids.shape}"
            )
        
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        
        B = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        assert self.decoder_start_token_id is not None
        assert self.eos_token_id is not None
        
        encoder_hidden = self.encode(
            encoder_input_ids=encoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
        )
        
        generated = torch.full(
            (B, 1),
            fill_value=self.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            decoder_attn_mask = torch.ones_like(generated, dtype=torch.long)
            
            decoder_hidden = self.decode(
                generated,
                encoder_hidden,
                decoder_attn_mask,
                encoder_attn_mask,
                use_cache=False,
            )
            
            logits = self.lm_head(decoder_hidden)
            next_tokens_logits = logits[:, -1, :]
            
            if temperature <= 0:
                raise ValueError(f"temperature must be positive, got {temperature}")
            
            next_tokens_logits = next_tokens_logits / temperature
            
            if do_sample:
                probs = F.softmax(next_tokens_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_tokens_logits, dim=-1, keepdim=True)
            
            next_token = torch.where(
                finished[:, None], # (B, 1)
                torch.full_like(next_token, self.eos_token_id),
                next_token,
            )
            
            generated = torch.cat([generated, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(1) == self.eos_token_id)
            
            if finished.all():
                break
        
        return generated
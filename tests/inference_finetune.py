'''
tests/inference_pretrain.py

Testing inference model finetune for one sample
'''

from pathlib import Path
import random
import torch

from basemodel.src.t5 import AlmondT5Model
from tokenizer.ulm import AlmondUnigramTokenizer
from finetune.data.loader import get_prefix, pad_1d, truncate_with_eos
from config.config import get_config

# ENV
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = get_config()
SEED   = 42

# HELPER
def load_model_and_tokenizer() -> tuple[AlmondT5Model, AlmondUnigramTokenizer]:
    '''Load model pretrain and tokenizer.'''
    tokenizer = AlmondUnigramTokenizer(
        whitespace_marker=CONFIG.tokenizer.whitespace_marker,
        num_extra_ids=CONFIG.tokenizer.num_extra_ids,
    )
    tokenizer.load(
        file_path=CONFIG.tokenizer.output_tokenizer_path
    )
    
    model = AlmondT5Model(CONFIG)
    
    best_model_path: Path = Path(CONFIG.finetune.finetune_model_path) / f"best_model_{CONFIG.model.position_encoding}.pt"
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def make_one_sample_inference(
    text: str,
    tokenizer: AlmondUnigramTokenizer,
    prefix_str: str,
    max_enc_len: int,
    max_piece_len: int,
):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    
    assert eos_id is not None
    assert pad_id is not None
    
    prefix = get_prefix(text=prefix_str, tokenizer=tokenizer, max_piece_length=max_piece_len)
    
    ids = tokenizer.encode(text=text, max_piece_length=max_piece_len)
    
    encoder_inputs = truncate_with_eos(
        ids=prefix + ids + [eos_id],
        max_len=max_enc_len,
        eos_id=eos_id
    )
    
    encoder_inputs, encoder_attn_mask = pad_1d(
        ids=encoder_inputs,
        pad_id=pad_id,
        max_len=max_enc_len
    )
    
    return encoder_inputs, encoder_attn_mask

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    
    texts = [
        ("I am sorry", "Aku minta maaf"),
        ("Thank you", "Terima kasih"),
        ("I'm here", "Aku disini"),
        ("Fuck you", "Persetan denganmu")
    ]
    
    for text in texts:
        
        encoder_inputs, encoder_attn_mask = make_one_sample_inference(
            text=text[0],
            tokenizer=tokenizer,
            prefix_str=CONFIG.finetune.prefix,
            max_enc_len=CONFIG.model.max_encoder_len,
            max_piece_len=CONFIG.tokenizer.max_piece_length
        )
        
        generated = model.generate(
            encoder_input_ids=torch.tensor(encoder_inputs, dtype=torch.long).unsqueeze(0).to(DEVICE),
            encoder_attn_mask=torch.tensor(encoder_attn_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )
        
        translate = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        
        print(f"ENGLISH TEXT      : {text[0]!r}")
        print(f"INDONESIA TEXT    : {translate!r}")
        print(f"English: {text[0]} ---> Indonesian: {translate}")
        print(f"GROUND TRUTH      : {text[1]}")
        print()
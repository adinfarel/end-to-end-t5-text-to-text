'''
tests/inference_pretrain.py

Testing inference model for one sample
'''

from pathlib import Path
import random
import torch

from basemodel.src.t5 import AlmondT5Model
from tokenizer.ulm import AlmondUnigramTokenizer
from basemodel.data.loader import make_t5_span_corruption_example, shift_right_t5, pad_1d, pad_labels
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
    
    best_model_path: Path = Path(CONFIG.pretrain.pretrain_model_path) / f"best_model_{CONFIG.model.position_encoding}.pt"
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def make_one_sample_to_span_corruption(
    text: str,
    tokenizer: AlmondUnigramTokenizer,
    max_encoder_length: int,
    max_decoder_length: int,
    max_piece_length: int,
    noise_density: float,
    mean_noise_span_length: float,
    rng: random.Random,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    decoder_start_id = tokenizer.decoder_start_token_id
    ignore_index = -100
    
    assert pad_id is not None
    assert decoder_start_id is not None
    
    encoder_ids, labels = make_t5_span_corruption_example(
        text=text,
        tokenizer=tokenizer,
        max_encoder_len=max_encoder_length,
        max_decoder_len=max_decoder_length,
        max_piece_length=max_piece_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        rng=rng
    )
    
    decoder_ids = shift_right_t5(
        labels=labels,
        decoder_start_token_id=decoder_start_id,
    )
    
    encoder_ids, enc_attn_mask = pad_1d(
        ids=encoder_ids,
        max_len=max_encoder_length,
        pad_id=pad_id
    )
    
    decoder_ids, dec_attn_mask = pad_1d(
        ids=decoder_ids,
        max_len=max_encoder_length,
        pad_id=pad_id
    )
    
    labels = pad_labels(
        labels=labels,
        max_len=max_decoder_length,
        ignore_index=ignore_index,
    )
    
    return {
        "encoder_inputs": torch.tensor(encoder_ids),
        "decoder_inputs": torch.tensor(decoder_ids),
        "enc_attn_mask": torch.tensor(enc_attn_mask),
        "dec_attn_mask": torch.tensor(dec_attn_mask),
        "labels": torch.tensor(labels),
    }
    
if __name__ == "__main__":
    print("INFERENCE TESTING")
    rng = random.Random(SEED)
    
    print(f"LOAD MODEL AND TOKENIZER = {CONFIG.model.position_encoding}")
    model, tokenizer = load_model_and_tokenizer()
    
    texts = [
        "The city is located near a river and has many schools, hospitals, and public buildings.",
        "The company was founded in the early twentieth century and later became one of the largest companies in the country.",
        "The village has a small population and most people work in agriculture, trade, and local services.",
        "The university offers programs in science, engineering, medicine, economics, and social studies.",
        "The film was released in several countries and received positive reviews from critics and audiences.",
    ]
    
    for text in texts:
        item = make_one_sample_to_span_corruption(
            text=text,
            tokenizer=tokenizer,
            max_encoder_length=CONFIG.model.max_encoder_len,
            max_decoder_length=CONFIG.model.max_decoder_len,
            max_piece_length=CONFIG.tokenizer.max_piece_length,
            noise_density=0.3,
            mean_noise_span_length=2.0,
            rng=rng
        )
        
        generated = model.generate(
            encoder_input_ids=item["encoder_inputs"].unsqueeze(0).to(DEVICE),
            encoder_attn_mask=item["enc_attn_mask"].unsqueeze(0).to(DEVICE),
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False
        )
        
        print("ENCODER TEXT     :", tokenizer.decode(item["encoder_inputs"].tolist(), skip_special_tokens=False))

        generated_ids = generated[0].detach().cpu().tolist()
        label_ids = [x for x in item["labels"].tolist() if x != -100]

        print("GENERATED IDS    :", generated_ids)
        print("GENERATED TEXT   :", tokenizer.decode(generated_ids, skip_special_tokens=False))
        print("GENERATED CLEAN  :", tokenizer.decode(generated_ids, skip_special_tokens=True))
        print("LABEL IDS        :", label_ids)
        print("LABEL TEXT       :", tokenizer.decode(label_ids, skip_special_tokens=False))
        print("LABEL CLEAN      :", tokenizer.decode(label_ids, skip_special_tokens=True))
        print("\n")
        print("="*60)
'''
eval/translation_eval.py

Evaluate fine-tuned AlmondT5Model to see how good model in term numeric >.<
'''

import torch
import sacrebleu
from pathlib import Path
from tqdm import tqdm

from basemodel.src.t5 import AlmondT5Model
from tokenizer.ulm import AlmondUnigramTokenizer
from finetune.data.loader import load_jsonl, train_val_split, pad_1d
from config.config import get_config

CONFIG = get_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def load_model_and_tokenizer() -> tuple[AlmondT5Model, AlmondUnigramTokenizer]:
    tokenizer = AlmondUnigramTokenizer(
        whitespace_marker=CONFIG.tokenizer.whitespace_marker,
        num_extra_ids=CONFIG.tokenizer.num_extra_ids,
    )
    tokenizer.load(CONFIG.tokenizer.output_tokenizer_path)

    CONFIG.tokenizer.vocab_size = len(tokenizer.piece_to_id)
    CONFIG.tokenizer.pad_token_id = tokenizer.pad_token_id
    CONFIG.tokenizer.unk_token_id = tokenizer.unk_token_id
    CONFIG.tokenizer.eos_token_id = tokenizer.eos_token_id
    CONFIG.tokenizer.decoder_start_token_id = tokenizer.decoder_start_token_id
    
    model = AlmondT5Model(CONFIG)

    ckpt_path = (
        Path(CONFIG.finetune.finetune_model_path)
        / f"best_model_{CONFIG.model.position_encoding}.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {str(ckpt_path)!r}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Best val loss    : {checkpoint.get('best_val_loss', 'unknown')}")

    return model, tokenizer

def encode_source(
    en_text: str,
    tokenizer: AlmondUnigramTokenizer,
    prefix: str,
    max_encoder_len: int,
    max_piece_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    
    assert pad_id is not None
    assert eos_id is not None
    
    source_text = prefix + en_text.strip()

    input_ids = tokenizer.encode(
        text=source_text,
        max_piece_length=max_piece_length,
        add_eos=False,
    )
    
    input_ids = input_ids + [eos_id]
    
    input_ids = input_ids[:max_encoder_len]
    if input_ids[-1] != eos_id:
        input_ids[-1] = eos_id
    
    input_ids, attn_mask = pad_1d(
        ids=input_ids,
        pad_id=pad_id,
        max_len=max_encoder_len,
    )
    
    return (
        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0),
    )

@torch.no_grad()
def translate_one(
    model: AlmondT5Model,
    tokenizer: AlmondUnigramTokenizer,
    en_text: str,
    prefix: str,
    max_new_tokens: int,
) -> str:
    encoder_input_ids, encoder_attn_mask = encode_source(
        en_text=en_text,
        tokenizer=tokenizer,
        prefix=prefix,
        max_encoder_len=CONFIG.model.max_encoder_len,
        max_piece_length=CONFIG.tokenizer.max_piece_length,
    )
    
    encoder_input_ids = encoder_input_ids.to(DEVICE)
    encoder_attn_mask = encoder_attn_mask.to(DEVICE)
    
    generated = model.generate(
        encoder_input_ids=encoder_input_ids,
        encoder_attn_mask=encoder_attn_mask,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        do_sample=False
    )
    
    pred = tokenizer.decode(
        generated[0].detach().cpu().tolist(),
        skip_special_tokens=True
    )
    
    return pred.strip()

def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    
    data_path = Path(CONFIG.finetune.output_dataset_raw)
    data = load_jsonl(data_path)
    
    _, val_data = train_val_split(
        data=data,
        train_ratio=CONFIG.finetune.train_ratio,
        seed=SEED
    )
    
    max_eval_samples = int(getattr(CONFIG.finetune, "max_eval_samples", 500))
    val_data = val_data[:max_eval_samples]
    
    prefix = str(CONFIG.finetune.prefix)
    
    if prefix and not prefix.endswith(" "):
        prefix = prefix + " "
    
    predictions: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    
    print(f"Evaluating {len(val_data)} samples...")
    
    for item in tqdm(val_data, desc="Generating"):
        en_text = item.get("en_lang", "").strip()
        id_text = item.get("id_lang", "").strip()
        
        if not en_text or not id_text:
            continue
        
        pred = translate_one(
            model=model,
            tokenizer=tokenizer,
            en_text=en_text,
            prefix=prefix,
            max_new_tokens=CONFIG.model.max_decoder_len,
        )
        
        sources.append(en_text)
        predictions.append(pred)
        references.append(id_text)
    
    if not predictions:
        raise ValueError(f"No predictions generated. Check eval loss.")
    
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )
    
    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
    )
    
    print()
    print("=" * 80)
    print("EVALUATION RESULT")
    print("=" * 80)
    print(f"Samples : {len(predictions)}")
    print(f"BLEU    : {bleu.score:.4f}")
    print(f"chrF    : {chrf.score:.4f}")
    print("=" * 80)
    
    print()
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    for i in range(min(10, len(predictions))):
        print(f"[{i}] EN  : {sources[i]}")
        print(f"    REF : {references[i]}")
        print(f"    PRED: {predictions[i]}")
        print("-" * 80)

if __name__ == "__main__":
    main()
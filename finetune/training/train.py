'''
training/train.py

Training pipeline for fine-tuning t5 for translation context
'''

import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from basemodel.src.t5 import AlmondT5Model
from tokenizer.ulm import AlmondUnigramTokenizer
from finetune.data.loader import create_dataloaders, load_jsonl, train_val_split
from config.config import get_config

# ENV
CONFIG = get_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
SEED = 42 # For reproducebility

# HELPER
@torch.no_grad()
def eval_loss(model: AlmondT5Model, val_loader: DataLoader) -> float:
    model.eval()
    losses = []
    
    for item in val_loader:
        encoder_input_ids = item['encoder_input_ids'].to(DEVICE)
        encoder_attn_mask = item['encoder_attn_mask'].to(DEVICE)
        decoder_input_ids = item['decoder_input_ids'].to(DEVICE)
        decoder_attn_mask = item['decoder_attn_mask'].to(DEVICE)
        labels            = item['labels'].to(DEVICE)
        
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
            _, loss = model(
                encoder_inputs=encoder_input_ids,
                decoder_inputs=decoder_input_ids,
                decoder_labels=labels,
                enc_attn_mask=encoder_attn_mask,
                dec_attn_mask=decoder_attn_mask,
                use_cache=False,
            )

        losses.append(loss.item())
        
    if not losses:
        raise ValueError(f"val_loader produced no batches")
    
    model.train()
    return sum(losses) / len(losses)

def load_model_and_tokenizer() -> tuple[AlmondT5Model, AlmondUnigramTokenizer]:
    model = AlmondT5Model(CONFIG)
    
    pretrain_path = Path(CONFIG.pretrain.pretrain_model_path) / f"ckpt_latest_{CONFIG.model.position_encoding}.pt"
    checkpoint = torch.load(pretrain_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(DEVICE)
    
    tokenizer = AlmondUnigramTokenizer(
        whitespace_marker=CONFIG.tokenizer.whitespace_marker,
        num_extra_ids=CONFIG.tokenizer.num_extra_ids,
    )
    tokenizer.load(
        CONFIG.tokenizer.output_tokenizer_path,
    )
    
    return model, tokenizer

def main() -> None:
    print(f"{'TRAINING FINE-TUNING MODEL':=^30}")
    print(f"Initialize Model {AlmondT5Model.__name__!r} and Tokenizer {AlmondUnigramTokenizer.__name__!r}")
    model, tokenizer = load_model_and_tokenizer()
    
    dataset_path = Path(CONFIG.finetune.output_dataset_raw)
    print("LOAD DATASET TRANSLATION EN -> ID")
    
    data = load_jsonl(dataset_path)
    
    print(f"TOTAL DATASET: {len(data)}")
    print(f"SAMPLE       : {data[0]}")
    
    print(f"SPLITTING DATASET INTO TRAIN AND TEST WITH RATIO {CONFIG.finetune.train_ratio}")
    train_data, val_data = train_val_split(
        data=data,
        train_ratio=float(CONFIG.finetune.train_ratio),
        seed=SEED
    )
    
    print("TOTAL TRAIN DATA : ", len(train_data))
    print("TOTAL VAL DATA   : ", len(val_data))
    
    print("CREATE DATALOADER FOR TRAIN AND VAL")
    train_loader = create_dataloaders(
        data=train_data,
        tokenizer=tokenizer,
        prefix_str=CONFIG.finetune.prefix,
        max_enc_len=CONFIG.model.max_encoder_len,
        max_dec_len=CONFIG.model.max_decoder_len,
        max_piece_length=CONFIG.tokenizer.max_piece_length,
        ignore_index=CONFIG.finetune.ignore_index,
        shuffle=CONFIG.finetune.shuffle,
        batch_size=CONFIG.finetune.batch_size,
        num_workers=CONFIG.finetune.num_workers,
    )
    val_loader = create_dataloaders(
        data=val_data,
        tokenizer=tokenizer,
        prefix_str=CONFIG.finetune.prefix,
        max_enc_len=CONFIG.model.max_encoder_len,
        max_dec_len=CONFIG.model.max_decoder_len,
        max_piece_length=CONFIG.tokenizer.max_piece_length,
        ignore_index=CONFIG.finetune.ignore_index,
        shuffle=False,
        batch_size=CONFIG.finetune.batch_size,
        num_workers=CONFIG.finetune.num_workers,
    )
    
    print("SAMPLE OF DATALOADER")
    sample = next(iter(train_loader))
    print("ENCODER INPUT IDS SHAPE : ", sample["encoder_input_ids"].shape)
    print("ENCODER ATTN MASK SHAPE : ", sample["encoder_attn_mask"].shape)
    print("DECODER INPUT IDS SHAPE : ", sample["decoder_input_ids"].shape)
    print("DECODER ATTN MASK SHAPE : ", sample["decoder_attn_mask"].shape)
    print("LABELS SHAPE            : ", sample["labels"].shape)
    
    print("ENCODER INPUT IDS : ", sample["encoder_input_ids"][0])
    print("ENCODER ATTN MASK : ", sample["encoder_attn_mask"][0])
    print("DECODER INPUT IDS : ", sample["decoder_input_ids"][0])
    print("DECODER ATTN MASK : ", sample["decoder_attn_mask"][0])
    print("LABELS            : ", sample["labels"][0])
    
    enc = sample["encoder_input_ids"][0].tolist()
    dec = sample["decoder_input_ids"][0].tolist()
    lab = [x for x in sample["labels"][0].tolist() if x != CONFIG.finetune.ignore_index]

    print("ENCODER TEXT:", tokenizer.decode(enc, skip_special_tokens=False))
    print("DECODER TEXT:", tokenizer.decode(dec, skip_special_tokens=False))
    print("LABEL TEXT  :", tokenizer.decode(lab, skip_special_tokens=False))
        
    print("SETUP COMPONENT")
    pos_enc = CONFIG.model.position_encoding
    best_model_path = Path(CONFIG.finetune.finetune_model_path) / f"best_model_{pos_enc}.pt"
    ckpt_model_path = Path(CONFIG.finetune.finetune_model_path) / f"ckpt_latest_{pos_enc}.pt"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG.finetune.learning_rate), weight_decay=float(CONFIG.finetune.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=int(CONFIG.finetune.epochs * len(train_loader)),
    )
    scaler = torch.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda" and DTYPE == torch.float16))
    
    early_stopping_counter = 0
    early_stopping_patience = CONFIG.finetune.early_stopping_patience
    
    # Checkpoints
    start_epochs = 0
    best_val_loss = float("inf")
    
    print("CHECK WHETHER CHECKPOINT EXISTS OR NOT")
    if ckpt_model_path.exists():
        print(f"--> Checkpoint found at: {str(ckpt_model_path)!r}. Resuming training...")
        checkpoint = torch.load(ckpt_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epochs = checkpoint.get('epoch', start_epochs)
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"--> Resuming training for epoch {start_epochs} with best validation loss {best_val_loss:.4f}")
    else:
        print(f"--> Checkpoint not found. Start training from scratch...")
    
    for epoch in range(start_epochs + 1, CONFIG.finetune.epochs + 1):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG.finetune.epochs}')
        for item in progress_bar:
            encoder_input_ids = item["encoder_input_ids"].to(DEVICE, non_blocking=True)
            encoder_attn_mask = item["encoder_attn_mask"].to(DEVICE, non_blocking=True)
            decoder_input_ids = item["decoder_input_ids"].to(DEVICE, non_blocking=True)
            decoder_attn_mask = item["decoder_attn_mask"].to(DEVICE, non_blocking=True)
            labels = item["labels"].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss = model(
                    encoder_inputs=encoder_input_ids,
                    decoder_inputs=decoder_input_ids,
                    decoder_labels=labels,
                    enc_attn_mask=encoder_attn_mask,
                    dec_attn_mask=decoder_attn_mask,
                    use_cache=False,
                )
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CONFIG.finetune.grad_clip))
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % CONFIG.finetune.eval_interval == 0:
            losses = eval_loss(model, val_loader)
            print(f"Epoch {epoch} | Eval loss {losses}")
            
            if losses < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = losses
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }
                
                if not best_model_path.parent.exists():
                    best_model_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save(checkpoint, best_model_path)
                
                print(f"--> New best model save at epoch {epoch} with loss {losses:.4f}")
            
            else:
                early_stopping_counter += 1
                if CONFIG.finetune.early_stopping and early_stopping_counter >= early_stopping_patience:
                    print(f"--> Early stopping at epoch {epoch} with loss {losses:.4f}")
                    break
                
        latest_checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }

        ckpt_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latest_checkpoint, ckpt_model_path)
    
    final_epoch = CONFIG.finetune.epochs if early_stopping_counter < early_stopping_patience else epoch
       
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': final_epoch,
        'best_val_loss': best_val_loss
    }
    
    if not ckpt_model_path.parent.exists():
        ckpt_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        checkpoint,
        ckpt_model_path
    )
    
    print("TRAINING FINISH. Yayyy >//<")

if __name__ == "__main__":
    main()
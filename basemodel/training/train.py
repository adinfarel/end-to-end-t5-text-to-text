'''
basemodel/src/train.py

Full training pipeline for Almond T5 Model
'''

import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from basemodel.src.t5 import AlmondT5Model
from tokenizer.ulm import AlmondUnigramTokenizer
from basemodel.data.loader import create_pretrain_dataloaders
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
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_labels=labels,
                encoder_attn_mask=encoder_attn_mask,
                decoder_attn_mask=decoder_attn_mask,
                use_cache=False,
            )

        losses.append(loss.item())
        
    if not losses:
        raise ValueError(f"val_loader produced no batches")
    
    model.train()
    return sum(losses) / len(losses)

def main() -> None:
    print(f"{'TRAINING PIPELINE MODEL':=^30}")
    print(f"Initialize Model {AlmondT5Model.__name__!r} and tokenizer {AlmondUnigramTokenizer.__name__!r}")
    # Tokenizer
    tokenizer = AlmondUnigramTokenizer(
        whitespace_marker=CONFIG.tokenizer.whitespace_marker,
        num_extra_ids=CONFIG.tokenizer.num_extra_ids,
    )
    tokenizer.load(CONFIG.tokenizer.output_tokenizer_path)
    # Model
    model = AlmondT5Model(config=CONFIG)
    model = model.to(DEVICE)
    
    corpus_path = Path(CONFIG.datasets.output_dataset_raw)
    print("CREATE DATA LOADERS FOR TRAIN AND VALIDATION")
    print("Component dataset:")
    print(f"Path                        --> {str(corpus_path)!r}")
    print(f"Batch Size                  --> {CONFIG.pretrain.batch_size}")
    print(f"Noise Density (Mask Prob)   --> {CONFIG.pretrain.mask_prob}")
    print(f"Mean noise span length      --> {CONFIG.pretrain.mean_noise_span_length}")
    
    train_loader, val_loader = create_pretrain_dataloaders(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        batch_size=CONFIG.pretrain.batch_size,
        max_encoder_len=CONFIG.model.max_encoder_len,
        max_decoder_len=CONFIG.model.max_decoder_len,
        max_piece_length=CONFIG.tokenizer.max_piece_length,
        train_ratio=float(CONFIG.pretrain.train_ratio),
        seed=SEED,
        min_chars=CONFIG.pretrain.min_chars,
        max_chars=CONFIG.pretrain.max_chars,
        noise_density=CONFIG.pretrain.mask_prob,
        mean_noise_span_length=CONFIG.pretrain.mean_noise_span_length,
    )
    
    print("SAMPLE OF DATA LOADER")
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
    
    # Setup component
    best_model_path = Path(CONFIG.pretrain.pretrain_model_path) / "best_model.pt"
    ckpt_model_path = Path(CONFIG.pretrain.pretrain_model_path) / "ckpt_latest.pt"
    
    print("SETUP COMPONENT TRAINING")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG.pretrain.learning_rate),
        weight_decay=CONFIG.pretrain.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=(CONFIG.pretrain.epochs * len(train_loader)),
    )
    scaler = torch.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda" and DTYPE == torch.float16))
    
    early_stopping_counter = 0
    early_stopping_patience = CONFIG.pretrain.early_stopping_patience
    
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
    
    for epoch in range(start_epochs + 1, CONFIG.pretrain.epochs + 1):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG.pretrain.epochs}')
        for item in progress_bar:
            encoder_input_ids = item["encoder_input_ids"].to(DEVICE, non_blocking=True)
            encoder_attn_mask = item["encoder_attn_mask"].to(DEVICE, non_blocking=True)
            decoder_input_ids = item["decoder_input_ids"].to(DEVICE, non_blocking=True)
            decoder_attn_mask = item["decoder_attn_mask"].to(DEVICE, non_blocking=True)
            labels = item["labels"].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss = model(
                    encoder_input_ids=encoder_input_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_labels=labels,
                    encoder_attn_mask=encoder_attn_mask,
                    decoder_attn_mask=decoder_attn_mask,
                    use_cache=False,
                )
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.pretrain.grad_clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if epoch % CONFIG.pretrain.eval_interval == 0:
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
                if CONFIG.pretrain.early_stopping and early_stopping_counter >= early_stopping_patience:
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
    
    final_epoch = CONFIG.pretrain.epochs if early_stopping_counter < early_stopping_patience else epoch
    
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

if __name__ == "__name__":
    main()
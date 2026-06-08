'''
data/dataset.py

Get datasets translation EN -> ID from HuggingFace
'''

import jsonlines
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path

from config.config import get_config

CONFIG = get_config()

def is_valid_pair(en_text: str, id_text: str) -> bool:
    if not en_text or not id_text:
        return False
    
    if len(en_text) < 2 or len(id_text) < 2:
        return False
    
    if len(en_text) > 400 or len(id_text) > 400:
        return False
    
    return True

def write_stream_to_jsonl(
    dataset_name: str,
    n_rows: int,
    output_path: Path,
) -> int:
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True
    )
    
    written = 0
    
    with jsonlines.open(output_path, mode='w') as f:
        for item in tqdm(dataset, desc="Downloading"):
            text = item.get('text', '').strip()
            
            if not text:
                continue
                
            if "###>" not in text:
                continue
            
            split = text.split("###>")
            id_text = split[0]
            en_text = split[1]
            
            f.write({'en_lang': en_text, 'id_lang': id_text})
            written += 1
            
            # Make sure we get dataset as many as n_rows
            if written >= n_rows:
                break
            
    return written

def load_from_hf() -> None:
    output_path = Path(CONFIG.finetune.output_dataset_raw)
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Dataset already exists at {str(output_path)!r}. Skipping download")
        return
    
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    
    if tmp_path.exists():
        tmp_path.unlink()
        
    tmp_path.touch()
    
    n_rows = int(CONFIG.finetune.max_dataset)
    
    try:
        count = write_stream_to_jsonl(
            dataset_name=CONFIG.finetune.dataset_name,
            n_rows=n_rows,
            output_path=tmp_path,
        )
        
        tmp_path.replace(output_path)
        
        print(
            f"Dataset raw corpus save at {str(output_path)!r}. total dataset {count}"
        )
    except:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    
if __name__ == "__main__":
    load_from_hf()
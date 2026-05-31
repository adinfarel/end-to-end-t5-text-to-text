'''
data/dataset.py

Get dataset wikipedia En and Id from HuggingFace platform.
'''

from itertools import islice
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from huggingface_hub import CONFIG_NAME
from config.config import get_config

CONFIG = get_config()

# NOTE: This is version naive how to load big large datasets
# def load_from_hf() -> None:
#     en_ds = load_dataset(CONFIG.datasets.name, "20231101.en", split=CONFIG.datasets.split)
#     id_ds = load_dataset(CONFIG.datasets.name, "20231101.id", split=CONFIG.datasets.split)
    
#     # Save and merge in one file .txt
#     output_save = Path(CONFIG.datasets.output_dataset_raw)
    
#     if not output_save.parent.exists():
#         output_save.parent.mkdir(parents=True, exist_ok=True)
    
#     with open(output_save, mode='w', encoding='utf-8') as f:
#         for item in en_ds:
#             f.write(item["text"] + "\n")
#         for item in id_ds:
#             f.write(item["text"] + "\n")
    
#     print(f"Dataset raw corpus {CONFIG.datasets.name} en-id successfully store at {CONFIG.datasets.output_dataset_raw}")

def write_stream_to_txt(
    dataset_name: str,
    config_name: str,
    n_rows: int,
    output_path: Path,
) -> int:
    ds = load_dataset(
        dataset_name,
        config_name,
        split="train",
        streaming=True
    )
    
    written = 0
    
    with output_path.open(mode="a", encoding="utf-8") as f:
        for item in tqdm(islice(ds, n_rows), desc="Downloading"):
            text = item.get("text", "")
            
            if not text:
                continue
                
            text = text.strip().replace("\n", " ")
            
            if not text:
                continue
            
            f.write(text + "\n")
            written += 1

    return written

def load_from_hf() -> None:
    output_path = Path(CONFIG.datasets.output_dataset_raw)
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Dataset already exists at {str(output_path)}. Skipping download.")
        return
    
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    
    if tmp_path.exists():
        tmp_path.unlink()
    
    tmp_path.touch()
    
    n_rows = int(CONFIG.datasets.max_row_datasets)
    
    try:
        en_count = write_stream_to_txt(
            dataset_name=CONFIG.datasets.name,
            config_name=CONFIG.datasets.en_config,
            n_rows=n_rows,
            output_path=tmp_path,
        )

        id_count = write_stream_to_txt(
            dataset_name=CONFIG.datasets.name,
            config_name=CONFIG.datasets.id_config,
            n_rows=n_rows,
            output_path=tmp_path,
        )
        
        tmp_path.replace(output_path)
        
        print(
            f"Dataset raw corpus saved at {output_path}. "
            f"EN rows: {en_count}, ID rows: {id_count}, total: {en_count + id_count}"
        )
    
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    
if __name__ == "__main__":
    load_from_hf()
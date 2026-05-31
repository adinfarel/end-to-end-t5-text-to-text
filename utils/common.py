'''
utils/common.py

Helper function or Utilities for production project
'''

import yaml
from pathlib import Path
from box import Box

def ensure_parent_exist(file: Path) -> None:
    '''Make sure parent file exists.'''
    file = Path(file)
    
    if file.parent != Path("."):
        file.parent.mkdir(parents=True, exist_ok=True)
    
def load_yaml(file: Path, use_box: bool = True) -> Box | dict:
    if not isinstance(file, Path):
        file = Path(file)
    
    if not file.exists():
        raise FileNotFoundError(f"File not found: {str(file)}")
    
    with open(file, mode='r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
    
    if use_box:
        return Box(content)
    
    return content
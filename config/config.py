'''
config/config.py

Load config once and use this config many-times
'''

from functools import lru_cache
from pathlib import Path
from box import Box

from utils.common import load_yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

@lru_cache(maxsize=1)
def get_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Box:
    return load_yaml(file=config_path, use_box=True) #type: ignore
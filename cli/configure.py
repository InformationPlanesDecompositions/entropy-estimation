import yaml

from os import path
from typing import Any


def read_config(config_path: str, *, config_encoding: str = 'utf-8') -> dict[str, Any]:
    if not path.exists(config_path):
        raise FileNotFoundError(config_path)

    if path.isdir(config_path):
        raise IsADirectoryError(config_path)

    if not (config_path.endswith('.yml') or config_path.endswith('.yaml')):
        raise ValueError(f'Provided file {config_path} is not a YAML file.')
    
    with open(config_path, 'r', encoding=config_encoding) as config_file:
        conf = yaml.load(config_file, Loader=yaml.SafeLoader)

    return conf

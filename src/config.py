import os
import yaml

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def ensure_exists(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_config(config_path):

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    config = {}
    for var_name, rel_path in raw_config.items():
        config[var_name] = ensure_exists(os.path.join(ROOT_PATH, rel_path))
        
    for var_name, value in config.items():
        if " " in value:
            raise ValueError(f"White space found in {var_name}")
    
    return config
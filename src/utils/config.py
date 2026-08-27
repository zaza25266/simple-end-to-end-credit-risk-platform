import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).resolve().parents[2]

def load_params(config_path: str = "config/params.yaml") -> dict:
    """
    Load parameters from YAML configuration file.
    """
    root = get_project_root()
    full_path = root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
        
    with open(full_path, "r") as f:
        params = yaml.safe_load(f)
        
    return params
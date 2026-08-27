from functools import lru_cache
import joblib
from pathlib import Path
from src.utils.config import load_params

@lru_cache(maxsize=1)
def get_production_artifacts():
    """
    Loads the trained production model and feature transformer into memory once
    using lru_cache to prevent repeated disk I/O overhead on live requests.
    """
    params = load_params()
    root = Path(__file__).resolve().parents[2]
    
    model_path = root / "models" / "champion_model.pkl"
    transformer_path = root / "models" / "feature_transformer.pkl"
    
    
    model = joblib.load(model_path) if model_path.exists() else None
    transformer = joblib.load(transformer_path) if transformer_path.exists() else None
    
    threshold = params.get("prediction", {}).get("threshold", 0.54)
    
    return {
        "model": model,
        "transformer": transformer,
        "threshold": threshold
    }
import pytest
from src.utils.config import load_params

def test_load_params():
    params = load_params()
    assert isinstance(params, dict)
    assert "project" in params
    assert "prediction" in params
    assert params["prediction"]["threshold"] == 0.54
    assert params["models"]["champion"] == "catboost_lightgbm_soft_voting"
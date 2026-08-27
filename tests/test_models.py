from src.models.candidates import get_candidate_models

def test_get_candidate_models():
    models = get_candidate_models()
    assert "catboost" in models
    assert "xgboost" in models
    assert "lightgbm" in models
    assert "ensemble" in models
    
    # Verify soft voting structure
    ensemble = models["ensemble"]
    assert ensemble.voting == "soft"
    assert len(ensemble.estimators) == 2
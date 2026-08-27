import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from src.utils.config import load_params

def get_candidate_models(y_train=None):
    params = load_params()
    model_params = params["models"]
    
    # Calculate scale_pos_weight for imbalanced classification if y_train is provided
    scale_pos_weight = 1.0
    if y_train is not None:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        if pos_count > 0:
            scale_pos_weight = float(neg_count / pos_count)

    catboost_model = CatBoostClassifier(
        iterations=model_params["catboost"].get("n_estimators", 500),
        depth=model_params["catboost"].get("max_depth", 6),
        learning_rate=model_params["catboost"].get("lr", 0.03),
        auto_class_weights="Balanced",
        verbose=0
    )

    xgboost_model = XGBClassifier(
        n_estimators=model_params["xgboost"].get("n_estimators", 500),
        max_depth=model_params["xgboost"].get("max_depth", 6),
        learning_rate=model_params["xgboost"].get("lr", 0.03),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss"
    )

    lightgbm_model = LGBMClassifier(
        n_estimators=model_params["lightgbm"].get("n_estimators", 500),
        max_depth=model_params["lightgbm"].get("max_depth", 6),
        learning_rate=model_params["lightgbm"].get("lr", 0.03),
        class_weight="balanced",
        verbose=-1
    )

    ensemble_model = VotingClassifier(
        estimators=[
            ('catboost', catboost_model),
            ('xgboost', xgboost_model)
        ],
        voting='soft'
    )

    return {
        "catboost": catboost_model,
        "xgboost": xgboost_model,
        "lightgbm": lightgbm_model,
        "ensemble": ensemble_model
    }
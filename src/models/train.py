import os
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
from src.data.load_and_validate import load_and_process_data
from src.models.candidates import get_candidate_models
from src.features.transformers import CreditRiskFeatureEngineer
from src.utils.config import load_params


def train_and_evaluate():
    """
    End-to-end training pipeline.
    """
    params = load_params()
    threshold = params["prediction"]["threshold"]

    # Set up local MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-experimentation")

    print("Loading and preprocessing raw data for training...")
    # Returns RAW feature splits — no transformation applied yet
    X_train_raw, X_test_raw, y_train, y_test = load_and_process_data()

    # ── Transformer lifecycle ──────────────────────────────────────────────────
    # Fit exclusively on raw X_train to avoid data leakage and the double-fit bug.
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting CreditRiskFeatureEngineer on raw training data...")
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(X_train_raw)

    print("Transforming train and test splits...")
    X_train = transformer.transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)

    joblib.dump(transformer, models_dir / "feature_transformer.pkl")
    print("Saved feature_transformer.pkl to models/ — consistent with training data.")
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Feature names: {list(X_train.columns)}")

    candidates = get_candidate_models(y_train=y_train)

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"{name}_training"):
            print(f"\n--- Training {name.upper()} ---")
            model.fit(X_train, y_train)

            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            y_pred = (y_proba >= threshold).astype(int)

            roc_auc = roc_auc_score(y_test, y_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("decision_threshold", threshold)

            # Log model with trusted types allowed for skops validation
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"{name}_model",
                serialization_format="cloudpickle"
            )

            # Save the champion ensemble locally for FastAPI serving
            if name == "ensemble":
                joblib.dump(model, models_dir / "champion_model.pkl")
                print("Saved champion_model.pkl to models/ for FastAPI serving.")

            print(f"{name.upper()} Results (Threshold: {threshold}):")
            print(f"  ROC-AUC: {roc_auc:.6f}")
            print(f"  PR-AUC:  {pr_auc:.6f}")
            print(classification_report(y_test, y_pred))

    print("\nModel training, local artifact export, and MLflow logging completed.")


if __name__ == "__main__":
    train_and_evaluate()
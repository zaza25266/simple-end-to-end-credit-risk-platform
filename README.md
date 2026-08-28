# End-to-End Credit Risk Prediction Platform

An end-to-end machine learning project for predicting **serious delinquency within two years** using the Give Me Some Credit dataset.

The project covers data validation, feature engineering, model training, experiment tracking, model serving and prediction telemetry.

### Key Features

* **Data validation:** Pandera
* **Preprocessing & feature engineering:** Scikit-learn + custom transformers
* **Models:** CatBoost, XGBoost, LightGBM
* **Ensemble:** VotingClassifier
* **Experiment tracking:** MLflow
* **Configuration:** YAML-based parameters
* **Explainability:** SHAP
* **Model serving:** FastAPI
* **Input validation:** Pydantic
* **Telemetry:** SQLAlchemy with SQLite
* **Rate limiting:** SlowAPI
* **Containerization:** Docker + Docker Compose
* **Testing:** Pytest
* **CI:** GitHub Actions
* **Deployment:** Render

### Architecture

```text
Raw Data
   ↓
Pandera Validation
   ↓
Preprocessing + Feature Engineering
   ↓
Model Training
   ├── CatBoost
   ├── XGBoost
   └── LightGBM
   ↓
Evaluation + Ensemble
   ↓
MLflow Tracking
   ↓
Model Artifacts
   ↓
FastAPI
   ↓
Prediction
   ↓
Database
```

### Configuration

Model and pipeline parameters are centralized in:

```text
config/params.yaml
```

Configuration includes dataset paths, preprocessing settings, model parameters, ensemble settings and prediction threshold.

### API

The FastAPI service provides:

* `GET /` — Web interface
* `GET /health` — API/model health
* `POST /predict` — Credit-risk prediction

Predictions return the estimated default probability, decision threshold and final prediction. Prediction information is also logged to the telemetry database.

### Project Structure

```text
├── config/          # YAML configuration
├── data/            # Dataset
├── models/          # Model artifacts
├── notebooks/       # EDA and experiments
├── scripts/         # Utility scripts
├── src/
│   ├── api/         # FastAPI + database
│   ├── data/        # Data loading + validation
│   ├── features/    # Feature engineering
│   ├── models/      # Training + explainability
│   ├── governance/  # Model approval logic
│   ├── monitoring/  # Prometheus + Evidently components
│   └── utils/       # Configuration
├── tests/
├── Dockerfile
├── docker-compose.yml
├── data.dvc
└── requirements.txt
```

### Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train:

```bash
python -m src.models.train
```

Run API:

```bash
uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

### Deployment

The API is deployed on Render:

**Live Demo:** [Credit Risk Platform](https://end-to-end-credit-risk-platform.onrender.com)

### Current Status

The core **ML → API → prediction telemetry** workflow is implemented.

The repository also contains Prometheus, Evidently, governance and PostgreSQL/Docker components as building blocks for a more complete production MLOps system. These components are **not currently presented as a fully automated production monitoring/retraining platform**.
# simple-end-to-end-credit-risk-platform

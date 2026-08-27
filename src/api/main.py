import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.schemas import LoanApplicationRequest, PredictionResponse
from src.api.dependencies import get_production_artifacts
from src.api.database import init_db, SessionLocal, PredictionTelemetry

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Credit Risk API",
    description="Scoring engine with telemetry and rate limiting.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

from fastapi.responses import FileResponse
import os

@app.get("/")
async def read_root():
    # Point this to where your App.html is stored in your project structure
    return FileResponse("src/App.html")

@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/health", tags=["Lifecycle"])
@limiter.limit("10/minute")
def health_check(request: Request):
    artifacts = get_production_artifacts()
    return {
        "status": "healthy" if artifacts["model"] is not None else "degraded_missing_model",
        "threshold": artifacts["threshold"],
        "model_loaded": artifacts["model"] is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("30/minute")
def predict_credit_risk(payload: LoanApplicationRequest, request: Request):
    artifacts = get_production_artifacts()
    model       = artifacts["model"]
    transformer = artifacts["transformer"]
    threshold   = artifacts["threshold"]

    if model is None or transformer is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run the training pipeline first.",
        )

    # by_alias=True is required — the transformer expects hyphenated column names
    # like "NumberOfTime30-59DaysPastDueNotWorse", not the underscore Python attr.
    input_data = pd.DataFrame([payload.model_dump(by_alias=True)])

    try:
        transformed = transformer.transform(input_data)
        probability = float(model.predict_proba(transformed)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

    decision = "Flagged for Default Risk" if probability >= threshold else "Approved"

    # Best-effort telemetry — a DB failure here must never kill a prediction.
    try:
        db = SessionLocal()
        db.add(PredictionTelemetry(
            revolving_utilization=payload.RevolvingUtilizationOfUnsecuredLines,
            age=payload.age,
            debt_ratio=payload.DebtRatio,
            monthly_income=payload.MonthlyIncome,
            default_probability=probability,
            prediction_decision=decision,
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return PredictionResponse(
        default_probability=round(probability, 4),
        decision_threshold=threshold,
        prediction=decision,
        risk_score_percentage=round(probability * 100, 2),
    )

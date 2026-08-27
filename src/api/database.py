import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

logger = logging.getLogger(__name__)

_DEFAULT_SQLITE = (
    "sqlite:///"
    + str(Path(__file__).resolve().parents[3] / "telemetry.db")
)
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_SQLITE)

# SQLite needs connect_args={"check_same_thread": False} for use with FastAPI
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionTelemetry(Base):
    """
    Database model to log live inference requests, feature payloads,
    and model outputs for monitoring and continuous retraining pipelines.
    """
    __tablename__ = "prediction_telemetry"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    revolving_utilization = Column(Float)
    age = Column(Integer)
    debt_ratio = Column(Float)
    monthly_income = Column(Float)
    default_probability = Column(Float)
    prediction_decision = Column(String(50))


def init_db():
    """Creates telemetry database tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        db_label = "SQLite (local)" if DATABASE_URL.startswith("sqlite") else "PostgreSQL"
        logger.info(f"Telemetry database schema initialised successfully ({db_label}).")
    except Exception as e:
        logger.warning(f"Database initialisation skipped or deferred: {e}")
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from src.utils.config import load_params

# Column order must match exactly what the model was trained on.
# Changing this list means you need to retrain.
FEATURE_ORDER = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    # engineered
    "EstimatedTotalDebt",
    "TotalDelinquencies",
    "CreditUtilizationPerLine",
    "IsYoungAdult",
    "IsSenior",
]


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Handles imputation and feature engineering for the credit risk model.
    Always outputs columns in FEATURE_ORDER so CatBoost gets them in the
    right positions.
    """

    def __init__(self, n_neighbors: int = None):
        params = load_params()
        self.n_neighbors = n_neighbors or params.get("preprocessing", {}).get(
            "knn_imputer_n_neighbors", 5
        )
        self.knn_imputer = None
        self.median_income_ = None
        self.median_dependents_ = None

    def fit(self, X: pd.DataFrame, y=None):
        impute_cols = ["MonthlyIncome", "NumberOfDependents"]
        available_cols = [c for c in impute_cols if c in X.columns]

        if available_cols and len(X) > 1:
            self.knn_imputer = KNNImputer(n_neighbors=min(self.n_neighbors, len(X) - 1))
            self.knn_imputer.fit(X[available_cols])

        if "MonthlyIncome" in X.columns:
            self.median_income_ = X["MonthlyIncome"].median()
        if "NumberOfDependents" in X.columns:
            self.median_dependents_ = X["NumberOfDependents"].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in ["Unnamed: 0", "Id"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        if self.knn_imputer is not None:
            if hasattr(self.knn_imputer, "feature_names_in_"):
                knn_cols = list(self.knn_imputer.feature_names_in_)
            else:
                knn_cols = [
                    c for c in ["MonthlyIncome", "NumberOfDependents"] if c in X.columns
                ][: self.knn_imputer.n_features_in_]

            cols_to_impute = [c for c in knn_cols if c in X.columns]
            if cols_to_impute:
                X[cols_to_impute] = self.knn_imputer.transform(X[cols_to_impute])

        # Median fallback for anything KNN didn't cover
        if "MonthlyIncome" in X.columns:
            X["MonthlyIncome"] = X["MonthlyIncome"].fillna(self.median_income_ or 0)
        if "NumberOfDependents" in X.columns:
            _med = getattr(self, "median_dependents_", None)
            X["NumberOfDependents"] = X["NumberOfDependents"].fillna(_med or 0)

        # Engineered features
        if "MonthlyIncome" in X.columns and "DebtRatio" in X.columns:
            X["EstimatedTotalDebt"] = X["MonthlyIncome"] * X["DebtRatio"]

        delinq_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ]
        present = [c for c in delinq_cols if c in X.columns]
        if present:
            X["TotalDelinquencies"] = X[present].sum(axis=1)

        if "RevolvingUtilizationOfUnsecuredLines" in X.columns and "NumberOfOpenCreditLinesAndLoans" in X.columns:
            X["CreditUtilizationPerLine"] = (
                X["RevolvingUtilizationOfUnsecuredLines"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1)
            )

        if "age" in X.columns:
            X["IsYoungAdult"] = (X["age"] < 30).astype(int)
            X["IsSenior"] = (X["age"] > 65).astype(int)

        # Fail loudly if any expected column is missing rather than letting
        # CatBoost crash with a confusing positional error downstream.
        missing = [c for c in FEATURE_ORDER if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing columns after feature engineering: {missing}. "
                f"Make sure the input DataFrame has all 10 raw features."
            )

        return X[FEATURE_ORDER]
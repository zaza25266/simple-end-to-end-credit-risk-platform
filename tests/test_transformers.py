import pandas as pd
import pytest
from src.features.transformers import CreditRiskFeatureEngineer, FEATURE_ORDER


def _make_raw_df(**overrides):
    """
    Helper that returns a minimal valid raw-feature DataFrame.
    All 10 raw columns required by the transformer are present.
    """
    base = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 40,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTimes90DaysLate": 1,  
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_feature_engineer_transform_produces_correct_columns():
    """Transformer must return exactly the 15 canonical features in FEATURE_ORDER."""
    df = _make_raw_df()
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(df)
    transformed = transformer.transform(df)

    assert list(transformed.columns) == FEATURE_ORDER
    assert len(transformed.columns) == 15


def test_feature_engineer_derived_features_values():
    """Spot-check the computed values of every engineered feature."""
    df = _make_raw_df(
        MonthlyIncome=5000.0,
        DebtRatio=0.4,
        # 30-59=2, 90+=1, 60-89=0 → TotalDelinquencies = 3
        **{"NumberOfTime30-59DaysPastDueNotWorse": 2,
           "NumberOfTimes90DaysLate": 1,
           "NumberOfTime60-89DaysPastDueNotWorse": 0}
    )
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(df)
    transformed = transformer.transform(df)

    assert transformed["EstimatedTotalDebt"].iloc[0] == pytest.approx(2000.0)
    assert transformed["TotalDelinquencies"].iloc[0] == 3   # 2+1+0
    assert "CreditUtilizationPerLine" in transformed.columns
    assert "IsYoungAdult" in transformed.columns
    assert "IsSenior" in transformed.columns


def test_feature_engineer_imputation():
    """NaN MonthlyIncome and NumberOfDependents must be imputed without error."""
    df = _make_raw_df(MonthlyIncome=None, NumberOfDependents=None)
    transformer = CreditRiskFeatureEngineer()
    transformer.fit(df)
    transformed = transformer.transform(df)

    assert not transformed["MonthlyIncome"].isnull().any()
    assert not transformed["NumberOfDependents"].isnull().any()


def test_feature_engineer_missing_required_column_raises():
    """
    Transformer must raise a ValueError — not a silent CatBoostError — when
    a required raw column is absent from the input DataFrame.
    """
    df = _make_raw_df()
    # Remove a required raw column
    df = df.drop(columns=["NumberOfTime60-89DaysPastDueNotWorse"])

    transformer = CreditRiskFeatureEngineer()
    transformer.fit(df)   # fit on partial data for test purposes

    with pytest.raises(ValueError, match="required columns are absent"):
        transformer.transform(df)


def test_feature_order_is_enforced():
    """Columns must be in FEATURE_ORDER regardless of input column order."""
    df = _make_raw_df()
    # Shuffle the input columns
    df = df[list(reversed(df.columns))]

    transformer = CreditRiskFeatureEngineer()
    transformer.fit(df)
    transformed = transformer.transform(df)

    assert list(transformed.columns) == FEATURE_ORDER
import pytest
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from src.data.validation import validate_raw_data

def test_validate_raw_data_success():
    valid_df = pd.DataFrame([{
        "SeriousDlqin2yrs": 0,
        "RevolvingUtilizationOfUnsecuredLines": 0.1,
        "age": 30,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.2,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfDependents": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0
    }])
    validated = validate_raw_data(valid_df)
    assert isinstance(validated, pd.DataFrame)

def test_validate_raw_data_failure():
    # Invalid age (> 120) and invalid target value (2)
    invalid_df = pd.DataFrame([{
        "SeriousDlqin2yrs": 2, 
        "RevolvingUtilizationOfUnsecuredLines": -0.5,
        "age": 150,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.2,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfDependents": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0
    }])
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_raw_data(invalid_df)